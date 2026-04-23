# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FSDP2 Migration Notes:
#   - FSDP1's FSDP(module, ...) wrapping replaced by fully_shard(module, ...) in-place call
#   - MixedPrecision -> MixedPrecisionPolicy (from torch.distributed._composable.fsdp)
#   - CPUOffload -> CpuOffloadPolicy / OffloadPolicy
#   - ShardingStrategy enum dropped; use reshard_after_forward=True/False
#   - flat_param / _all_handles removed; parameters are now regular DTensors
#   - offload/load helpers rewritten to iterate named_parameters() directly
#   - get_fsdp_wrap_policy returns a callable used by _apply_fsdp2 helper

import gc
from collections import defaultdict
from functools import partial
from typing import Callable, Union

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as _traversal_utils
from torch import nn
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from torch.optim import Optimizer
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name


def get_init_fn(model: nn.Module, device: Union[str, torch.device]) -> Callable[[nn.Module], None]:
    """Return a param-init function for non-rank-0 workers in rank-0-init mode.

    FSDP2 uses the same pattern as FSDP1 here: rank-0 loads real weights,
    other ranks allocate empty tensors that are then broadcast via
    sync_module_states (handled by fully_shard's sync_module_states=True).
    """
    param_occurrence = defaultdict(int)
    for _, param in model.named_parameters(remove_duplicate=False):
        param_occurrence[param] += 1

    duplicated_params = {param for param in param_occurrence.keys() if param_occurrence[param] > 1}
    materialized_params = {}

    def init_fn(module: nn.Module):
        for name, param in module.named_parameters(recurse=False):
            if param in duplicated_params:
                module._parameters[name] = materialized_params.setdefault(
                    param,
                    nn.Parameter(
                        torch.empty_like(param.data, device=device),
                        requires_grad=param.requires_grad,
                    ),
                )
            else:
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param.data, device=device),
                    requires_grad=param.requires_grad,
                )

    return init_fn


def get_fsdp_wrap_policy(model: PreTrainedModel, is_lora_model: bool = False):
    """Return an FSDP auto-wrap policy callable.

    The returned policy is the same structure as FSDP1's, and is consumed by
    _apply_fsdp2() below via transformer_auto_wrap_policy + optional lambda
    policy for LoRA leaf modules.

    Args:
        model: The pre-trained model.
        is_lora_model: Whether LoRA is applied (adds a leaf-level lambda policy).
    """
    transformer_cls_to_wrap = set()
    for module_name in model._no_split_modules:
        transformer_cls = get_module_class_from_name(model, module_name)
        if transformer_cls is None:
            raise Exception(f"Cannot find {module_name} in pretrained model.")
        transformer_cls_to_wrap.add(transformer_cls)

    policies = []

    if is_lora_model:
        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
        lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    transformer_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap
    )
    policies.append(transformer_policy)

    auto_wrap_policy = partial(_or_policy, policies=policies)
    return auto_wrap_policy


def apply_fsdp2(
    model: nn.Module,
    device_mesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy=None,
    reshard_after_forward: bool = True,
    wrap_policy=None,
) -> nn.Module:
    """Apply FSDP2 (fully_shard) recursively to a model.

    FSDP2 is a composable API: ``fully_shard`` is called on individual
    sub-modules bottom-up, then on the root.  This replaces the single
    ``FSDP(model, auto_wrap_policy=...)`` call of FSDP1.

    Args:
        model: Root nn.Module to shard.
        device_mesh: DeviceMesh (shape (fsdp,) for FSDP, (ddp, fsdp) for HSDP).
        mp_policy: Mixed-precision policy.
        offload_policy: Optional CpuOffloadPolicy; pass None to disable.
        reshard_after_forward: Equivalent to FULL_SHARD (True) or SHARD_GRAD_OP (False).
        wrap_policy: Callable(module) -> bool deciding whether to shard a submodule.
                     Typically the result of get_fsdp_wrap_policy().
    """
    fsdp_kwargs = dict(
        mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
    )

    # Bottom-up application: shard eligible child modules before the root
    for name, submodule in reversed(list(model.named_modules())):
        if submodule is model:
            continue  # root handled last
        if wrap_policy is not None and not wrap_policy(submodule):
            continue
        fully_shard(submodule, **fsdp_kwargs)

    # Shard the root module
    fully_shard(model, **fsdp_kwargs)
    return model


# ---------------------------------------------------------------------------
# Manual param offload helpers
# ---------------------------------------------------------------------------
# FSDP2 no longer exposes _all_handles / flat_param.  Parameters are regular
# DTensors sharded across the device mesh.  We iterate named_parameters() and
# move the local shards to CPU / CUDA as needed.
# ---------------------------------------------------------------------------

@torch.no_grad()
def offload_fsdp_model(model: nn.Module, empty_cache: bool = True) -> None:
    """Move all FSDP2-sharded parameters to CPU (manual offload)."""
    for param in model.parameters():
        if param.data.device.type == "cuda":
            param.data = param.data.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_model(model: nn.Module, empty_cache: bool = True) -> None:
    """Move all FSDP2-sharded parameters back to CUDA."""
    for param in model.parameters():
        if param.data.device.type == "cpu":
            param.data = param.data.to("cuda", non_blocking=True)

    if empty_cache:
        gc.collect()


@torch.no_grad()
def offload_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True) -> None:
    """Move optimizer state tensors to CPU."""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_optimizer(optimizer: Optimizer, empty_cache: bool = True) -> None:
    """Move optimizer state tensors back to CUDA."""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cuda", non_blocking=True)

    if empty_cache:
        gc.collect()


# ---------------------------------------------------------------------------
# Submodule-level offload helpers (used by sharding_manager for LoRA layers)
# ---------------------------------------------------------------------------

@torch.no_grad()
def offload_fsdp_submodule(module: nn.Module, empty_cache: bool = True) -> None:
    """Move a specific submodule's parameters to CPU."""
    for param in module.parameters():
        if param.data.device.type == "cuda":
            param.data = param.data.to("cpu", non_blocking=True)

    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp_submodule(module: nn.Module, empty_cache: bool = True) -> None:
    """Move a specific submodule's parameters back to CUDA."""
    for param in module.parameters():
        if param.data.device.type == "cpu":
            param.data = param.data.to("cuda", non_blocking=True)

    if empty_cache:
        gc.collect()