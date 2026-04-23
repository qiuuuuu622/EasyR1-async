from contextlib import nullcontext
from typing import Literal, Optional, Union, cast

import numpy as np
import peft
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from peft import TaskType, get_peft_model
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.device_mesh import init_device_mesh
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp2_utils import (
    apply_fsdp2,
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import (
    AnyPrecisionAdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout import vLLMRollout, SGLangRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from .sharding_manager.fsdp_vllm_server import FSDPVLLMServerShardingManager


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._lora_rank = self.config.actor.model.lora.rank
        self._is_lora = self._lora_rank > 0

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:
            self._use_ref_param_offload = self.config.ref.offload.offload_params

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()

        # Main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            # Pure FSDP: 1-D mesh
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:
            # HSDP (replicated DDP outer, sharded FSDP inner): 2-D mesh
            self.device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // fsdp_size, fsdp_size),
                mesh_dim_names=("ddp", "fsdp"),
            )

        # Ulysses sequence-parallel mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # Validate / normalise batch-size config
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = config.global_batch_size // (world_size // config.ulysses_size)
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        # FSDP2's CpuOffloadPolicy handles offload internally; gradient accumulation
        # is still valid. The FSDP1 restriction no longer applies.

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        # ------------------------------------------------------------------
        # Tokenizer / processor (skipped for ref which reuses actor's)
        # ------------------------------------------------------------------
        if role != "ref":
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        # ------------------------------------------------------------------
        # dtype resolution
        # ------------------------------------------------------------------
        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        # ------------------------------------------------------------------
        # Select AutoClass
        # ------------------------------------------------------------------
        if role == "critic":
            AutoClass = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM

        # ------------------------------------------------------------------
        # Load weights (rank-0 init pattern unchanged)
        # ------------------------------------------------------------------
        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        model = cast(PreTrainedModel, model)
        model.tie_weights()

        if role == "ref":
            model.requires_grad_(False)

        # ------------------------------------------------------------------
        # LoRA
        # ------------------------------------------------------------------
        is_lora_model = self._is_lora and role == "actor"
        if is_lora_model:
            self.print_rank0("Applying LoRA to actor module")
            model.enable_input_require_grads()
            if model_config.lora.target_modules == "all-linear":
                target_modules = model_config.lora.target_modules
            else:
                target_modules = [
                    item.strip()
                    for item in model_config.lora.target_modules.split(",")
                    if item.strip()
                ]

            lora_config = peft.LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=model_config.lora.rank,
                lora_alpha=model_config.lora.alpha,
                target_modules=target_modules,
                exclude_modules=model_config.lora.exclude_modules,
            )
            model = get_peft_model(model, lora_config)
            for p in model.parameters():
                p.data = p.to(torch.bfloat16 if not p.requires_grad else torch_dtype)
        else:
            model = model.to(torch_dtype)

        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")

        # ------------------------------------------------------------------
        # FSDP2: build MixedPrecisionPolicy and optional CpuOffloadPolicy
        # ------------------------------------------------------------------
        mp_policy = MixedPrecisionPolicy(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            output_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
            cast_forward_inputs=True,
        )

        # CpuOffloadPolicy replaces FSDP1's CPUOffload; FSDP2 handles gradient
        # accumulation transparently so we remove the old restriction.
        if fsdp_config.enable_cpu_offload:
            offload_policy = CPUOffloadPolicy(offload_params=True)
        else:
            offload_policy = None

        # reshard_after_forward=True  ~ FULL_SHARD
        # reshard_after_forward=False ~ SHARD_GRAD_OP
        # HSDP (2-D mesh) is supported natively by passing the 2-D mesh.
        reshard_after_forward = fsdp_config.enable_full_shard

        wrap_policy = get_fsdp_wrap_policy(model, is_lora_model=is_lora_model)
        self.print_rank0(f"FSDP2 wrap policy: {wrap_policy}.")

        # sync_module_states for rank-0 init: pass device_id so FSDP2 can
        # broadcast parameters from rank-0 after materialisation.
        param_init_fn = None
        if fsdp_config.enable_rank0_init and self.rank != 0:
            param_init_fn = get_init_fn(model, device="cuda")

        if param_init_fn is not None:
            # Materialise empty params on non-0 ranks before apply_fsdp2
            for module in model.modules():
                param_init_fn(module)

        # apply_fsdp2 calls fully_shard bottom-up then on the root
        model = apply_fsdp2(
            model,
            device_mesh=self.device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
            wrap_policy=wrap_policy,
        )

        if fsdp_config.enable_rank0_init:
            # Broadcast rank-0 weights to all other ranks
            for param in model.parameters():
                dist.broadcast(param.data, src=0)

        print_gpu_memory_usage("After FSDP2 module init")

        # ------------------------------------------------------------------
        # Optimizer + LR scheduler (actor / critic only)
        # ------------------------------------------------------------------
        if role in ["actor", "critic"]:
            self.fsdp_module = model
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            if optim_config.lr_scheduler_type == "constant":
                self.lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                )
            elif optim_config.lr_scheduler_type == "cosine":
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=optim_config.training_steps,
                    min_lr_ratio=optim_config.min_lr_ratio,
                    num_cycles=0.5,
                )
            else:
                raise NotImplementedError(
                    f"LR scheduler type {optim_config.lr_scheduler_type} is not supported"
                )

            print_gpu_memory_usage("After optimizer init")

            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = model
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    # -----------------------------------------------------------------------
    # Rollout helpers (unchanged in logic, kept for completeness)
    # -----------------------------------------------------------------------

    def _build_rollout(self) -> None:
        rollout_server_urls = getattr(self.config.rollout, "server_urls", None)
        if rollout_server_urls:
            self._build_rollout_cluster_mode(rollout_server_urls)
            return

        rollout_server_url = getattr(self.config.rollout, "server_url", None)
        if rollout_server_url:
            self._build_rollout_server_mode(rollout_server_url)
            return

        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        lora_kwargs = (
            {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self._lora_rank}}
            if self._is_lora
            else {}
        )
        self.rollout = SGLangRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
            **lora_kwargs,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
        )
        print_gpu_memory_usage("After vllm init")

    def _build_rollout_cluster_mode(self, server_urls: list[str]) -> None:
        from .rollout.cluster.rollout_cluster_client import RolloutClusterClient

        self.print_rank0(
            f"[Cluster Mode] Connecting to {len(server_urls)} RolloutServers: {server_urls}"
        )
        rank = dist.get_rank()
        self.rollout_client = RolloutClusterClient(
            server_urls=server_urls,
            fsdp_module=self.fsdp_module,
            rank=rank,
        )
        self.rollout = None
        self.rollout_sharding_manager = FSDPVLLMServerShardingManager(
            rollout_client=self.rollout_client,
            use_param_offload=self._use_param_offload,
        )
        self.print_rank0(
            f"[Cluster Mode] RolloutClusterClient initialized with {len(server_urls)} servers."
        )

    def _build_rollout_server_mode(self, server_url: str) -> None:
        from .rollout.rollout_client import RolloutClient

        self.print_rank0(f"[Server Mode] Connecting to RolloutServer at {server_url}")
        rank = dist.get_rank()
        self.rollout_client = RolloutClient(
            server_url=server_url,
            fsdp_module=self.fsdp_module,
            rank=rank,
        )
        self.rollout = None
        self.rollout_sharding_manager = FSDPVLLMServerShardingManager(
            rollout_client=self.rollout_client,
            use_param_offload=self._use_param_offload,
        )
        self.print_rank0("[Server Mode] RolloutClient initialized.")

    # -----------------------------------------------------------------------
    # Registered worker methods
    # -----------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                padding_free=self.config.critic.padding_free,
                role="critic",
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )

        if self._has_ref:
            if self._is_lora:
                self.ref_fsdp_module = self.fsdp_module
            else:
                self._build_model_optimizer(
                    model_config=self.config.actor.model,
                    fsdp_config=self.config.ref.fsdp,
                    optim_config=None,
                    padding_free=self.config.ref.padding_free,
                    role="ref",
                )

        if self._has_actor:
            from .actor.dp_actor import DataParallelPPOActor

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:
            self._build_rollout()

        if self._has_ref:
            from .actor.dp_actor import DataParallelPPOActor

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_fsdp_module,
            )

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        if "uid" in self._cache and not np.all(data.non_tensor_batch["uid"] == self._cache["uid"]):
            self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            min_pixels = data.meta_info["min_pixels"]
            max_pixels = data.meta_info["max_pixels"]
            video_fps = data.meta_info["video_fps"]
            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}
            for index, multi_modal_data in zip(
                data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]
            ):
                if index not in multi_modal_inputs_cache:
                    images, videos = [], []
                    if "images" in multi_modal_data:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, min_pixels, max_pixels))
                    if "videos" in multi_modal_data:
                        for video in multi_modal_data["videos"]:
                            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

                    if images:
                        multi_modal_inputs = dict(
                            self.processor.image_processor(images=images, return_tensors="pt")
                        )
                    elif videos:
                        multi_modal_inputs = dict(
                            self.processor.image_processor(images=None, videos=videos, return_tensors="pt")
                        )
                    else:
                        multi_modal_inputs = {}

                    multi_modal_inputs_cache[index] = multi_modal_inputs

                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()

            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value)
                    for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_server_pool_status(self):
        if self.rollout is None:
            return self.rollout_client.get_pool_status()
        raise RuntimeError("get_server_pool_status is only available in server mode.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def submit_prompts(self, new_batch: DataProto, gen_batch: DataProto):
        assert self._has_rollout
        if self.rollout is None:
            self.rollout_sharding_manager.submit_prompts(new_batch, gen_batch)
        else:
            raise RuntimeError("submit_prompts is only available in server mode.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pull_samples(self, target_samples: int, min_weight_version: int = 0, timeout: float = 300.0):
        assert self._has_rollout
        if self.rollout is None:
            return self.rollout_sharding_manager.pull_samples(
                target_samples=target_samples,
                min_weight_version=min_weight_version,
                timeout=timeout,
            )
        else:
            raise RuntimeError("pull_samples is only available in server mode.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def push_weights_only(self, flush_cache: bool = False):
        if self.rollout is None:
            self.rollout_sharding_manager.load_vllm_and_sync_weights(flush_cache)
        else:
            raise RuntimeError("push_weights_only is only available in server mode.")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout

        meta_info = {
            "eos_token_id": (
                self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id
            ),
            "pad_token_id": (
                self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id
            ),
        }
        prompts.meta_info.update(meta_info)

        if self.rollout is None:
            output = self.rollout_client.generate(prompts)
            output = output.to("cpu")
            return output

        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"prox_log_probs": output},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # FSDP2: no _handle.reshard(); the reshard is driven by
        # reshard_after_forward set at construction time.  The explicit
        # FSDP1 workaround is removed.

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        adapter_ctx = self.ref_fsdp_module.disable_adapter() if self._is_lora else nullcontext()

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_ref_param_offload or (self._is_lora and self._use_param_offload):
            load_fsdp_model(self.ref_fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager, adapter_ctx:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # FSDP2: reshard handled automatically; no _handle.reshard() needed.

        if self._use_ref_param_offload or (self._is_lora and self._use_param_offload):
            offload_fsdp_model(self.ref_fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value)
                    for key, value in metrics.items()
                }
            )

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output