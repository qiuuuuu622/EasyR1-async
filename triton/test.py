import triton
import triton.language as tl
import torch


@triton.jit
def _gemm_mantul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):

    pid = tl.program_id(0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    A_ptr = A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    B_ptr = B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):

        mask_k = offsets_k + k < K

        a = tl.load(A_ptr, mask=mask_m[:,None] & mask_k[None,:], other=0.0)
        b = tl.load(B_ptr, mask=mask_k[:,None] & mask_n[None,:], other=0.0)

        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

        A_ptr += BLOCK_SIZE_K * stride_ak
        B_ptr += BLOCK_SIZE_K * stride_bk

    C_ptr = C + offsets_m[:,None] * stride_cm + offsets_n[None,:] * stride_cn
    tl.store(C_ptr, acc, mask=mask_m[:,None] & mask_n[None,:])


def triton_matmul(A, B):
    M, K = A.shape
    K, N = B.shape
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    _gemm_mantul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return C
    



A = torch.randn((128, 128), device="cuda:1", dtype=torch.float32)
B = torch.randn((128, 128), device="cuda:1", dtype=torch.float32)

C_triton = triton_matmul(A, B)
C_torch = torch.matmul(A, B)

print(torch.allclose(C_triton, C_torch, atol=1e-5))
print((C_triton - C_torch).abs().max())