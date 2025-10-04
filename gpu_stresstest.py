#!/usr/bin/env python3
import torch
import time

def stress_gpu(matrix_size=8192, iters=None):
    """
    Continuously multiply two large matrices on the GPU.
    - matrix_size: controls how big the matrices are (8K×8K uses ~256 GB of memory ops per multiply).
    - iters: if None, runs forever until you Ctrl+C.
    """
    assert torch.cuda.is_available(), "CUDA-capable GPU not found"
    device = torch.device('cuda')
    # allocate two random matrices once:
    a = torch.randn((matrix_size, matrix_size), device=device)
    b = torch.randn((matrix_size, matrix_size), device=device)

    print(f"Allocated {matrix_size}×{matrix_size} matrices on {device}. Starting stress test…")
    count = 0
    try:
        while True:
            # c = a @ b
            c = torch.matmul(a, b)
            # ensure all kernels finish before next iteration
            torch.cuda.synchronize()
            count += 1
            if iters and count >= iters:
                break
    except KeyboardInterrupt:
        pass

    print(f"Completed {count} multiplies. Exiting.")

if __name__ == "__main__":
    # you can pass size and iterations on the command line if you like:
    # e.g. python stress.py 4096 100
    import sys
    size = int(sys.argv[1]) if len(sys.argv) >= 2 else 8192
    its  = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    stress_gpu(matrix_size=size, iters=its)
