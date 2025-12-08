import cupy as cp
import numpy as np
import time


def _gflops(n: int, seconds: float) -> float:
    """Compute GFLOPs/s for an NxN matmul (2 * N^3 ops)."""

    if seconds <= 0:
        return float("inf")
    return (2 * n**3) / (seconds * 1e9)


def test_cuda_cupy():
    print("Checking CUDA availability...")
    try:
        if not cp.cuda.is_available():
            print("CUDA is NOT available.")
            return
    except Exception as e:
        print(f"Error checking CUDA availability: {e}")
        return

    print("CUDA is available!")

    # Get device info
    try:
        device_id = cp.cuda.Device().id
        # Note: getDeviceProperties returns a dictionary-like object in recent cupy versions or struct
        # We'll just print the ID to be safe across versions, or try to get name
        print(f"Using Device ID: {device_id}")

        # Attempt to get more info if possible
        try:
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = props['name'].decode('utf-8')
            print(f"Device Name: {name}")
        except Exception:
            pass

    except Exception as e:
        print(f"Could not retrieve device info: {e}")

    # Matrix size
    N = 2000

    print(f"\nPerforming matrix multiplication (size {N}x{N})...")

    # Create random matrices on GPU
    print("Allocating and computing on GPU...")
    start_gpu = time.time()
    a_gpu = cp.random.rand(N, N, dtype=cp.float32)
    b_gpu = cp.random.rand(N, N, dtype=cp.float32)
    _ = cp.matmul(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
    end_gpu = time.time()

    gpu_time = end_gpu - start_gpu
    print(f"GPU time: {gpu_time:.4f} seconds")

    # Verify with CPU (NumPy)
    # We use a smaller size for verification to avoid waiting too long on CPU if N is huge
    N_verify = 500
    print(f"\nVerifying correctness with smaller matrix ({N_verify}x{N_verify}) on CPU...")

    a_gpu_small = cp.random.rand(N_verify, N_verify, dtype=cp.float32)
    b_gpu_small = cp.random.rand(N_verify, N_verify, dtype=cp.float32)

    # GPU calc
    c_gpu_small = cp.matmul(a_gpu_small, b_gpu_small)
    c_gpu_result = cp.asnumpy(c_gpu_small)

    # CPU calc
    start_cpu = time.time()
    a_cpu = cp.asnumpy(a_gpu_small)
    b_cpu = cp.asnumpy(b_gpu_small)
    c_cpu = np.matmul(a_cpu, b_cpu)
    end_cpu = time.time()

    cpu_time = end_cpu - start_cpu
    print(f"CPU time ({N_verify}x{N_verify}): {cpu_time:.4f} seconds")

    # Check difference
    diff = np.linalg.norm(c_cpu - c_gpu_result) / (N_verify * N_verify)
    print(f"Mean difference between CPU and GPU result: {diff:.8f}")

    if diff < 1e-4:
        print("\nSUCCESS: CuPy test passed!")
    else:
        print("\nWARNING: Large difference detected.")


def benchmark_matmul(sizes=(512, 1024, 2048, 4096), repeats=3):
    """Benchmark CPU vs GPU matmul across sizes to show speedup."""

    if not cp.cuda.is_available():
        print("CUDA is NOT available; skipping benchmark.")
        return

    device_id = cp.cuda.Device().id
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    print(f"\nBenchmarking on GPU {device_id}: {props['name'].decode('utf-8')}")
    print(f"Repeats per size: {repeats}\n")

    # Warmup to trigger kernel compilation and allocator setup.
    _ = cp.matmul(cp.ones((16, 16), dtype=cp.float32), cp.ones((16, 16), dtype=cp.float32))
    cp.cuda.Stream.null.synchronize()

    header = f"{'N':>6} | {'CPU s':>8} | {'CPU GF/s':>9} | {'GPU s':>8} | {'GPU GF/s':>9} | {'Speedup':>7}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        # CPU timing
        cpu_times = []
        for _ in range(repeats):
            a_cpu = np.random.rand(n, n).astype(np.float32)
            b_cpu = np.random.rand(n, n).astype(np.float32)
            t0 = time.perf_counter()
            _ = np.matmul(a_cpu, b_cpu)
            t1 = time.perf_counter()
            cpu_times.append(t1 - t0)
        cpu_time = min(cpu_times)
        cpu_gflops = _gflops(n, cpu_time)

        # GPU timing
        gpu_times = []
        for _ in range(repeats):
            a_gpu = cp.random.rand(n, n, dtype=cp.float32)
            b_gpu = cp.random.rand(n, n, dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            _ = cp.matmul(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            gpu_times.append(t1 - t0)
        gpu_time = min(gpu_times)
        gpu_gflops = _gflops(n, gpu_time)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(f"{n:6d} | {cpu_time:8.4f} | {cpu_gflops:9.1f} | {gpu_time:8.4f} | {gpu_gflops:9.1f} | {speedup:7.2f}x")


if __name__ == "__main__":
    try:
        test_cuda_cupy()
        benchmark_matmul()
    except ImportError:
        print("Error: 'cupy' is not installed. Please install it via pip or conda.")
    except Exception as e:
        print(f"An error occurred: {e}")
