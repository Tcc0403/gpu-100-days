import os
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


@torch.inference_mode()
def bench_inference():
    from mojo_kernels.softmax import softmax

    torch.manual_seed(42)
    for seq_len in [1, 16, 128, 1024]:
        for hidden_size in [128, 512, 1024, 4096]:
            dtype = torch.float32
            x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")

            @torch.cuda.nvtx.range(f"mojo_softmax {seq_len=}, {hidden_size=}, {dtype=}")
            def mojo_fn():
                softmax(x)

            @torch.cuda.nvtx.range(
                f"torch_softmax {seq_len=}, {hidden_size=}, {dtype=}"
            )
            def torch_fn():
                _ = torch.softmax(x, dim=-1)

            mojo_ms = do_bench(mojo_fn, return_mode="median")
            torch_ms = do_bench(torch_fn, return_mode="median")
            io = x.numel() * x.element_size() * 2

            print(f"{seq_len=}, {hidden_size=}, {dtype=}")
            print("Latency:")
            print(f"torch: {torch_ms * 1e3:7.3f}us")
            print(f"Mojo : {mojo_ms * 1e3:7.3f}us")
            print("Throughput:")
            print(f"torch: {(io * 1e-9) / (torch_ms * 1e-3):7.3f}GB/s")
            print(f"Mojo : {(io * 1e-9) / (mojo_ms * 1e-3):7.3f}GB/s")

            print(f"Mojo Speedup: {torch_ms / mojo_ms:2.3f}x")

            print("-" * 10)

        print("=" * 20)


if __name__ == "__main__":
    bench_inference()
