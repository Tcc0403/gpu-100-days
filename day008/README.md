# Day 008 (2025/6/23)
- Created a [directory](https://github.com/Tcc0403/gpu-100-days/tree/main/pytorch_custom_ops) for pytorch custom ops ([mojo doc](https://docs.modular.com/max/tutorials/custom-kernels-pytorch))
    - Set up environment with pixi
        - To enable CUDA, we need to add system-requirements in `pixi.toml' ([link](https://pixi.sh/dev/workspace/system_requirements/#using-cuda-in-pixi))
    - Added unit tests (pytest) for it

## Pitfalls
### Checking dtype is important!
#### Context

 I was trying to generate random input tensors to test the kernel correctness. Although it can sucessfully generate a grayscale image given a color image, it just kept running into Segmentation Fault when I passed my randomized tensors

#### What goes wrong

 The custom op defined in `grayscale.mojo` restricts input tensor's dtype `img_in.dtype` to be `uint8`. However, when I was generating input tensors with `torch.randint()`, I didn't pass the desired `dtype`, so it would [return `torch.uint64`](https://docs.pytorch.org/docs/stable/generated/torch.randint.html#:~:text=dtype%20(torch.dtype%2C%20optional)%20%E2%80%93%20if%20None%2C%20this%20function%20returns%20a%20tensor%20with%20dtype%20torch.int64.) instead, not `torch.uint8`!

#### What I can do better

- Always check the assumption we made, such as `shape`, `dtype`, `stride` and so on, before launching kernels. It can help us debugging by giving an informative error message instead of a brutal SEGFAULT ERROR.
- I haven't looked up where the SEGFAULT happened. Does mojo not automatically parse the input/output even we've given type hint?