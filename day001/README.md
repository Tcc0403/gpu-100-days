# Day 001 (2025/6/16)
- Done [mojo gpu puzzle](https://builds.modular.com/puzzles/introduction.html) p18-p20([my solutions](https://github.com/Tcc0403/mojo-gpu-puzzles/tree/solution/problems))
## What I learned
- PyTorch Custom Ops integration
    - Way simpler than MAX Graph
    - Binding with `torch.compile()`
- Fusing Layernorm and Linear
    - Making it correct is easy, each thread computes layernorm on one row, and a dot product with weight columns after can generate results.
    - Contraints:
        - The puzzle is designed for small problem size. It only works when hidden size is smaller than `THREADS_PER_BLOCK`.