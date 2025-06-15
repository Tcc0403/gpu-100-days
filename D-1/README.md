# Warmup D-1 (2025/6/15)
- Done [mojo puzzle](https://builds.modular.com/puzzles/introduction.html) p15 & p16 ([my solution](https://github.com/Tcc0403/mojo-gpu-puzzles/tree/solution/problems))

## What I learned
- Using MAX Graph to custom ops to call mojo kernel in Python 
    1. Write a kernel in mojo
    2. Custom op registration
    3. Packaging custom ops
    4. Python integration with MAX Graph

## Pitfalls
- In Puzzle 16, `uv run poe p16-test-kernels` somehow failed to load shared libraries: libMojoJupyter.so, but `pixi run p16-test-kernel` works. 
[`pixi`](https://pixi.sh/latest/) package manager has better compatiblity it seems. The original package manager for modular project is `Magic` which is built on top of `pixi`. [`Magic`](https://docs.modular.com/magic/) is now deprecated and replaced with `pixi`.
- In Puzzle 16, `ops.custom` is missing a required paramater `device`. It's been fixed by a recent [commit](https://github.com/modular/mojo-gpu-puzzles/commit/ba4acf07bce65cc49cab25554141812858375931) (partly, idk why the problem script wasn't patched). Remember to keep your mojo-gpu-puzzle repo up-to-date.