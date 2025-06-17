# Day 002 (2025/6/17)
- Done mojo [gpu puzzle p20](https://builds.modular.com/puzzles/puzzle_20/backward_pass.html): fused-layernorm-linear-op backward pass
- Shamelessly copy-pasted most code for learning the mojo language itself. I feel it is almost impossible to know what tools/functions it requires without taking a look beforehand.

## What I learned
- [Atomic](https://docs.modular.com/mojo/stdlib/os/atomic/Atomic/) addition in mojo
    1. Calculate the pointer of the element by `.ptr` and `.offset()`
    2. `Atomic[dtype].fetch_add()` to perform atomic in-place addition. It can return the original value, but we don't need it here so we can handle it with _ like what we do in python

## TODO
- It's pretty annoying to spam `rebind[dtype]()` everywhere. I wonder whether there's another way to cast the values without hurting the performance.