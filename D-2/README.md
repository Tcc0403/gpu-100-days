# Warmup D-2 (2025/6/14)
- Done [mojo puzzle](https://builds.modular.com/puzzles/introduction.html) p11 - p13 ([my solution](https://github.com/Tcc0403/mojo-gpu-puzzles/tree/solution/problems))

## Note
### Proper type inference with `var` in mojo
Variables ([doc](https://docs.modular.com/mojo/manual/variables))
```mojo
var local_sum: output.element_type = 0 # Using var allows type inference
```

### @parameter deocrator on an `if` or `for` statement
>The @parameter for construct unrolls at the beginning of compilation, which might explode the size of the program that still needs to be compiled, depending on the amount of code that's unrolled.
([doc](https://docs.modular.com/mojo/manual/decorators/parameter/))

Useful to unroll loop
Example: puzzle 11
```mojo
@parameter  # Unrolls loop at compile time since CONV is constant
for j in range(CONV):
    if local_i + j < SIZE:
        local_sum += shared_a[local_i + j] * shared_b[j]
```
### Host-side synchronization
`barrier()` is block-level synchronization

If we want to perform a synchronzation across blocks, we must do a host-side synchronization
`DeviceContext.synchronize()` ([doc](https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceContext/#synchronize))
```mojo
from gpu.host import DeviceContext

with DeviceContext() as ctx:
    ctx.synchronize()
```

### Race condition
Puzzle 13 says