# Warmup D-2 (2025/6/14)
- Done [mojo puzzle](https://builds.modular.com/puzzles/introduction.html) p11 - p14 ([my solution](https://github.com/Tcc0403/mojo-gpu-puzzles/tree/solution/problems))

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

In Puzzle 13, when performing a block-wise sum, the tutorial says it has a potential race condition:

```mojo
stride = TPB // 2
while stride > 0:
    if local_i < stride:
        cache[local_i] += cache[local_i + stride]
    barrier()
    stride //= 2
```

> Note: This implementation has a potential race condition where threads simultaneously read from and write to shared memory during the same iteration. A safer approach would separate the read and write phases:

```mojo
stride = TPB // 2
while stride > 0:
    var temp_val: output.element_type = 0
    if local_i < stride:
        temp_val = cache[local_i + stride]  # Read phase
    barrier()
    if local_i < stride:
        cache[local_i] += temp_val  # Write phase
    barrier()
    stride //= 2
```

Personally I believe it never happens since
1. we already limit active threads with `local_i < stride` and 
2. the indices they access are `cache[local_i]` `cache[local_i + stride]` which are always different across threads. 

There should be no read-write hazards, but surely we can make it safer at the cost of extra synchronization `barrier()` each loop.

## Tiled MatMul

Mojo has high-level APIs, maintaining the performance benefits of tiling while providing cleaner abstractions. ([example in Puzzle 14](https://builds.modular.com/puzzles/puzzle_14/tiled.html#solution-idiomatic-layouttensor-tiling))

### LayoutTensor tile API

Directly call `.tile[]()` and it can return the specific tile given `[TILE_ROW_DIM, TILE_COL_DIM]` and `(tile_position_in_row, tile_position_in_col)`. 

For instance, `a.tile[16, 32](2, 3)` will return a $16 \times 32$ tile that is located at position (2, 3) from block-wise perspective.

Another example from Puzzle 14:
```mojo
out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
a_tile = a.tile[TPB, TPB](block_idx.y, idx)
b_tile = b.tile[TPB, TPB](idx, block_idx.x)
```

### Asynchronous memory operations

`copy_dram_to_sram_async[]()` ([doc](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/))

Passing destination and source `LayoutTensor` and everything is done, no need to calculate indices by hands. 

`[thread_layout=...]` determines how the workload is distributed among threads. 

It also provides advanced features such as `swizzle`, `fill`, `eviction_policy`, etc...

Example from Puzzle 14:
```mojo
alias load_a_layout = Layout.row_major(1, TPB) # Each thread loads a slice of a row
alias load_b_layout = Layout.row_major(TPB, 1) # Each thread loads a slice of a column

copy_dram_to_sram_async[thread_layout=load_a_layout](a_shared, a_tile)
copy_dram_to_sram_async[thread_layout=load_b_layout](b_shared, b_tile)
async_copy_wait_all()
```

## TODO
- Try high-level APIs and figure out what exactly `thread_layout` works in `copy_dram_to_sram_async[]()` 
- Benchmark Tiled MatMul with those APIs and see the performance