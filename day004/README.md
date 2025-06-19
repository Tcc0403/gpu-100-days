# Day 004 (2025/6/19)
- Done mojo [gpu puzzle p21](https://builds.modular.com/puzzles/puzzle_21/puzzle_21.html) ([my solution](https://github.com/Tcc0403/mojo-gpu-puzzles))

## What I learned
### Mojo's `elementwise` function

Think in SIMD, not threads

Define your SIMD function and perform SIMD vecotrization automatically.

Take p21 for example to see the `elementwise` + SIMD structure: (Only list parts of parameters for brevity)

```mojo
fn elementwise_add[simd_width: Int, rank: Int, size: Int](output, a, b, ctx: DeviceContext):
    # Define a SIMD function
    @parameter
    @always_inline
    fn add[
        simd_width: Int, rank: Int # Neccesary parameters
    ](
        indices: IndexList[rank]   # `elementwise()` can genereate indices automatically
    ) capturing -> None:
        # Just read the indices passed by `elementwise()`
        idx = indices[0]

        # Vectorized load with `.load[simd_width](idx, 0)`, (idx, 0) is (row index, column index)
        a_simd = a.load[simd_width](idx, 0)
        b_simd = b.load[simd_width](idx, 0)
        # Actual computation happens here, simply using `+` to perform addition with two SIMD vectors
        result = a_simd + b_simd
        # Vecotrized store
        output.store[simd_width](idx, 0, result)
    
    # Dispatch the defined function with `elementwise()`, it can automatically perform SIMD vectorization without traditional indexing, guards
    elementwise[add, SIMD_WIDTH, target="gpu"](a.size(), ctx)
```

## TODO
- Learn more about `vecotrize` + `elementwise` usage
- See how benchmark works in Mojo