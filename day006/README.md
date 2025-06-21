# Day 006 (2025/6/21)
- Done mojo gpu puzzles p22 p23 ([my solutions](https://github.com/Tcc0403/mojo-gpu-puzzles/tree/solution))


## What I learned
- Warp programming (SIMT execution model):
    - 32/64 (based on architecture) threads within a warp
    - SIMT: all threads within a warp execute the same instruction simultaneously.
        - If warp splits execution, some lanes have to wait. It's called **warp divergnece**
        - Warp efficiency is crucial for performance. 
    - Thread 0, Thread 1, ... in a warp can also be called lane 0, lane 1, ...
- Warp functions in mojo ([doc](https://docs.modular.com/mojo/stdlib/gpu/warp/#functions)) 
    - Reduction 
        - Common patterns such as `sum()`, `max()`, `min()`, `prefix_sum()` are provided
        - `reduce()` provides a generic warp-wide reduction operation using shuffle operations
            - For example: `reduce[shuffle_down, add](val)`
    - Communication ([link](https://builds.modular.com/puzzles/puzzle_23/puzzle_23.html#warp-communication-operations-in-mojo))
    - `broadcast(value)`: Share lane 0's value with all other lanes
    - `shuffle_down(value, offset)`: Get value from lane at higher index. 
        - For example: lane 0 gets lane (0 + offset)'s value, lane 1 gets lane (1 + offset)'s value, and so on
        - If lane_id + offset >= WARP_SIZE, the return value is undefined
    - `shuffle_high(value, offset)` and `shuffle_idx(value, lane)` is similar to `shuffle_down()`