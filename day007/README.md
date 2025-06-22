# Day 007 (2025/6/22)
- Done [mojo-gpu-puzzles p24](https://builds.modular.com/puzzles/puzzle_24/puzzle_24.html) ([my solution](https://github.com/Tcc0403/mojo-gpu-puzzles/blob/solution/solutions/p24/p24.mojo))

## What I learned
### Two advanced warp communication primitives
1. `shuffle_xor(value, mask)`
    - Lane N gets value from Lane (N XOR mask) <--> Lane (N XOR mask) gets value from Lane N
    - Butterfly patterns / reductions
    - Tree reduction: offset is power-of-2, which means we can set offset as mask and right shift 1 bit each iteration until 1
2. `prefix_sum(value)`
    - Hardware support (warp primitives)
    - Inclusive scan & exclusive scan

### Warp partition
`shuffle_xor()` + `prefix_sum()` can implement many useful parallel algorithms
#### Psuedo code
Step 1: Compute the current value belongs to left or right partition
```mojo
predicate_left = current < pivot ? 1 : 0
predicate_right = current >- pivot ? 1 : 0
```
Step 2 Prefix_sum(): Compute the indices of the elements with exclusive scan
```mojo
# Exclusive scan on predicate_left
# = How many lower lanes are predicated to left
# = The absolute position for those elements in left partition
left_pos = prefix_sum[exclusive=True](predicate_left)
# Exclusive scan on predicate_right
# = How many lower lanes are predicated to right
# = The relative position to the pivot for those elements in right partition
right_pos = prefix_sum[exlcusive=True](predicate_right)
```

Step 3 Reduction: Compute the pivot position
```mojo
# Sum of predicate_left
# = Total # of elements in left partition
warp_left_total = predicate_left
offset = WARP_SIZE // 2
while offset >= 1:
    warp_left_total += shuffle_xor(warp_left_total, offset)
    offset //= 2
```
Question: Can we just get the last lane's left pos?

Step 4: write to output position
```mojo
if current < pivot:
    # Left partition
    output[warp_left_pos] = current
else:
    # Right partition: idx = pivot position + relative position to pivot
    outpu[warp_left_total + warp_right_pos] = current
```