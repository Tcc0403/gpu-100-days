# Warmup D-3 (2025/6/13)
- Planning to go through some tutorials and figure out potential pitfalls before D-Day(6/17)
- Done [mojo puzzle](https://builds.modular.com/puzzles/introduction.html) ([my solutions](https://github.com/Tcc0403/mojo-gpu-puzzles/tree/solution/problems))

## Note
### Memory allocation in mojo
#### Device memory (global memory)
Use `DeviceContext` method [`enqueue_create_buffer[]()`](https://docs.modular.com/mojo/stdlib/gpu/host/device_context/DeviceContext/#enqueue_create_buffer) 
```mojo
from gpu.host import DeviceContext
with DeviceContext as ctx:
    dtype = Dtype.float32
    size = 10 * 10
    # Allocate devcice's global memory
    tensor = ctx.enqueue_create_buffer[dtype](size)
    # Fill the buffer with a specidfied value
    tensor = tensor.enqueue_fill(0)
```
#### Shared memory (on-chip memory)
stack_allocation ([doc](https://docs.modular.com/mojo/stdlib/memory/memory/stack_allocation/))
```mojo
from memory import UnsafePointer, stack_allocation
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import AddressSpace

alias TPB = 4 # thread per block, block_dim = (4, 1)
fn kernel_func(
    output: UnsafePointer[Scalar[dtype]]
    a: UnsafePointer[Scalar[dtype]]
    size: Int,
):
    # Calculate index
    global_idx = block_dim.x * block_idx.x + thread_idx.x
    local_idx = thread_idx.x
    # Allocate shared memory
    shared = stack_allocation[
        TPB,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    # Access shared memory
    if global_idx < size:
        shared[local_idx] = a[global_idx]
    # Synchronize
    barrier()
```


### LayoutTenor in mojo
#### Basic usage example ([ref](https://builds.modular.com/puzzles/puzzle_04/introduction_layout_tensor.html#basic-usage-example))
```mojo
from layout import Layout, LayoutTensor
# Define layouot
alias HIEGHT = 2
alias WIDTH = 3
alias layout = Layout.row_major(HEIGHT, WIDTH)

# Create tensor
tensor = LayoutTensor[dtype, layout](buffer.unsafe_ptr())

# Access elements naturally
tensor[0, 0] = 1.0
tensor[1, 2] = 2.0
```
Advanced features
```mojo
# Column-major layout
# [1 2 3]
# [4 5 6] -> [1 4 2 5 3 6]
layout_col = Layout.col_major(2, 3)

# Tiled layout
layout_tiled = Layout.tiled[2, 2](4, 4)
```
#### Allocate shared memory using layout tensor builder (LayoutTensorBuild)
LayoutTensorBuild ([doc](https://docs.modular.com/mojo/kernels/layout/tensor_builder/LayoutTensorBuild/))
```mojo
from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb

alias TPB = 4 # thread per block, block_dim = (4, 1)
alias SIZE = 8
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE)

fn kernel_func[
    layout: Layout
](
    output: UnsafePointer[Scalar[dtype]]
    a: UnsafePointer[Scalar[dtype]]
    size: Int,
):
    # Calculate index
    global_idx = block_dim.x * block_idx.x + thread_idx.x
    local_idx = thread_idx.x
    # Allocate shared memory using tensor builder
    # what dtype -> what layout -> place where(local/shared) -> allocate using given information
    shared = tb[dtype]().row_major[TPB]().shared().alloc()
    
    # Access shared memory
    if global_idx < size:
        shared[local_idx] = a[global_idx]
    # Synchronize
    barrier()
```

## Next plan
- Finish mojo gpu puzzle
- Learn how to call mojo from python (pytroch) [link](https://docs.modular.com/mojo/manual/python/mojo-from-python)