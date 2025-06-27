from gpu.host import DeviceContext
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx, lane_id
from gpu import warp
from compiler import register
from layout import Layout, LayoutTensor
from max.tensor import InputTensor, OutputTensor
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList
from utils.numerics import min_finite
from math import exp


fn softmax_kernel[
    output_layout: Layout,
    input_layout: Layout,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, output_layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, input_layout, MutableAnyOrigin],
    n_cols: Int,
):
    """
    Each block processes 1 row with 1 warp
    grid_dim = (seq_len, 1, 1)
    block_dim = (WARP_SIZE, 1, 1).
    """
    seq_idx = block_idx.x
    lane = lane_id()
    # Both statistics calculated in float32 by default
    var global_max: Scalar[dtype] = min_finite[dtype]()
    var global_sum: Scalar[dtype] = 0

    for i in range((n_cols + WARP_SIZE - 1) // WARP_SIZE):
        var last_dim_idx: Int = i * WARP_SIZE + lane
        var x_i: input.element_type = min_finite[dtype]()
        if last_dim_idx < n_cols:
            x_i = input[seq_idx, last_dim_idx].cast[dtype]()

        local_max = warp.max(x_i).reduce_max()
        local_max = max(local_max, global_max)
        local_sum = warp.sum(exp((x_i - local_max))).reduce_add()

        # Update global statistics
        global_sum = global_sum * exp(global_max - local_max) + local_sum
        global_max = local_max

    for i in range((n_cols + WARP_SIZE - 1) // WARP_SIZE):
        var last_dim_idx: Int = i * WARP_SIZE + lane
        if last_dim_idx < n_cols:
            output[seq_idx, last_dim_idx] = (
                exp(input[seq_idx, last_dim_idx].cast[dtype]() - global_max)
                / global_sum
            ).cast[output.element_type.dtype]()


@register("softmax")
struct Softmax:
    @staticmethod
    fn execute[
        target: StaticString,
        # dtype: DType = DType.float32,  This parameter must be given when calling from torch.compile()
    ](output: OutputTensor, input: InputTensor, ctx: DeviceContextPtr,) raises:
        @parameter
        if target == "gpu":
            output_tensor = output.to_layout_tensor()
            input_tensor = input.to_layout_tensor()

            if output.rank == 2:
                seq_len = output_tensor.dim[0]()
            else:
                raise Error("Softmax only supported rank-2 input for now")

            gpu_ctx = ctx.get_device_context()

            gpu_ctx.enqueue_function[
                softmax_kernel[
                    output_tensor.layout,
                    input_tensor.layout,
                ]
            ](
                output_tensor,
                input_tensor,
                input_tensor.shape[1](),
                grid_dim=(seq_len, 1, 1),
                block_dim=(WARP_SIZE, 1, 1),
            )
        else:
            raise Error("cpu softmax is not implemented")


# def main():
#     alias seq_len = 8
#     alias hidden_size = 1024
#     alias dtype = DType.float32
#     alias layout = Layout.row_major(seq_len, hidden_size)
#     with DeviceContext() as ctx:
#         out_buf = ctx.enqueue_create_buffer[dtype](
#             seq_len * hidden_size
#         ).enqueue_fill(0)
#         out_tensor = LayoutTensor[
#             mut=True, dtype, Layout.row_major(seq_len, hidden_size)
#         ](out_buf.unsafe_ptr()).reshape[layout]()

#         input_buf = ctx.enqueue_create_buffer[dtype](
#             seq_len * hidden_size
#         ).enqueue_fill(0)
#         with input_buf.map_to_host() as input_host:
#             for i in range(seq_len * hidden_size):
#                 input_host[i] = i
#         input_tensor = LayoutTensor[mut=True, dtype, layout](
#             input_buf.unsafe_ptr()
#         ).reshape[layout]()

#         print("generated input")
#         ctx.enqueue_function[
#             softmax_kernel[
#                 layout,
#                 layout,
#                 dtype,
#             ]
#         ](
#             out_tensor,
#             input_tensor,
#             grid_dim=(seq_len, 1, 1),
#             block_dim=(WARP_SIZE, 1, 1),
#         )

#         ctx.synchronize()
#         print("kernel finished")
