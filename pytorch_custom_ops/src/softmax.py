from pathlib import Path

import torch
from max.torch import CustomOpLibrary

# Load the compiled Mojo package containing our custom operations
mojo_kernels = Path(__file__).parent.joinpath("ops")
ops = CustomOpLibrary(mojo_kernels)

@torch.compile
def softmax(input: torch.Tensor):
    output = torch.empty_like(input)
    ops.softmax(output, input)
    return output


