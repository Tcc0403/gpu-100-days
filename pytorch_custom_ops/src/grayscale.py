from pathlib import Path

import torch
from max.torch import CustomOpLibrary

# Load the compiled Mojo package containing our custom operations
mojo_kernels = Path(__file__).parent.joinpath("ops")
ops = CustomOpLibrary(mojo_kernels)

@torch.compile
def grayscale(pic):
    output = pic.new_empty(pic.shape[:-1])  # Remove color channel dimension
    ops.grayscale(output, pic)  # Call our Mojo custom op
    return output