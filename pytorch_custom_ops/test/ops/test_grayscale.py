import pytest
import torch
from src.grayscale import grayscale

DEVICE = "cuda"

def ref_grayscale(pic: torch.Tensor):
    output = pic.new_empty(pic.shape[:-1]) # Remove color channel dimension
    r = pic[:, :, 0]
    g = pic[:, :, 1]
    b = pic[:, :, 2]
    output = 0.21 * r + 0.71 * g + 0.007 * b 
    return output.to(torch.uint8)

@pytest.mark.parametrize(
    "H, W",
    [
        (128, 128),
        (128, 256),
        (300, 300),
    ]
)
def test_grayscale_cpu(H, W):
    torch.manual_seed(0)

    input = torch.randint(0, 255, (H, W, 3)).cpu()
    ref_output = ref_grayscale(input)
    actual_output = grayscale(input)


    torch.testing.assert_close(ref_output, actual_output)


@pytest.mark.parametrize(
    "H, W",
    [
        (128, 128),
        (128, 256),
        (300, 300),
    ]
)
def test_grayscale_gpu(H, W):
    torch.manual_seed(0)

    input = torch.randint(0, 255, (H, W, 3), device=DEVICE)
    ref_output = ref_grayscale(input)
    actual_output = grayscale(input)


    torch.testing.assert_close(ref_output, actual_output)