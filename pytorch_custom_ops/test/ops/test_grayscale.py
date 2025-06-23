import pytest
import torch

from src.grayscale import grayscale

DEVICE = "cuda"

def ref_grayscale(pic: torch.Tensor):
    output = pic.new_empty(pic.shape[:-1]) # Remove color channel dimension
    r = pic[:, :, 0].to(torch.float32)
    g = pic[:, :, 1].to(torch.float32)
    b = pic[:, :, 2].to(torch.float32)
    output = 0.21 * r + 0.71 * g + 0.07 * b 
    return output.clip(max=255).to(torch.uint8)

@pytest.mark.parametrize(
    "H, W",
    [
        (5, 5),
        (128, 128),
        (128, 256),
        (300, 300),
    ]
)
def test_grayscale_cpu(H, W):
    torch.manual_seed(0)

    input = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8).cpu()
    ref_output = ref_grayscale(input)
    actual_output = grayscale(input)

    # Gray scale transformation is done in float32, so there might be rounding error
    torch.testing.assert_close(ref_output, actual_output, atol=1, rtol=0)


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

    input = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, device=DEVICE)
    ref_output = ref_grayscale(input)
    actual_output = grayscale(input)

    # Gray scale transformation is done in float32, so there might be rounding error
    torch.testing.assert_close(ref_output, actual_output, atol=1, rtol=0)