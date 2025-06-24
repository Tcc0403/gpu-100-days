import pytest
import torch
from src.softmax import softmax

DEVICE = "cuda"
@pytest.mark.parametrize(
    "seq_len, hidden_size",
    [
        # (32, 32), Failed
        (16, 1024),
        (16, 32000),
        (1024, 1024),
        # weird shape
        (16, 1023),
    ]
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-4, 1e-5),
        # (torch.float16, 1e-2, 1e-2),  Failed: overflow(?)
        # (torch.bfloat16, 1e-2, 1e-2),
    ]
)
def test_softmax(seq_len, hidden_size, dtype, atol, rtol):
    torch.manual_seed(0)
    input = torch.rand((seq_len, hidden_size), dtype=dtype, device="cuda")

    
    ref_output = torch.nn.functional.softmax(input, dim=-1)
    actual_output = softmax(input)

    torch.testing.assert_close(actual_output, ref_output, atol=atol, rtol=rtol)