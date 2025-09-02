import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from beignet.nn import TemplateEmbedder


@given(
    batch_size=st.integers(min_value=1, max_value=2),
    n_tokens=st.integers(min_value=8, max_value=16),
    c_z=st.integers(min_value=8, max_value=32).filter(lambda x: x % 4 == 0),
    c_template=st.integers(min_value=8, max_value=32),
    n_head=st.integers(min_value=1, max_value=4),
    dtype=st.sampled_from([torch.float32]),
)
@settings(deadline=None, max_examples=3)
def test__template_embedder(batch_size, n_tokens, c_z, c_template, n_head, dtype):
    """Test AlphaFold3TemplateEmbedder with various input configurations."""

    # Ensure c_z is divisible by n_head
    c_z = (c_z // n_head) * n_head if c_z % n_head != 0 else c_z

    module = TemplateEmbedder(
        c_z=c_z,
        c_template=c_template,
        n_head=n_head,
    ).to(dtype=dtype)

    # Create test inputs
    z_ij = torch.randn(batch_size, n_tokens, n_tokens, c_z, dtype=dtype)

    # Test without template features
    f_star_empty = {}
    output_empty = module(f_star_empty, z_ij)

    # Should return unchanged when no templates
    assert torch.allclose(output_empty, z_ij)
    assert output_empty.shape == (batch_size, n_tokens, n_tokens, c_z)
    assert output_empty.dtype == dtype

    # Test with template features
    f_star_with_template = {
        "template_features": torch.randn(
            batch_size, n_tokens, n_tokens, c_template, dtype=dtype
        )
    }
    output_with_template = module(f_star_with_template, z_ij)

    # Check output shape and properties
    assert output_with_template.shape == (batch_size, n_tokens, n_tokens, c_z)
    assert output_with_template.dtype == dtype
    assert torch.all(torch.isfinite(output_with_template))

    # Output should be different when templates are provided
    assert not torch.allclose(output_with_template, z_ij, atol=1e-4)

    # Test gradient computation
    z_grad = z_ij.clone().requires_grad_(True)
    template_grad = (
        f_star_with_template["template_features"].clone().requires_grad_(True)
    )
    f_star_grad = {"template_features": template_grad}

    output_grad = module(f_star_grad, z_grad)
    loss = output_grad.sum()
    loss.backward()

    # Check gradients exist
    assert z_grad.grad is not None
    assert template_grad.grad is not None
    assert torch.all(torch.isfinite(z_grad.grad))
    assert torch.all(torch.isfinite(template_grad.grad))

    # Test different template sizes
    if c_template != c_z:
        f_star_diff_size = {
            "template_features": torch.randn(
                batch_size, n_tokens, n_tokens, c_template, dtype=dtype
            )
        }
        output_diff_size = module(f_star_diff_size, z_ij)
        assert output_diff_size.shape == (batch_size, n_tokens, n_tokens, c_z)

    # Test edge case with single token
    if n_tokens >= 1:
        z_single = torch.randn(batch_size, 1, 1, c_z, dtype=dtype)
        template_single = torch.randn(batch_size, 1, 1, c_template, dtype=dtype)
        f_star_single = {"template_features": template_single}

        output_single = module(f_star_single, z_single)
        assert output_single.shape == (batch_size, 1, 1, c_z)
        assert torch.all(torch.isfinite(output_single))
