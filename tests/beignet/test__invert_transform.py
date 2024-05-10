import beignet
import torch.testing


def test_invert_transform():
    input = torch.randn([32, 3])

    transform = torch.randn([3, 3])

    torch.testing.assert_close(
        input,
        beignet.apply_transform(
            beignet.apply_transform(input, transform),
            beignet.invert_transform(transform),
        ),
    )
