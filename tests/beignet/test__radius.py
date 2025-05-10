import torch

from beignet import radius


def test_radius():
    N = 500
    M = 100
    device = None
    x = torch.randn(N, 3, device=device)
    y = torch.randn(M, 3, device=device)
    batch_x = torch.randint(2, (N,), device=device).sort().values
    batch_y = torch.randint(2, (M,), device=device).sort().values

    max_radius = 1.0

    edge_index = radius(
        x, y, max_radius, batch_x=batch_x, batch_y=batch_y, ignore_same_index=False
    )

    src, dst = edge_index.unbind(dim=0)

    assert src.max() <= M
    assert dst.max() <= N

    r_vec = y[src] - x[dst]
    r = r_vec.pow(2).sum(dim=-1)

    assert r.max() <= max_radius

    pdist = (y[:, None] - x[None]).pow(2).sum(dim=-1)
    assert pdist.shape == (M, N)

    same_batch = batch_y[:, None] == batch_x[None]
    assert same_batch.shape == (M, N)

    src_ref, dst_ref = torch.nonzero((pdist <= max_radius) & same_batch, as_tuple=True)
    r_ref = (y[src_ref] - x[dst_ref]).pow(2).sum(dim=-1)

    assert torch.equal(r_ref.sort().values, r.sort().values)


def test_radius_chunked():
    N = 500
    M = 100
    device = None
    x = torch.randn(N, 3, device=device)
    y = torch.randn(M, 3, device=device)
    batch_x = torch.randint(2, (N,), device=device).sort().values
    batch_y = torch.randint(2, (M,), device=device).sort().values

    max_radius = 1.0

    edge_index = radius(
        x,
        y,
        max_radius,
        batch_x=batch_x,
        batch_y=batch_y,
        ignore_same_index=False,
        chunk_size=32,
    )

    src, dst = edge_index.unbind(dim=0)

    assert src.max() <= M
    assert dst.max() <= N

    r_vec = y[src] - x[dst]
    r = r_vec.pow(2).sum(dim=-1)

    assert r.max() <= max_radius

    pdist = (y[:, None] - x[None]).pow(2).sum(dim=-1)
    assert pdist.shape == (M, N)

    same_batch = batch_y[:, None] == batch_x[None]
    assert same_batch.shape == (M, N)

    src_ref, dst_ref = torch.nonzero((pdist <= max_radius) & same_batch, as_tuple=True)
    r_ref = (y[src_ref] - x[dst_ref]).pow(2).sum(dim=-1)

    assert torch.equal(r_ref.sort().values, r.sort().values)
