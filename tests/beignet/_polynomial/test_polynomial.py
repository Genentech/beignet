import torch

chebyshev_polynomial_coefficients = [
    torch.tensor([1]),
    torch.tensor([0, 1]),
    torch.tensor([-1, 0, 2]),
    torch.tensor([0, -3, 0, 4]),
    torch.tensor([1, 0, -8, 0, 8]),
    torch.tensor([0, 5, 0, -20, 0, 16]),
    torch.tensor([-1, 0, 18, 0, -48, 0, 32]),
    torch.tensor([0, -7, 0, 56, 0, -112, 0, 64]),
    torch.tensor([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    torch.tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]

hermite_polynomial_coefficients = [
    (torch.tensor([1])),
    (torch.tensor([0, 2])),
    (torch.tensor([-2, 0, 4])),
    (torch.tensor([0, -12, 0, 8])),
    (torch.tensor([12, 0, -48, 0, 16])),
    (torch.tensor([0, 120, 0, -160, 0, 32])),
    (torch.tensor([-120, 0, 720, 0, -480, 0, 64])),
    (torch.tensor([0, -1680, 0, 3360, 0, -1344, 0, 128])),
    (torch.tensor([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])),
    (torch.tensor([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])),
]

hermite_e_polynomial_coefficients = [
    (torch.tensor([1], dtype=torch.float64)),
    (torch.tensor([0, 1], dtype=torch.float64)),
    (torch.tensor([-1, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, -3, 0, 1], dtype=torch.float64)),
    (torch.tensor([3, 0, -6, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, 15, 0, -10, 0, 1], dtype=torch.float64)),
    (torch.tensor([-15, 0, 45, 0, -15, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, -105, 0, 105, 0, -21, 0, 1], dtype=torch.float64)),
    (torch.tensor([105, 0, -420, 0, 210, 0, -28, 0, 1], dtype=torch.float64)),
    (torch.tensor([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1], dtype=torch.float64)),
]

laguerre_polynomial_coefficients = [
    (torch.tensor([1]) / 1),
    (torch.tensor([1, -1]) / 1),
    (torch.tensor([2, -4, 1]) / 2),
    (torch.tensor([6, -18, 9, -1]) / 6),
    (torch.tensor([24, -96, 72, -16, 1]) / 24),
    (torch.tensor([120, -600, 600, -200, 25, -1]) / 120),
    (torch.tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720),
]

legendre_polynomial_coefficients = [
    torch.tensor([1], dtype=torch.float64),
    torch.tensor([0, 1], dtype=torch.float64),
    torch.tensor([-1, 0, 3], dtype=torch.float64) / 2,
    torch.tensor([0, -3, 0, 5], dtype=torch.float64) / 2,
    torch.tensor([3, 0, -30, 0, 35], dtype=torch.float64) / 8,
    torch.tensor([0, 15, 0, -70, 0, 63], dtype=torch.float64) / 8,
    torch.tensor([-5, 0, 105, 0, -315, 0, 231], dtype=torch.float64) / 16,
    torch.tensor([0, -35, 0, 315, 0, -693, 0, 429], dtype=torch.float64) / 16,
    torch.tensor([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435], dtype=torch.float64)
    / 128,
    torch.tensor([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155], dtype=torch.float64)
    / 128,
]

polynomial_coefficients = [
    torch.tensor([1]),
    torch.tensor([0, 1]),
    torch.tensor([-1, 0, 2]),
    torch.tensor([0, -3, 0, 4]),
    torch.tensor([1, 0, -8, 0, 8]),
    torch.tensor([0, 5, 0, -20, 0, 16]),
    torch.tensor([-1, 0, 18, 0, -48, 0, 32]),
    torch.tensor([0, -7, 0, 56, 0, -112, 0, 64]),
    torch.tensor([1, 0, -32, 0, 160, 0, -256, 0, 128]),
    torch.tensor([0, 9, 0, -120, 0, 432, 0, -576, 0, 256]),
]
