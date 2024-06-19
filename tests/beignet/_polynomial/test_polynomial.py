import torch

chebyshev_polynomial_coefficients = [
    [1],
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8],
    [0, 5, 0, -20, 0, 16],
    [-1, 0, 18, 0, -48, 0, 32],
    [0, -7, 0, 56, 0, -112, 0, 64],
    [1, 0, -32, 0, 160, 0, -256, 0, 128],
    [0, 9, 0, -120, 0, 432, 0, -576, 0, 256],
]

hermite_polynomial_H0 = torch.tensor([1])
hermite_polynomial_H1 = torch.tensor([0, 2])
hermite_polynomial_H2 = torch.tensor([-2, 0, 4])
hermite_polynomial_H3 = torch.tensor([0, -12, 0, 8])
hermite_polynomial_H4 = torch.tensor([12, 0, -48, 0, 16])
hermite_polynomial_H5 = torch.tensor([0, 120, 0, -160, 0, 32])
hermite_polynomial_H6 = torch.tensor([-120, 0, 720, 0, -480, 0, 64])
hermite_polynomial_H7 = torch.tensor([0, -1680, 0, 3360, 0, -1344, 0, 128])
hermite_polynomial_H8 = torch.tensor([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])
hermite_polynomial_H9 = torch.tensor([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])

hermite_polynomial_coefficients = [
    hermite_polynomial_H0,
    hermite_polynomial_H1,
    hermite_polynomial_H2,
    hermite_polynomial_H3,
    hermite_polynomial_H4,
    hermite_polynomial_H5,
    hermite_polynomial_H6,
    hermite_polynomial_H7,
    hermite_polynomial_H8,
    hermite_polynomial_H9,
]

hermite_e_polynomial_He0 = torch.tensor([1])
hermite_e_polynomial_He1 = torch.tensor([0, 1])
hermite_e_polynomial_He2 = torch.tensor([-1, 0, 1])
hermite_e_polynomial_He3 = torch.tensor([0, -3, 0, 1])
hermite_e_polynomial_He4 = torch.tensor([3, 0, -6, 0, 1])
hermite_e_polynomial_He5 = torch.tensor([0, 15, 0, -10, 0, 1])
hermite_e_polynomial_He6 = torch.tensor([-15, 0, 45, 0, -15, 0, 1])
hermite_e_polynomial_He7 = torch.tensor([0, -105, 0, 105, 0, -21, 0, 1])
hermite_e_polynomial_He8 = torch.tensor([105, 0, -420, 0, 210, 0, -28, 0, 1])
hermite_e_polynomial_He9 = torch.tensor([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])

hermite_e_polynomial_coefficients = [
    hermite_e_polynomial_He0,
    hermite_e_polynomial_He1,
    hermite_e_polynomial_He2,
    hermite_e_polynomial_He3,
    hermite_e_polynomial_He4,
    hermite_e_polynomial_He5,
    hermite_e_polynomial_He6,
    hermite_e_polynomial_He7,
    hermite_e_polynomial_He8,
    hermite_e_polynomial_He9,
]

laguerre_polynomial_L0 = torch.tensor([1]) / 1
laguerre_polynomial_L1 = torch.tensor([1, -1]) / 1
laguerre_polynomial_L2 = torch.tensor([2, -4, 1]) / 2
laguerre_polynomial_L3 = torch.tensor([6, -18, 9, -1]) / 6
laguerre_polynomial_L4 = torch.tensor([24, -96, 72, -16, 1]) / 24
laguerre_polynomial_L5 = torch.tensor([120, -600, 600, -200, 25, -1]) / 120
laguerre_polynomial_L6 = torch.tensor([720, -4320, 5400, -2400, 450, -36, 1]) / 720

laguerre_polynomial_coefficients = [
    laguerre_polynomial_L0,
    laguerre_polynomial_L1,
    laguerre_polynomial_L2,
    laguerre_polynomial_L3,
    laguerre_polynomial_L4,
    laguerre_polynomial_L5,
    laguerre_polynomial_L6,
]

legendre_polynomial_L0 = torch.tensor([1])
legendre_polynomial_L1 = torch.tensor([0, 1])
legendre_polynomial_L2 = torch.tensor([-1, 0, 3]) / 2
legendre_polynomial_L3 = torch.tensor([0, -3, 0, 5]) / 2
legendre_polynomial_L4 = torch.tensor([3, 0, -30, 0, 35]) / 8
legendre_polynomial_L5 = torch.tensor([0, 15, 0, -70, 0, 63]) / 8
legendre_polynomial_L6 = torch.tensor([-5, 0, 105, 0, -315, 0, 231]) / 16
legendre_polynomial_L7 = torch.tensor([0, -35, 0, 315, 0, -693, 0, 429]) / 16
legendre_polynomial_L8 = torch.tensor([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128
legendre_polynomial_L9 = (
    torch.tensor([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128
)

legendre_polynomial_coefficients = [
    legendre_polynomial_L0,
    legendre_polynomial_L1,
    legendre_polynomial_L2,
    legendre_polynomial_L3,
    legendre_polynomial_L4,
    legendre_polynomial_L5,
    legendre_polynomial_L6,
    legendre_polynomial_L7,
    legendre_polynomial_L8,
    legendre_polynomial_L9,
]

polynomial_coefficients = [
    [1],
    [0, 1],
    [-1, 0, 2],
    [0, -3, 0, 4],
    [1, 0, -8, 0, 8],
    [0, 5, 0, -20, 0, 16],
    [-1, 0, 18, 0, -48, 0, 32],
    [0, -7, 0, 56, 0, -112, 0, 64],
    [1, 0, -32, 0, 160, 0, -256, 0, 128],
    [0, 9, 0, -120, 0, 432, 0, -576, 0, 256],
]
