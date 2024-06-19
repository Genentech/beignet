import numpy.testing

chebyshev_polynomial_T0 = [1]
chebyshev_polynomial_T1 = [0, 1]
chebyshev_polynomial_T2 = [-1, 0, 2]
chebyshev_polynomial_T3 = [0, -3, 0, 4]
chebyshev_polynomial_T4 = [1, 0, -8, 0, 8]
chebyshev_polynomial_T5 = [0, 5, 0, -20, 0, 16]
chebyshev_polynomial_T6 = [-1, 0, 18, 0, -48, 0, 32]
chebyshev_polynomial_T7 = [0, -7, 0, 56, 0, -112, 0, 64]
chebyshev_polynomial_T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
chebyshev_polynomial_T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

chebyshev_polynomial_Tlist = [
    chebyshev_polynomial_T0,
    chebyshev_polynomial_T1,
    chebyshev_polynomial_T2,
    chebyshev_polynomial_T3,
    chebyshev_polynomial_T4,
    chebyshev_polynomial_T5,
    chebyshev_polynomial_T6,
    chebyshev_polynomial_T7,
    chebyshev_polynomial_T8,
    chebyshev_polynomial_T9,
]

hermite_polynomial_H0 = numpy.array([1])
hermite_polynomial_H1 = numpy.array([0, 2])
hermite_polynomial_H2 = numpy.array([-2, 0, 4])
hermite_polynomial_H3 = numpy.array([0, -12, 0, 8])
hermite_polynomial_H4 = numpy.array([12, 0, -48, 0, 16])
hermite_polynomial_H5 = numpy.array([0, 120, 0, -160, 0, 32])
hermite_polynomial_H6 = numpy.array([-120, 0, 720, 0, -480, 0, 64])
hermite_polynomial_H7 = numpy.array([0, -1680, 0, 3360, 0, -1344, 0, 128])
hermite_polynomial_H8 = numpy.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])
hermite_polynomial_H9 = numpy.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])

hermite_polynomial_Hlist = [
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

hermite_e_polynomial_He0 = numpy.array([1])
hermite_e_polynomial_He1 = numpy.array([0, 1])
hermite_e_polynomial_He2 = numpy.array([-1, 0, 1])
hermite_e_polynomial_He3 = numpy.array([0, -3, 0, 1])
hermite_e_polynomial_He4 = numpy.array([3, 0, -6, 0, 1])
hermite_e_polynomial_He5 = numpy.array([0, 15, 0, -10, 0, 1])
hermite_e_polynomial_He6 = numpy.array([-15, 0, 45, 0, -15, 0, 1])
hermite_e_polynomial_He7 = numpy.array([0, -105, 0, 105, 0, -21, 0, 1])
hermite_e_polynomial_He8 = numpy.array([105, 0, -420, 0, 210, 0, -28, 0, 1])
hermite_e_polynomial_He9 = numpy.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])

hermite_e_polynomial_Helist = [
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

laguerre_polynomial_L0 = numpy.array([1]) / 1
laguerre_polynomial_L1 = numpy.array([1, -1]) / 1
laguerre_polynomial_L2 = numpy.array([2, -4, 1]) / 2
laguerre_polynomial_L3 = numpy.array([6, -18, 9, -1]) / 6
laguerre_polynomial_L4 = numpy.array([24, -96, 72, -16, 1]) / 24
laguerre_polynomial_L5 = numpy.array([120, -600, 600, -200, 25, -1]) / 120
laguerre_polynomial_L6 = numpy.array([720, -4320, 5400, -2400, 450, -36, 1]) / 720

laguerre_polynomial_Llist = [
    laguerre_polynomial_L0,
    laguerre_polynomial_L1,
    laguerre_polynomial_L2,
    laguerre_polynomial_L3,
    laguerre_polynomial_L4,
    laguerre_polynomial_L5,
    laguerre_polynomial_L6,
]

legendre_polynomial_L0 = numpy.array([1])
legendre_polynomial_L1 = numpy.array([0, 1])
legendre_polynomial_L2 = numpy.array([-1, 0, 3]) / 2
legendre_polynomial_L3 = numpy.array([0, -3, 0, 5]) / 2
legendre_polynomial_L4 = numpy.array([3, 0, -30, 0, 35]) / 8
legendre_polynomial_L5 = numpy.array([0, 15, 0, -70, 0, 63]) / 8
legendre_polynomial_L6 = numpy.array([-5, 0, 105, 0, -315, 0, 231]) / 16
legendre_polynomial_L7 = numpy.array([0, -35, 0, 315, 0, -693, 0, 429]) / 16
legendre_polynomial_L8 = numpy.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435]) / 128
legendre_polynomial_L9 = (
    numpy.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155]) / 128
)

legendre_polynomial_Llist = [
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

polynomial_T0 = [1]
polynomial_T1 = [0, 1]
polynomial_T2 = [-1, 0, 2]
polynomial_T3 = [0, -3, 0, 4]
polynomial_T4 = [1, 0, -8, 0, 8]
polynomial_T5 = [0, 5, 0, -20, 0, 16]
polynomial_T6 = [-1, 0, 18, 0, -48, 0, 32]
polynomial_T7 = [0, -7, 0, 56, 0, -112, 0, 64]
polynomial_T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
polynomial_T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

polynomial_Tlist = [
    polynomial_T0,
    polynomial_T1,
    polynomial_T2,
    polynomial_T3,
    polynomial_T4,
    polynomial_T5,
    polynomial_T6,
    polynomial_T7,
    polynomial_T8,
    polynomial_T9,
]
