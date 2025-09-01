# def test_divide_chebyshev_polynomial():
#     for j in range(5):
#         for k in range(5):
#             input = torch.tensor([0.0] * j + [1.0])
#             other = torch.tensor([0.0] * k + [1.0])
#
#             quotient, remainder = beignet.polynomials.divide_chebyshev_polynomial(
#                 beignet.polynomials.add_chebyshev_polynomial(
#                     input,
#                     other,
#                 ),
#                 input,
#             )
#
#             torch.testing.assert_close(
#                 beignet.polynomials.trim_chebyshev_polynomial_coefficients(
#                     beignet.polynomials.add_chebyshev_polynomial(
#                         beignet.polynomials.multiply_chebyshev_polynomial(
#                             quotient,
#                             input,
#                         ),
#                         remainder,
#                     ),
#                     tol=0.000001,
#                 ),
#                 beignet.polynomials.trim_chebyshev_polynomial_coefficients(
#                     beignet.polynomials.add_chebyshev_polynomial(
#                         input,
#                         other,
#                     ),
#                     tol=0.000001,
#                 ),
#             )
