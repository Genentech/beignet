# def test_polynomial_roots():
#     torch.testing.assert_close(
#         beignet.polynomials.polynomial_roots(torch.tensor([1.0])),
#         torch.tensor([]),
#     )
#
#     torch.testing.assert_close(
#         beignet.polynomials.polynomial_roots(torch.tensor([1.0, 2.0])),
#         torch.tensor([-0.5]),
#     )
#
#     for index in range(2, 5):
#         input = torch.linspace(-1, 1, index)
#
#         torch.testing.assert_close(
#             beignet.polynomials.trim_polynomial_coefficients(
#                 beignet.polynomials.polynomial_roots(
#                     beignet.polynomials.polynomial_from_roots(
#                         input,
#                     ),
#                 ),
#                 tol=0.000001,
#             ),
#             beignet.polynomials.trim_polynomial_coefficients(
#                 input,
#                 tol=0.000001,
#             ),
#         )
