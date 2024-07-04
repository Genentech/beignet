import torch

from ._root import root


def nelder_mead(
    func,
    x0,
    max_iterations=200,
    tolerance=1e-8,
    alpha=1.0,
    gamma=2.0,
    rho=-0.5,
    sigma=0.5,
):
    def solve(f, x0):
        n = x0.numel()

        # Create initial simplex
        simplex = torch.zeros((n + 1, n), dtype=x0.dtype)
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.clone()
            if simplex[i, i - 1] != 0:
                simplex[i, i - 1] *= 1.05
            else:
                simplex[i, i - 1] = 0.00025

        # Compute function values
        f_values = torch.stack([torch.sum(f(x) ** 2) for x in simplex])

        for _ in range(max_iterations):
            # Sort simplex
            order = torch.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]

            # Check for convergence
            if torch.max(torch.abs(f_values[0] - f_values[1:])) <= tolerance:
                break

            # Compute centroid
            centroid = torch.mean(simplex[:-1], dim=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = torch.sum(f(x_r) ** 2)

            if f_values[0] <= f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            elif f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = torch.sum(f(x_e) ** 2)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = torch.sum(f(x_c) ** 2)
                if f_c < f_values[-1]:
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
                    f_values[1:] = torch.stack(
                        [torch.sum(f(x) ** 2) for x in simplex[1:]]
                    )

        return simplex[0]

    def tangent_solve(g, y):
        # For Nelder-Mead, we can use a simple approximation
        # This is not as accurate as for Newton's method, but should work for many cases
        eps = 1e-8
        jac = torch.stack(
            [g(y + eps * torch.eye(y.shape[0])[i]) for i in range(y.shape[0])]
        )
        return torch.linalg.solve(jac.T @ jac, jac.T @ y)

    return root(func, x0, solve, tangent_solve)
