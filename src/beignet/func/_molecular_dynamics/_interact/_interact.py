from typing import Callable, Literal, Optional, Tuple, Union

from torch import Tensor

from .__angle_interaction import _angle_interaction
from .__bond_interaction import _bond_interaction
from .__dihedral_interaction import _dihedral_interaction
from .__long_range_interaction import _long_range_interaction
from .__mesh_interaction import _mesh_interaction
from .__neighbor_list_interaction import _neighbor_list_interaction
from .__pair_interaction import _pair_interaction


def interact(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    interaction: Literal[
        "angle",
        "bond",
        "dihedral",
        "long-range",
        "mesh",
        "neighbor_list",
        "pair",
    ],
    *,
    bonds: Optional[Tensor] = None,
    kinds: Optional[Tensor] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    ignore_unused_parameters: bool = True,
    **kwargs,
) -> Callable[..., Tensor]:
    r"""
    Define interactions between elements of a system.

    For a collection of $N$ elements, $\vec{r}_i \in \mathbb{R}^{D}$, where $1 \leq i \leq N$, energy is the function $U : \mathbb{R}^{N \times D} \rightarrow \mathbb{R}$. Energy is used by a simulation by applying Newton's laws: $m \vec{\ddot{r}}_{i} = - \nabla_{\vec{r}_{i}} U$ where $m$ is mass. Rather than defining an energy as an interaction between all the elements in the simulation space simultaneously, it's preferable to use a pairwise energy function based on the displacement between a pair of elements, $u(\vec{r}_{i} - \vec{r}_{j})$. Total energy is defined by the sum over pairwise interactions:

    $$U = \frac{1}{2} \sum_{i \neq j} u(\vec{r}_{i} - \vec{r}_{j}).$$

    To facilitate the construction of functions from interactions, `interact` returns a function to map bonds, neighbors, pairs, or triplets interactions and transforms them to operate on an entire simulation.

    Parameters
    ----------
    fn : Callable[..., Array]
        Function that takes distances or displacements of shape `(n, m)` or `(n, m, spatial_dimension)` and `kwargs` and returns values of shape `(n, m, spatial_dimension)`. The function must be a differentiable function as the force is computed using automatic differentiation (see `prescient.func.force`).

    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        Displacement function that takes positions of shape `(spatial_dimension)` and `(spatial_dimension)` and returns distances or displacements of shape `()` or `(spatial_dimension)`.

    interaction : Literal["bond", "neighbor_list", "pair", "triplet"]
        One of the following types of interactions:

        -   `"angle"`,

        -   `"bond"`, transforms a function that acts on a single pair of elements to a function that acts on a set of bonds.

        -  `"dihedral"`,

        -   `"long-range"`,

        -   `"neighbor_list"`, transforms a function that acts on pairs of elements to a function that acts on neighbor lists.

        -   `"pair"`, transforms a function that acts on a pair of elements to a function that acts on a system of interacting elements.

        -   `"triplet"`, transforms a function that acts on triplets of elements to a function that acts on a system of interacting elements. Many common empirical potentials include three-body terms, this type of pairwise interaction simplifies the loss computation by transforming a loss function that acts on two pairwise displacements or distances to a loss function that acts on a system of interacting elements.

    bonds : Optional[Tensor], default=None

    kinds : Optional[Tensor], default=None
        Kinds for the different elements. Should either be `None` (in which case it is assumed that all the elements have the same kind) or labels of shape `(n)`. If `intraction` is `"pair"` or `"triplet"`, kinds can be dynamically specified by passing the `kinds` keyword argument to the mapped function.

    dim : Optional[Union[int, Tuple[int, ...]]], default=None
        Dimension or dimensions to reduce. If `None`, all dimensions are reduced.

    keepdim : bool, default=False
        Whether the output has `dim` retained or not.

    ignore_unused_parameters : bool, default=True

    kwargs :
        `kwargs` passed to the function. Depends on the `interaction` type:

        *   If `interaction` is `"bond"` and `kinds` is `None`, must be a scalar or a tensor of shape `(n)`. If `interaction` is `"bond"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"neighbor"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)`, tensor of shape `(n, n)`, a PyTree of parameters and corresponding mapping, or a binary function that determines how per-element parameters are combined. If `kinds` is `None`, `kinds` is defined as the average of the two per-element parameters. If `interaction` is `"neighbor"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds, kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"pair"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)`, tensor of shape `(n, n)`, a PyTree of parameters and corresponding mapping, or a binary function that determines how per-element parameters are combined. If `kinds` is `None`, `kinds` is defined as the average of the two per-element parameters. If `interaction` is `"pair"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds, kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"triplet"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)` based on the central element, or a tensor of shape `(n, n, n)` defining triplet interactions. If `interaction` is `"triplet"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds)`, or a tensor of shape `(kinds, kinds, kinds)` defining triplet interactions.

    Returns
    -------
    : Callable[..., Tensor]
        Signature of the return function depends on `interaction`:

        *   `"bond"`:
                `(positions, bonds, kinds) -> Tensor`

            The return function can optionally take the keyword arguments `bonds` and `kinds` to dynamically allocate bonds.

        *   `"neighbor"`:
                `(positions, neighbors) -> Tensor`

            The return function takes positions of shape `(n, spatial_dimension)` and neighbor labels of shape `(n, neighbors)`.

        *   `"pair"`:
                `(positions, kinds, maximum_kind, **kwargs) -> Tensor`

            If `kinds` is `None` or static, the return function takes positions of shape `(n, spatial_dimension)`. If `kinds` is dynamic, the return function takes positions of shape `(n, spatial_dimension)`, integer labels of shape (n), and an integer specifying the maximum kind. The return function can optionally take keyword arguments to pass to the displacement function.

        *   `"triplet"`:
                `(positions, kinds, maximum_kind, **kwargs) -> Tensor`

            If `kinds` is `None` or static, the return function takes positions of shape `(n, spatial_dimension)`. If `kinds` is dynamic, the return function takes positions of shape `(n, spatial_dimension)`, integer labels of shape (n), and an integer specifying the maximum kind. The return function can optionally take keyword arguments to pass to the displacement function.

    Examples
    --------
    Create a pairwise interaction from a potential function:

        def fn(x: Tensor, a: float, e: float, s: float, **_) -> Tensor:
            return e / a * (1.0 - x / s) ** a

        displacement_fn, _ = prescient.func.space([10.0], parallelpiped=False)

        fn = prescient.func.interact(
            fn,
            displacement_fn,
            interaction="pair",
            a=2.0,
            e=1.0,
            s=1.0,
        )
    """
    match interaction:
        case "angle":
            return _angle_interaction(
                fn,
                displacement_fn,
            )
        case "bond":
            return _bond_interaction(
                fn,
                displacement_fn,
                static_bonds=bonds,
                static_kinds=kinds,
                ignore_unused_parameters=ignore_unused_parameters,
                **kwargs,
            )
        case "dihedral":
            return _dihedral_interaction(
                fn,
                displacement_fn,
            )
        case "long-range":
            return _long_range_interaction(
                fn,
                displacement_fn,
            )
        case "mesh":
            return _mesh_interaction(
                fn,
                displacement_fn,
            )
        case "neighbor_list":
            return _neighbor_list_interaction(
                fn,
                displacement_fn,
                kinds=kinds,
                dim=dim,
                ignore_unused_parameters=ignore_unused_parameters,
            )
        case "pair":
            return _pair_interaction(
                fn,
                displacement_fn,
                kinds=kinds,
                dim=dim,
                keepdim=keepdim,
                ignore_unused_parameters=ignore_unused_parameters,
                **kwargs,
            )
        case _:
            raise ValueError
