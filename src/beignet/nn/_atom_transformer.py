import torch
import torch.nn as nn
from torch import Tensor

from ._diffusion_transformer import DiffusionTransformer


class AtomTransformer(nn.Module):
    r"""
    Atom Transformer for AlphaFold 3.

    This module implements sequence-local atom attention using rectangular blocks
    along the diagonal. It applies the DiffusionTransformer with sequence-local
    attention masking based on query and key positions.

    Parameters
    ----------
    n_block : int, default=3
        Number of transformer blocks
    n_head : int, default=4
        Number of attention heads
    n_queries : int, default=32
        Number of queries per block
    n_keys : int, default=128
        Number of keys per block
    subset_centres : list, default=[15.5, 47.5, 79.5, ...]
        Centers for subset selection
    c_q : int, default=None
        Query dimension (inferred from input if None)
    c_kv : int, default=None
        Key-value dimension (inferred from input if None)
    c_pair : int, default=None
        Pair dimension (inferred from input if None)

    Examples
    --------
    >>> import torch
    >>> from beignet.nn import AtomTransformer
    >>> batch_size, n_atoms = 2, 1000
    >>> module = AtomTransformer()
    >>> q = torch.randn(batch_size, n_atoms, 128)
    >>> c = torch.randn(batch_size, n_atoms, 64)
    >>> p = torch.randn(batch_size, n_atoms, n_atoms, 16)
    >>> output = module(q, c, p)
    >>> output.shape
    torch.Size([2, 1000, 128])

    References
    ----------
    .. [1] AlphaFold 3 Algorithm 7: Atom Transformer
    """

    def __init__(
        self,
        n_block: int = 3,
        n_head: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        subset_centres: list = None,
        c_q: int = None,
        c_kv: int = None,
        c_pair: int = None,
    ):
        super().__init__()

        self.n_block = n_block
        self.n_head = n_head
        self.n_queries = n_queries
        self.n_keys = n_keys

        if subset_centres is None:
            # Default subset centers as specified in the algorithm
            self.subset_centres = [15.5, 47.5, 79.5]  # Can be extended as needed
        else:
            self.subset_centres = subset_centres

        # Store dimensions (will be inferred from input if not provided)
        self.c_q = c_q
        self.c_kv = c_kv
        self.c_pair = c_pair

        # Will be initialized in first forward pass
        self.diffusion_transformer = None

    def _create_sequence_local_mask(self, q: Tensor, beta_lm: Tensor) -> Tensor:
        """
        Create sequence-local attention mask based on Algorithm 7.

        Parameters
        ----------
        q : Tensor, shape=(batch_size, n_atoms, c_q)
            Query tensor
        beta_lm : Tensor, shape=(batch_size, n_atoms, n_atoms, n_head)
            Base attention bias

        Returns
        -------
        beta_lm : Tensor, shape=(batch_size, n_atoms, n_atoms, n_head)
            Modified attention bias with sequence-local masking
        """
        batch_size, n_atoms = q.shape[:2]
        device = q.device

        # Create position indices
        l_idx = torch.arange(n_atoms, device=device)  # (n_atoms,)
        m_idx = torch.arange(n_atoms, device=device)  # (n_atoms,)

        # Create meshgrid for all pairs
        l_grid, m_grid = torch.meshgrid(
            l_idx, m_idx, indexing="ij"
        )  # (n_atoms, n_atoms)

        # Initialize mask with -10^10 (effectively -inf)
        mask = torch.full_like(beta_lm, -1e10)

        # For each subset center, create rectangular blocks along diagonal
        for c in self.subset_centres:
            # Condition: |l - c| < n_queries/2 ∧ |m - c| < n_keys/2
            l_condition = torch.abs(l_grid - c) < (self.n_queries / 2)
            m_condition = torch.abs(m_grid - c) < (self.n_keys / 2)

            # Combined condition for this subset
            subset_condition = l_condition & m_condition  # (n_atoms, n_atoms)

            # Expand to match beta_lm shape
            subset_condition = subset_condition.unsqueeze(0).unsqueeze(
                -1
            )  # (1, n_atoms, n_atoms, 1)
            subset_condition = subset_condition.expand(batch_size, -1, -1, self.n_head)

            # Set mask to 0 where condition is satisfied
            mask = torch.where(subset_condition, 0.0, mask)

        # Apply mask to beta_lm
        beta_lm = beta_lm + mask

        return beta_lm

    def forward(self, q: Tensor, c: Tensor, p: Tensor) -> Tensor:
        r"""
        Forward pass of Atom Transformer.

        Parameters
        ----------
        q : Tensor, shape=(batch_size, n_atoms, c_q)
            Query representations
        c : Tensor, shape=(batch_size, n_atoms, c_kv)
            Context (single) representations
        p : Tensor, shape=(batch_size, n_atoms, n_atoms, c_pair)
            Pair representations

        Returns
        -------
        q : Tensor, shape=(batch_size, n_atoms, c_q)
            Updated query representations
        """
        # Infer dimensions from input if not provided
        if self.diffusion_transformer is None:
            c_q = q.shape[-1] if self.c_q is None else self.c_q
            c_kv = c.shape[-1] if self.c_kv is None else self.c_kv
            c_pair = p.shape[-1] if self.c_pair is None else self.c_pair

            self.diffusion_transformer = DiffusionTransformer(
                c_a=c_q,  # Use query dimension as token dimension
                c_s=c_kv,  # Use context dimension as single dimension
                c_z=c_pair,  # Use pair dimension
                n_head=self.n_head,
                n_block=self.n_block,
            )

            # Move to same device as input
            self.diffusion_transformer = self.diffusion_transformer.to(q.device)

        # Create initial beta_lm (starts as zeros, will be modified by masking)
        batch_size, n_atoms = q.shape[:2]
        beta_lm = torch.zeros(
            batch_size, n_atoms, n_atoms, self.n_head, device=q.device, dtype=q.dtype
        )

        # Apply sequence-local masking
        beta_lm = self._create_sequence_local_mask(q, beta_lm)

        # Apply DiffusionTransformer
        q = self.diffusion_transformer(q, c, p, beta_lm)

        return q
