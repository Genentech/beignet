import importlib

from openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.geometry.quat_rigid import QuatRigid
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module

from ._backbone_update import BackboneUpdate
from ._monomer_invariant_point_attention import MonomerInvariantPointAttention
from ._resnet import ResNet
from ._transition import Transition

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")


class Structure(Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        is_multimer=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        c_s: int
            Single representation channel dimension

        c_z: int
            Pair representation channel dimension

        c_ipa: int
            IPA hidden channel dimension

        c_resnet: int
            Angle resnet (Alg. 23 lines 11-14) hidden channel dimension

        no_heads_ipa: int
            Number of IPA heads

        no_qk_points: int
            Number of query/key points to generate during IPA

        no_v_points: int
            Number of value points to generate during IPA

        dropout_rate: float
            Dropout rate used throughout the layer

        no_blocks: int
            Number of structure module blocks

        no_transition_layers: int
            Number of layers in the single representation transition
            (Alg. 23 lines 8-9)

        no_resnet_blocks: int
            Number of blocks in the angle resnet

        no_angles: int
            Number of angles to generate in the angle resnet

        trans_scale_factor:
            Scale of single representation transition hidden dimension

        epsilon:
            Small number used in angle resnet normalization

        inf:
            Large number used for attention masking
        """
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf
        self.is_multimer = is_multimer

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = MonomerInvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
            is_multimer=self.is_multimer,
        )

        self.ipa_dropout = Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = Transition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        if self.is_multimer:
            self.backbone_update = QuatRigid(self.c_s, full_quat=False)
        else:
            self.backbone_update = BackboneUpdate(self.c_s)

        self.resnet = ResNet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                Tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                Tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                Tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                Tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self,
        r,
        f,  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.dtype, r.device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
