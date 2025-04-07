import importlib
import sys

import torch
import torch.nn as nn
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
from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.utils.tensor_utils import (
    dict_multimap,
)
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Dropout

from ._backbone_update import BackboneUpdate
from ._invariant_point_attention import InvariantPointAttention
from ._resnet import ResNet
from ._transition import Transition

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")


class MonomerStructure(Module):
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
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(MonomerStructure, self).__init__()

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

        self.ipa = InvariantPointAttention(
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

    def _forward_monomer(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            self.training,
            fmt="quat",
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids.compose_q_update_vec(self.backbone_update(s))

            # To hew as closely as possible to AlphaFold, we convert our
            # quaternion-based transformations to rotation-matrix ones
            # here
            backb_to_global = Rigid(
                Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(self.trans_scale_factor)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global,
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if _offload_inference:
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].to(s.device)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def _forward_multimer(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if _offload_inference:
            assert sys.getrefcount(evoformer_output_dict["pair"]) == 2
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid3Array.identity(
            s.shape[:-1],
            s.device,
        )
        outputs = []
        for _ in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference,
                _z_reference_list=z_reference_list,
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids @ self.backbone_update(s)

            # [*, N, 7, 2]
            unnormalized_angles, angles = self.resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                rigids.scale_translation(self.trans_scale_factor),
                angles,
                aatype,
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
            )

            preds = {
                "frames": rigids.scale_translation(self.trans_scale_factor).to_tensor(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }

            preds = {k: v.to(dtype=s.dtype) for k, v in preds.items()}

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z, z_reference_list

        if _offload_inference:
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].to(s.device)

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        if self.is_multimer:
            outputs = self._forward_multimer(
                evoformer_output_dict, aatype, mask, inplace_safe, _offload_inference
            )
        else:
            outputs = self._forward_monomer(
                evoformer_output_dict, aatype, mask, inplace_safe, _offload_inference
            )

        return outputs

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
