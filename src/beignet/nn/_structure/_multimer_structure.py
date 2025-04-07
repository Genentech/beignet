import torch
from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array

from .__structure import Structure


def dict_multimap(func, dictionaries):
    updated_dictionary = {}

    for k, v in dictionaries[0].items():
        vs = []

        for dictionary in dictionaries:
            vs.append(dictionary[k])

        if isinstance(v, dict):
            updated_dictionary[k] = dict_multimap(func, vs)
        else:
            updated_dictionary[k] = func(vs)

    return updated_dictionary


class MultimerStructure(Structure):
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
        super().__init__(
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
        )

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
    ):
        s = evoformer_output_dict["single"]

        if mask is None:
            mask = s.new_ones(s.shape[:-1])  # [*, n]

        s = self.layer_norm_s(s)  # [*, n, C_s]

        z = evoformer_output_dict["pair"]
        z = self.layer_norm_z(z)  # [*, n, n, C_z]

        z_references = None

        # [*, n, C_s]
        s_initial = s
        s = self.linear_in(s)

        rigids = Rigid3Array.identity(s.shape[:-1], s.device)  # [*, n]

        outputs = []

        for _ in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(
                s,
                z,
                rigids,
                mask,
                _z_reference_list=z_references,
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

        del z, z_references

        outputs = dict_multimap(torch.stack, outputs)

        outputs["single"] = s

        return outputs
