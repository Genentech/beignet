import torch

from beignet.nn import MultipleSequenceAlignment


class TimeMultipleSequenceAlignment:
    params = ([1, 2], [1, 2], [64], [128], [torch.float32])
    param_names = ["batch_size", "n_block", "c_m", "c_z", "dtype"]

    def setup(self, batch_size, n_block, c_m, c_z, dtype):
        torch.manual_seed(42)

        device = torch.device("cpu")
        seq_len = 32  # Reasonable sequence length for benchmarking
        n_seq = 16  # Number of MSA sequences
        c_s = 256  # Single representation dimension

        # Create module
        self.module = (
            MultipleSequenceAlignment(
                n_block=n_block,
                c_m=c_m,
                c_z=c_z,
                c_s=c_s,
                n_head_msa=8,
                n_head_pair=4,
                dropout_rate=0.15,
            )
            .to(device)
            .to(dtype)
        )

        # Generate input features according to Algorithm 8
        self.f_msa = torch.randn(
            batch_size, seq_len, n_seq, 23, dtype=dtype, device=device
        )
        self.f_has_deletion = torch.randn(
            batch_size, seq_len, n_seq, 1, dtype=dtype, device=device
        )
        self.f_deletion_value = torch.randn(
            batch_size, seq_len, n_seq, 1, dtype=dtype, device=device
        )
        self.s_inputs = torch.randn(
            batch_size, seq_len, c_s, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )

        # Try to compile the module for optimal performance
        try:
            self.compiled_module = torch.compile(self.module, fullgraph=True)
        except Exception:
            self.compiled_module = self.module

    def time_multiple_sequence_alignment(self, batch_size, n_block, c_m, c_z, dtype):
        return self.module(
            self.f_msa,
            self.f_has_deletion,
            self.f_deletion_value,
            self.s_inputs,
            self.z_ij,
        )

    def time_multiple_sequence_alignment_compiled(
        self, batch_size, n_block, c_m, c_z, dtype
    ):
        return self.compiled_module(
            self.f_msa,
            self.f_has_deletion,
            self.f_deletion_value,
            self.s_inputs,
            self.z_ij,
        )


class PeakMemoryMultipleSequenceAlignment:
    params = ([1], [1], [64], [128], [torch.float32])
    param_names = ["batch_size", "n_block", "c_m", "c_z", "dtype"]

    def setup(self, batch_size, n_block, c_m, c_z, dtype):
        torch.manual_seed(42)

        device = torch.device("cpu")
        seq_len = 24  # Smaller for memory benchmarking
        n_seq = 12  # Fewer MSA sequences
        c_s = 256  # Single representation dimension

        # Create module
        self.module = (
            MultipleSequenceAlignment(
                n_block=n_block,
                c_m=c_m,
                c_z=c_z,
                c_s=c_s,
                n_head_msa=8,
                n_head_pair=4,
                dropout_rate=0.15,
            )
            .to(device)
            .to(dtype)
        )

        # Generate input features
        self.f_msa = torch.randn(
            batch_size, seq_len, n_seq, 23, dtype=dtype, device=device
        )
        self.f_has_deletion = torch.randn(
            batch_size, seq_len, n_seq, 1, dtype=dtype, device=device
        )
        self.f_deletion_value = torch.randn(
            batch_size, seq_len, n_seq, 1, dtype=dtype, device=device
        )
        self.s_inputs = torch.randn(
            batch_size, seq_len, c_s, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )

    def peakmem_multiple_sequence_alignment(self, batch_size, n_block, c_m, c_z, dtype):
        return self.module(
            self.f_msa,
            self.f_has_deletion,
            self.f_deletion_value,
            self.s_inputs,
            self.z_ij,
        )
