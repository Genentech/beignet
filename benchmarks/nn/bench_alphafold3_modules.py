import torch

from beignet.nn import (
    MSAPairWeightedAveraging,
    Transition,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class TimeTriangleMultiplicationOutgoing:
    params = ([1, 4], [32, 64], [torch.float32, torch.float64])
    param_names = ["batch_size", "c", "dtype"]

    def setup(self, batch_size, c, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 16

        self.module = TriangleMultiplicationOutgoing(c=c).to(device).to(dtype)
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c, dtype=dtype, device=device
        )

        try:
            self.compiled_module = torch.compile(self.module, fullgraph=True)
        except Exception:
            self.compiled_module = self.module

    def time_triangle_multiplication_outgoing(self, batch_size, c, dtype):
        return self.module(self.z_ij)

    def time_triangle_multiplication_outgoing_compiled(self, batch_size, c, dtype):
        return self.compiled_module(self.z_ij)


class TimeTriangleMultiplicationIncoming:
    params = ([1, 4], [32, 64], [torch.float32, torch.float64])
    param_names = ["batch_size", "c", "dtype"]

    def setup(self, batch_size, c, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 16

        self.module = TriangleMultiplicationIncoming(c=c).to(device).to(dtype)
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c, dtype=dtype, device=device
        )

    def time_triangle_multiplication_incoming(self, batch_size, c, dtype):
        return self.module(self.z_ij)


class TimeTriangleAttentionStartingNode:
    params = ([1, 4], [32], [4], [torch.float32])
    param_names = ["batch_size", "c", "n_head", "dtype"]

    def setup(self, batch_size, c, n_head, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 12

        self.module = (
            TriangleAttentionStartingNode(c=c, n_head=n_head).to(device).to(dtype)
        )
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c, dtype=dtype, device=device
        )

    def time_triangle_attention_starting_node(self, batch_size, c, n_head, dtype):
        return self.module(self.z_ij)


class TimeTriangleAttentionEndingNode:
    params = ([1, 4], [32], [4], [torch.float32])
    param_names = ["batch_size", "c", "n_head", "dtype"]

    def setup(self, batch_size, c, n_head, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 12

        self.module = (
            TriangleAttentionEndingNode(c=c, n_head=n_head).to(device).to(dtype)
        )
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c, dtype=dtype, device=device
        )

    def time_triangle_attention_ending_node(self, batch_size, c, n_head, dtype):
        return self.module(self.z_ij)


class TimeMSAPairWeightedAveraging:
    params = ([1, 2], [32], [32], [4], [torch.float32])
    param_names = ["batch_size", "c_m", "c_z", "n_head", "dtype"]

    def setup(self, batch_size, c_m, c_z, n_head, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 16
        n_seq = 8

        self.module = (
            MSAPairWeightedAveraging(c_m=c_m, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.m_si = torch.randn(
            batch_size, seq_len, n_seq, c_m, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )

    def time_msa_pair_weighted_averaging(self, batch_size, c_m, c_z, n_head, dtype):
        return self.module(self.m_si, self.z_ij)


class TimeTransition:
    params = ([1, 8], [64, 128], [2, 4], [torch.float32, torch.float64])
    param_names = ["batch_size", "c", "n", "dtype"]

    def setup(self, batch_size, c, n, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 64

        self.module = Transition(c=c, n=n).to(device).to(dtype)
        self.x = torch.randn(batch_size, seq_len, c, dtype=dtype, device=device)

        try:
            self.compiled_module = torch.compile(self.module, fullgraph=True)
        except Exception:
            self.compiled_module = self.module

    def time_transition(self, batch_size, c, n, dtype):
        return self.module(self.x)

    def time_transition_compiled(self, batch_size, c, n, dtype):
        return self.compiled_module(self.x)


# Memory benchmarks for smaller configurations
class PeakMemoryTriangleMultiplicationOutgoing:
    params = ([1, 2], [32], [torch.float32])
    param_names = ["batch_size", "c", "dtype"]

    def setup(self, batch_size, c, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 16

        self.module = TriangleMultiplicationOutgoing(c=c).to(device).to(dtype)
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c, dtype=dtype, device=device
        )

    def peakmem_triangle_multiplication_outgoing(self, batch_size, c, dtype):
        return self.module(self.z_ij)


class PeakMemoryMSAPairWeightedAveraging:
    params = ([1], [32], [32], [4], [torch.float32])
    param_names = ["batch_size", "c_m", "c_z", "n_head", "dtype"]

    def setup(self, batch_size, c_m, c_z, n_head, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 16
        n_seq = 8

        self.module = (
            MSAPairWeightedAveraging(c_m=c_m, c_z=c_z, n_head=n_head)
            .to(device)
            .to(dtype)
        )
        self.m_si = torch.randn(
            batch_size, seq_len, n_seq, c_m, dtype=dtype, device=device
        )
        self.z_ij = torch.randn(
            batch_size, seq_len, seq_len, c_z, dtype=dtype, device=device
        )

    def peakmem_msa_pair_weighted_averaging(self, batch_size, c_m, c_z, n_head, dtype):
        return self.module(self.m_si, self.z_ij)


class PeakMemoryTransition:
    params = ([1, 4], [64], [2], [torch.float32])
    param_names = ["batch_size", "c", "n", "dtype"]

    def setup(self, batch_size, c, n, dtype):
        torch.manual_seed(42)
        device = torch.device("cpu")
        seq_len = 64

        self.module = Transition(c=c, n=n).to(device).to(dtype)
        self.x = torch.randn(batch_size, seq_len, c, dtype=dtype, device=device)

    def peakmem_transition(self, batch_size, c, n, dtype):
        return self.module(self.x)
