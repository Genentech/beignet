import torch

from beignet.nn import OuterProductMean


class TimeOuterProductMean:
    params = ([1, 10], [32, 64], [128, 256], [torch.float32, torch.float64])
    param_names = ["batch_size", "c", "c_z", "dtype"]

    def setup(self, batch_size, c, c_z, dtype):
        torch.manual_seed(42)

        device = torch.device("cpu")
        seq_len = 50  # Fixed sequence length for benchmarking
        n_seq = 16  # Number of MSA sequences

        # Create module
        self.module = OuterProductMean(c=c, c_z=c_z).to(device).to(dtype)

        # Generate input - MSA representation
        self.m_si = torch.randn(
            batch_size, seq_len, n_seq, c, dtype=dtype, device=device
        )

        # Compile the module for optimal performance
        try:
            self.compiled_module = torch.compile(self.module, fullgraph=True)
        except Exception:
            self.compiled_module = self.module

    def time_outer_product_mean(self, batch_size, c, c_z, dtype):
        return self.module(self.m_si)

    def time_outer_product_mean_compiled(self, batch_size, c, c_z, dtype):
        return self.compiled_module(self.m_si)


class PeakMemoryOuterProductMean:
    params = ([1, 5], [32], [128], [torch.float32])
    param_names = ["batch_size", "c", "c_z", "dtype"]

    def setup(self, batch_size, c, c_z, dtype):
        torch.manual_seed(42)

        device = torch.device("cpu")
        seq_len = 50  # Fixed sequence length for memory benchmarking
        n_seq = 16  # Number of MSA sequences

        # Create module
        self.module = OuterProductMean(c=c, c_z=c_z).to(device).to(dtype)

        # Generate input - MSA representation
        self.m_si = torch.randn(
            batch_size, seq_len, n_seq, c, dtype=dtype, device=device
        )

    def peakmem_outer_product_mean(self, batch_size, c, c_z, dtype):
        return self.module(self.m_si)
