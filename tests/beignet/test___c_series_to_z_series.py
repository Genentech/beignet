import beignet.polynomial
import beignet.polynomial.__c_series_to_z_series
import torch


def test__c_series_to_z_series():
    for i in range(5):
        torch.testing.assert_close(
            beignet.polynomial.__c_series_to_z_series._c_series_to_z_series(
                torch.tensor(
                    [2] + [1] * i,
                    dtype=torch.float64,
                )
            ),
            torch.tensor(
                [0.5] * i + [2] + [0.5] * i,
                dtype=torch.float64,
            ),
        )
