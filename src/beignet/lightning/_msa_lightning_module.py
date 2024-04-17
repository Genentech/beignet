from lightning import LightningModule

from beignet.nn import MSA


class MSALightningModule(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        kernel_size: int = 18,
        *,
        gap_penalty: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.module = MSA(
            in_channels,
            out_channels,
            kernel_size,
            gap_penalty=gap_penalty,
            temperature=temperature,
        )

    def forward(self, x):
        return self.model(x)
