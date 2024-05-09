_Beignet is a work-in-progress. Contributions and feedback are encouraged!_

# Beignet

A standard library for biological research.

## Installation

```bash
pip install beignet
```

_Requires Python 3.10 or later and PyTorch 2.2.0 or later._

## Layout

*   `beignet`: PyTorch operators
    *   `beignet.datasets`: PyTorch [datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
    *   `beignet.func`: [composable PyTorch function transforms](https://pytorch.org/docs/stable/func.html)
    *   `beignet.lightning`: Lightning [modules](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
        *   `beignet.lightning.datamodules`: Lightning [datamodules](https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
    *   `beignet.metrics`: [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) metrics
        *   `beignet.metrics.functional`: functional [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) metrics
    *   `beignet.nn`: PyTorch [modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
        *   `beignet.nn.functional`: 
    *   `beignet.optim`: PyTorch [optimizers](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
    *   `beignet.samplers`: PyTorch [samplers](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)
    *   `beignet.transforms`: 
        *   `beignet.transforms.functional`:
