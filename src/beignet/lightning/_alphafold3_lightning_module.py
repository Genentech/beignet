from lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from beignet import smooth_local_distance_difference_test
from beignet.nn import AlphaFold3


class AlphaFold3LightningModule(LightningModule):
    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        *,
        module: Module = AlphaFold3,
    ):
        super().__init__()

        self.loss_weights = {
            "aligned_error": 0.5,
            "diffusion": 1.0,
            "distance_error": 0.25,
            "distogram": 1.0,
            "experimentally_resolved": 0.25,
            "local_distance_difference_test": 0.0,
        }

        self.module = module

        self.optimizer, self.scheduler = optimizer, scheduler

        self.save_hyperparameters(logger=False, ignore=["module"])

    def forward(
        self,
        x: Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
        return self.module(x)

    def loss(
        self,
        input: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor),
        target: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor),
    ) -> Tensor:
        (
            input,
            local_distance_difference_test,
            aligned_error,
            distance_error,
            experimentally_resolved,
            distogram,
        ) = input

        (
            target,
            target_local_distance_difference_test,
            target_aligned_error,
            target_distance_error,
            target_experimentally_resolved,
            target_distogram,
        ) = target

        smooth_local_distance_difference_test(
            local_distance_difference_test,
            target_local_distance_difference_test,
        )

        return local_distance_difference_test

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        output = self(inputs)

        loss = self.loss(output, targets)

        self.log("Loss (Train)", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        outputs = self(inputs)

        loss = self.loss(outputs, targets)

        self.log("Loss (Validation)", loss)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters())

        if self.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)

            if hasattr(scheduler, "T_max"):
                scheduler.T_max = self.trainer.max_epochs

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "Loss (Validation)",
                },
            }

        return optimizer
