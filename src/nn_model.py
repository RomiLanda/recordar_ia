import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import F1Score
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningModule


LEARNING_RATE = 0.0001


class Model(LightningModule):
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        hidden_channels: int,
        n_features: int,
        n_classes: int,
    ):
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.lin1 = nn.Linear(512, n_features // 2)
        self.sig1 = nn.Sigmoid()

        self.sage_conv1 = SAGEConv(
            n_features,
            hidden_channels,
            aggr="mean",
        )

        self.sage_conv2 = SAGEConv(hidden_channels, n_classes, aggr="mean")

        self.ce_loss = CrossEntropyLoss()
        self.f1 = F1Score('multiclass', num_classes = n_classes, top_k=1, average="macro")

    
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
    ) -> torch.Tensor:

        x = self.sage_conv1(x, edge_index)
        x = x.relu()

        x = self.sage_conv2(x, edge_index)
        return x


    def training_step(
        self, batch: torch.Tensor, batch_index: torch.Tensor
    ) -> torch.Tensor:

        x, edge_index = (
            batch.x,
            batch.edge_index,
        )

        x_out = self.forward(x, edge_index)
        loss = self.ce_loss(x_out, batch.y)

        preds = x_out.argmax(dim=1)
        self.f1(preds, batch.y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.f1, prog_bar=True)

        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        """
        When the validation_step() is called,
        the model has been put in eval mode
        and PyTorch gradients have been disabled.
        At the end of validation, the model goes back to training mode
        and gradients are enabled.
        """

        x, edge_index = (
            batch.x,
            batch.edge_index,
        )

        x_out = self.forward(x,edge_index)
        loss = self.ce_loss(x_out, batch.y)

        preds = x_out.argmax(dim=1)
        self.f1(preds, batch.y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.f1, prog_bar=True)


    def predict_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> list:

        x, edge_index = (
            batch.x,
            batch.edge_index,
        )
        pred = self(x, edge_index)

        pred = pred.softmax(dim=1)
        confidences = pred.max(dim=1)
        pred = pred.argmax(dim=1)

        return pred, confidences


    def train_dataloader(self):
        return self.train_loader


    def val_dataloader(self):
        return self.val_loader


    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer