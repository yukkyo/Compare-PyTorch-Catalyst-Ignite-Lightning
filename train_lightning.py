import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from share_funcs import get_model, get_loaders, get_criterion, get_optimizer


class MyLightninModule(pl.LightningModule):
    def __init__(self, num_class):
        super(MyLightninModule, self).__init__()
        self.model = get_model(num_class=num_class)
        self.criterion = get_criterion()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        preds = torch.argmax(y_hat, dim=1)
        return {'val_loss': self.criterion(y_hat, y), 'correct': (preds == y).float()}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.cat([x['correct'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        # REQUIRED
        optimizer, scheduler = get_optimizer(model=self.model)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return get_loaders()[0]

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return get_loaders()[1]


def main():
    epochs = 5
    num_class = 10
    output_path = './output/lightning'

    model = MyLightninModule(num_class=num_class)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=epochs,
        default_save_path=output_path,
        gpus=[0],
        # use_amp=False,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
