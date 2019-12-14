import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar

from share_funcs import get_model, get_loaders, get_criterion, get_optimizer


def run(epochs, model, criterion, optimizer, scheduler,
        train_loader, val_loader, device):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(), 'nll': Loss(criterion)},
        device=device
    )

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        scheduler.step()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)


def main():
    epochs = 10
    train_loader, val_loader = get_loaders()
    model = get_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer, scheduler = get_optimizer(model)
    criterion = get_criterion()

    run(
        epochs=epochs,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )


if __name__ == "__main__":
    main()
