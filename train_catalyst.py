import catalyst
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback

from share_funcs import get_model, get_loaders, get_criterion, get_optimizer


def main():
    epochs = 5
    num_class = 10
    output_path = './output/catalyst'

    # Use if you want to fix seed
    # catalyst.utils.set_global_seed(42)
    # catalyst.utils.prepare_cudnn(deterministic=True)

    model = get_model()
    train_loader, val_loader = get_loaders()
    loaders = {"train": train_loader, "valid": val_loader}

    optimizer, lr_scheduler = get_optimizer(model=model)
    criterion = get_criterion()

    runner = SupervisedRunner(device=catalyst.utils.get_device())
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loaders=loaders,
        logdir=output_path,
        callbacks=[AccuracyCallback(num_classes=num_class, accuracy_args=[1])],
        num_epochs=epochs,
        main_metric="accuracy01",
        minimize_metric=False,
        fp16=None,
        verbose=True
    )


if __name__ == '__main__':
    main()
