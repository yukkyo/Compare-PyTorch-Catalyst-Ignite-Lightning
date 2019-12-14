import torch
from tqdm import tqdm

from share_funcs import get_model, get_loaders, get_criterion, get_optimizer


def train(model, data_loader, criterion, optimizer, device, grad_acc=1):
    model.train()

    # zero the parameter gradients
    optimizer.zero_grad()

    total_loss = 0.
    for i, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient accumulation
        if (i % grad_acc) == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    total_loss /= len(data_loader)
    metrics = {'train_loss': total_loss}
    return metrics


def eval(model, data_loader, criterion, device):
    model.eval()
    num_correct = 0.

    with torch.no_grad():
        total_loss = 0.
        for inputs, labels in tqdm(data_loader, total=len(data_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_correct += torch.sum(preds == labels.data)

        total_loss /= len(data_loader)
        num_correct /= len(data_loader.dataset)
        metrics = {'valid_loss': total_loss, 'val_acc': num_correct}
    return metrics


def main():
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    train_loader, val_loader = get_loaders()
    optimizer, lr_scheduler = get_optimizer(model=model)
    criterion = get_criterion()

    # Model を multi-gpu したり、FP16 対応したりする
    model = model.to(device)

    print('Train start !')
    for epoch in range(epochs):
        print(f'epoch {epoch} start !')
        metrics_train = train(model, train_loader, criterion, optimizer, device)
        metrics_eval = eval(model, val_loader, criterion, device)

        lr_scheduler.step()

        # Logger 周りの処理
        # print するためのごちゃごちゃした処理
        print(f'epoch: {epoch} ', metrics_train, metrics_eval)

        # tqdm 使ってたらさらにごちゃごちゃする処理をここに書く
        # Model を保存するための処理
        # Multi-GPU の場合さらに注意して書く


if __name__ == '__main__':
    main()
