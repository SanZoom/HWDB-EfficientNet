import os
from torch.utils.tensorboard import SummaryWriter

from network import *
import dataset


def train_loop(dataloader, model, loss_fn, optimizer, device, logger_writer=None, epoch=0):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if logger_writer is not None:
            correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            correct /= len(y)
            logger_writer.add_scalar(tag='loss/train', scalar_value=loss.item(),
                                     global_step=epoch * len(dataloader) + batch)
            logger_writer.add_scalar(tag='Accuracy/train', scalar_value=correct,
                                     global_step=epoch * len(dataloader) + batch)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            correct /= len(y)
            print(f"loss: {loss:>7f}    Accuracy:{correct * 100:>7f}%  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, device, logger_writer=None, epoch=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if logger_writer is not None:
        logger_writer.add_scalar(tag='Accuracy/test', scalar_value=correct,
                                 global_step=epoch)
        logger_writer.add_scalar(tag='loss/test', scalar_value=test_loss,
                                 global_step=epoch)

    # global best_accuracy, best_module_file
    # if correct > best_accuracy:
    #     best_accuracy = correct
        # torch.save(module.state_dict(), best_module_file)


epochs = 500
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
module_file = './model/b0_weight.pth'
best_module_file = './model/best_b0_weight.pth'

train_dataloader, test_dataloader = dataset.get_dataloader()
module = efficientnet_b0(3, 3926, use_stn=True, input_size=64).to(device)


if os.path.exists(best_module_file):
    module.load_state_dict(torch.load(best_module_file))

optimizer = torch.optim.Adam(module.parameters(), lr)
loss_fn = torch.nn.CrossEntropyLoss()
writer = SummaryWriter()
best_accuracy = 0

for t in range(epochs):
    print(f'Epoch{t + 1}\n--------------------')
    # train_loop(train_dataloader, module, loss_fn, optimizer, device, writer, t)
    test_loop(test_dataloader, module, loss_fn, device, writer, t)

    # torch.save(module.state_dict(), module_file)

