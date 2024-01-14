import torch
from exercise_code.networks.segmentation_nn import SegmentationNN
from tqdm import tqdm


def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


model = SegmentationNN(hparams=hparams)

import torch.optim as optim

num_epochs = 15
log_nth = 5  # log_nth: log training accuracy and loss every nth iteration
batch_size = 8

train_loss_history = []
train_acc_history = []
val_acc_history = []
val_loss_history = []

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

iter_per_epoch = len(train_loader)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model.to(device)
loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-04,
                                                steps_per_epoch=len(
                                                    train_loader),
                                                epochs=num_epochs,
                                                div_factor=2, pct_start=0.05)

print('START TRAIN.')

for epoch in range(num_epochs):
    # TRAINING
    train_acc_epoch = []
    train_loss_epoch = []
    train_loop = create_tqdm_bar(train_loader,
                                 desc=f'Train, Epoch [{epoch + 1}/{num_epochs}]')
    for i, (inputs, targets) in train_loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        scheduler.step()

        train_loss_epoch.append(loss.item())

        _, preds = torch.max(outputs, 1)

        # Only allow images/pixels with label >= 0 e.g. for segmentation
        targets_mask = targets >= 0
        train_acc = np.mean((preds == targets)[
                                targets_mask].detach().cpu().numpy())
        train_acc_epoch.append(train_acc)
        train_loop.set_postfix(train_loss=np.mean(train_loss_epoch),
                               train_acc=np.mean(train_acc_epoch))

    # VALIDATION
    val_losses = []
    val_scores = []
    model.eval()
    eval_loop = create_tqdm_bar(val_loader,
                                desc=f'Eval, Epoch [{epoch + 1}/{num_epochs}]')
    for i, (inputs, targets) in eval_loop:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model.forward(inputs)
        loss = loss_func(outputs, targets)
        val_losses.append(loss.item())

        _, preds = torch.max(outputs, 1)

        # Only allow images/pixels with target >= 0 e.g. for
        # segmentation
        targets_mask = targets >= 0
        scores = np.mean((preds == targets)[
                             targets_mask].detach().cpu().numpy())
        val_scores.append(scores)

        eval_loop.set_postfix(val_loss=loss.item(),
                              val_acc=np.mean(val_scores))

    model.train()