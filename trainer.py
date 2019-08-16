import torch
from collections import defaultdict

import const

def train(net, optimizer, criterion_dict, train_loader, epoch, logger, flatten=True):
    net.train()

    num_correct_dict = defaultdict(int)
    loss_dict = defaultdict(float)
    for batch, (x, target) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        x, target = x.to(const.DEVICE), target.to(const.DEVICE)

        if flatten: 
            x = x.view(x.shape[0], -1)
        output = net(x, target)

        # Compute loss for each head, and propagate gradients based on weighted sum of losses.
        loss = None
        for head, (weight, criterion) in criterion_dict.items():
            tmp_l = weight * criterion(output[head], target)
            loss = loss + tmp_l if loss is not None else tmp_l
            loss_dict[head] += tmp_l.item()
        loss.backward()
        optimizer.step()

        # Add number of correct predictions, separately for each head.
        for head_key in criterion_dict.keys():
            num_correct_dict[head_key] += output[head_key].argmax(1).eq(target).sum().item()

        # Time to log losses.
        # Log losses of each head separately, with suffix '_{head_name}'.
        if batch % const.LOG_LOSS_EVERY == 0:
            loss_dict_for_log = {'loss_' + h:l / const.LOG_LOSS_EVERY for h, l in loss_dict.items()}
            loss_dict = {h:l / const.LOG_LOSS_EVERY for h, l in loss_dict.items()}

            avg_loss = 0.0
            for h, (w, _) in criterion_dict.items():
                avg_loss += w * loss_dict[h]

            print(f'Epoch {epoch}, Batch {batch}, loss={avg_loss}')
            logger.update_log(epoch=epoch, batch=batch, is_validation=False, **loss_dict_for_log)

            # Clear loss dict for next logging.
            loss_dict = defaultdict(float)
    
    acc_dict = {h:num_correct / const.NUM_TRAIN_DATA for h, num_correct in num_correct_dict.items()}
    acc_dict_for_log = {'acc_' + h:num_correct / const.NUM_TRAIN_DATA for h, num_correct in num_correct_dict.items()}
    for h, acc in acc_dict.items():
        print(f'Epoch {epoch}, Accuracy for head {h}={acc}')
    logger.update_log(epoch=epoch, batch=None, is_validation=False, **acc_dict_for_log)

def validate(net, optimizer, criterion_dict, val_loader, epoch, logger, flatten=True):
    net.eval()

    num_correct_dict, loss_dict = defaultdict(int), defaultdict(float)
    with torch.no_grad():
        for _, (x, target) in enumerate(val_loader, 1):
            x, target = x.to(const.DEVICE), target.to(const.DEVICE)
            x = x.view(x.shape[0], -1)
            output = net(x, target)

            # Compute loss for each head, and propagate gradients based on weighted sum of losses.
            loss = None
            for head, (weight, criterion) in criterion_dict.items():
                tmp_l = weight * criterion(output[head], target)
                loss = loss + tmp_l if loss is not None else tmp_l
                loss_dict[head] += tmp_l.item()

            # Add number of correct predictions, separately for each head.
            for head in loss_dict.keys():
                num_correct_dict[head] += output[head].argmax(1).eq(target).sum().item()

    acc_dict_for_log = {'acc_' + h:num_correct / const.NUM_VAL_DATA for h, num_correct in num_correct_dict.items()}
    acc_dict = {h:num_correct / const.NUM_VAL_DATA for h, num_correct in num_correct_dict.items()}
    loss_dict_for_log = {'loss_' + h:l / len(val_loader) for h, l in loss_dict.items()}
    loss_dict = {h:l / len(val_loader) for h, l in loss_dict.items()}

    for h in loss_dict.keys():
        print('Accuracy: %d/%d (%.2f), Loss: %.2f' % (num_correct_dict[h], const.NUM_VAL_DATA, acc_dict[h], loss_dict[h]))
    logger.update_log(epoch=epoch, batch=None, is_validation=True, **acc_dict_for_log, **loss_dict_for_log)
