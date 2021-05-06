import numpy as np
import torch
from tqdm import tqdm
import os
from utility.distributed import apply_gradient_allreduce, reduce_tensor

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(option, rank, epoch, model, criterion, optimizer, tr_loader, scaler, save_module, neptune, save_folder):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    for tr_data in tqdm(tr_loader):
        high_img, low_img = tr_data
        high_img, low_img = high_img.to(rank), low_img.to(rank)

        optimizer.zero_grad()

        # Update Optimizer
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = criterion(output, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            high_out = model.feature_extract(high_img)
            low_out = model.feature_extract(low_img)

            loss = criterion(output, label)

            loss.backward()
            optimizer.step()











        acc_result = accuracy(output, label, topk=(1, 5))

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss += reduce_tensor(loss.data, num_gpu).item()
            mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
            mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

        else:
            mean_loss += loss.item()
            mean_acc1 += acc_result[0]
            mean_acc5 += acc_result[1]


    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_acc5 /= len(tr_loader)
    mean_loss /= len(tr_loader)

    # Saving Network Params
    if multi_gpu:
        model_param = model.module.state_dict()
    else:
        model_param = model.state_dict()

    save_module.save_dict['model'] = [model_param]
    save_module.save_dict['optimizer'] = [optimizer.state_dict()]
    save_module.save_dict['save_epoch'] = epoch

    if (rank == 0) or (rank == 'cuda'):
        # Loggin
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_ACC@5-%.2f, tr_loss:%.3f' %(epoch, option.result['train']['total_epoch'], \
                                                                            mean_acc1, mean_acc5, mean_loss))
        neptune.log_metric('tr_loss', mean_loss)
        neptune.log_metric('tr_acc1', mean_acc1)
        neptune.log_metric('tr_acc5', mean_acc5)

        # Save
        if epoch % option.result['train']['save_epoch'] == 0:
            torch.save(model_param, os.path.join(save_folder, 'epoch%d_model.pt' %epoch))

    return model, optimizer, save_module


def validation(option, rank, epoch, model, criterion, val_loader, neptune):
    num_gpu = len(option.result['train']['gpu'].split(','))

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    with torch.no_grad():
        for val_data in tqdm(val_loader):
            input, label = val_data
            input, label = input.to(rank), label.to(rank)

            output = model(input)
            loss = criterion(output, label)

            acc_result = accuracy(output, label, topk=(1, 5))

            if (num_gpu > 1) and (option.result['train']['ddp']):
                mean_loss += reduce_tensor(loss.data, num_gpu).item()
                mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
                mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

            else:
                mean_loss += loss.item()
                mean_acc1 += acc_result[0]
                mean_acc5 += acc_result[1]

        # Train Result
        mean_acc1 /= len(val_loader)
        mean_acc5 /= len(val_loader)
        mean_loss /= len(val_loader)

        # Logging
        if (rank == 0) or (rank == 'cuda'):
            print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_ACC@5-%.2f, val_loss:%.3f' % (epoch, option.result['train']['total_epoch'], \
                                                                                    mean_acc1, mean_acc5, mean_loss))
            neptune.log_metric('val_loss', mean_loss)
            neptune.log_metric('val_acc1', mean_acc1)
            neptune.log_metric('val_acc5', mean_acc5)
            neptune.log_metric('epoch', epoch)

    result = {'acc1':mean_acc1, 'acc5':mean_acc5, 'val_loss':mean_loss}
    return result

def test():
    pass