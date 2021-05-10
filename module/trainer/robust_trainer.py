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

    for tr_data in tqdm(tr_loader):
        high_img, low_img = tr_data
        high_img, low_img = high_img.to(rank), low_img.to(rank)

        optimizer.zero_grad()

        # Update Optimizer
        if scaler is not None:
            pass
            # with torch.cuda.amp.autocast():
            #     output = model(input)
            #     loss = criterion(output, label)
            #
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
        else:
            # Extract High Features
            if multi_gpu:
                _ = model.module.feature_extract(high_img)
                high_features = model.module.features
                high_feats, high_ids = model.module.sample_patch(high_features)
            else:
                _ = model.feature_extract(high_img)
                high_features = model.features
                high_feats, high_ids = model.sample_patch(high_features)


            # Extract Low Features
            if multi_gpu:
                _ = model.module.feature_extract(low_img)
                low_features = model.module.features
                low_feats, _ = model.module.sample_patch(low_features, high_ids)
            else:
                _ = model.feature_extract(low_img)
                low_features = model.features
                low_feats, _ = model.sample_patch(low_features, high_ids)


            # Update Loss
            for ix in range(len(high_feats)):
                if ix == 0:
                    loss = criterion(low_feats[ix], high_feats[ix])
                else:
                    loss += criterion(low_feats[ix], high_feats[ix])
                print(loss)

            loss.backward()
            optimizer.step()

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss += reduce_tensor(loss.data, num_gpu).item()
        else:
            mean_loss += loss.item()

    # Train Result
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
        print('Epoch-(%d/%d) - tr_loss:%.3f' %(epoch, option.result['train']['total_epoch'], mean_loss))
        neptune.log_metric('tr_loss', mean_loss)

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


def run(option, model, tr_loader, val_loader, optimizer, criterion, scaler, scheduler, early, \
        save_folder, save_module, multi_gpu, rank, neptune):

    # Get Hook
    target_layers = option.result['train']['target_layers']
    if multi_gpu:
        model.module.get_hook(target_layers)
        model.module.detector.get_hook(target_layers)
    else:
        model.get_hook(target_layers)
        model.detector.get_hook(target_layers)

    for epoch in range(save_module.init_epoch, save_module.total_epoch):
        model.train()
        model, optimizer, save_module = train(option, rank, epoch, model, criterion, optimizer,
                                              tr_loader, scaler, save_module, neptune, save_folder)

        model.eval()
        result = validation(option, rank, epoch, model, criterion, val_loader, neptune)

        if scheduler is not None:
            scheduler.step()
            save_module.save_dict['scheduler'] = [scheduler.state_dict()]
        else:
            save_module.save_dict['scheduler'] = None

        # Save the last-epoch module
        if (rank == 0) or (rank == 'cuda'):
            save_module_path = os.path.join(save_folder, 'last_dict.pt')
            save_module.export_module(save_module_path)

            save_config_path = os.path.join(save_folder, 'last_config.json')
            option.export_config(save_config_path)

        # Early Stopping
        if multi_gpu:
            param = deepcopy(model.module.state_dict())
        else:
            param = deepcopy(model.state_dict())

        if early is not None:
            if option.result['train']['early_loss']:
                early(result['val_loss'], param, result)
            else:
                early(-result['acc1'], param, result)

            if early.early_stop == True:
                break

    # Remove Hook
    if multi_gpu:
        model.module.remove_hook(target_layers)
        model.module.detector.remove_hook(target_layers)
    else:
        model.remove_hook(target_layers)
        model.detector.remove_hook(target_layers)

    # Save the Model
    if (rank == 0) or (rank == 'cuda'):
        if early is not None:
            torch.save(early.model, os.path.join(save_folder, 'best_model.pt'))
        else:
            totch.save(model, os.path.join(save_folder, 'best_model.pt'))
    return early, save_module, option

