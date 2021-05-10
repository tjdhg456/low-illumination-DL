import numpy as np
import torch
from tqdm import tqdm
import os
from utility.distributed import apply_gradient_allreduce, reduce_tensor
from copy import deepcopy

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

def robust_train(option, rank, epoch, model, criterion, optimizer, tr_loader, scaler, save_module, neptune, save_folder, save_robust):
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
                model.module.clear_features()
                _ = model.module.feature_extract(high_img)
                high_features = model.module.features
                high_feats, high_ids = model.module.sample_patch(high_features)
            else:
                model.clear_features()
                _ = model.feature_extract(high_img)
                high_features = model.features
                high_feats, high_ids = model.sample_patch(high_features)

            # Extract Low Features
            if multi_gpu:
                model.module.clear_features()
                _ = model.module.feature_extract(low_img)
                low_features = model.module.features
                low_feats, _ = model.module.sample_patch(low_features, high_ids)
            else:
                model.clear_features()
                _ = model.feature_extract(low_img)
                low_features = model.features
                low_feats, _ = model.sample_patch(low_features, high_ids)

            # Update Loss
            for ix in range(len(high_feats)):
                if ix == 0:
                    loss = criterion(low_feats[ix], high_feats[ix])
                else:
                    loss += criterion(low_feats[ix], high_feats[ix])

            loss.backward()
            optimizer.step()

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss += reduce_tensor(loss.data, num_gpu).item()
        else:
            mean_loss += loss.item()

    # Train Result
    mean_loss /= len(tr_loader)

    if (rank == 0) or (rank == 'cuda'):
        # Logging
        print('Epoch-(%d/%d) - tr_robust_loss:%.3f' %(epoch, option.result['train']['total_epoch'], mean_loss))
        neptune.log_metric('tr_robust_loss', mean_loss)

        if save_robust:
            # Saving Network Params
            if multi_gpu:
                model_param = model.module.state_dict()
            else:
                model_param = model.state_dict()

            # Update Save Module
            save_module.save_dict['model'] = [model_param]
            save_module.save_dict['optimizer'] = [optimizer.state_dict()]
            save_module.save_dict['save_epoch'] = epoch

            # Save Params at save_epoch
            if epoch % option.result['train']['save_epoch'] == 0:
                torch.save(model_param, os.path.join(save_folder, 'epoch%d_model.pt' %epoch))

    return model, optimizer, save_module


def detection_train(option, rank, epoch, model, criterion, optimizer, tr_loader, type, scaler, save_module, neptune, save_folder, save_detection=False):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.

    for tr_data in tqdm(tr_loader):
        tr_img, tr_target = tr_data

        # To GPU
        tr_img = tr_img.to(rank)
        with torch.no_grad():
            tr_target = [ann.to(rank) for ann in tr_target]

        # Backprop
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
            # Forward
            if multi_gpu:
                model.module.detector.clear_features()
            else:
                model.detector.clear_features()

            out = model(tr_img)
            loss_l, loss_c = criterion(out, tr_target)

            # Total Loss
            loss = loss_l + loss_c

            # Back Prop
            loss.backward()
            optimizer.step()

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss += reduce_tensor(loss.data, num_gpu).item()
        else:
            mean_loss += loss.item()

    # Train Result
    mean_loss /= len(tr_loader)

    if (rank == 0) or (rank == 'cuda'):
        # Logging
        print('Epoch-(%d/%d) - (%s) tr_detection_loss:%.3f' %(epoch, option.result['train']['total_epoch'], type, mean_loss))
        neptune.log_metric('%s_tr_detection_loss' %type, mean_loss)

        if save_detection:
            # Saving Network Params
            if multi_gpu:
                model_param = model.module.state_dict()
            else:
                model_param = model.state_dict()

            # Update Save Module
            save_module.save_dict['model'] = [model_param]
            save_module.save_dict['optimizer'] = [optimizer.state_dict()]
            save_module.save_dict['save_epoch'] = epoch

            # Save Params at save_epoch
            if epoch % option.result['train']['save_epoch'] == 0:
                torch.save(model_param, os.path.join(save_folder, 'epoch%d_model.pt' %epoch))

    return model, optimizer, save_module


def run(option, model, tr_robust_loader, tr_coco_loader, tr_ex_loader, val_robust_loader, val_coco_loader, val_ex_loader,
        optimizer, patch_criterion, detection_criterion, scaler, scheduler, early, save_folder, save_module, multi_gpu, rank, neptune):

    # Training Type
    last = ''
    if tr_robust_loader is not None:
        robust = True
        last = 'robust'
    else:
        robust = False

    if tr_coco_loader is not None:
        coco = True
        last = 'coco'
    else:
        coco = False

    if tr_ex_loader is not None:
        ex = True
        last = 'ex'
    else:
        ex = False


    # Get Hook
    target_layers = option.result['train']['target_layers']
    if multi_gpu:
        if robust:
            model.module.get_hook(target_layers)
        if coco or ex:
            model.module.detector.get_hook(target_layers=['23'])
    else:
        if robust:
            model.get_hook(target_layers)
        if coco or ex:
            model.detector.get_hook(target_layers=['23'])


    # Trainer
    for epoch in range(save_module.init_epoch, save_module.total_epoch):
        # Training
        model.train()

        if robust:
            if last == 'robust':
                save_robust = True
            else:
                save_robust = False

            model, optimizer, save_module = robust_train(option, rank, epoch, model, patch_criterion, optimizer,
                                                         tr_robust_loader, scaler, save_module, neptune, save_folder, save_robust=save_robust)
        if coco:
            if last == 'coco':
                save_coco = True
            else:
                save_coco = False

            model, optimizer, save_module = detection_train(option, rank, epoch, model, detection_criterion, optimizer,
                                                            tr_coco_loader, 'coco', scaler, save_module, neptune, save_folder, save_detection=save_coco)
        if ex:
            if last == 'ex':
                save_ex = True
            else:
                save_ex = False

            model, optimizer, save_module = detection_train(option, rank, epoch, model, detection_criterion, optimizer,
                                                            tr_ex_loader, 'ex', scaler, save_module, neptune, save_folder, save_detection=save_ex)

        # Validation
        model.eval()
    #
    #     if robust and detection:
    #         _ = robust_validation(option, rank, epoch, model, criterion, val_robust_loader, neptune)
    #         result = detection_validation(option, rank, epoch, model, criterion, val_detection_loader, neptune)
    #     elif robust and (not detection):
    #         result = robust_validation(option, rank, epoch, model, criterion, val_robust_loader, neptune)
    #     elif (not robust) and detection:
    #         result = detection_validation(option, rank, epoch, model, criterion, val_detection_loader, neptune)
    #     else:
    #         raise('Select Proper Train Type')
    #
    #
    #     # Scheduler
    #     if scheduler is not None:
    #         scheduler.step()
    #         save_module.save_dict['scheduler'] = [scheduler.state_dict()]
    #     else:
    #         save_module.save_dict['scheduler'] = None
    #
    #
    #     # Save the last-epoch module
    #     if (rank == 0) or (rank == 'cuda'):
    #         save_module_path = os.path.join(save_folder, 'last_dict.pt')
    #         save_module.export_module(save_module_path)
    #
    #         save_config_path = os.path.join(save_folder, 'last_config.json')
    #         option.export_config(save_config_path)
    #
    #
    #     # Early Stopping
    #     if multi_gpu:
    #         param = deepcopy(model.module.state_dict())
    #     else:
    #         param = deepcopy(model.state_dict())
    #
    #     if early is not None:
    #         if option.result['train']['early_loss']:
    #             early(result['val_loss'], param, result)
    #         else:
    #             early(-result['acc1'], param, result)
    #
    #         if early.early_stop == True:
    #             break
    #
    # # Remove Hook
    # if multi_gpu:
    #     if robust:
    #         model.module.remove_hook(target_layers)
    #     if detection:
    #         model.module.detector.remove_hook(target_layers=['23'])
    # else:
    #     if robust:
    #         model.remove_hook(target_layers)
    #     if detection:
    #         model.detector.remove_hook(target_layers=['23'])
    #
    # # Save the Model
    # if (rank == 0) or (rank == 'cuda'):
    #     if early is not None:
    #         torch.save(early.model, os.path.join(save_folder, 'best_model.pt'))
    #     else:
    #         totch.save(model, os.path.join(save_folder, 'best_model.pt'))
    # return early, save_module, option

