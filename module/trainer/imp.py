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