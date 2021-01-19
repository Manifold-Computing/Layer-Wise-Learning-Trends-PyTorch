import torch
import numpy as np

from utils.delta import compute_delta


def training(epochs, loaders, model, optimizer, scheduler, criterion, prev_list,
             rmae_delta_dict, configs, experiment):
    """
    Performs training and evaluation.
    """

    min_test_loss = np.Inf

    train_acc_arr, test_acc_arr = [], []

    for epoch in range(1, epochs+1):

        train_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        test_correct = 0.0
        test_total = 0.0
        test_loss = 0.0

        # train the model
        model.train()

        for data, labels in loaders['train']:
            # move the data and labels to gpu
            data, labels = data.cuda(), labels.cuda()

            optimizer.zero_grad()
            # get model outputs
            output = model(data)
            # calculate the loss
            loss = criterion(output, labels)
            # backprop
            loss.backward()
            # optimize the weights
            optimizer.step()
            # update the training loss for the batch
            train_loss += loss.item()*data.size(0)
            # get the predictions for each image in the batch
            preds = torch.max(output, 1)[1]
            # get the number of correct predictions in the batch
            train_correct += np.sum(np.squeeze(
                preds.eq(labels.data.view_as(preds))).cpu().numpy())

            # accumulate total number of examples
            train_total += data.size(0)

        train_loss = round(train_loss/len(loaders['train'].dataset), 4)
        train_acc = round(((train_correct/train_total) * 100.0), 4)

        # compute layer deltas after epoch.
        rmae_delta_dict, prev_list = compute_delta(
            model, prev_list, rmae_delta_dict, experiment, epoch)

        model.eval()
        with torch.no_grad():
            for data, labels in loaders['test']:

                data, labels = data.cuda(), labels.cuda()

                output = model(data)
                loss = criterion(output, labels)

                test_loss += loss.item()*data.size(0)

                # get the predictions for each image in the batch
                preds = torch.max(output, 1)[1]
                # get the number of correct predictions in the batch
                test_correct += np.sum(np.squeeze(
                    preds.eq(labels.data.view_as(preds))).cpu().numpy())

                # accumulate total number of examples
                test_total += data.size(0)

        if test_loss < min_test_loss:
            print(f"Saving model at Epoch: {epoch}")
            # torch.save(model.state_dict(), 'drive/My Drive/cifar10-resnet18-gradual-adam')

        test_loss = round(test_loss/len(loaders['test'].dataset), 4)
        test_acc = round(((test_correct/test_total) * 100), 4)

        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

        print(
            f"Epoch: {epoch} \tTrain Loss: {train_loss} \tTrain Acc: {train_acc}% \tTest Loss: {test_loss} \tTest Acc: {test_acc}%")
        if float(test_acc) >= configs.target_val_acc:
            break
            
        scheduler.step()

    return rmae_delta_dict, np.asarray(train_acc_arr), np.asarray(test_acc_arr)
