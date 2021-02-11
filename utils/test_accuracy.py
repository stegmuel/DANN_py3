import os
import torch.backends.cudnn as cudnn
# import torch.utils.data
import torch

def test_accuracy(dataloader_valid, model, cuda):
    cudnn.benchmark = True
    alpha = 0
    model.eval()
    data_iter = iter(dataloader_valid)
    n_total = 0
    n_correct = 0

    for image, label in data_iter:
        batch_size = len(label)
        if cuda:
            image = image.cuda()
            label = label.cuda()

        with torch.no_grad():
            class_output, _ = model(input_data=image, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct.data.numpy() * 1.0 / n_total
    model.train()
    return accu
