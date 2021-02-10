import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets


def test_accuracy(dataloader_valid, model, cuda):
    # assert dataset_name in ['MNIST', 'mnist_m']

    # image_root = os.path.join(dataset_dir, dataset_name)

    cudnn.benchmark = True
    # batch_size = 128
    # image_size = 28
    alpha = 0

    # """load data"""
    #
    # img_transform_source = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    # ])
    #
    # img_transform_target = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])
    #
    # if dataset_name == 'mnist_m':
    #     test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
    #
    #     dataset = GetLoader(
    #         data_root=os.path.join(image_root, 'mnist_m_test'),
    #         data_list=test_list,
    #         transform=img_transform_target
    #     )
    # else:
    #     dataset = datasets.MNIST(
    #         root=dataset_dir,
    #         train=False,
    #         transform=img_transform_source,
    #     )
    #
    # dataloader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=8
    # )

    """ test """
    # my_net = torch.load(os.path.join(models_dir, 'mnist_mnistm_model_epoch_current.pth'))
    # my_net = my_net.eval()
    model.eval()

    # if cuda:
    #     my_net = my_net.cuda()

    len_dataloader = len(dataloader_valid)
    data_iter = iter(dataloader_valid)

    i = 0
    n_total = 0
    n_correct = 0

    for image, label in data_iter:
        batch_size = len(label)

        if cuda:
            image = image.cuda()
            label = label.cuda()

        class_output, _ = model(input_data=image, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    model.train()
    return accu