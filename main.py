import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.kather_dataset import KatherHDF
import torch.optim as optim
from model import CNNModel
import torch.utils.data
from utils.test_accuracy import test_accuracy
import numpy as np
import random
from itertools import cycle
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

source_dataset_path = '/media/thomas/Samsung_T5/colorectal/kather_datasets/100k_dataset.h5'
target_dataset_path = '/media/thomas/Samsung_T5/colorectal/kather16_target_3500.h5'

model_root = 'models'
cuda = torch.cuda.is_available()
cudnn.benchmark = True
lr = 1e-3
batch_size = 8
image_size = 224
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Prepare the source datasets
dataset_source_train = KatherHDF(hdf5_filepath=source_dataset_path,
                                 phase='train',
                                 batch_size=batch_size,
                                 use_cache=False)
dataset_source_valid = KatherHDF(hdf5_filepath=source_dataset_path,
                                 phase='valid',
                                 batch_size=batch_size,
                                 use_cache=False)
dataloader_source_train = DataLoader(dataset=dataset_source_train,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=2)
dataloader_source_valid = DataLoader(dataset=dataset_source_valid,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=2)

# Prepare the target datasets
dataset_target_train = KatherHDF(hdf5_filepath=target_dataset_path,
                                 phase='train',
                                 batch_size=batch_size,
                                 use_cache=False)
dataset_target_valid = KatherHDF(hdf5_filepath=target_dataset_path,
                                 phase='valid',
                                 batch_size=batch_size,
                                 use_cache=False)
dataloader_target_train = DataLoader(dataset=dataset_target_train,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=2)
dataloader_target_valid = DataLoader(dataset=dataset_target_valid,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=2)

# load model
model = CNNModel()

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_class = torch.nn.CrossEntropyLoss()
loss_domain = torch.nn.CrossEntropyLoss()

if cuda:
    model = model.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in model.parameters():
    p.requires_grad = True

# training
best_accu_s = 0.
best_accu_t = 0.
len_dataloader = min(len(dataloader_source_train), len(dataloader_target_train))
data_source_iter = cycle(iter(dataloader_source_train))
data_target_iter = cycle(iter(dataloader_target_train))
for epoch in range(n_epoch):
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = next(data_source_iter)
        s_img, s_label = data_source

        model.zero_grad()
        batch_size = len(s_label)

        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()

        class_output, domain_output = model(input_data=s_img, alpha=alpha)
        err_s_label = loss_class(class_output, s_label.squeeze())
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = next(data_source_iter)
        t_img, _ = data_target

        batch_size = len(t_img)

        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = model(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
        sys.stdout.flush()
        torch.save(model, '{0}/mnist_mnistm_model_epoch_current.pth'.format(model_root))

    print('\n')
    accu_s = test_accuracy(dataloader_source_valid, model, cuda)
    print('Accuracy of the %s dataset: %f' % ('mnist', accu_s))
    accu_t = test_accuracy(dataloader_target_valid, model, cuda)
    print('Accuracy of the %s dataset: %f\n' % ('mnist_m', accu_t))
    if accu_t > best_accu_t:
        best_accu_s = accu_s
        best_accu_t = accu_t
        torch.save(model, '{0}/mnist_mnistm_model_epoch_best.pth'.format(model_root))

print('============ Summary ============= \n')
print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
print('Corresponding model was save in ' + model_root + '/mnist_mnistm_model_epoch_best.pth')