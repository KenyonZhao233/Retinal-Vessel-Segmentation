# %%
import os
import cv2
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from albumentations import (Resize, RandomCrop, VerticalFlip, HorizontalFlip, Normalize, Compose, CLAHE, Rotate)
from albumentations.pytorch import ToTensor
from torch.autograd import Variable
from PIL import Image
import segmentation_models_pytorch as smp
import imageio
from fcn import *
# %%

seed = 42
random.seed(seed)
torch.manual_seed(seed)
os.environ["CUDA_VISIBLE_DEVICE"] = '0'
print(torch.cuda.get_device_name(0))


# %%

def get_transforms(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(),
                VerticalFlip(),
                Rotate(),
            ])
    list_transforms.extend(
        [Resize(480, 480, interpolation=Image.BILINEAR), CLAHE(), Normalize(mean=mean, std=std, p=1), ToTensor(), ])
    list_trfms = Compose(list_transforms)
    return list_trfms


# %%

def readImg(im_fn):
    im = cv2.imread(im_fn)
    if im is None:
        tmp = imageio.mimread(im_fn)
        if tmp is not None:
            im = np.array(tmp)
            im = im.transpose(1, 2, 0)
        else:
            image = Image.open(im_fn)
            im = np.asarray(image)
    else:
        im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)

    return im


# %%

class RetinalDataset(Dataset):
    def __init__(self, name, img_root, gt_root, phase):
        super().__init__()
        self.inputs = []
        self.gts = []
        self.transform = get_transforms(phase)

        for root in img_root:
            file_list = os.getcwd() + root
            list_image = os.listdir(file_list)
            list_image.sort()

            for i, image_path in enumerate(list_image):
                img = os.path.join(file_list, list_image[i])
                self.inputs.append(img)

        for root in gt_root:
            file_list = os.getcwd() + root
            list_image = os.listdir(file_list)
            list_image.sort()

            for i, image_path in enumerate(list_image):
                img = os.path.join(file_list, list_image[i])
                self.gts.append(img)

        print('Load %s: %d samples for %s' % (name, len(self.inputs), phase))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        image = readImg(self.inputs[index])
        mask = readImg(self.gts[index])
        if mask.shape[2] == 3:
            mask = mask[:, :, 0]

        augmented = self.transform(image=image, mask=mask.squeeze())
        return augmented["image"], augmented["mask"]


# %%

# DRIVE 数据集
dr_train_loader = RetinalDataset('DRIVE', ['\\data\\DRIVE\\train\\images'],
                                 ['\\data\\DRIVE\\train\\1st_manual'], 'train')
dr_test_loader = RetinalDataset('DRIVE', ['\\data\\DRIVE\\test\\images'],
                                ['\\data\\DRIVE\\test\\1st_manual'], 'test')

# STARE 数据集
st_train_loader = RetinalDataset('STARE', ['\\data\\STARE\\train\\image'],
                                 ['\\data\\STARE\\train\\labels-ah'], 'train')
st_test_loader = RetinalDataset('STARE', ['\\data\\STARE\\test\\image'],
                                ['\\data\\STARE\\test\\labels-ah'], 'test')

# CHASEDB1 数据集
st_train_loader = RetinalDataset('CHASEDB1', ['\\data\\CHASEDB1\\train\\image'],
                                 ['\\data\\CHASEDB1\\train\\1st'], 'train')
st_test_loader = RetinalDataset('CHASEDB1', ['\\data\\CHASEDB1\\test\\image'],
                                ['\\data\\CHASEDB1\\test\\1st'], 'test')

# HRF 数据集
hr_train_loader = RetinalDataset('HRF', ['\\data\\HRF\\train\\images'],
                                 ['\\data\\HRF\\train\\manual1'], 'train')
hr_test_loader = RetinalDataset('HRF', ['\\data\\HRF\\test\\images'],
                                ['\\data\\HRF\\test\\manual1'], 'test')

# 混合训练集
all_train_loader = RetinalDataset('all', ['\\data\\DRIVE\\train\\images', '\\data\\STARE\\train\\image',
                                          '\\data\\CHASEDB1\\train\\image', '\\data\\HRF\\train\\images'],
                                  ['\\data\\DRIVE\\train\\1st_manual', '\\data\\STARE\\train\\labels-ah',
                                   '\\data\\CHASEDB1\\train\\1st', '\\data\\HRF\\train\\manual1'], 'train')

all_test_loader = RetinalDataset('all', ['\\data\\DRIVE\\test\\images', '\\data\\STARE\\test\\image',
                                         '\\data\\CHASEDB1\\test\\image', '\\data\\HRF\\test\\images'],
                                 ['\\data\\DRIVE\\test\\1st_manual', '\\data\\STARE\\test\\labels-ah',
                                  '\\data\\CHASEDB1\\test\\1st', '\\data\\HRF\\test\\manual1'], 'test')

# %%

batch_size = 8
epochs = 500
lr = 0.001
batch_iter = math.ceil(len(all_train_loader) / batch_size)

vgg_model = VGGNet(requires_grad=True, show_params=False)
net = FCN8s(pretrained_net=vgg_model, n_class=1)
net.cuda()

net_name = 'FCN8s'
loss_fuc = 'BCEL'

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=4, verbose=True)

dataset = "all"
trainloader = DataLoader(all_train_loader, batch_size=batch_size, shuffle=True, pin_memory=True)
testloader = DataLoader(all_test_loader, batch_size=1, shuffle=False, pin_memory=True)

# %%

result_path = 'results'
if not os.path.exists(result_path):
    os.makedirs(result_path)
weights_path = "weights"
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
image_path = os.path.join(result_path, dataset)
if not os.path.exists(image_path):
    os.makedirs(image_path)

f_loss = open(os.path.join(result_path, "log_%s_%s_%s.txt" % (dataset, loss_fuc, net_name)), 'w')
f_loss.write('Dataset : %s\n' % dataset)
f_loss.write('Loss : %s\n' % loss_fuc)
f_loss.write('Net : %s\n' % net_name)
f_loss.write('Learning rate: %05f\n' % lr)
f_loss.write('batch-size: %s\n' % batch_size)
f_loss.close()


# %%

def train(e):
    print('start train epoch: %d' % e)
    net.train()

    loss_plot = []

    for i, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        x = net(x)
        loss = criterion(x.squeeze(), y.squeeze())
        print('Epoch:%d  Batch:%d/%d  loss:%08f' % (e, i + 1, batch_iter, loss.data))

        loss_plot.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_plot


# %%

def test():
    net.eval()
    acc = torch.tensor(0)
    tpr = torch.tensor(0)
    fpr = torch.tensor(0)
    sn = torch.tensor(0)
    sp = torch.tensor(0)

    for i, (x, y) in enumerate(testloader):
        optimizer.zero_grad()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        x = net(x)

        x = torch.sigmoid(x).squeeze()
        y = y.squeeze().int().long()

        x = torch.where(x > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())

        temp = x + torch.tensor(2).cuda().long() * y
        tp = torch.sum(torch.where(temp == 3, torch.tensor(1).cuda(), torch.tensor(0).cuda())).float()
        fp = torch.sum(torch.where(temp == 1, torch.tensor(1).cuda(), torch.tensor(0).cuda())).float()
        tn = torch.sum(torch.where(temp == 0, torch.tensor(1).cuda(), torch.tensor(0).cuda())).float()
        fn = torch.sum(torch.where(temp == 2, torch.tensor(1).cuda(), torch.tensor(0).cuda())).float()

        acc = acc + (tp + tn) / (tp + fp + tn + fn)
        tpr = tpr + tp / (tp + fn)
        fpr = fpr + fp / (tn + fp)
        sn = sn + tn / (tn + fp)
        sp = sp + tp / (tp + fn)

    acc = (acc / len(testloader)).cpu().numpy()
    tpr = (tpr / len(testloader)).cpu().numpy()
    fpr = (fpr / len(testloader)).cpu().numpy()
    sn = (sn / len(testloader)).cpu().numpy()
    sp = (sp / len(testloader)).cpu().numpy()

    print('ACC:', acc)
    print('TPR:', tpr)
    print('FPR:', fpr)
    print('SN:', sn)
    print('SP:', sp)

    f_log = open(os.path.join(result_path, "log_%s_%s_%s.txt" % (dataset, loss_fuc, net_name)), 'a')
    f_log.write('Epoch:%d  acc:%08f\n' % (e, acc))
    f_log.write('Epoch:%d  TPR:%08f\n' % (e, tpr))
    f_log.write('Epoch:%d  FPR:%08f\n' % (e, fpr))
    f_log.write('Epoch:%d  SN:%08f\n' % (e, sn))
    f_log.write('Epoch:%d  SP:%08f\n' % (e, sp))
    f_log.close()

    return acc


# %%

best_acc = 0
loss_plot = [0]
for e in range(1, epochs + 1):
    loss_plot = loss_plot + train(e)
    if e % 10 == 0:
        acc = test()
        if acc > best_acc:
            if best_acc != 0:
                os.remove(os.path.join(weights_path,
                                       'net_%s_%s_%s_%f.pth' % (dataset, loss_fuc, net_name, best_acc)))
            torch.save(net.state_dict(), os.path.join(weights_path,
                                                      'net_%s_%s_%s_%f.pth' % (dataset, loss_fuc, net_name, acc)))
            best_acc = acc

# %%

plt.plot(loss_plot[1:])


# %%

def test_plot():
    net.eval()
    res = []
    for i, (x, y) in enumerate(testloader):
        optimizer.zero_grad()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        x = net(x)

        x = torch.sigmoid(x).squeeze()
        y = y.squeeze().int().long().cpu().detach().numpy()

        x = torch.where(x > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda()).cpu().detach().numpy()

        acc = np.sum(np.where(x == y, 1, 0)) / np.sum(np.where(x == x, 1, 0))
        res.append(acc)
        im = cv2.merge([x * 255, y * 255, y * 255])
        plt.imsave(os.path.join(image_path, (str(i) + '_' + '%4f' % acc + '.png')), im.astype('uint8'), format="png")

    return res


# %%

resume = os.path.join(weights_path,
                      'net_%s_%s_%s_%f.pth' % (dataset, loss_fuc, net_name, best_acc))
pre_params = torch.load(resume)
net.load_state_dict(pre_params)
res = test_plot()

# %%

res = np.array(res)
print('ACC of DRIVE:',np.mean(res[0:20]))  # DRIVE
print('ACC of STARE:',np.mean(res[20:30]))  # STARE
print('ACC of CHASEDB1:',np.mean(res[30:44]))  # CHASEDB1
print('ACC of HRF:',np.mean(res[44:68]))  # HRF
