import sys
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from demo import *
import cv2
from PIL import Image
import imageio
import numpy as np
import torch
import os
from albumentations import Resize, Normalize, Compose, CLAHE
from albumentations.pytorch import ToTensor
import segmentation_models_pytorch as smp

class DrawingWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.im = None
        self.res = None
        self.transform = self.get_transforms()
        self.net = smp.Unet('resnet18', classes=1, activation=None)
        self.net.cuda()
        resume = os.path.join("..//weights//net_all_BCEL_Unet-Resnet18_0.970091.pth")
        pre_params = torch.load(resume)
        self.net.load_state_dict(pre_params)
        self.net.eval()

    def readImg(self, im_fn):
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

    def get_transforms(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        list_transforms = []
        list_transforms.extend(
            [Resize(480, 480, interpolation=Image.BILINEAR), CLAHE(), Normalize(mean=mean, std=std, p=1), ToTensor(), ])
        list_trfms = Compose(list_transforms)
        return list_trfms

    def input(self):
        path = QFileDialog.getOpenFileName(self, "选择文件", "./", 'Images(*.png *jpg *JPG *gif *tif *.ppm)')
        if path[0] is "":
            return
        self.im = self.readImg(path[0])
        img = QtGui.QPixmap(path[0]).scaled(self.originalLabel.width(), self.originalLabel.height())
        self.originalLabel.setPixmap(img)
        augmented = self.transform(image=self.im)
        x = augmented["image"]
        x = torch.unsqueeze(x,0)
        x = x.cuda()
        x = self.net(x)
        x = torch.sigmoid(x).squeeze()
        x = torch.where(x > 0.5, torch.tensor(255).cuda(), torch.tensor(0).cuda()).cpu().detach().numpy()
        x = cv2.merge([x, x, x])
        self.res = x.astype('uint8')
        cv2.imwrite("temp.png", x.astype('uint8'))
        img = QtGui.QPixmap("temp.png").scaled(self.segLabel.width(), self.segLabel.height())
        self.segLabel.setPixmap(img)
        os.remove("temp.png")

    def output(self):
        if self.res is None:
            return
        path = QFileDialog.getSaveFileName(self, "选择文件", "res.png", 'Images(*.png *jpg)')
        if path[0] is "":
            return
        cv2.imwrite(path[0], self.res)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DrawingWidget()
    win.show()
    sys.exit(app.exec_())
