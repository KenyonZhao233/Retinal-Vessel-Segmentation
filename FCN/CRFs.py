import numpy as np
import pydensecrf.densecrf as dcrf

try:
    from cv2 import imread, imwrite
except ImportError:
    # 如果没有安装OpenCV，就是用skimage
    from skimage.io import imread, imsave

    imwrite = imsave

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import numpy as np
np.set_printoptions(threshold=np.inf)

"""
original_image_path  原始图像路径
predicted_image_path  之前用自己的模型预测的图像路径
CRF_image_path  即将进行CRF后处理得到的结果图像保存路径
"""


def CRFs(original_image_path, predicted_image_path, CRF_image_path):
    print("original_image_path: ", original_image_path)
    img = imread(original_image_path)

    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    anno_rgb = imread(predicted_image_path).astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)
    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    # HAS_UNK = 0 in colors
    # if HAS_UNK:
    #     colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    ###     设置CRF模型     ###
    ###########################
    # use_2d = False
    use_2d = True
    ###########################################################
    ##不是很清楚什么情况用2D
    ##作者说“对于图像，使用此库的最简单方法是使用DenseCRF2D类”
    ##作者还说“DenseCRF类可用于通用（非二维）密集CRF”
    ##但是根据我的测试结果一般情况用DenseCRF比较对
    #########################################################33
    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，只是位置-----会惩罚空间上孤立的小块分割,即强制执行空间上更一致的分割
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)-----使用局部颜色特征来细化它们
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        '''
        addPairwiseGaussian函数里的sxy为公式中的 $\theta_{\gamma}$, 
        addPairwiseBilateral函数里的sxy、srgb为$\theta_{\alpha}$ 和 $\theta_{\beta}$
        '''
    else:
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.5, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=8, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(10)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    MAP = colorize[MAP, :]
    imwrite(CRF_image_path, MAP.reshape(img.shape))
    print("CRF图像保存在", CRF_image_path, "!")
CRFs("D://image//0_.bmp","D://image//0_.bmp","D://image//test.png")