"""
Plug-and-Play Algorithm for Magnetic Resonance Image Reconstruction
Authors:JinCheng Li  (201971138@yangtzeu.edu.cn)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.PGD_L1  import PGD_L1
from utils.PNP_PGD_L1_BM3D  import PNP_PGD_L1_BM3D
from utils.PNP_PGD_L1_D  import PNP_PGD_L1_D
from utils.PGD_CNC  import PGD_CNC
from utils.PNP_PGD_CNC_BM3D  import PNP_PGD_CNC_BM3D
from utils.PNP_PGD_CNC_D  import PNP_PGD_CNC_D

from utils.PGD_L1  import analyze_parse_PGD_L1
from utils.PNP_PGD_L1_BM3D  import analyze_parse_PNP_PGD_L1_BM3D
from utils.PNP_PGD_L1_D  import analyze_parse_PNP_PGD_L1_D
from utils.PGD_CNC  import analyze_parse_PGD_CNC
from utils.PNP_PGD_CNC_BM3D  import analyze_parse_PNP_PGD_CNC_BM3D
from utils.PNP_PGD_CNC_D  import analyze_parse_PNP_PGD_CNC_D
from utils.utils  import psnr
from utils.utils  import enlargement

def sub_plot_org(n,img,title):
    plt.subplot(2, 3, n)
    plt.imshow(img, cmap = plt.cm.gray)
    plt.axis('off')
    plt.title(title, fontsize = 'x-large', y=-0.1)

def sub_plot_noi(n, x_noises, M ,N ):
    plt.subplot(M ,N, n)
    xx= cv2.putText(x_noises, '%.2f dB' % psnr(x_noises_init, im_orig[j]), (160, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (1, 1, 1), 2, cv2.LINE_AA)
    plt.imshow(xx, cmap=plt.cm.gray)
    plt.axis('off')

def sub_plot_res(num, a, X, M, N):
    plt.subplot(M, N, num)
    xx= cv2.putText(a, '%.2f dB' % psnr(X, im_orig[j]), (160, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (1, 1, 1), 2, cv2.LINE_AA)
    plt.imshow(xx, cmap = plt.cm.gray)
    plt.axis('off')

def sub_plot_small(n, x_noises,MM, NN):
    plt.subplot(MM, NN, n)
    plt.imshow(x_noises, cmap=plt.cm.gray)
    plt.axis('off')

def sub_plot_error_img(num, X, min, j, M ,N):
    """
    This function is used to plot error images of three denoising methods.
    [min, max] is the relative error range of pixels.
    """
    s = [0.2, 0.2, 0.2]
    max = s[j]
    plt.subplot(M ,N, num)
    plt.imshow(X, vmin = min, vmax = max, cmap = plt.cm.gray)
    plt.axis('off')

def sub_plot_error_img_small(num, X, min, j, MM, NN):
    """
    This function is used to plot error images of three denoising methods.
    [min, max] is the relative error range of pixels.
    """
    s = [0.2, 0.2, 0.2]
    max = s[j]
    plt.subplot(MM, NN, num)
    plt.imshow(X, vmin = min, vmax = max, cmap = plt.cm.gray)
    plt.axis('off')

# ---- input arguments ----

PGD_L1_opt = analyze_parse_PGD_L1(0.4, 2, 0.07)
#the arguments are default_alpha, default_max iteration，default_lambda1
PNP_PGD_L1_BM3D_opt = analyze_parse_PNP_PGD_L1_BM3D(0.4, 12)
# the arguments are default sigma, default alpha and default max iteration.
PNP_PGD_L1_D_opt1 = analyze_parse_PNP_PGD_L1_D(0.25, 50)   #IRCNN
# the arguments are default sigma, default alpha and default max iteration.
PNP_PGD_L1_D_opt2 = analyze_parse_PNP_PGD_L1_D(0.4, 6)   #FFDNet
# # the arguments are default sigma, default alpha and default max iteration.
PNP_PGD_L1_D_opt3 = analyze_parse_PNP_PGD_L1_D(0.4, 50)   #DRUNet
# # the arguments are default sigma, default alpha and default max iteration.

PGD_CNC_opt = analyze_parse_PGD_CNC(0.4, 4, 0.04, 1)
# the arguments are default_alpha, default_max iteration, default_lambda1, default_b
PNP_PGD_CNC_BM3D_opt = analyze_parse_PNP_PGD_CNC_BM3D(0.4, 50, 0.05, 36)
# the arguments are default_alpha, default_max iteration, default_lambda1, default_b
PNP_PGD_CNC_D_opt1 = analyze_parse_PNP_PGD_CNC_D(0.25, 50, 2.5, 1)   #IRCNN
# the arguments are default_alpha, default_max iteration, default_lambda1, default_b
PNP_PGD_CNC_D_opt2 = analyze_parse_PNP_PGD_CNC_D(0.4, 6, 1.25, 1)  #FFDNet
# # the arguments are default_alpha, default_max iteration, default_lambda1, default_b
PNP_PGD_CNC_D_opt3 = analyze_parse_PNP_PGD_CNC_D(0.4, 50, 2.5, 1)  #DRUNet
# # the arguments are default_alpha, default_max iteration, default_lambda1, default_b


j, k = 0, 0  # j = im_orig number, k = mask number


with torch.no_grad():

    # ---- load mask matrix ----
    mat = np.array([sio.loadmat('CS_MRI/Q_Random30.mat'),
                    sio.loadmat('CS_MRI/Q_Radial30.mat'),
                    sio.loadmat('CS_MRI/Q_Cartesian30.mat')])
    mask = np.array([mat[0].get('Q1').astype(np.float64),
                     mat[1].get('Q1').astype(np.float64),
                     mat[2].get('Q1').astype(np.float64)])
    mask1 = np.fft.fftshift(mask)

    # ---- load noises -----
    noises = sio.loadmat('CS_MRI/noises.mat')
    noises = noises.get('noises').astype(np.complex128) * 3.0

    # ---- set options -----
    PGD_L1_opts = dict(alpha=PGD_L1_opt.alpha, iter_num=PGD_L1_opt.iter_num, lambda1=PGD_L1_opt.lambda1)
    PNP_PGD_L1_BM3D_opts = dict(alpha=PNP_PGD_L1_BM3D_opt.alpha, iter_num=PNP_PGD_L1_BM3D_opt.iter_num)
    PNP_PGD_L1_D_opts1 = dict(alpha=PNP_PGD_L1_D_opt1.alpha, iter_num=PNP_PGD_L1_D_opt1.iter_num)
    PNP_PGD_L1_D_opts2 = dict(alpha=PNP_PGD_L1_D_opt2.alpha, iter_num=PNP_PGD_L1_D_opt2.iter_num)
    PNP_PGD_L1_D_opts3 = dict(alpha=PNP_PGD_L1_D_opt3.alpha, iter_num=PNP_PGD_L1_D_opt3.iter_num)
    PGD_CNC_opts = dict(alpha=PGD_CNC_opt.alpha, iter_num=PGD_CNC_opt.iter_num,
                             lambda1=PGD_CNC_opt.lambda1, b=PGD_CNC_opt.b)
    PNP_PGD_CNC_BM3D_opts = dict(alpha=PNP_PGD_CNC_BM3D_opt.alpha, iter_num=PNP_PGD_CNC_BM3D_opt.iter_num,
                             lambda1=PNP_PGD_CNC_BM3D_opt.lambda1, b=PNP_PGD_CNC_BM3D_opt.b)
    PNP_PGD_CNC_D_opts1 = dict(alpha=PNP_PGD_CNC_D_opt1.alpha, iter_num=PNP_PGD_CNC_D_opt1.iter_num,
                             lambda1=PNP_PGD_CNC_D_opt1.lambda1, b=PNP_PGD_CNC_D_opt1.b)
    PNP_PGD_CNC_D_opts2 = dict(alpha=PNP_PGD_CNC_D_opt2.alpha, iter_num=PNP_PGD_CNC_D_opt2.iter_num,
                             lambda1=PNP_PGD_CNC_D_opt2.lambda1, b=PNP_PGD_CNC_D_opt2.b)
    PNP_PGD_CNC_D_opts3 = dict(alpha=PNP_PGD_CNC_D_opt3.alpha, iter_num=PNP_PGD_CNC_D_opt3.iter_num,
                             lambda1=PNP_PGD_CNC_D_opt3.lambda1, b=PNP_PGD_CNC_D_opt3.b)

    #  load demo synthetic block image and demo noisy image
    im_orig = np.array([cv2.imread('testsets/set/1 brain.bmp', 0) / 255.0,
                        cv2.imread('testsets/set/2 Brain angiography.jpg', 0) / 255.0,
                        cv2.imread('testsets/set/3 Bust.jpg', 0) / 255.0])
    img1 = mpimg.imread('testsets/set/1 brain.bmp')
    img2 = mpimg.imread('testsets/set/2 Brain angiography.jpg')
    img3 = mpimg.imread('testsets/set/3 Bust.jpg')
    img = [img1, img2, img3]

    y = np.fft.fft2(im_orig[j]) * mask[k] + noises  # observed value
    x_noises_init = np.fft.ifft2(y)
    x_noises = np.reshape(np.real(x_noises_init), (im_orig[j].shape))

    # ---- denoising work -----
    name = ['ircnn_gray', 'ffdnet_gray', 'drunet_gray']

    " PGD-L1 "
    # out = PGD_L1(mask[k], noises, **PGD_L1_opts)
    # out = out[j]

    " PNP-PGD-L1 "        " 1  BM3D "
    # out = PNP_PGD_L1_BM3D(mask[k], **PNP_PGD_L1_BM3D_opts)
    # out = out[j]

    " PNP-PGD-L1 "      " 2  neural network "
    # out = PNP_PGD_L1_D(name[0], mask[k], noises, **PNP_PGD_L1_D_opts1)
    # out = out[j]

    # out = PNP_PGD_L1_D(name[1], mask[k], noises, **PNP_PGD_L1_D_opts2)
    # out = out[j]

    out = PNP_PGD_L1_D(name[2], mask[k], noises, **PNP_PGD_L1_D_opts3)
    out = out[j]


    " PGD-CNC "
    # out = PGD_CNC(mask[k], noises, **PGD_CNC_opts)
    # out = out[j]

    " PNP-PGD-CNC "     " 1  BM3D "
    # out = PNP_PGD_CNC_BM3D(mask[k], **PNP_PGD_CNC_BM3D_opts)
    # out = out[j]

    " PNP-PGD-CNC "     " 2  neural network "
    # out = PNP_PGD_CNC_D(name[0], mask[k], noises, **PNP_PGD_CNC_D_opts1)
    # out = out[j]

    # out = PNP_PGD_CNC_D(name[1], mask[k], noises, **PNP_PGD_CNC_D_opts2)
    # out = out[j]

    # out = PNP_PGD_CNC_D(name[2], mask[k], noises, **PNP_PGD_CNC_D_opts3)
    # out = out[j]

    '''' plot demo result figure '''
    plt.figure(1)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)

    sub_plot_org(1, im_orig[0], '(a)Brain')
    sub_plot_org(2, im_orig[1], '(b)Brain angiography')
    sub_plot_org(3, im_orig[2], '(c)Bust')
    sub_plot_org(4, mask1[1], '(d)Random Sampling')
    sub_plot_org(5, mask1[2], '(e)Radial Sampling')
    sub_plot_org(6, mask1[0], '(f)Cartesian Sampling')


    ''' plot demo result figure '''
    region = [[40, 50], [40, 50],  [0, 160],
              [90, 50], [90, 50],  [50, 160],
              [40, 100], [40, 100], [0, 210],
              [90, 100], [90, 100],  [50, 210]]

    region1 = [[180, 150], [180, 120],  [140,  30],
              [230, 150], [230, 120],  [190, 30],
              [180, 200], [180, 170],  [140, 80],
              [230, 200], [230, 170],  [190, 80]]

    plt.figure(2)        # noises image
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    noises1 = enlargement(j, x_noises, region)
    noises2 = enlargement(j, noises1[0], region1)
    sub_plot_noi(1, noises2[0], 2, 1)
    sub_plot_small(5, noises1[1], 4, 2)
    sub_plot_small(6, noises2[1], 4, 2)


    plt.figure(3)       # noises diff image
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    x_noises1 = np.fabs(x_noises - im_orig[j])
    noises1 = enlargement(j, x_noises1, region)
    noises2 = enlargement(j, noises1[0], region1)
    sub_plot_error_img(1, noises2[0], 0, j, 2, 1)
    sub_plot_error_img_small(5, noises1[1], 0, j, 4, 2)
    sub_plot_error_img_small(6, noises2[1], 0, j, 4, 2)


    plt.figure(4)       # out image
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    x1 = enlargement(j, out, region)
    x1a = enlargement(j, x1[0], region1)
    sub_plot_res(1, x1a[0], out, 2, 1)
    sub_plot_small(5, x1[1], 4, 2)
    sub_plot_small(6, x1a[1], 4, 2)


    plt.figure(5)       # out diff image
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                        wspace=0.05, hspace=0.05)
    out1A = np.fabs(out - im_orig[j])
    out1A = enlargement(j, out1A, region)
    out1AA = enlargement(j, out1A[0], region1)
    sub_plot_error_img(1, out1AA[0], 0, j,  2, 1)
    sub_plot_error_img_small(5, out1A[1], 0, j, 4, 2)
    sub_plot_error_img_small(6, out1AA[1], 0, j, 4, 2)


    plt.rcParams['savefig.dpi'] = 600  # image pixel
    plt.rcParams['figure.dpi'] = 600  # resolution ratio
    plt.show()