"""
The definition is about the use of neural network denoising under the PNP-PGD-CNC framework
Authors:JinCheng Li  (201971138@yangtzeu.edu.cn)
"""

import os.path
import cv2
import logging
import argparse
import numpy as np

import torch
from collections import OrderedDict
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_image as util


def denoising_step2(model_name, x, x8, sigmas, i, model, noises, device, noise_level_model):
    if 'dncnn' in model_name and 'fdncnn' not in model_name:
        if not x8:
            x = model(x)
        else:
            x = utils_model.test_mode(model, x, mode=3)

    elif 'fdncnn' in model_name:

        noises1 = np.absolute(noises)
        noises1 = torch.from_numpy(noises1).float()
        noises1 = np.reshape(noises1, (256, 256, 1))
        noises1 = util.single2tensor4(noises1).to(device) / 255.
        x = torch.cat((x, noises1), dim=1).to(device)
        x = x.to(device)

        if not x8:
            x = model(x)
        else:
            x = utils_model.test_mode(model, x, mode=3)

    elif 'drunet' in model_name:
        if x8:
            x = util.augment_img_tensor4(x, i % 8)

        x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
        x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)

        if x8:
            if i % 8 == 3 or i % 8 == 5:
                x = util.augment_img_tensor4(x, 8 - i % 8)
            else:
                x = util.augment_img_tensor4(x, i % 8)

    elif 'ircnn' in model_name:
        if x8:
            x = util.augment_img_tensor4(x, i % 8)

        x = model(x)

        if x8:
            if i % 8 == 3 or i % 8 == 5:
                x = util.augment_img_tensor4(x, 8 - i % 8)
            else:
                x = util.augment_img_tensor4(x, i % 8)


    elif 'ffdnet' in model_name:
        ffdnet_sigma = torch.full((1, 1, 1, 1), noise_level_model / 255.).type_as(x)
        x = model(x, ffdnet_sigma)

    return x

def analyze_parse_PNP_PGD_CNC_D(default_alpha, default_iter_num, default_lambda1, default_b):
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=default_alpha, help="Step size in Plug-and Play")
    parser.add_argument("--iter_num", type=int, default=default_iter_num, help="Number of iterations")
    parser.add_argument("--lambda1", type=float, default=default_lambda1, help="regularization parameter")
    parser.add_argument("--b", type=float, default=default_b, help="convex parameter")

    PNP_PGD_CNC_D_opt = parser.parse_args()
    return PNP_PGD_CNC_D_opt

def PNP_PGD_CNC_D(model_name, mask, noises, **PNP_PGD_CNC_D_opts):
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    alpha = PNP_PGD_CNC_D_opts.get('alpha', 0.4)
    iter_num = PNP_PGD_CNC_D_opts.get('iter_num', 46)
    lambda1 = PNP_PGD_CNC_D_opts.get('lambda1', 2.75)
    b = PNP_PGD_CNC_D_opts.get('b', 1)           # Here b is b^2 in the original article

    a = b * alpha * lambda1

    task_current = 'dn'  # 'dn' for denoising
    testset_name = 'Set'
    x8 = True  # default: False, x8 to boost performance
    iter_k = 3
    n_channels = 1
    sf = 1  # unused for denoising
    show_img = False  # default: False
    save_L = True  # save LR image
    save_LEH = False  # save zoomed LR, E and H images
    border = 0
    n = 0
    sigmas = 0
    A = np.zeros((256, 256),dtype='uint8')
    out = [A, A, A]
    use_clip = True
    model_zoo = 'model_zoo'  # fixed
    testsets = 'testsets'  # fixed
    results = 'results'  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name
    model_path = os.path.join(model_zoo, model_name + '.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name)  # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)  # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    if 'dncnn' in model_name and 'fdncnn' not in model_name:
        from models.network_dncnn import DnCNN as net
        noise_level_img = 15
        noise_level_model = noise_level_img  # noise level of model, default 0
        if model_name in ['dncnn_gray_blind', 'dncnn_color_blind', 'dncnn3']:
            nb = 20  # fixed
        else:
            nb = 17  # fixed
        x8 = False
        border = sf if task_current == 'sr' else 0  # shave boader to calculate PSNR and SSIM
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    elif 'fdncnn' in model_name:
        from models.network_dncnn import FDnCNN as net
        border = sf if task_current == 'sr' else 0  # shave boader to calculate PSNR and SSIM
        x8 = False
        noise_level_img = 15  # default: 0, noise level for LR image
        noise_level_model = noise_level_img  # noise level of model, default 0

        if 'clip' in model_name:
            use_clip = True  # clip the intensities into range of [0, 1]
        else:
            use_clip = False

        model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=64, nb=20, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    elif 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        noise_level_img = 15 / 255.0  # default: 0, noise level for LR image
        noise_level_model = noise_level_img  # noise level of model, default 0
        n_channels = 3 if 'color' in model_name else 1  # fixed
        modelSigma1 = 49
        modelSigma2 = noise_level_model * 255.
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=iter_num,
                                         modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    elif 'ircnn' in model_name:
        from models.network_dncnn import IRCNN as net
        noise_level_img = 15 / 255.0  # default: 0, noise level for LR image
        noise_level_model = noise_level_img  # noise level of model, default 0
        modelSigma1 = 49
        modelSigma2 = noise_level_model * 255.
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255 / 255., noise_level_model), iter_num=iter_num,
                                         modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        model25 = torch.load(model_path)
        former_idx = 0

    elif 'ffdnet' in model_name:
        from models.network_ffdnet import FFDNet as net
        noise_level_img = 15  # noise level for noisy image
        noise_level_model = noise_level_img  # noise level for model
        nc = 64  # setting for grayscale image
        nb = 15  # setting for grayscale image
        border = sf if task_current == 'sr' else 0  # shave boader to calculate PSNR

        if 'clip' in model_name:
            use_clip = True  # clip the intensities into range of [0, 1]
        else:
            use_clip = False

        model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    test_results = OrderedDict()
    test_results['psnr'] = []

    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []

    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_L
        # --------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_H = util.modcrop(img_H, 8)
        img_L = util.uint2single(img_H)

        if use_clip:
            img_L = util.uint2single(util.single2uint(img_L))
        util.imshow(img_L) if show_img else None

        # --------------------------------
        # (2) initialize x
        # --------------------------------

        img_L = img_L.squeeze()
        y = np.fft.fft2(img_L) * mask + noises  # observed value
        img_L_init = np.fft.ifft2(y)
        x = np.copy(img_L_init)

        # --------------------------------
        # (3) main iterations
        # --------------------------------

        for i in range(iter_num):

            """ Gradient step  """
            x = np.reshape(x, (256, 256))
            res = np.fft.fft2(x) * mask
            index = np.nonzero(mask)
            res[index] = res[index] - y[index]
            z = x - alpha * np.fft.ifft2(res)
            z = np.absolute(z)
            z = np.reshape(z, (256, 256, 1))
            z = util.single2tensor4(z).to(device)

            x = np.absolute(x)
            x = torch.from_numpy(x).float()
            x = np.reshape(x, (256, 256, 1))
            x = util.single2tensor4(x).to(device)

            """ Denoising step  """

            if 'ircnn' in model_name:
                current_idx = np.int(np.ceil(sigmas[i].cpu().numpy() * 255. / 2.) - 1)

                if current_idx != former_idx:
                    model.load_state_dict(model25[str(current_idx)], strict=True)
                    model.eval()
                    for _, v in model.named_parameters():
                        v.requires_grad = False
                    model = model.to(device)
                former_idx = current_idx

            K = 0
            for k in range(iter_k):
                q = denoising_step2(model_name, x, x8, sigmas, i, model, noises, device, noise_level_model)
                t = z + a * (x - q)
                x = denoising_step2(model_name, t, x8, sigmas, i, model, noises, device, noise_level_model)
                K += 1
            x = x.data.squeeze().float().clamp_(0, 1).cpu().numpy()

        # --------------------------------
        # (4) img_E
        # --------------------------------
        if n_channels == 1:
            img_H = img_H.squeeze()

        out[n] = x
        img_E = np.uint8((x * 255.0).round())

        # --------------------------------
        # (5) img_LEH
        # --------------------------------

        if save_LEH:
            img_L = util.single2uint(img_L)
            k_v = k / np.max(k) * 1.0
            k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
            k_v = cv2.resize(k_v, (3 * k_v.shape[1], 3 * k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I = cv2.resize(img_L, (sf * img_L.shape[1], sf * img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
            img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
            img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
            util.imshow(np.concatenate([img_I, img_E, img_H], axis=1),
                        title='LR / Recovered / Ground-truth') if show_img else None
            util.imsave(np.concatenate([img_I, img_E, img_H], axis=1),
                        os.path.join(E_path, img_name + '_k' + '_LEH.png'))

        if save_L:
            util.imsave(util.single2uint(img_L), os.path.join(E_path, img_name + '_LR.png'))

        psnr = util.calculate_psnr(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)

        logger.info('{:s} - PSNR: {:.2f} dB'.format(img_name + ext, psnr))
        util.imshow(np.concatenate([img_E, img_H], axis=1),
                    title='Recovered / Ground-truth') if show_img else None

        n += 1

    return out
