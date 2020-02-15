"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.

Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""
import numpy as np
import os.path as osp
import argparse
import cv2
import torch
from torch.nn import functional as F

import torchreid
from torchreid.utils import (
    check_isfile, mkdir_if_missing, load_pretrained_weights
)
from torchreid.models import resnet_orig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
        model,
        test_loader,
        save_dir,
        width,
        height,
        use_gpu,
        img_mean=None,
        img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    for target in list(test_loader.keys()):
        data_loader = test_loader[target]['query']  # only process query images
        # original images and activation maps are saved individually
        actmap_dir = osp.join(save_dir, 'actmap_' + target)
        mkdir_if_missing(actmap_dir)
        print('Visualizing activation maps for {} ...'.format(target))

        for batch_idx, data in enumerate(data_loader):
            imgs, paths = data[0], data[3]
            if use_gpu:
                imgs = imgs.cuda()

            outputs = model(imgs)
            # for ABD-Net, use the last conv layer of attentive branch
            outputs = outputs[-1]['after'][0]

            if outputs.dim() != 4:
                raise ValueError(
                    'The model output is supposed to have '
                    'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                    'Please make sure you set the model output at eval mode '
                    'to be the last convolutional feature maps'.format(
                        outputs.dim()
                    )
                )

            # compute activation maps
            outputs = (outputs ** 2).sum(1)
            b, h, w = outputs.size()
            outputs = outputs.view(b, h * w)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.view(b, h, w)

            if use_gpu:
                imgs, outputs = imgs.cpu(), outputs.cpu()

            for j in range(outputs.size(0)):
                # get image name
                path = paths[j]
                imname = osp.basename(osp.splitext(path)[0])

                # RGB image
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

                # activation map
                am = outputs[j, ...].numpy()
                am = cv2.resize(am, (width, height))
                am = 255 * (am - np.min(am)) / (
                        np.max(am) - np.min(am) + 1e-12
                )
                am = np.uint8(np.floor(am))
                am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                # overlapped
                overlapped = img_np * 0.3 + am * 0.7
                overlapped[overlapped > 255] = 255
                overlapped = overlapped.astype(np.uint8)

                # save images in a single figure (add white spacing between images)
                # from left to right: original image, activation map, overlapped image
                grid_img = 255 * np.ones(
                    (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8
                )
                grid_img[:, :width, :] = img_np[:, :, ::-1]
                grid_img[:,
                width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
                grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

            if (batch_idx + 1) % 10 == 0:
                print(
                    '- done batch {}/{}'.format(
                        batch_idx + 1, len(data_loader)
                    )
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/remote_data/Market1501')
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-m', '--model', type=str, default='abd_resnet50')
    # /home/ddj2/PycharmProjects/deep-person-reid-master/saves/checkpoint_best.pth.tar
    parser.add_argument('--weights', type=str,
                        default='/media/ddj2/8b1bfd93-3f3f-4475-b279-6a9ae59c6639/'
                                'remote_dir/checkpoint/market_checkpoint_best.pth.tar')
    parser.add_argument('--save-dir', type=str,
                        default='/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/remote_data/Market1501/log/abd'
                                '-official-pretrained')
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=128)

    args = parser.parse_args()

    new_args = {
        'shallow_cam': True,
        'compatibility': False,
        'branches': ['global', 'abd'],
        'abd_dim': 1024,
        'global_dim': 1024,
        'abd_np': 2,
        'abd_dan': ['cam', 'pam'],
        'abd_dan_no_head': False,
        'dropout': 0.5,
        'global_max_pooling': False,
        'use_ow': True,
        'margin': 1.2,
        'label_smooth': True,
        'flip_eval': True,
    }

    use_gpu = torch.cuda.is_available()

    datamanager = torchreid.data.ImageDataManager(
        root=args.root,
        sources=args.dataset,
        height=args.height,
        width=args.width,
        batch_size_train=100,
        batch_size_test=100,
        transforms=None,
        train_sampler='SequentialSampler'
    )
    test_loader = datamanager.test_loader

    # model = resnet_orig.resnet50(num_classes=1501)
    model = torchreid.models.build_model(
        name=args.model,
        num_classes=datamanager.num_train_pids,
        use_gpu=use_gpu,
        args=new_args
    )

    if use_gpu:
        model = model.cuda()

    if args.weights and check_isfile(args.weights):
        load_pretrained_weights(model, args.weights)

    visactmap(
        model, test_loader, args.save_dir, args.width, args.height, use_gpu
    )


if __name__ == '__main__':
    main()
