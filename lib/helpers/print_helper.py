import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import decode_detections
from lib.datasets.kitti_utils import get_affine_transform

class Printer(object):
    def __init__(self, cfg, cfg_dataset, model, data_loader, logger):
        self.cfg = cfg
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label_dir = cfg_dataset['label_dir']
        self.eval_cls = cfg_dataset['eval_cls']

        if self.cfg.get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = cfg['resume_model'],
                        logger = self.logger,
                        map_location=self.device)
        self.model.to(self.device)

    def print(self, idx):

        dataset = self.data_loader.dataset

        img = dataset.get_image(idx)
        calib = dataset.get_calib(idx)
        img_size = np.array(img.size)


        center = np.array(img_size) / 2
        crop_size = img_size
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)

        trans, trans_inv = get_affine_transform(center, crop_size, 0, dataset.resolution, inv=1)
        inputs = img.transform(tuple(dataset.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        inputs = np.array(inputs).astype(np.float32) / 255.0
        inputs = (inputs - dataset.mean) / dataset.std
        inputs = inputs.transpose(2, 0, 1)  # C * H * W

        features_size = dataset.resolution // dataset.downsample
        info = {'img_id': idx,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}


        torch.set_grad_enabled(False)
        self.model.eval()
        inputs = torch.Tensor(inputs).unsqueeze(0).to(self.device)
        calib = torch.Tensor(calib.P2).unsqueeze(0).to(self.device)
        coord_ranges = torch.Tensor(coord_range).unsqueeze(0).to(self.device)
        output = self.model(inputs, coord_ranges, calib, K=50, mode='test')

        # get corresponding calibs & transform tensor to numpy
        calibs = dataset.get_calib(idx)
        info = {key: val.detach().cpu().numpy() for key, val in info.items()}
        cls_mean_size = dataset.cls_mean_size
        dets = decode_detections(dets = dets,
                                info = info,
                                calibs = calibs,
                                cls_mean_size=cls_mean_size,
                                threshold = self.cfg['threshold']
                                )


        print(dets)

        plt.axis('off')
        plt.imshow(img)
        plt.show()