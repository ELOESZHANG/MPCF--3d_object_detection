from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector
# from torch.cuda.amp import autocast as autocast


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'loss_rpn', 'loss_rcnn'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        # with autocast():
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        loss_rpn = ret_dict['loss_rpn'].mean()
        loss_rcnn = ret_dict['loss_rcnn'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict, loss_rpn, loss_rcnn)

    return model_func
