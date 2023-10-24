'''
ArcFace MS1MV3 r50
https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
'''
import math
import os.path as osp
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm
import sys
sys.path.append('.')
import numpy as np
import cv2
import torch.nn as nn
cv2.setNumThreads(1)
import torchvision
from scripts.utils import pad_tensor_back


@MODEL_REGISTRY.register()
class REGRESSIONSRModel(BaseModel):

    def __init__(self, opt):
        super(REGRESSIONSRModel, self).__init__(opt)

        # define u-net network
        self.unet = build_network(opt['network_unet'])
        self.unet = self.model_to_device(self.unet)
        self.print_network(self.unet)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.unet, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if 'lpips' in self.opt['val']['metrics']:
            import lpips
            self.lpips = lpips.LPIPS(net='alex')
            self.lpips = self.model_to_device(self.lpips)


        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.unet.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        for _, param in self.unet.named_parameters():
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.LR = data['LR'].to(self.device)
        self.HR = data['HR'].to(self.device)
        if 'pad_left' in data:
            self.pad_left = data['pad_left'].to(self.device)
            self.pad_right = data['pad_right'].to(self.device)
            self.pad_top = data['pad_top'].to(self.device)
            self.pad_bottom = data['pad_bottom'].to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        self.optimizer_g.zero_grad()
        self.output = self.unet(self.LR)
        l_g_total = 0
        loss_dict = OrderedDict()
        l_g_pix = F.l1_loss(self.output, self.HR)
        loss_dict['l_g_pix'] = l_g_pix
        l_g_total += l_g_pix
        l_g_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        if self.opt['val'].get('test_speed', False):
            with torch.no_grad():
                iterations = self.opt['val'].get('iterations', 100)
                input_size = self.opt['val'].get('input_size', [400, 600])

                LR = torch.randn(1, 6, input_size[0], input_size[1]).to(self.device)
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # self.bare_model.denoise_fn.eval()
                
                # GPU 预热
                print('GPU 预热')
                for _ in tqdm(range(50)):
                    self.output = self.unet(LR)
                
                # 测速
                times = torch.zeros(iterations)     # 存储每轮iteration的时间
                for iter in tqdm(range(iterations)):
                    starter.record()
                    self.output = self.unet(LR)
                    ender.record()
                    # 同步GPU时间
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender) # 计算时间
                    times[iter] = curr_time
                    # print(curr_time)

                mean_time = times.mean().item()
                logger = get_root_logger()
                logger.info("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
                import sys
                sys.exit()
        with torch.no_grad():
            self.unet.eval()
            self.output = self.unet(self.LR)
            if hasattr(self, 'pad_left'):
                self.output = pad_tensor_back(self.output, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.LR = pad_tensor_back(self.LR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.HR = pad_tensor_back(self.HR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)

            self.unet.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        metric_data = dict()
        metric_data_pytorch = dict()
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            if self.opt['datasets']['val'].get('use_01', False):
                sr_img = tensor2img([visuals['sr']], min_max=(0, 1))
                gt_img = tensor2img([visuals['gt']], min_max=(0, 1))
                lq_img = tensor2img([visuals['lq']], min_max=(0, 1))
            else:
                sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))
                gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
                lq_img = tensor2img([visuals['lq']], min_max=(-1, 1))

            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img
            metric_data_pytorch['img'] = self.output
            metric_data_pytorch['img2'] = self.HR
            path = val_data['lq_path'][0]
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                # print(save_img_path)
                if idx < 10 or self.opt['val'].get('cal_score', False):
                    if self.opt['val'].get('only_save_sr', False):
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                        f'{img_name}.png')
                        imwrite(sr_img, save_img_path)
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}.png')
                        imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'lpips' in opt_['type']:
                        opt_['device'] = self.device
                        opt_['model'] = self.lpips
                    if 'pytorch' in opt_['type']:
                        self.metric_results[name] += calculate_metric(metric_data_pytorch, opt_).item()
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            # tentative for out of GPU memory
            del self.LR
            del self.output
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if self.opt['val'].get('cal_score', False):
            import sys
            sys.exit()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.HR.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        out_dict['lq'] = self.LR[:, :3, :, :].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.unet], 'net_g', current_iter, param_key=['params'])
