import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import ssim
# from pytorch_msssimm import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from models import DepthNet
from pytorch_ssim import ssim
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
from utils.metrics import psnr
import time

import setproctitle
setproctitle.setproctitle("Dehaze_Test")

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--dehaze_result_dir', default='./results/dehaze_result/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='1', type=str, help='GPUs used for training')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def test_dehaze(test_loader, network, dehaze_result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(dehaze_result_dir, 'imgs'), exist_ok=True)
	f_result = open(os.path.join(dehaze_result_dir, 'dehaze_results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		start_time = time.time()
		input = batch['source'].cuda()
		target = batch['target'].cuda()
		filename = batch['filename'][0]

		with torch.no_grad():
			# output = network(input)[0].clamp_(-1, 1)
			output = network(input)[0].clamp_(-1, 1)

			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			# psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()
			psnr_val = psnr(output, target)

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
							 size_average=False).item()

			# ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
			# 				F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
			# 				data_range=1, size_average=False).item()

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

		print('Test_dehaze: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(dehaze_result_dir, 'imgs', filename), out_img)

	f_result.close()

	os.rename(os.path.join(dehaze_result_dir, 'dehaze_results.csv'),
			  os.path.join(dehaze_result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	DEPTH_NET = DepthNet.DN().cuda()
	network.cuda()
	DEPTH_NET.cuda()

	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):

		print('==> Start testing, current model name: ' + args.model)

		dehaze_state_dict = torch.load(saved_model_dir)
		depth_state_dict = torch.load('')   #DE weigth path
		network.load_state_dict(dehaze_state_dict['dehaze_net'])
		DEPTH_NET.load_state_dict(depth_state_dict['depth_net'])

	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	dehaze_result_dir = os.path.join(args.dehaze_result_dir, args.dataset, args.model)
	test_dehaze(test_loader, network, dehaze_result_dir)