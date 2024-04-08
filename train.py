import os
import argparse
import json
from torch import nn
from depth.networks import *
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models import DepthNet, DIACMPN, DIACMPN_dehaze_Indoor
from models.UNet import UNet
from pytorch_ssim import ssim
from utils import AverageMeter
from datasets.loader import PairLoader
from loss.CR_loss import ContrastLoss as crloss
import setproctitle
setproctitle.setproctitle("Dehaze")


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DIACMPN-dehaze-Indoor', type=str, help='dehaze model name')
parser.add_argument('--model_depth', default='DIACMPN-depth-Indoor', type=str, help='depth model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0，1，2，3', type=str, help='GPUs used for training')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def train(train_loader, network, DEPTH_NET, criterion_dehaze, criterion_dehaze_cr, criterion_depth,
				   optimizer_dehaze, optimizer_depth, scaler):
	Dehaze_loss = AverageMeter()
	Depth_loss = AverageMeter()
	torch.cuda.empty_cache()

	network.train()
	DEPTH_NET.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			dehaze_output_img, d11, d22, d33 = network(source_img)
			dehaze_output_img_2_depth_img = DEPTH_NET(dehaze_output_img)

			real_img_2_depth_img = depth_decoder(encoder(target_img))
			real_img_2_depth_img = real_img_2_depth_img[("disp", 0)]

			diff_dehaze = torch.sub(dehaze_output_img, target_img)
			B, C, H, W = diff_dehaze.shape
			diff_dehaze = diff_dehaze.permute(0, 2, 3, 1)
			diff_dehaze = diff_dehaze.reshape(-1, C * H * W)
			epsilon = 1e-7
			diff_d_w = F.softmax(diff_dehaze, dim=-1) + epsilon
			diff_d_w = diff_d_w.reshape(B, H, W, C).permute(0, 3, 1, 2)
			diff_dehaze_w = torch.sum(diff_d_w, dim=1, keepdim=True)
			weighted_depth_output_img = torch.mul(dehaze_output_img_2_depth_img, diff_dehaze_w)
			weighted_real_img_2_depth_img = torch.mul(real_img_2_depth_img, diff_dehaze_w)

			loss_depth_consis = criterion_depth(weighted_depth_output_img, weighted_real_img_2_depth_img)
			loss_depth_consis_w = criterion_depth(dehaze_output_img_2_depth_img, real_img_2_depth_img)
			loss_total_depth = loss_depth_consis + loss_depth_consis_w

			t_d1, t_d2, t_d3 = deep_estimate_net(target_img)
			o_d1, o_d2, o_d3 = deep_estimate_net(source_img)

			loss_dehaza_consis = criterion_dehaze(dehaze_output_img, target_img)
			loss_dehaze_consis_w = criterion_dehaze(dehaze_output_img_2_depth_img, real_img_2_depth_img)
			loss_dehaze_cr = criterion_dehaze_cr(dehaze_output_img, target_img, source_img)
			loss_dehaze_u1 = criterion_dehaze(t_d1, o_d1)
			loss_dehaze_u2 = criterion_dehaze(t_d2, o_d2)
			loss_dehaze_u3 = criterion_dehaze(t_d3, o_d3)

			loss_dehaze_total = loss_dehaza_consis + 0.1*loss_dehaze_consis_w + loss_dehaze_cr + \
								loss_dehaze_u1 + loss_dehaze_u2 + loss_dehaze_u3

		Dehaze_loss.update(loss_dehaze_total.item())
		Depth_loss.update(loss_total_depth.item())

		optimizer_dehaze.zero_grad()
		optimizer_depth.zero_grad()

		scaler.scale(loss_dehaze_total + loss_total_depth).backward()

		scaler.step(optimizer_dehaze)
		scaler.step(optimizer_depth)

		scaler.update()

	return Dehaze_loss.avg, Depth_loss.avg


def valid(val_loader, network):

	PSNR = AverageMeter()
	SSIM = AverageMeter()

	torch.cuda.empty_cache()
	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():
			output = network(source_img)[0].clamp_(-1, 1)
		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))
		# ssim_value = ssim(output, target_img, data_range=1.0, size_average=True)
		ssim_value = ssim(output, target_img, size_average=True)
		SSIM.update(ssim_value.item(), source_img.size(0))

	return PSNR.avg, SSIM.avg

if __name__ == '__main__':
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	with torch.no_grad():
		model_path = os.path.join("./depth/models", 'RA-Depth')
		assert os.path.isdir(model_path), \
			"Cannot find a folder at {}".format(model_path)
		print("-> Loading weights from {}".format(model_path))

		encoder_path = os.path.join(model_path, "encoder.pth")
		decoder_path = os.path.join(model_path, "depth.pth")
		encoder_dict = torch.load(encoder_path)
		encoder = hrnet18(False)
		depth_decoder = DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1)
		model_dict = encoder.state_dict()
		encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
		depth_decoder.load_state_dict(torch.load(decoder_path))
		encoder.cuda()
		encoder.eval()
		depth_decoder.cuda()
		depth_decoder.eval()

	network = DIACMPN_dehaze_Indoor().cuda()
	DEPTH_NET = DepthNet.DN().cuda()
	deep_estimate_net = UNet().cuda()

	criterion_dehaze = nn.L1Loss()
	criterion_dehaze_cr = crloss()
	criterion_depth = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer_dehaze = torch.optim.Adam(network.parameters(), lr=setting['lr_dehaze'])
		optimizer_depth = torch.optim.Adam(DEPTH_NET.parameters(), lr=setting['lr_depth'])
	elif setting['optimizer'] == 'adamw':
		optimizer_dehaze = torch.optim.AdamW(network.parameters(), lr=setting['lr_dehaze'])
		optimizer_depth = torch.optim.AdamW(DEPTH_NET.parameters(), lr=setting['lr_depth'])
	else:
		raise Exception("ERROR: unsupported optimizer")

	scheduler_dehaze = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dehaze, T_max=setting['epochs'], eta_min=setting['lr_dehaze'] * 1e-2)
	scheduler_depth = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_depth, T_max=setting['epochs'], eta_min=setting['lr_depth'] * 1e-2)
	scaler = GradScaler()

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train',
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
							  batch_size=setting['batch_size'],
							  shuffle=True,
							  num_workers=args.num_workers,
							  pin_memory=True,
							  drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
							batch_size=setting['batch_size'],
							num_workers=args.num_workers,
							pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):

		print('==> Start training, current model name: ' + args.model)
		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0

		for epoch in tqdm(range(setting['epochs'] + 1)):

			dehaze_loss, depth_loss = train(train_loader, network, DEPTH_NET, criterion_dehaze, criterion_dehaze_cr, criterion_depth,
				   optimizer_dehaze, optimizer_depth, scaler)

			writer.add_scalar('dehaze_train_loss', dehaze_loss, epoch)
			writer.add_scalar('depth_train_loss', depth_loss, epoch)

			print('Dehaze_loss: ', dehaze_loss, epoch)
			print('Depth_loss: ', depth_loss, epoch)
			scheduler_dehaze.step()
			scheduler_depth.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr, avg_ssim = valid(val_loader, network)
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				print('valid_psnr', avg_psnr, epoch)
				print('valid_ssim', avg_ssim, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					Dehaze_state_dict = {'dehaze_net': network.state_dict(), 'dehaze_optimizer': optimizer_dehaze.state_dict(), 'epoch_dehaze': epoch, 'dehaze_net_best_psnr':best_psnr}
					depth_state_dict = {'depth_net': DEPTH_NET.state_dict(), 'depth_optimizer': optimizer_depth.state_dict(),'epoch_depth': epoch}
					torch.save(Dehaze_state_dict, os.path.join(save_dir, args.model+'.pth'))
					torch.save(depth_state_dict, os.path.join(save_dir, args.model_depth + '.pth'))
				print('best_psnr', best_psnr, epoch)

				writer.add_scalar('best_psnr', best_psnr, epoch)

				with open('./checkpoint/psnr_loss.txt', 'a') as file:
					file.write('Epoch [{}/{}], Loss_Dehaze: {:.4f}, Loss_Depth: {:.4f}'
							   .format(epoch + 1, setting['epochs'], dehaze_loss, depth_loss))
					file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
					file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
					file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
					file.write('\n')
	else:
		print('==> 1')

#############################################################################################################################
		dehaze_checkpoint = torch.load(os.path.join(save_dir, args.model + '.pth'))
		depth_checkpoint = torch.load(os.path.join(save_dir, args.model_depth + '.pth'))
		network.load_state_dict(dehaze_checkpoint['dehaze_net'])
		DEPTH_NET.load_state_dict(depth_checkpoint['depth_net'])
		optimizer_dehaze.load_state_dict(dehaze_checkpoint['dehaze_optimizer'])
		optimizer_depth.load_state_dict(depth_checkpoint['depth_optimizer'])
		best_psnr = dehaze_checkpoint['dehaze_net_best_psnr']
		start_epoch = dehaze_checkpoint['epoch_dehaze'] + 1

		print('Load start_epoch {} ！'.format(start_epoch))
		print('Load depth_optimizer {} ！'.format(optimizer_dehaze))
		print('Load depth_optimizer {} ！'.format(optimizer_depth))

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))
		#
		# best_psnr = 0

		for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):
			dehaze_loss, depth_loss = train(train_loader, network, DEPTH_NET, criterion_dehaze, criterion_dehaze_cr, criterion_depth,
				   optimizer_dehaze, optimizer_depth, scaler)

			writer.add_scalar('dehaze_train_loss', dehaze_loss, epoch)
			writer.add_scalar('depth_train_loss', depth_loss, epoch)

			print('Dehaze_loss: ', dehaze_loss, epoch)
			print('Depth_loss: ', depth_loss, epoch)
			scheduler_dehaze.step()
			scheduler_depth.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr, avg_ssim = valid(val_loader, network)

				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				print('valid_psnr', avg_psnr, epoch)
				print('valid_ssim', avg_ssim, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					Dehaze_state_dict = {'dehaze_net': network.state_dict(),
										 'dehaze_optimizer': optimizer_dehaze.state_dict(),
										 'epoch_dehaze': epoch,
										 'loss_dehaze': dehaze_loss,
										 'dehaze_net_best_psnr':best_psnr}
					depth_state_dict = {'depth_net': DEPTH_NET.state_dict(),
										'depth_optimizer': optimizer_depth.state_dict(),
										'epoch_depth': epoch,
										'loss_depth': depth_loss}
					torch.save(Dehaze_state_dict, os.path.join(save_dir, args.model + '.pth'))
					torch.save(depth_state_dict, os.path.join(save_dir, args.model_depth + '.pth'))

				writer.add_scalar('best_psnr', best_psnr, epoch)
				print('best_psnr', best_psnr, epoch)

				with open('./checkpoint/psnr_loss.txt', 'a') as file:
					file.write('Epoch [{}/{}], Loss_Dehaze: {:.4f}, Loss_Depth: {:.4f}'
							   .format(epoch + 1, setting['epochs'], dehaze_loss, depth_loss))
					file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
					file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
					file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
					file.write('\n')