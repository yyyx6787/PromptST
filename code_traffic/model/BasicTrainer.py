import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
import torch.nn.functional as F
from model.MTGNN import MTGNN as Network1
# from model.MTGNN import PGNN as Network2
import torch.nn as nn 




class Trainer(object):
	def __init__(self, model, pmodel, loss, optimizer, optimizerp, train_loader, val_loader, test_loader, prompt_loader, 
				 scaler, args, lr_scheduler=None, lr_schedulerp=None):
		super(Trainer, self).__init__()
		self.model = model
		self.pmodel = pmodel
		self.args = args
		self.loss = loss
		self.optimizer = optimizer
		self.optimizerp = optimizerp
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.prompt_loader = prompt_loader
		self.scaler = scaler
		self.args = args
		self.lr_scheduler = lr_scheduler
		self.lr_schedulerp = lr_schedulerp
		self.train_per_epoch = len(train_loader)
		if val_loader != None:
			self.val_per_epoch = len(val_loader)
		self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
		self.best_pathP = os.path.join(self.args.log_dir, 'best_model_prompt.pth')
		self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
		#log
		if os.path.isdir(args.log_dir) == False and not args.debug:
			os.makedirs(args.log_dir, exist_ok=True)
		self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
		self.logger.info('Experiment log path in: {}'.format(args.log_dir))
		#if not args.debug:
		#self.logger.info("Argument: %r", args)
		# for arg, value in sorted(vars(args).items()):
		#     self.logger.info("Argument %s: %r", arg, value)
		if self.args.teacher:
			self.tmodel = self.loadTeacher(self.args)
	# load teacher model
	def loadTeacher(self,args):
		# teacherModel = torch.load('./SAVE/PEMS08/best_model.pth')
		# ckp = t.load('./SAVE/PEMS08/best_model.pth')
		# teacher = ckp['model']
		tmodel = Network1(args)
		tmodel = tmodel.to(args.device)
		# tmodel = tmodel.load_state_dict(torch.load('./SAVE/PEMS08/best_model.pth'), strict=False)
		tmodel.load_state_dict(torch.load('./SAVE/PEMS04/best_model.pth'), strict=False)
		return tmodel
	def val_epoch(self, epoch, val_dataloader):
		self.pmodel.eval()
		total_val_loss = 0

		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(val_dataloader):
				data = data[..., :self.args.input_dim]
				label = target[..., :self.args.output_dim]
				output, _, _ = self.pmodel(data)
				output, _, _ = self.model(output)
				if self.args.real_value:
					label = self.scaler.inverse_transform(label)
				loss = self.loss(output.cuda(), label)
				#a whole batch of Metr_LA is filtered
				if not torch.isnan(loss):
					total_val_loss += loss.item()
		val_loss = total_val_loss / len(val_dataloader)
		self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
		return val_loss
	def val_epochi(self, epoch, val_dataloader):
		self.model.eval()
		total_val_loss = 0

		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(val_dataloader):
				data = data[..., :self.args.input_dim]
				label = target[..., :self.args.output_dim]
				
				output, _, _ = self.model(data)
				if self.args.real_value:
					label = self.scaler.inverse_transform(label)
				loss = self.loss(output.cuda(), label)
				#a whole batch of Metr_LA is filtered
				if not torch.isnan(loss):
					total_val_loss += loss.item()
		val_loss = total_val_loss / len(val_dataloader)
		self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
		return val_loss

	def train_epoch(self, epoch):
		self.model.train()
		total_loss = 0
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data = data[..., :self.args.input_dim]
			label = target[..., :self.args.output_dim]  # (..., 1)
			self.optimizer.zero_grad()
			out, _, _ = self.model(data)

			if self.args.real_value:
				label = self.scaler.inverse_transform(label)

			loss = self.loss(out.cuda(), label)
			loss.backward()

			# add max grad clipping
			if self.args.grad_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
			self.optimizer.step()
			total_loss += loss.item()

			#log information
			if batch_idx % self.args.log_step == 0:
				self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
					epoch, batch_idx, self.train_per_epoch, loss.item()))
		train_epoch_loss = total_loss/self.train_per_epoch
		self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

		#learning rate decay
		if self.args.lr_decay:
			self.lr_scheduler.step()
		return train_epoch_loss

	def train_epochp(self, epoch, model):
		# s_time = time.time()
		self.pmodel.train()
		# print("1 step:", time.time() - s_time)
		# for param in model.parameters():
		# 	param.requires_grad = False
		for param in model.parameters():
			param.requires_grad = False
		# self.optimizer(filter(lambda param: param.requires_grad, model.parameters()), 
		# 	lr=self.args.lr_init, eps=1.0e-8, weight_decay=0)
		total_loss = 0
		for batch_idx, (data, target) in enumerate(self.prompt_loader):
			data = data[..., :self.args.input_dim]
			label = target[..., :self.args.output_dim]  # (..., 1)
			self.optimizerp.zero_grad()
			# a_time = time.time()
			# s1 = time.time()
			out, embeds, _ = self.pmodel(data)  #out dim: torch.Size([128, 12, 307, 1])
			# s2 = time.time()


			# print("lightgcn:", out.size())
			# s5 = time.time()
			out, _, _ = model(out)
			# s6 = time.time()

			# s3 = time.time()
			# out, embeds, _ = self.pmodel(data)  #out dim: torch.Size([128, 12, 307, 1])
			# s4 = time.time()
			# print("3 step:", s2 -s1, s6-s5)
			# println()
			if self.args.real_value:
				label = self.scaler.inverse_transform(label)

			loss = self.loss(out.cuda(), label)
			loss.backward()

			# add max grad clipping
			if self.args.grad_norm:
				torch.nn.utils.clip_grad_norm_(self.pmodel.parameters(), self.args.max_grad_norm)
			self.optimizerp.step()
			total_loss += loss.item()

			#log information
			if batch_idx % self.args.log_step == 0:
				self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
					epoch, batch_idx, self.train_per_epoch, loss.item()))
		train_epoch_loss = total_loss/self.train_per_epoch
		self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

		#learning rate decay
		if self.args.lr_decay:
			self.lr_schedulerp.step()
		return train_epoch_loss
	
	def train_epochf(self, epoch, model):
		model_fit = model
		for param in model_fit.parameters():
			param.requires_grad = True
		total_loss = 0
		for batch_idx, (data, target) in enumerate(self.prompt_loader):
			data = data[..., :self.args.input_dim]
			label = target[..., :self.args.output_dim]  # (..., 1)
			self.optimizer.zero_grad()
			# s1 = time.time()
			out, _, _ = model_fit(data)
			# s2 = time.time()
			# print("step for fine-tune:", s2-s1)
			
			
			if self.args.real_value:
				label = self.scaler.inverse_transform(label)

			loss = self.loss(out.cuda(), label)
			loss.backward()

			# add max grad clipping
			if self.args.grad_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
			self.optimizer.step()
			total_loss += loss.item()

			#log information
			if batch_idx % self.args.log_step == 0:
				self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
					epoch, batch_idx, self.train_per_epoch, loss.item()))
		train_epoch_loss = total_loss/self.train_per_epoch
		self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
		#learning rate decay
		if self.args.lr_decay:
			self.lr_scheduler.step()
		return train_epoch_loss

	def train(self):
		best_model = None
		best_loss = float('inf')
		not_improved_count = 0
		train_loss_list = []
		val_loss_list = []
		start_time = time.time()
		for epoch in range(1, self.args.epochs + 1):
			epoch_time = time.time()
			train_epoch_loss = self.train_epoch(epoch)
			print(time.time()-epoch_time)
			#exit()
			if self.val_loader == None:
				val_dataloader = self.test_loader
			else:
				val_dataloader = self.val_loader
			val_epoch_loss = self.val_epochi(epoch, val_dataloader)

			#print('LR:', self.optimizer.param_groups[0]['lr'])
			train_loss_list.append(train_epoch_loss)
			val_loss_list.append(val_epoch_loss)
			if train_epoch_loss > 1e6:
				self.logger.warning('Gradient explosion detected. Ending...')
				break
			#if self.val_loader == None:
			#val_epoch_loss = train_epoch_loss
			if val_epoch_loss < best_loss:
				best_loss = val_epoch_loss
				not_improved_count = 0
				best_state = True
			else:
				not_improved_count += 1
				best_state = False
			# early stop
			if self.args.early_stop:
				if not_improved_count == self.args.early_stop_patience:
					self.logger.info("Validation performance didn\'t improve for {} epochs. "
									"Training stops.".format(self.args.early_stop_patience))
					break
			# save the best state
			if best_state == True:
				self.logger.info('*********************************Current best model saved!')
				best_model = copy.deepcopy(self.model.state_dict())

		training_time = time.time() - start_time
		self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

		#save the best model to file
		# if not self.args.debug:
		if self.args.debug:
			torch.save(best_model, self.best_path)
			self.logger.info("Saving current best model to " + self.best_path)

		#test
		self.model.load_state_dict(best_model)
		#self.val_epoch(self.args.epochs, self.test_loader)
		self.test_initial(self.model, self.args, self.test_loader, self.scaler, self.logger)



	def trainP(self, model, path=None):
		if path != None:
			check_point = torch.load(path)
			state_dict = check_point['state_dict']
			args = check_point['config']
			model.load_state_dict(state_dict)
			model.to(args.device)
		model.eval()

		best_model = None
		best_loss = float('inf')
		not_improved_count = 0
		train_loss_list = []
		val_loss_list = []
		start_time = time.time()
		for epoch in range(1, self.args.epochs + 1):
			epoch_time = time.time()
			train_epoch_loss = self.train_epochp(epoch, model)
			print(time.time()-epoch_time)
			#exit()
			if self.val_loader == None:
				val_dataloader = self.test_loader
			else:
				val_dataloader = self.val_loader
			val_epoch_loss = self.val_epoch(epoch, val_dataloader)

			#print('LR:', self.optimizer.param_groups[0]['lr'])
			train_loss_list.append(train_epoch_loss)
			val_loss_list.append(val_epoch_loss)
			if train_epoch_loss > 1e6:
				self.logger.warning('Gradient explosion detected. Ending...')
				break
			#if self.val_loader == None:
			#val_epoch_loss = train_epoch_loss
			if val_epoch_loss < best_loss:
				best_loss = val_epoch_loss
				not_improved_count = 0
				best_state = True
			else:
				not_improved_count += 1
				best_state = False
			# early stop
			if self.args.early_stop:
				if not_improved_count == self.args.early_stop_patience:
					self.logger.info("Validation performance didn\'t improve for {} epochs. "
									"Training stops.".format(self.args.early_stop_patience))
					break
			# save the best state
			if best_state == True:
				self.logger.info('*********************************Current best model saved!')
				best_model = copy.deepcopy(self.pmodel.state_dict())
				best_modelo = copy.deepcopy(self.model.state_dict())

		training_time = time.time() - start_time
		self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

		#save the best model to file
		# if not self.args.debug:
		if self.args.debug:
			torch.save(best_model, self.best_pathP)
			self.logger.info("Saving current best model to " + self.best_pathP)
		

		#test
		self.model.load_state_dict(best_modelo)
		self.pmodel.load_state_dict(best_model)
		#self.val_epoch(self.args.epochs, self.test_loader)
		self.test(self.model, self.pmodel, self.args, self.test_loader, self.scaler, self.logger)

	def trainF(self, model, path=None):
		if path != None:
			check_point = torch.load(path)
			state_dict = check_point['state_dict']
			args = check_point['config']
			model.load_state_dict(state_dict)
			model.to(args.device)
		model.eval()

		best_model = None
		best_loss = float('inf')
		not_improved_count = 0
		train_loss_list = []
		val_loss_list = []
		start_time = time.time()
		for epoch in range(1, self.args.epochs + 1):
			epoch_time = time.time()
			train_epoch_loss = self.train_epochf(epoch, model)
			print(time.time()-epoch_time)
			#exit()
			if self.val_loader == None:
				val_dataloader = self.test_loader
			else:
				val_dataloader = self.val_loader
			val_epoch_loss = self.val_epochi(epoch, val_dataloader)

			#print('LR:', self.optimizer.param_groups[0]['lr'])
			train_loss_list.append(train_epoch_loss)
			val_loss_list.append(val_epoch_loss)
			if train_epoch_loss > 1e6:
				self.logger.warning('Gradient explosion detected. Ending...')
				break
			#if self.val_loader == None:
			#val_epoch_loss = train_epoch_loss
			if val_epoch_loss < best_loss:
				best_loss = val_epoch_loss
				not_improved_count = 0
				best_state = True
			else:
				not_improved_count += 1
				best_state = False
			# early stop
			if self.args.early_stop:
				if not_improved_count == self.args.early_stop_patience:
					self.logger.info("Validation performance didn\'t improve for {} epochs. "
									"Training stops.".format(self.args.early_stop_patience))
					break
			# save the best state
			if best_state == True:
				self.logger.info('*********************************Current best model saved!')
				# best_model = copy.deepcopy(self.pmodel.state_dict())
				best_modelo = copy.deepcopy(self.model.state_dict())

		training_time = time.time() - start_time
		self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

		#save the best model to file
		# if not self.args.debug:
		if self.args.debug:
			torch.save(best_model, self.best_pathP)
			self.logger.info("Saving current best model to " + self.best_pathP)
		

		#test
		self.model.load_state_dict(best_modelo)
		# self.pmodel.load_state_dict(best_model)
		#self.val_epoch(self.args.epochs, self.test_loader)
		
		self.test_initial(self.model, self.args, self.test_loader, self.scaler, self.logger)
		
	def save_checkpoint(self):
		state = {
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'config': self.args
		}
		torch.save(state, self.best_path)
		self.logger.info("Saving current best model to " + self.best_path)

	@staticmethod
	def test_initial(model, args, data_loader, scaler, logger, path=None):
		if path != None:
			check_point = torch.load(path)
			state_dict = check_point['state_dict']
			args = check_point['config']
			model.load_state_dict(state_dict)
			model.to(args.device)

		model.eval()
		y_pred = []
		y_true = []
		y_emb = []
		start_time = time.time()
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data = data[..., :args.input_dim]
				label = target[..., :args.output_dim]
				output, vectors, _ = model(data)
				y_true.append(label)
				y_pred.append(output)
				y_emb.append(vectors)
		y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
		if args.real_value:
			y_pred = torch.cat(y_pred, dim=0)
		else:
			y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
		y_emb = torch.cat(y_emb, dim=0)
		np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
		np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
		np.save('./{}_emb.npy'.format(args.dataset), y_emb.cpu().numpy())
		for t in range(y_true.shape[1]):
			mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
												args.mae_thresh, args.mape_thresh)
			logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
				t + 1, mae, rmse, mape*100))
		mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
		time_p = time.time()- start_time
		logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, Test Time:{:.7f}".format(
					mae, rmse, mape*100, time_p))
	@staticmethod
	def test(model, pmodel, args, data_loader, scaler, logger, path=None):
		if path != None:
			check_point = torch.load(path)
			state_dict = check_point['state_dict']
			args = check_point['config']
			model.load_state_dict(state_dict)
			model.to(args.device)
			pmodel.load_state_dict(state_dict)
			pmodel.to(args.device)

		model.eval()
		pmodel.eval()

		y_pred = []
		y_true = []
		y_emb = []
		start_time = time.time()
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data = data[..., :args.input_dim]
				label = target[..., :args.output_dim]
				output, _, _ = pmodel(data)
				output, vectors, _ = model(output)
				y_true.append(label)
				y_pred.append(output)
				y_emb.append(vectors)
		y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
		if args.real_value:
			y_pred = torch.cat(y_pred, dim=0)
		else:
			y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
		y_emb = torch.cat(y_emb, dim=0)
		np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
		np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
		np.save('./{}_emb.npy'.format(args.dataset), y_emb.cpu().numpy())
		for t in range(y_true.shape[1]):
			mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
												args.mae_thresh, args.mape_thresh)
			logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
				t + 1, mae, rmse, mape*100))
		mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
		time_p = time.time()- start_time
		logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, Test Time:{:.7f}".format(
					mae, rmse, mape*100, time_p))
	

	@staticmethod
	def _compute_sampling_threshold(global_step, k):
		"""
		Computes the sampling probability for scheduled sampling using inverse sigmoid.
		:param global_step:
		:param k:
		:return:
		"""
		return k / (k + math.exp(global_step / k))