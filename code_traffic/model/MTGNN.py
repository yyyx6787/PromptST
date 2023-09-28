from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F


class NConv(nn.Module):
	def __init__(self):
		super(NConv, self).__init__()

	def forward(self, x, adj):
		x = torch.einsum('ncwl,vw->ncvl', (x, adj))
		return x.contiguous()


class DyNconv(nn.Module):
	def __init__(self):
		super(DyNconv, self).__init__()

	def forward(self, x, adj):
		x = torch.einsum('ncvl,nvwl->ncwl', (x, adj))
		return x.contiguous()


class Linear(nn.Module):
	def __init__(self, c_in, c_out, bias=True):
		super(Linear, self).__init__()
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

	def forward(self, x):
		return self.mlp(x)


class Prop(nn.Module):
	def __init__(self, c_in, c_out, gdep, dropout, alpha):
		super(Prop, self).__init__()
		self.nconv = NConv()
		self.mlp = Linear(c_in, c_out)
		self.gdep = gdep
		self.dropout = dropout
		self.alpha = alpha

	def forward(self, x, adj):
		adj = adj + torch.eye(adj.size(0)).to(x.device)
		d = adj.sum(1)
		h = x
		dv = d
		a = adj / dv.view(-1, 1)
		for i in range(self.gdep):
			h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
		ho = self.mlp(h)
		return ho


class MixProp(nn.Module):
	def __init__(self, c_in, c_out, gdep, dropout, alpha):
		super(MixProp, self).__init__()
		# self.nconv = nn.Sequential(GCNLayer())
		self.nconv = NConv()
		self.mlp = Linear((gdep+1)*c_in, c_out)
		self.gdep = gdep
		self.dropout = dropout
		self.alpha = alpha

	def forward(self, x, adj):
		adj = adj + torch.eye(adj.size(0)).to(x.device)
		d = adj.sum(1)
		h = x
		out = [h]
		a = adj / d.view(-1, 1)
		for i in range(self.gdep):
			h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
			out.append(h)
		ho = torch.cat(out, dim=1)
		ho = self.mlp(ho)
		return ho


class DyMixprop(nn.Module):
	def __init__(self, c_in, c_out, gdep, dropout, alpha):
		super(DyMixprop, self).__init__()
		self.nconv = DyNconv()
		self.mlp1 = Linear((gdep+1)*c_in, c_out)
		self.mlp2 = Linear((gdep+1)*c_in, c_out)

		self.gdep = gdep
		self.dropout = dropout
		self.alpha = alpha
		self.lin1 = Linear(c_in, c_in)
		self.lin2 = Linear(c_in, c_in)

	def forward(self, x):
		x1 = torch.tanh(self.lin1(x))
		x2 = torch.tanh(self.lin2(x))
		adj = self.nconv(x1.transpose(2, 1), x2)
		adj0 = torch.softmax(adj, dim=2)
		adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

		h = x
		out = [h]
		for i in range(self.gdep):
			h = self.alpha*x + (1-self.alpha)*self.nconv(h, adj0)
			out.append(h)
		ho = torch.cat(out, dim=1)
		ho1 = self.mlp1(ho)

		h = x
		out = [h]
		for i in range(self.gdep):
			h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
			out.append(h)
		ho = torch.cat(out, dim=1)
		ho2 = self.mlp2(ho)
		return ho1+ho2


class Dilated1D(nn.Module):
	def __init__(self, cin, cout, dilation_factor=2):
		super(Dilated1D, self).__init__()
		self.tconv = nn.ModuleList()
		self.kernel_set = [2, 3, 6, 7]
		self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

	def forward(self, inputs):
		x = self.tconv(inputs)
		return x


class DilatedInception(nn.Module):
	def __init__(self, cin, cout, dilation_factor=2):
		super(DilatedInception, self).__init__()
		self.tconv = nn.ModuleList()
		self.kernel_set = [2, 3, 6, 7]
		cout = int(cout/len(self.kernel_set))
		for kern in self.kernel_set:
			self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

	def forward(self, input):
		x = []
		for i in range(len(self.kernel_set)):
			x.append(self.tconv[i](input))
		for i in range(len(self.kernel_set)):
			x[i] = x[i][..., -x[-1].size(3):]
		x = torch.cat(x, dim=1)
		return x


class GraphConstructor(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphConstructor, self).__init__()
		self.nnodes = nnodes
		if static_feat is not None:
			xd = static_feat.shape[1]
			self.lin1 = nn.Linear(xd, dim)
			self.lin2 = nn.Linear(xd, dim)
		else:
			self.emb1 = nn.Embedding(nnodes, dim)
			self.emb2 = nn.Embedding(nnodes, dim)
			self.lin1 = nn.Linear(dim, dim)
			self.lin2 = nn.Linear(dim, dim)

		self.device = device
		self.k = k
		self.dim = dim
		self.alpha = alpha
		self.static_feat = static_feat

	def forward(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb2(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
		mask.fill_(float('0'))
		s1, t1 = adj.topk(self.k, 1)
		mask.scatter_(1, t1, s1.fill_(1))
		adj = adj*mask
		return adj

	def fulla(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb2(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		return adj


class GraphGlobal(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphGlobal, self).__init__()
		self.nnodes = nnodes
		self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

	def forward(self, idx):
		return F.relu(self.A)


class GraphUndirected(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphUndirected, self).__init__()
		self.nnodes = nnodes
		if static_feat is not None:
			xd = static_feat.shape[1]
			self.lin1 = nn.Linear(xd, dim)
		else:
			self.emb1 = nn.Embedding(nnodes, dim)
			self.lin1 = nn.Linear(dim, dim)

		self.device = device
		self.k = k
		self.dim = dim
		self.alpha = alpha
		self.static_feat = static_feat

	def forward(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb1(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
		mask.fill_(float('0'))
		s1, t1 = adj.topk(self.k, 1)
		mask.scatter_(1, t1, s1.fill_(1))
		adj = adj*mask
		return adj


class GraphDirected(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphDirected, self).__init__()
		self.nnodes = nnodes
		if static_feat is not None:
			xd = static_feat.shape[1]
			self.lin1 = nn.Linear(xd, dim)
			self.lin2 = nn.Linear(xd, dim)
		else:
			self.emb1 = nn.Embedding(nnodes, dim)
			self.emb2 = nn.Embedding(nnodes, dim)
			self.lin1 = nn.Linear(dim, dim)
			self.lin2 = nn.Linear(dim, dim)

		self.device = device
		self.k = k
		self.dim = dim
		self.alpha = alpha
		self.static_feat = static_feat

	def forward(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb2(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
		mask.fill_(float('0'))
		s1, t1 = adj.topk(self.k, 1)
		mask.scatter_(1, t1, s1.fill_(1))
		adj = adj*mask
		return adj


class LayerNorm(nn.Module):
	__constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

	def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
		super(LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		self.normalized_shape = tuple(normalized_shape)
		self.eps = eps
		self.elementwise_affine = elementwise_affine
		if self.elementwise_affine:
			self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
			self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		if self.elementwise_affine:
			init.ones_(self.weight)
			init.zeros_(self.bias)

	def forward(self, inputs, idx):
		if self.elementwise_affine:
			return F.layer_norm(inputs, tuple(inputs.shape[1:]),
								self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
		else:
			return F.layer_norm(inputs, tuple(inputs.shape[1:]),
								self.weight, self.bias, self.eps)

	def extra_repr(self):
		return '{normalized_shape}, eps={eps}, ' \
			'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class MTGNN(nn.Module):
	def __init__(self, args):
		super(MTGNN, self).__init__()
		self.adj_mx = args.adj_mx
		self.num_nodes = args.num_nodes
		self.feature_dim = args.input_dim

		self.input_window = args.input_window
		self.output_window = args.output_window
		self.output_dim = args.output_dim
		self.device = args.device

		self.gcn_true = args.gcn_true
		self.buildA_true = args.buildA_true
		self.gcn_depth = args.gcn_depth
		self.dropout = args.dropout
		self.subgraph_size = args.subgraph_size
		self.node_dim = args.node_dim
		self.dilation_exponential = args.dilation_exponential

		self.conv_channels = args.conv_channels
		self.residual_channels = args.residual_channels
		self.skip_channels = args.skip_channels
		self.end_channels = args.end_channels

		self.layers = args.layers
		self.propalpha = args.propalpha
		self.tanhalpha = args.tanhalpha
		self.layer_norm_affline = args.layer_norm_affline

		self.use_curriculum_learning = args.use_curriculum_learning

		self.task_level = args.task_level
		self.idx = torch.arange(self.num_nodes).to(self.device)

		if self.adj_mx is None:
			self.predefined_A = None
		else:
			self.predefined_A = torch.tensor(self.adj_mx) - torch.eye(self.num_nodes)
			self.predefined_A = self.predefined_A.to(self.device)
		self.static_feat = None

		# transformer attention neural network
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=4)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

		self.filter_convs = nn.ModuleList()
		self.gate_convs = nn.ModuleList()
		self.residual_convs = nn.ModuleList()
		self.skip_convs = nn.ModuleList()
		self.gconv1 = nn.ModuleList()
		self.gconv2 = nn.ModuleList()
		self.norm = nn.ModuleList()
		self.stu_mlp = nn.ModuleList()
		self.stu_mlp.append(nn.Sequential(nn.Linear(13,13),nn.Linear(13,13),nn.Linear(13,13)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(7,7),nn.Linear(7,7),nn.Linear(7,7)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(1,1),nn.Linear(1,1),nn.Linear(1,1)))
		self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
									out_channels=self.residual_channels,
									kernel_size=(1, 1))
		self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
								   self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

		kernel_size = 7
		if self.dilation_exponential > 1:
			self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
									   / (self.dilation_exponential - 1))
		else:
			self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

		for i in range(1):
			if self.dilation_exponential > 1:
				rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
								/ (self.dilation_exponential - 1))
			else:
				rf_size_i = i * self.layers * (kernel_size - 1) + 1
			new_dilation = 1
			for j in range(1, self.layers+1):
				if self.dilation_exponential > 1:
					rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
									/ (self.dilation_exponential - 1))
				else:
					rf_size_j = rf_size_i+j*(kernel_size-1)

				self.filter_convs.append(DilatedInception(self.residual_channels,
														  self.conv_channels, dilation_factor=new_dilation))
				self.gate_convs.append(DilatedInception(self.residual_channels,
														self.conv_channels, dilation_factor=new_dilation))
				self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
													 out_channels=self.residual_channels, kernel_size=(1, 1)))

				if self.input_window > self.receptive_field:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.input_window-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)
				else:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.receptive_field-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)

				if self.gcn_true:
					self.gconv1.append(MixProp(self.conv_channels, self.residual_channels,
											   self.gcn_depth, self.dropout, self.propalpha))
					self.gconv2.append(MixProp(self.conv_channels, self.residual_channels,
											   self.gcn_depth, self.dropout, self.propalpha))

				if self.input_window > self.receptive_field:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.input_window - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))
				else:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.receptive_field - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))

				new_dilation *= self.dilation_exponential

		self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
									out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
		self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
									out_channels=self.output_window, kernel_size=(1, 1), bias=True)
		if self.input_window > self.receptive_field:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window-self.receptive_field+1), bias=True)
		else:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)

		# self._logger.info('receptive_field: ' + str(self.receptive_field))
		

	
	def forward(self, source, idx=None):
		# inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
		sout = []
		tout = []
		inputs = source
		import copy

		inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window) #64, 1, 170, 12
		# odata = copy.deepcopy(inputs)
		assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'
		# inputs = inputs.view(-1, self.num_nodes, self.input_window)
		
		if self.input_window < self.receptive_field:
			inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))

		if self.gcn_true:
			if self.buildA_true:
				if idx is None:
					adp = self.gc(self.idx)
				else:
					adp = self.gc(idx)
			else:
				adp = self.predefined_A

		
		x = self.start_conv(inputs)
		skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
		for i in range(self.layers):
			residual = x
			filters = self.filter_convs[i](x)
			filters = torch.tanh(filters)
			gate = self.gate_convs[i](x)
			gate = torch.sigmoid(gate)
			x = filters * gate
			x = F.dropout(x, self.dropout, training=self.training)
			tout.append(x)
			s = x
			s = self.skip_convs[i](s)
			skip = s + skip
			if self.gcn_true:
				# print("gcn in x:", x.size())
				x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1, 0)) # in :64, 32, 170, 13, out: 64, 32, 170, 13 , 64, 32, 170, 7], 64, 32, 170, 1]
				# print("gcn out x:", x.size())
				# println()
			else:
				# x = self.residual_convs[i](x)
				x = self.stu_mlp[i](x)
				# print("mlp out x:", x.size())

			x = x + residual[:, :, :, -x.size(3):]
			if idx is None:
				x = self.norm[i](x, self.idx)
			else:
				x = self.norm[i](x, idx)
			sout.append(x)
		skip = self.skipE(x) + skip
		x = F.relu(skip)
		x = F.relu(self.end_conv_1(x))
		x = self.end_conv_2(x)
		# x = nn.Linear(self.num_nodes, self.input_window).cuda()(x.view(-1, self.input_window, self.num_nodes))
		# x = self.transformer_encoder(x)
		# x = nn.Linear(self.input_window, self.num_nodes).cuda()(x.view(-1, self.input_window, self.input_window)).view(-1, self.input_window, self.num_nodes, self.feature_dim)
		# print("x.size():", x.size())
		# println()
		ttout = nn.Linear(1, 32).cuda()(nn.Linear(self.residual_channels, self.input_window).cuda()(tout[-1].transpose(1,3)).transpose(1,3))
		ssout = nn.Linear(1, 32).cuda()(nn.Linear(self.residual_channels, self.input_window).cuda()(sout[-1].transpose(1,3)).transpose(1,3))
		# print(ttout.size(), ssout.size())
		# println()
		return x, ttout, ssout

class LGModel(nn.Module):
	def __init__(self, args):
		super(LGModel, self).__init__()
		self.device = args.device
		self.node_dim = args.node_dim
		self.feature_dim = args.input_dim
		self.output_dim = args.output_dim
		self.residual_channels = args.residual_channels
		self.output_window = args.output_window
		self.dilation_exponential = args.dilation_exponential
		self.conv_channels = args.conv_channels
	
		self.skip_channels = args.skip_channels
		self.end_channels = args.end_channels
		self.num_nodes = args.num_nodes
		self.dilation_exponential = args.dilation_exponential

		self.conv_channels = args.conv_channels
		
		self.propalpha = args.propalpha
		self.tanhalpha = args.tanhalpha
		self.layers = args.layers
		self.layer_norm_affline = args.layer_norm_affline

		self.use_curriculum_learning = args.use_curriculum_learning

		self.task_level = args.task_level
		self.idx = torch.arange(self.num_nodes).to(self.device)


		kernel_size = 7
		if self.dilation_exponential > 1:
			self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
									   / (self.dilation_exponential - 1))
		else:
			self.receptive_field = self.layers * (kernel_size-1) + self.output_dim
		for i in range(1):
			if self.dilation_exponential > 1:
				rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
								/ (self.dilation_exponential - 1))
			else:
				rf_size_i = i * self.layers * (kernel_size - 1) + 1
			new_dilation = 1
			for j in range(1, self.layers+1):
				if self.dilation_exponential > 1:
					rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
									/ (self.dilation_exponential - 1))
				else:
					rf_size_j = rf_size_i+j*(kernel_size-1)
		self.norm = nn.ModuleList()
		self.norm.append(LayerNorm((self.output_window, self.num_nodes,
												self.receptive_field - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))

		self.gcnLayers = nn.Sequential(*[LightGCNLayer(args) for i in range(args.gcn_depth)])
		self.start_conv = nn.Conv2d(in_channels=self.feature_dim, out_channels=self.output_window, kernel_size=(1, 1))
		self.mlp = nn.Linear(12, self.output_dim).to(self.device)
		# self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
		# 							out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
		# self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
		# 							out_channels=self.output_window, kernel_size=(1, 1), bias=True)
	def forward(self, embeds, idx = None):

		embeds = embeds.transpose(1, 3)
		x = self.start_conv(embeds)
		embeds = x.transpose(1,3).transpose(2,3)

		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst).transpose(2,3) # need dimension : torch.Size([128, 12, 307, 1])
		
		x = self.mlp(embeds)
		if idx is None:
				x = self.norm[0](x, self.idx)
		else:
				x = self.norm[0](x, idx)
		out = F.relu(x)
		# x = F.relu(self.end_conv_1(x))
		# x = self.end_conv_2(x)
		# print("x:", x.size())
		# println()
		
		return out, x, x


class STMLP(nn.Module):
	def __init__(self, args):
		super(STMLP, self).__init__()

		self.adj_mx = args.adj_mx
		self.num_nodes = args.num_nodes
		self.feature_dim = args.input_dim

		self.input_window = args.input_window
		self.output_window = args.output_window
		self.output_dim = args.output_dim
		self.device = args.device

		self.gcn_true = args.gcn_true
		self.buildA_true = args.buildA_true
		self.gcn_depth = args.gcn_depth
		self.dropout = args.dropout
		self.subgraph_size = args.subgraph_size
		self.node_dim = args.node_dim
		self.dilation_exponential = args.dilation_exponential

		self.conv_channels = args.conv_channels
		self.residual_channels = args.residual_channels
		self.skip_channels = args.skip_channels
		self.end_channels = args.end_channels

		self.layers = args.layers
		self.propalpha = args.propalpha
		self.tanhalpha = args.tanhalpha
		self.layer_norm_affline = args.layer_norm_affline

		self.use_curriculum_learning = args.use_curriculum_learning

		self.task_level = args.task_level
		self.idx = torch.arange(self.num_nodes).to(self.device)

		if self.adj_mx is None:
			self.predefined_A = None
		else:
			self.predefined_A = torch.tensor(self.adj_mx) - torch.eye(self.num_nodes)
			self.predefined_A = self.predefined_A.to(self.device)
		self.static_feat = None

		# transformer attention neural network
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=4)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

		self.filter_convs = nn.ModuleList()
		self.gate_convs = nn.ModuleList()
		self.residual_convs = nn.ModuleList()
		self.skip_convs = nn.ModuleList()
		self.gconv1 = nn.ModuleList()
		self.gconv2 = nn.ModuleList()
		self.norm = nn.ModuleList()
		self.stu_mlp = nn.ModuleList()
		# self.stu_mlp.append(nn.Linear(13, 13))
		# self.stu_mlp.append(nn.Linear(7, 7))
		# self.stu_mlp.append(nn.Linear(1, 1))
		self.stu_mlp.append(nn.Sequential(nn.Linear(13,13),nn.Linear(13,13),nn.Linear(13,13)))
		# self.stu_mlp.append(nn.Sequential(nn.Linear(6,13),nn.Linear(13,13),nn.Linear(13,6)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(7,7),nn.Linear(7,7),nn.Linear(7,7)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(1,1),nn.Linear(1,1),nn.Linear(1,1)))
		self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
									out_channels=self.residual_channels,
									kernel_size=(1, 1))
		# self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
		# 						   self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

		kernel_size = 7
		if self.dilation_exponential > 1:
			self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
									   / (self.dilation_exponential - 1))
		else:
			self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

		for i in range(1):
			if self.dilation_exponential > 1:
				rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
								/ (self.dilation_exponential - 1))
			else:
				rf_size_i = i * self.layers * (kernel_size - 1) + 1
			new_dilation = 1
			for j in range(1, self.layers+1):
				if self.dilation_exponential > 1:
					rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
									/ (self.dilation_exponential - 1))
				else:
					rf_size_j = rf_size_i+j*(kernel_size-1)

				self.filter_convs.append(DilatedInception(self.residual_channels,
														  self.conv_channels, dilation_factor=new_dilation))
				self.gate_convs.append(DilatedInception(self.residual_channels,
														self.conv_channels, dilation_factor=new_dilation))
				self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
													 out_channels=self.residual_channels, kernel_size=(1, 1)))

				if self.input_window > self.receptive_field:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.input_window-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)
				else:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.receptive_field-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)

				# if self.gcn_true:
				# 	self.gconv1.append(MixProp(self.conv_channels, self.residual_channels,
				# 							   self.gcn_depth, self.dropout, self.propalpha))
				# 	self.gconv2.append(MixProp(self.conv_channels, self.residual_channels,
				# 							   self.gcn_depth, self.dropout, self.propalpha))

				if self.input_window > self.receptive_field:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.input_window - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))
				else:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.receptive_field - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))

				new_dilation *= self.dilation_exponential

		self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
									out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
		self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
									out_channels=self.output_window, kernel_size=(1, 1), bias=True)
		if self.input_window > self.receptive_field:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window-self.receptive_field+1), bias=True)
		else:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)
	def forward(self, source, idx=None):
		# inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
		# print("source:", source.size())
		# println()


		sout = []
		tout = []
		inputs = source
		inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window) #64, 1, 170, 12
		
		assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'
		
		if self.input_window < self.receptive_field:
			inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))
		x = self.start_conv(inputs)
		skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
		
		for i in range(self.layers):
			residual = x
			filters = self.filter_convs[i](x)
			filters = torch.tanh(filters)
			gate = self.gate_convs[i](x)
			gate = torch.sigmoid(gate)
			x = filters * gate
			x = F.dropout(x, self.dropout, training=self.training)
			tout.append(x)
			s = x
			s = self.skip_convs[i](s)
			skip = s + skip
			x = self.stu_mlp[i](x)
			x = x + residual[:, :, :, -x.size(3):] 

			if idx is None:
				x = self.norm[i](x, self.idx)
			else:
				x = self.norm[i](x, idx)
		# 	sout.append(x)
		skip = self.skipE(x) + skip
		x = F.relu(skip)
		x = F.relu(self.end_conv_1(x))
		x = self.end_conv_2(x)
		x_ = nn.Linear(1, 32).cuda()(x)  #final out:  64, 12, 170, 1
		return x, x_, x


# class MulP(nn.Module):
# 	"""Multi-Layer Perceptron with residual links."""
# 	def __init__(self, args) -> None:
# 		super(MulP, self).__init__()
# 		self.fc1 = nn.Linear(args.input_window-6,  12)
# 		self.fc2 = nn.Linear(12, args.input_window)
# 		self.act = nn.ReLU()
# 		self.args = args
# 		self.skip0 = nn.Conv2d(in_channels=args.batch_size,
# 								   out_channels=args.batch_size, kernel_size=(1, 7), bias=True)
# 		self.skip1 = nn.Conv2d(in_channels=args.batch_size,
# 								   out_channels=args.batch_size, kernel_size=(1, 1), bias=True)
# 		self.layer_norm1 = nn.LayerNorm((args.batch_size, self.args.num_nodes, 6)).to(self.args.device)
# 		self.layer_norm = nn.LayerNorm((args.batch_size, self.args.num_nodes, 12)).to(self.args.device)
# 		# self.layer_norm2 = nn.LayerNorm((args.batch_size, self.args.num_nodes, 12)).to(self.args.device)
# 	def forward(self, history_data) -> torch.Tensor:
# 		"""Feedforward function of AGCRN.
# 		Args:
# 			history_data (torch.Tensor): inputs with shape [B, L, N, C].
# 		Returns:
# 			torch.Tensor: outputs with shape [B, L, N, C]
# 		"""
# 		import copy
# 		odata = copy.deepcopy(history_data)
# 		history_data = history_data[..., 0].transpose(1, 2)     # B, N, L
# 		s1 = self.skip0(history_data)
# 		s2 = self.skip1(s1)
# 		# print("**:", s1.size(), s2.size())
# 		# println()
# 		out1 = self.layer_norm1(s2)
# 		embedding = self.act(self.fc1(out1))
# 		prediction = self.layer_norm(embedding).transpose(1, 2)
		
# 		# prediction = self.fc2(embedding).transpose(1, 2)     # B, L, N
# 		out = prediction.unsqueeze(-1)+odata
# 		# print("out:", out.size())
# 		# println()
# 		return out, out, out         # B, L, N, C


# class MulP(nn.Module):
# 	"""Multi-Layer Perceptron with residual links."""
# 	def __init__(self, args) -> None:
# 		super(MulP, self).__init__()
# 		self.fc1 = nn.Linear(args.input_window-6,  12)
# 		self.fc3 = nn.Linear(169, 170)
# 		self.fc2 = nn.Linear(12, args.input_window)
# 		self.act = nn.ReLU()
# 		self.args = args
# 		self.skip0 = nn.Conv2d(in_channels=args.batch_size,
# 								   out_channels=args.batch_size, kernel_size=(2, 7), bias=True)
# 		self.layer_norm1 = nn.LayerNorm((args.batch_size, self.args.num_nodes, 6)).to(self.args.device)
# 		self.layer_norm = nn.LayerNorm((args.batch_size, self.args.num_nodes, 12)).to(self.args.device)
# 		# self.layer_norm2 = nn.LayerNorm((args.batch_size, self.args.num_nodes, 12)).to(self.args.device)
# 	def forward(self, history_data) -> torch.Tensor:
# 		"""Feedforward function of AGCRN.
# 		Args:
# 			history_data (torch.Tensor): inputs with shape [B, L, N, C].
# 		Returns:
# 			torch.Tensor: outputs with shape [B, L, N, C]
# 		"""
# 		import copy
# 		odata = copy.deepcopy(history_data)
# 		history_data = history_data[..., 0].transpose(1, 2)     # B, N, L
# 		# print("&:", history_data.size())
# 		# out1 = self.act(self.fc3(self.layer_norm1(self.skip0(history_data)).transpose(1,2))).transpose(1,2)
# 		out1 = self.layer_norm1(self.act(self.fc3(self.skip0(history_data).transpose(1,2))).transpose(1,2))
		
# 		# print("&:", out1.size())
# 		# println()
# 		embedding = self.act(self.fc1(out1)).transpose(1, 2)
# 		# embedding = self.layer_norm(embedding).transpose(1, 2)
# 		prediction = embedding
# 		# prediction = self.fc2(embedding).transpose(1, 2)     # B, L, N
# 		out = prediction.unsqueeze(-1)+odata
# 		return out, out, out   



class MulP(nn.Module):
	"""Multi-Layer Perceptron with residual links."""
	def __init__(self, args) -> None:
		super(MulP, self).__init__()
		self.dim = 32
		self.kernel = 7
		self.time = 12
		self.fc1 = nn.Linear(self.time-self.kernel+1,  self.time)
		self.fc3 = nn.Linear(1, self.dim)
		self.fc2 = nn.Linear(self.dim, 1)
		# self.fc4 = nn.Linear(12, 6)
		self.act = nn.ReLU()
		self.args = args
		self.skip0 = nn.Conv2d(in_channels=self.dim,
								   out_channels=self.dim, kernel_size=(1, self.kernel), bias=True)
		self.skip1 = nn.Conv2d(in_channels=12,
								   out_channels=12, kernel_size=(1, self.dim), bias=True)
		self.layer_norm1 = nn.LayerNorm((self.dim, self.args.num_nodes, self.time-self.kernel+1)).to(self.args.device)
		# self.layer_norm = nn.LayerNorm((1, self.args.num_nodes, 12)).to(self.args.device)
		# self.layer_norm2 = nn.LayerNorm((args.batch_size, self.args.num_nodes, 12)).to(self.args.device)
	def forward(self, history_data) -> torch.Tensor:
		"""Feedforward function of AGCRN.
		Args:
			history_data (torch.Tensor): inputs with shape [B, L, N, C].
		Returns:
			torch.Tensor: outputs with shape [B, L, N, C]
		"""
		import copy
		odata = copy.deepcopy(history_data)
		   
		history_data = self.fc3(history_data).transpose(1, 3) # B, N, L (B,N, 12, d)--> B N , 12, 26
		print("*:", history_data.size())
		# history_data = history_data.transpose(1, 3)
		out1 = self.layer_norm1(self.skip0(history_data))
		# out1 = self.layer_norm1(self.fc4(history_data))
		print("*:", out1.size())
		embedding = self.act(self.fc1(out1))
		print("*:", embedding.size())

		# prediction = self.fc2(embedding.transpose(1, 3))     # B, L, N
		prediction = self.skip1(embedding.transpose(1, 3))
		print(prediction.size())
		println()
		out = prediction + odata
		# print("out:", out.size())
		# println()
		return out, out, out         # B, L, N, C









