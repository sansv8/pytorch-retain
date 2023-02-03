""" Matplotlib backend configuration """
from fileinput import filename
import matplotlib
matplotlib.use('PS')  # generate postscript output by default

""" Imports """
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import os

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

import sys
import argparse
import pickle
import time
from tqdm import tnrange, trange, tqdm_notebook

""" Arguments """
parser = argparse.ArgumentParser()

# Path to the dataset
parser.add_argument('data_path', metavar='DATA_PATH', help="Path to the dataset")

# Learning rate
# Iniial: 0.001
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR',
		    help='learning rate (default: 1e-3)')
# Weight Decay/Number of weights to drop out
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 0)')

# Number of epochs
# Initial: 10
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')

# Batch-Size
# Initial: 128
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
					help='mini-batch size for train (default: 64)')
# Evalutation Batch size. Use for evaluation
# Initial: 256
parser.add_argument('--eval-batch-size', type=int, default=256, help='mini-batch size for eval (default: 1000)')

# Tells the program to not use cuda
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')

# Do not plot the dataset
parser.add_argument('--no-plot', dest='plot', action='store_false', help='no plot')

# Allows the use of threads
# Initial: number of cpus. 2 for my computer so 4 threads
parser.add_argument('--threads', type=int, default=-1,
					help='number of threads for data loader to use (default: -1 = (multiprocessing.cpu_count()-1 or 1))')

# Random Seed
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')

# Saving both the model and checkpoints
parser.add_argument('--save', default='./', type=str, metavar='SAVE_PATH',
					help='path to save checkpoints (default: none)')


					# Embeded Size
parser.add_argument('--embSize', type=int, default=512, help='Embedded Size for Embedding Layer')

# Embeded Dropout
parser.add_argument('--embDropout', type=float, default=0.5, help='Embedded Dropout for Embedding Layer')

# Context Dropout
parser.add_argument('--contextDropout', type=float, default=0.5, help='Dropout for Context Layer')

# Alpha Heads
parser.add_argument('--heads', type=int, default=8, help='Number of heads to divide up embedded input')

# Alpha Attention Dropout
parser.add_argument('--attentionDropout', type=float, default=0.2, help="Dropout for attention layer")

# Alpha Feed Forward Dropout
parser.add_argument('--ffDropout', type=float, default=0.2, help="Dropout for Feed Forward layer")

# Alpha Norm Eps
parser.add_argument('--normEps', type=float, default=1e-6, help="Epslion value for norm")

# Alpha Attention Epslion
parser.add_argument('--attEps', type=float, default=-1e-9, help="Epslion value for attention layer")

# Alpah Feed Forward Size
parser.add_argument('--ffSize', type=int, default=2048, help="Size of Feed Forward layer")

# Alpha Transformer Dropout
parser.add_argument('--transDropout', type=float, default=0.2, help="Dropout for transformer layer")

# Alpha Number of layers
parser.add_argument('--numLayers', type=int, default=1, help="Number of layers for transformer")

# Beta Heads
parser.add_argument('--headsBeta', type=int, default=8, help='Number of heads to divide up embedded input')

# Beta Attention Dropout
parser.add_argument('--attentionDropoutBeta', type=float, default=0.2, help="Dropout for attention layer")

# Beta Feed Forward Dropout
parser.add_argument('--ffDropoutBeta', type=float, default=0.2, help="Dropout for Feed Forward layer")

# Beta Norm Eps
parser.add_argument('--normEpsBeta', type=float, default=1e-6, help="Epslion value for norm")

# Beta Attention Epslion
parser.add_argument('--attEpsBeta', type=float, default=-1e-9, help="Epslion value for attention layer")

# Beta Feed Forward Size
parser.add_argument('--ffSizeBeta', type=int, default=2048, help="Size of Feed Forward layer")

# Beta Transformer Dropout
parser.add_argument('--transDropoutBeta', type=float, default=0.2, help="Dropout for transformer layer")

# Beta Number of layers
parser.add_argument('--numLayersBeta', type=int, default=1, help="Number of layers for transformer")

# Type of best we are fidning
parser.add_argument('--metric', type=str, default="valid_loss", help="Metric to find best epoch with")

# Name of File
parser.add_argument('--name', type=str, default="1", help="Name of file")

# GPUs in enviornment
# Used to determine what gpus will be used
parser.add_argument('--gpu', type=str, default="0,1,2", help="GPUs used for program")

# Set both cuda and plot
parser.set_defaults(cuda=True, plot=True)

""" Helper Functions """

# Stores the current value
class AverageMeter(object):
	"""Computes and stores the average and current value"""
	
	# Intialize the value by resetting
	# Contains 4 memeber variables
	# value
	# average
	# sum
	# and count
	def __init__(self):
		self.reset()

	# Does a reset by setting all member variables to 0
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	# Update by passing in a new value and number of values
	def update(self, val, n=1):
		# Set self.val to value
		self.val = val
		
		# Add value * n to current sum
		self.sum += val * n
		
		# Add n to count
		self.count += n
		
		# Caculate the current average by dividing sum by count
		self.avg = self.sum / self.count


""" Custom Dataset """

# Creates a custom dataset
# It takes in sequences, labels, num of features
class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features, reverse=True):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
			reverse (bool): If true, reverse the order of sequence (for RETAIN)
		"""
		
		# If the number of sequences and labels are not the same length
		# Then do not make the dataset
		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		# Set sequences
		self.seqs = []
		# self.labels = []

		# For each sequence and label from seqs/labels
		for seq, label in zip(seqs, labels):

			# If reverse is true
			# Then reverse the sequence
			if reverse:
				sequence = list(reversed(seq))
			else:
				sequence = seq

			# Create three lists, row, col and val
			row = []
			col = []
			val = []
			
			# For each visit in sequence
			for i, visit in enumerate(sequence):
				# For each code in visit
				for code in visit:
					# If code is less than number of features
					# Then add it
					if code < num_features:
						# Add the index of the visit to row
						row.append(i)
						
						# Add the code to col
						col.append(code)
						
						# Add 1.0 to val
						val.append(1.0)
						
			# Add the sequence to seqs by converting sequence to a coo matrix
			self.seqs.append(coo_matrix((np.array(val, dtype=np.float32), (np.array(row), np.array(col))),
										shape=(len(sequence), num_features)))
		self.labels = labels

	# Return the length of labels
	def __len__(self):
		return len(self.labels)

	# Get the sequence and label at index
	def __getitem__(self, index):
		return self.seqs[index], self.labels[index]


""" Custom collate_fn for DataLoader"""
# @profile
def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor

	:returns
		seqs
		labels
		lengths
	"""
	# Get both bactch seq and label
	batch_seq, batch_label = zip(*batch)

	# Get number of features from batch_seq
	num_features = batch_seq[0].shape[1]

	# Get length of number of visits for each seqence in banch
	seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))

	# Get the maximum length
	max_length = max(seq_lengths)

	# Sort the lengths of seq_lengths and store their original indicies
	sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))

	# Create two lists, sorted_padded_seqs and sorted_labels
	sorted_padded_seqs = []
	sorted_labels = []

	# Go through each index in sorted_indices
	for i in sorted_indices:
		# Get the length of the batch
		length = batch_seq[i].shape[0]

		# If length is less than max length
		if length < max_length:
			# Pad the sequence with zeroes
			padded = np.concatenate(
				(batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
		else:
			# Otherwise, set padded directly to batch_seq[i]
			padded = batch_seq[i].toarray()

		# Add padded to sorted_padded and batch_label[i] to sorted_labels
		sorted_padded_seqs.append(padded)
		sorted_labels.append(batch_label[i])

	# Get seq tensor from sorted_padded_seqs
	seq_tensor = np.stack(sorted_padded_seqs, axis=0)

	# Get label tensor from sorted labels
	label_tensor = torch.LongTensor(sorted_labels)

	# Return the seq_tensor, label_tensor< and sorted lengths
	return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths)

# The Positional Encoder
class PositionalEncoder(nn.Module):
	# Intializer
	# Inputs: dim_emb - Embedded Size
	# max_seq_len - Max length of sequence
	def __init__(self, dim_emb, max_seq_len):
		# Intialize as nn.Module
		super().__init__()

		# Add dim_emb 
		self.dim_emb = dim_emb

		# Create constant Positional Embedding
		# (pe) matrix with values dependent on pos and i
		pe = torch.zeros(max_seq_len, dim_emb)

		# Go through each possible position of a sequence
		for pos in range(max_seq_len):
			# For each position in emebedded system
			for i in range(0, dim_emb):
				# If i is even
				if i % 2 == 0:
					# Set the value at (pos, i)
					pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/dim_emb)))
				# If i is odd
				else:
					# Set the value at (pos, i)
					pe[pos, i] = math.cos(pos / (10000 ** ((2*i)/dim_emb)))
		
		# Unsequeeze pe 
		pe = pe.unsqueeze(0)

		# Register pe as a buffer
		# Buffers will not be updated during
		# gradient descent
		self.register_buffer('pe', pe)

	# Forward method
	# Inputs: x - Input
	def forward(self, x):
		# Make embeddings relatively larger
		x = x * math.sqrt(self.dim_emb)

		# Add pe to embedding
		seq_len = x.size(1)
		x = x + Variable(self.pe[:, :seq_len], requires_grad=False)

		# Return x 
		return x

# The Self Attention Layer
class MultiHeadAttention(nn.Module):
	# Constructor
	# Inputs: heads - Number of heads to divided up input
	# emb_size: Embedded Size of Iinput
	# dropout: Amount to dropout
	# eps: Value of episolon for attention function
	def __init__(self, heads, emb_size, dropout, eps=-1e9):
		# Initialize nn.Moduele
		super().__init__()

		# Set emb_size, heads, and k_size
		self.emb_size = emb_size
		self.heads = heads

		# k_size is defined as emb_size / heads
		self.k_size = emb_size // heads

		# 3 linears layers for Query, Key, and Value
		self.q_linear = nn.Linear(emb_size, emb_size)
		self.v_linear = nn.Linear(emb_size, emb_size)
		self.k_linear = nn.Linear(emb_size, emb_size)

		# Inialize all three layers
		# Let weights be in xavier normal
		# Set bias to be 0
		init.xavier_normal_(self.q_linear.weight)
		self.q_linear.bias.data.zero_()
		init.xavier_normal_(self.v_linear.weight)
		self.v_linear.bias.data.zero_()
		init.xavier_normal_(self.k_linear.weight)
		self.k_linear.bias.data.zero_()

		# Dropout layer
		self.dropout = nn.Dropout(dropout)

		# Output Linear Layer
		self.out = nn.Linear(emb_size, emb_size)

		# Epslion use for mask
		self.eps = eps

		# Intialize weights and biases of output layer
		init.xavier_normal_(self.out.weight)
		self.out.bias.data.zero_()
	
	# Attention Function
	# Inputs: query, key, value, mask, k_size, mask, dropout
	def attention(self, qA, kA, vA, k_size, mask, dropout, eps):
		# Caculare scores
		# scores = softmax(q^k / sqrt(k_size)) * value
		# Step 1: q^k / sqrt(k_size)
		scores = torch.matmul(qA, kA.transpose(-2, -1)) / math.sqrt(k_size)

		# Step 2: Apply mask
		mask = torch.unsqueeze(mask, 1)
		scores = scores.masked_fill(mask == 0, eps)

		# Step 3: Softamx
		scores = F.softmax(scores, dim=-1)

		# Step 4: Dropout
		scores = dropout(scores)

		# Step 5: Mutilpy scores wilth value
		output = torch.matmul(scores, vA)

		# Return output
		return output
	
	# Forward fucntion
	# Inputs, q, k, v, mask
	def forward(self, q, k, v, mask):
		# Get batach size
		bs = q.size(0)

		# Peform linear operations on q, k, and v and split them into heads
		k = self.k_linear(k).view(bs, -1, self.heads, self.k_size)
		q = self.q_linear(q).view(bs, -1, self.heads, self.k_size)
		v = self.v_linear(v).view(bs, -1, self.heads, self.k_size)

		# Transpose to get shap bs x heads x max_len x k_size
		k = torch.transpose(k, 1, 2)
		q = torch.transpose(q, 1, 2)
		v = torch.transpose(v, 1, 2)

		# Caculate scores
		scores = self.attention(qA=q, kA=k, vA=v, k_size=self.k_size, mask=mask, dropout=self.dropout, eps=self.eps)

		# Concated the scores
		concat = scores.transpose(1,2).contiguous().view(bs, -1, self.emb_size)

		# Pass concat through layer to fet final output
		output = self.out(concat)

		# Return output
		return output

# Feed Forward Layer
class FeedForward(nn.Module):
	# Inputs: emb_size: Size of embedded input
	# d_ff: Size of each linear layer
	# dropout: Dropout amount for each layer
	def __init__(self, emb_size, d_ff, dropout):
		# Construct as nn.Module
		super().__init__()

		# Create first linear layer with size (emb_size, d_ff)
		self.linear_1 = nn.Linear(emb_size, d_ff)

		# Intialize weights as xavier_normal_ and biases as zero
		init.xavier_normal_(self.linear_1.weight)
		self.linear_1.bias.data.zero_()

		# Create dropout out layer with dropout
		self.dropout = nn.Dropout(dropout)

		# Create second linear layer with size (d_ff, emb_size)
		self.linear_2 = nn.Linear(d_ff, emb_size)

		# Intialize weights as xavier_normal_ and biases as zero
		init.xavier_normal_(self.linear_2.weight)
		self.linear_2.bias.data.zero_()
	# Forward Function
	def forward(self, x):
		# Pass x through firest linear layer
		# Then pass the output to relu function
		# Finally, pass output through droput out layer
		x = self.dropout(F.relu(self.linear_1(x)))

		# Pass caculated x into second linear layer
		x = self.linear_2(x)

		# Return x
		return x

# Normalizer
class Norm(nn.Module):
	# Intializer
	# Input: emb_size: The size of the embedded input
	# eps: Used for caculating the norm
	def __init__(self, emb_size, eps = 1e-6):
		# Initialize as Neural Layer
		super().__init__()

		# Set size to emb_size
		self.size = emb_size

		# Create two learnable parameters from normalisation
		self.alpha = nn.Parameter(torch.ones(self.size))
		self.bias = nn.Parameter(torch.ones(self.size))

		# Store eps
		self.eps = eps
	
	# Forward model
	def forward(self, x):
		# Caculate the norm
		norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

		# Return norm
		return norm

# Create transformer Layer
class TransformerLayer(nn.Module):
	# Intializer
	# Input: emb_size: Size of embedded input
	# d_ff: Size of Feed Forward Layer
	# head: Number of heads in attention layer
	# transDropout: Dropout for Transformer
	# attentionDropout: Dropout for attention layer
	# ffDropout: Dropout for Feed Forward Layer
	# normEps: Epsilon for normalizer layers
	# attEps: Epsilon for attention layer
	def __init__(self, emb_size, d_ff, heads, transDropout, attentionDropout, ffDropout, normEps, attEps):
		# Intialize as Neural Network
		super().__init__()

		# Initialize two norms, multheadattention, feedforward, and two drouput layers
		# Norm layers
		self.norm_1 = Norm(emb_size, normEps)
		self.norm_2 = Norm(emb_size, normEps)

		# Attention Layer
		self.attn = MultiHeadAttention(heads, emb_size, attentionDropout, attEps)

		# Feed Forward Layer
		self.ff = FeedForward(emb_size, d_ff, ffDropout)

		# Dropout layer
		self.dropout_1 = nn.Dropout(transDropout)
		self.dropout_2 = nn.Dropout(transDropout)

	# Forward
	def forward(self, x, mask):
		# Firstly, input x into first Normalizer Layer
		x2 = self.norm_1.forward(x)

		# Next, pass in both x2 and mask into attention layer
		# The output then goes through the first dropout layer
		# Then add original x to output
		x = x + self.dropout_1(self.attn.forward(q=x2, k=x2, v=x2, mask=mask))

		# Pass x through normalizer layer
		x2 = self.norm_2(x)

		# Pass x2 through feed forward layer, then dropout layer, and add it ot x
		x = x + self.dropout_2(self.ff.forward(x2))

		# Return x
		return x

# Define cloning function and clone Transformer Layer
def cloning(modules, n):
	return nn.ModuleList([copy.deepcopy(modules) for i in range(n)])
	
# Create Transformer
class Transformer(nn.Module):
	# Initializer
	# Input: emb_size: Size of embedded input
	# d_ff: Size of Feed Forward Layer
	# head: Number of heads in attention layer
	# transDropout: Dropout for Transformer
	# attentionDropout: Dropout for attention layer
	# ffDropout: Dropout for Feed Forward Layer
	# normEps: Epsilon for normalizer layers
	# attEps: Epsilon for attention layer
	# numLayers: The number of transformer layers
	def __init__(self, emb_size, d_ff, heads, transDropout, attentionDropout, ffDropout, normEps, attEps, numLayers):
		# Intiailize as neaur network
		super().__init__()

		# Initialize layers
		self.layers = cloning(TransformerLayer(emb_size, d_ff, heads, transDropout, attentionDropout, ffDropout, normEps, attEps), numLayers)

		# Norm Layer
		self.norm = Norm(emb_size, normEps)

	# Forward
	def forward(self, x, mask):
		# For layer in layers
		for layer in self.layers:
			# Input x and mask through layer
			x = layer.forward(x, mask)
		
		# Return x
		return self.norm.forward(x)


""" RETAIN model class """

# The Retain Model
class RETAIN(nn.Module):
	# Initialize the model with following parameters
	# Input Dimensions
	# Embedding Dimenstions dim_emb (Default 128) 
	# Transformer Dropout transDropout (Default 0.2)
	# Embedded Dropout dropout_emb (Default 0.2)
	# Context Layer Dropout dropout_context (Default 0.8)
	# Maximum Length max_len
	# Alpha ransformer Feed Forward Layer Size d_ff (Default 2048)
	# Output Layer Size dim_output (Default 1)
	# l2 (Default 0.0001)
	# Alpha Transfomer Attention Layer Head heads (Default 8)
	# Alpha Transformer Attention Layer Dropout (Default 0.2)
	# Alpha Transfomer Feed Forward Layer Dropout (Default 0.2)
	# Alpha Transformer Normal Layer Epsilon (Default: 1e-6)
	# Alpha Transformer Attention Layer Epsilon (Default: -1e-9)
	# Alpha Transfromer Number of Layers numLayers (Default:1)
	# Beta Transformer Feed Forward Layer Size beta_d_ff (Default 2048)
	# Beta Transformer Attention Layer Heads beta_heads (Default 8)
	# Beta Transformer Dropout beta_transDropout (Default 0.2)
	# Beta Transformer Attention Layer Dropout beta_attentionDropout (Default 0.2)
	# Beta Transformer Feed Forward Layer Dropout (Default 0.2)
	# Beta Transformer Normalizer Layer Epsilon (Default 1e-6)
	# Beta Transformer Attention Layer Epsilon (Default 1e-9)
	# Beta Transformer Number of Layers (Default 1)
	# batch_first (Default True)
	def __init__(self, dim_input, dim_emb=128, transDropout=0.2, dropout_emb=0.2,
				 dropout_context=0.8, dim_output=1, l2=0.0001, batch_first=True, max_len=0, d_ff=2048, heads=8, 
				 attentionDropout=0.2, ffDropout=0.2, normEps=1e-6, attEps=-1e-9, numLayers=1,
				 beta_d_ff=2048, beta_heads=8, beta_transDropout=0.2, beta_attentionDropout=0.2, beta_ffDropout=0.2, 
				 beta_normEps=1e-6, beta_attEps=1e-9, beta_numLayers=1):
		# Construct RETAIN and nn.Module
		super(RETAIN, self).__init__()

		# Set batch_first
		self.batch_first = batch_first
		
		# Embedding Layer 
		# Input Dropout Layer
		# Linear Layer v = W(emb) * X
		# Embedded Dropout Layer
		self.embedding = nn.Sequential(
			nn.Linear(dim_input, dim_emb, bias=False),
			nn.Dropout(p=dropout_emb)
		)

		# Positional Embedding Layer
		# Positional Encoder
		# Input batch_size x max_len x emb_size
		self.pos_embedding = PositionalEncoder(dim_emb, max_len)

		# Initailze weight using xavier normal
		init.xavier_normal_(self.embedding[0].weight)

		# Set RNN Alpha as GRU
		# input_size = dim_emb
		# hidden_size = dim_alpha
		# num_layers = 1
		# g(i) .... g(1) = RRN_alpha(v(i)...v(1))
		#self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)
		self.rnn_alpha = Transformer(dim_emb, d_ff, heads, transDropout, attentionDropout, ffDropout, normEps ,attEps, numLayers)

		# e(j) = w(alpha) * g(j) + b(alpha) for j = 1 ... i
		# Input_Dim = dim_alpha
		# Output_Dim = 1
		self.alpha_fc = nn.Linear(in_features=dim_emb, out_features=1)
		
		# Initialize xavier nomral weights
		init.xavier_normal_(self.alpha_fc.weight)
		
		# Set bias to be 0
		self.alpha_fc.bias.data.zero_()

		# Set RNN Beta as GRU
		# input_size = dim_emb
		# hidden_size = dim_beta
		# num_layers = 1
		# h(i) ... h(1) = RNN_betta(v(i) ... v(1))
		# self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)
		self.rnn_beta = Transformer(dim_emb, beta_d_ff, beta_heads, beta_transDropout, beta_attentionDropout, beta_ffDropout, beta_normEps , beta_attEps, beta_numLayers)

		# b(j) = W(beta) * h(j) + b(j) for j = 1 .. i
		# input_dim = dim_beta
		# output_dim = dim_emb
		self.beta_fc = nn.Linear(in_features=dim_emb, out_features=dim_emb)
		
		# Initialize weights as xavier_normal for tanh
		init.xavier_normal_(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
		
		# Set bias to be 0
		self.beta_fc.bias.data.zero_()

		# Output Layer
		# Dropout layer
		# Linear layer y(i) = W * c(i) + b
		self.output = nn.Sequential(
			nn.Dropout(p=dropout_context),
			nn.Linear(in_features=dim_emb, out_features=dim_output)
		)
		
		# Initialize weights and biases for output layer
		init.xavier_normal_(self.output[1].weight)
		self.output[1].bias.data.zero_()

	# Forward function by getting input and lengths
	def forward(self, x, lengths):
		# If batch_first, get batch_size then max_len
		if self.batch_first:
			batch_size, max_len = x.size()[:2]
		# Otherwise, get max_len and batch_size
		else:
			max_len, batch_size = x.size()[:2]

		# Create mask for transformer
		# Creates mask with 0s wherever there is padding in the input
		input_msk = (x.mean(2) != 0)
		input_msk = input_msk.unsqueeze(1)

		# emb -> batch_size X max_len X dim_emb
		emb = self.embedding(x)

		# pos_emb -> batch_size x max_len x dim_emb
		packed_input = self.pos_embedding.forward(emb)

		# Input packed_input into RNN_Alpha
		g = self.rnn_alpha.forward(packed_input, input_msk)

		# Set alpha_unpacked to be g
		alpha_unpacked = g

		# mask -> batch_size X max_len X 1
		mask = Variable(torch.FloatTensor(
			[[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]).unsqueeze(2),
						requires_grad=False)
		if next(self.parameters()).is_cuda:  # returns a boolean
			mask = mask.cuda()

		# e => batch_size X max_len X 1
		e = self.alpha_fc(alpha_unpacked)

		# Softmax Function using mask
		def masked_softmax(batch_tensor, mask):
			exp = torch.exp(batch_tensor)
			masked_exp = exp * mask
			sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
			return masked_exp / sum_masked_exp

		# Alpha = batch_size X max_len X 1
		# alpha value for padded visits (zero) will be zero
		alpha = masked_softmax(e, mask)

		# Input packed_input into RNN Beta
		# h  = self.rnn_beta(packed_input)
		beta_unpacked = self.rnn_beta.forward(packed_input, input_msk)

		# Beta -> batch_size X max_len X dim_emb
		# beta for padded visits will be zero-vectors
		beta = torch.tanh(self.beta_fc(beta_unpacked) * mask)

		# context -> batch_size X (1) X dim_emb (squeezed)
		# Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
		# Vectorized sum
		context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

		# without applying non-linearity
		logit = self.output(context)
		logit = torch.squeeze(logit, 1)

		# Return logit, alpha, and beta
		return logit, alpha, beta


""" Epoch function """

# Epoch function
# optimizer is needed for train
def epoch(loader, model, criterion, optimizer=None, train=False):
	if train and not optimizer:
		raise AttributeError("Optimizer should be given for training")

	# Training Mode
	if train:
		model.train()
		mode = 'Train'
	# Evaluation Mode
	else:
		model.eval()
		mode = 'Eval'

	# Keeps track of all losses
	losses = AverageMeter()
	
	# Keeps track of labels and ouputs
	labels = []
	outputs = []

	# For each batch from loader
	for bi, batch in enumerate(tqdm_notebook(loader, desc="{} batches".format(mode), leave=False)):
		# Get inputs, targets, and lengths of each input
		inputs, targets, lengths = batch

		# Get the variables for inputs and ouputs
		input_var = torch.autograd.Variable(inputs)
		target_var = torch.autograd.Variable(targets)
		if args.cuda:
			input_var = input_var.cuda()
			target_var = target_var.cuda()

		# Run the model
		output, alpha, beta = model(input_var, lengths)
		
		# Get the loss function
		loss = criterion(output, target_var.float())
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		# Add targets to labels
		labels.append(targets)

		# since the outputs are logit, not probabilities
		# Replace sigmoid with softamxin future
		outputs.append(F.sigmoid(output).round().data)

		# record loss
		losses.update(loss.item(), inputs.size(0))

		# compute gradient and do update step
		if train:
			optimizer.zero_grad()
			loss.backward()
			# Prevent exploding graidents
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
			optimizer.step()

	# Return labels, ouptuts, and average of losses
	return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg


""" Main function """

# Main function, take in arguments
def main(argv):
	global args

	# Set threads
	args = parser.parse_args(argv)

	# Get visible devices
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
	
	if args.threads == -1:
		args.threads = torch.multiprocessing.cpu_count() - 1 or 1
	print('===> Configuration')
	print(args)

	# Set cuda
	cuda = args.cuda
	if cuda:
		if torch.cuda.is_available():
			print('===> {} GPUs are available'.format(torch.cuda.device_count()))
		else:
			raise Exception("No GPU found, please run with --no-cuda")

	# Fix the random seed for reproducibility
	# random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if cuda:
		torch.cuda.manual_seed(args.seed)

	# Data loading
	print('===> Loading entire datasets')
	with open(args.data_path + 'train.seqs', 'rb') as f:
		train_seqs = pickle.load(f)
	with open(args.data_path + 'train.labels', 'rb') as f:
		train_labels = pickle.load(f)
	with open(args.data_path + 'valid.seqs', 'rb') as f:
		valid_seqs = pickle.load(f)
	with open(args.data_path + 'valid.labels', 'rb') as f:
		valid_labels = pickle.load(f)
	with open(args.data_path + 'test.seqs', 'rb') as f:
		test_seqs = pickle.load(f)
	with open(args.data_path + 'test.labels', 'rb') as f:
		test_labels = pickle.load(f)
	
	# Get maximum code from seqs
	max_code = max(map(lambda p: max(map(lambda v: max(v), p)), train_seqs + valid_seqs + test_seqs))

	# Get number of features by adding 1 of amx_code
	num_features = max_code + 1

	# Find the max length of all sequences
	max_length = max(map(lambda p: len(p), train_seqs + valid_seqs + test_seqs))

	# Get the rest of the arguments
	argsEmbSize = args.embSize
	argsFFSize = args.ffSize
	argsHeads = args.heads
	assert argsFFSize > argsEmbSize
	assert argsEmbSize % argsHeads == 0
	
	# Create dataset for training, validation, and test sets
	print("     ===> Construct train set")
	train_set = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features)
	print("     ===> Construct validation set")
	valid_set = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
	print("     ===> Construct test set")
	test_set = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features)

	# Create loaders for training, validation, and test sets
	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=visit_collate_fn,
							  num_workers=args.threads)
	valid_loader = DataLoader(dataset=valid_set, batch_size=args.eval_batch_size, shuffle=False,
							  collate_fn=visit_collate_fn, num_workers=args.threads)
	test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False,
							 collate_fn=visit_collate_fn, num_workers=args.threads)
	print('===> Dataset loaded!')

	# Create model
	print('===> Building a Model')
	model = RETAIN(dim_input=num_features,
				   dim_emb=args.embSize,
				   dropout_emb=args.embDropout,
				   dropout_context=args.contextDropout,
				   dim_output=1, max_len=max_length,
				   ffDropout=args.ffDropout,
				   heads=args.heads,
				   attentionDropout=args.attentionDropout,
				   d_ff=args.ffSize,
				   normEps=args.normEps,
				   attEps=args.attEps,
				   transDropout=args.transDropout,
				   numLayers=args.numLayers,
				   beta_d_ff=args.ffSizeBeta, 
				   beta_heads=args.headsBeta, 
				   beta_transDropout=args.transDropoutBeta, 
				   beta_attentionDropout=args.attentionDropoutBeta, 
				   beta_ffDropout=args.ffDropoutBeta, 
				   beta_normEps=args.normEpsBeta, 
				   beta_attEps=args.attEpsBeta, 
				   beta_numLayers=args.numLayersBeta
				   )

	# Save model to cuda 
	if cuda:
		model = model.cuda()
	print(model)

	# for name, param in model.named_parameters():
	#    print("{}: {}".format(name, param.size()))

	print('===> Model built!')

	# Set loss function as BCEWithLogitsLoss
	criterion = nn.BCEWithLogitsLoss()
	if args.cuda:
		criterion = criterion.cuda()
	
	# Set optimizer as Adam Optimizer and learning rate
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# Kep track of best validation epochs
	best_valid_epoch = 0

	# Use best_valid_loss when we are finding best validation epoch for validation loss
	best_valid_loss = sys.float_info.max

	# Use best_valid_aurc when we are finding best validation epoch for validation aurc
	best_valid_auc = 0.0

	# Store best validation aupr
	best_valid_aupr = 0.0

	# Use best_train_loss when we are finding best validation epoch for training loss
	best_train_loss = sys.float_info.max

	# Store training loss and validation loss
	train_losses = []
	valid_losses = []

	# If plot is true, plot
	if args.plot:
		# initialise the graph and settings

		fig = plt.figure(figsize=(12, 9))  # , facecolor='w', edgecolor='k')
		ax = fig.add_subplot(111)
		plt.ion()
		fig.show()
		fig.canvas.draw()

	# For each peoch
	for ei in tnrange(args.epochs, desc="Epochs"):
		# Train
		train_y_true, train_y_pred, train_loss = epoch(train_loader, model, criterion=criterion, optimizer=optimizer,
													   train=True)
		train_losses.append(train_loss)

		# Eval
		valid_y_true, valid_y_pred, valid_loss = epoch(valid_loader, model, criterion=criterion)
		valid_losses.append(valid_loss)

		# print("Epoch {} - Loss train: {}, valid: {}".format(ei, train_loss, valid_loss))

		# Save validation and training true and predict values to cuda
		if args.cuda:
			valid_y_true = valid_y_true.cpu()
			valid_y_pred = valid_y_pred.cpu()

		if args.cuda:
				train_y_true = train_y_true.cpu()
				train_y_pred = train_y_pred.cpu()

		# Caculation validation aur and aupr
		valid_auc = roc_auc_score(valid_y_true.numpy(), valid_y_pred.numpy()[:], average="weighted")
		valid_aupr = average_precision_score(valid_y_true.numpy(), valid_y_pred.numpy()[:], average="weighted")
		
		# Find confusion matrix, f1, precision, and recall for validation set
		valid_confusion_matrix = confusion_matrix(valid_y_true.numpy(), valid_y_pred.numpy())
		valid_f1 = f1_score(valid_y_true.numpy(), valid_y_pred.numpy()[:])
		valid_precision = precision_score(valid_y_true.numpy(), valid_y_pred.numpy()[:])
		valid_recall = recall_score(valid_y_true.numpy(), valid_y_pred.numpy()[:])

		# Cacuclat auc, aupr, confusion matrix, f1, precions, and recall for training
		train_auc = roc_auc_score(train_y_true.numpy(), train_y_pred.numpy()[:], average="weighted")
		train_aupr = average_precision_score(train_y_true.numpy(), train_y_pred.numpy()[:], average="weighted")
		train_confusion_matrix = confusion_matrix(train_y_true.numpy(), train_y_pred.numpy()[:])
		train_f1 = f1_score(train_y_true.numpy(), train_y_pred.numpy()[:])
		train_precision = precision_score(train_y_true.numpy(), train_y_pred.numpy()[:])
		train_recall = recall_score(train_y_true.numpy(), train_y_pred.numpy()[:])

		# Create is best
		is_best = False

		# If args metrics is valid_loss
		if(args.metric == "valid_loss"):
			is_best = valid_loss < best_valid_loss
		# If metric is train loss
		elif(args.metric == "train_loss"):
			is_best = train_loss < best_train_loss
		# Otherwise if metric is valid_auc
		elif(args.metric == "valid_auc"):
			is_best = valid_auc > best_valid_auc
		# Otherwise if metric is none existant
		else:
			# Throw runtime error
			raise Exception("Metric does not exist")

		# If is_best is true
		if is_best:
			# Set best validation epoch and its metrics to current validation epoch
			best_valid_epoch = ei
			best_valid_loss = valid_loss
			best_valid_auc = valid_auc
			best_valid_aupr = valid_aupr
			best_valid_confusion_matrix = valid_confusion_matrix
			best_valid_recall = valid_recall
			best_valid_precision = valid_precision
			best_valid_f1 = valid_f1
			best_train_loss = train_loss

			# print("\t New best validation AUC!")
			# print('\t Evaluation on the test set')

			# evaluate on the test set
			test_y_true, test_y_pred, test_loss = epoch(test_loader, model, criterion=criterion)

			# Save true and pedicted value for testing set in cuda
			if args.cuda:
				test_y_true = test_y_true.cpu()
				test_y_pred = test_y_pred.cpu()

			# Caculate metrics for test set
			test_auc = roc_auc_score(test_y_true.numpy(), test_y_pred.numpy()[:], average="weighted")
			test_aupr = average_precision_score(test_y_true.numpy(), test_y_pred.numpy()[:], average="weighted")
			test_confusion_matrix = confusion_matrix(test_y_true.numpy(), test_y_pred.numpy()[:])
			test_f1 = f1_score(test_y_true.numpy(), test_y_pred.numpy()[:])
			test_precision = precision_score(test_y_true.numpy(), test_y_pred.numpy()[:])
			test_recall = recall_score(test_y_true.numpy(), test_y_pred.numpy()[:])

			# print("Train - Loss: {}, AUC: {}".format(train_loss, train_auc))
			# print("Valid - Loss: {}, AUC: {}".format(valid_loss, valid_auc))
			# print(" Test - Loss: {}, AUC: {}".format(valid_loss, test_auc))

			# Open file and save Transformer Arguments and metrics for training, validation, and test sets
			filename = "train_result{}.txt".format(args.name)
			with open(args.save + filename, 'w') as f:
				f.write('Embeded Size: {}\n'.format(args.embSize))
				f.write('Embeded Dropout: {}\n'.format(args.embDropout))
				f.write('Context Dropout: {}\n'.format(args.contextDropout))
				f.write('Alpha Number of Heads: {}\n'.format(args.heads))
				f.write('Alpha Attention Dropout: {}\n'.format(args.attentionDropout))
				f.write('Alpha Feed Forward Dropout {}\n'.format(args.ffDropout))
				f.write('Alpha Norm Eps: {}\n'.format(args.normEps))
				f.write('Alpha Attention Eps: {}\n'.format(args.attEps))
				f.write('Alpha Feed Forward Size: {}\n'.format(args.ffSize))
				f.write('Alpha Transformer Dropout: {}\n'.format(args.transDropout))
				f.write('Alpha Number of Layers: {}\n'.format(args.numLayers))
				f.write('Beta Number of Heads: {}\n'.format(args.headsBeta))
				f.write('Beta Attention Dropout: {}\n'.format(args.attentionDropoutBeta))
				f.write('Beta Feed Forward Dropout {}\n'.format(args.ffDropoutBeta))
				f.write('Beta Norm Eps: {}\n'.format(args.normEpsBeta))
				f.write('Beta Attention Eps: {}\n'.format(args.attEpsBeta))
				f.write('Beta Feed Forward Size: {}\n'.format(args.ffSizeBeta))
				f.write('Beta Transformer Dropout: {}\n'.format(args.transDropoutBeta))
				f.write('Beta Number of Layers: {}\n'.format(args.numLayersBeta))
				f.write('Batch Size: {}\n'.format(args.batch_size))
				f.write('Evaluation Batch Size: {}\n'.format(args.eval_batch_size))
				f.write('Best Validation Epoch: {}\n'.format(ei))
				f.write('Best Validation Loss: {}\n'.format(best_valid_loss))
				f.write('Best Validation AUROC: {}\n'.format(best_valid_auc))
				f.write('Best Validation AUPR: {}\n'.format(best_valid_aupr))
				f.write('Best Validation True Negatives: {}\n'.format(best_valid_confusion_matrix[0][0]))
				f.write('Best Validation False Positives: {}\n'.format(best_valid_confusion_matrix[0][1]))
				f.write('Best Validation False Negatives: {}\n'.format(best_valid_confusion_matrix[1][0]))
				f.write('Best Validation True Positives: {}\n'.format(best_valid_confusion_matrix[1][1]))
				f.write('Best Validation Recall: {}\n'.format(best_valid_recall))
				f.write('Best Validation Precision: {}\n'.format(best_valid_precision))
				f.write('Best Validation F1 Score: {}\n'.format(best_valid_f1))
				f.write('Train Loss: {}\n'.format(train_loss))
				f.write('Train AUROC: {}\n'.format(train_auc))
				f.write('Train AUPR: {}\n'.format(train_aupr))
				f.write('Train True Negatives: {}\n'.format(train_confusion_matrix[0][0]))
				f.write('Train False Positives: {}\n'.format(train_confusion_matrix[0][1]))
				f.write('Train False Negatives: {}\n'.format(train_confusion_matrix[1][0]))
				f.write('Train True Positives: {}\n'.format(train_confusion_matrix[1][1]))
				f.write('Train Recall: {}\n'.format(train_recall))
				f.write('Train Precision: {}\n'.format(train_precision))
				f.write('Train F1 Score: {}\n'.format(train_f1))
				f.write('Test Loss: {}\n'.format(test_loss))
				f.write('Test AUROC: {}\n'.format(test_auc))
				f.write('Test AUPR: {}\n'.format(test_aupr))
				f.write('Test True Negatives: {}\n'.format(test_confusion_matrix[0][0]))
				f.write('Test False Positives: {}\n'.format(test_confusion_matrix[0][1]))
				f.write('Test False Negatives: {}\n'.format(test_confusion_matrix[1][0]))
				f.write('Test True Positives: {}\n'.format(test_confusion_matrix[1][1]))
				f.write('Test Recall: {}\n'.format(test_recall))
				f.write('Test Precision: {}\n'.format(test_precision))
				f.write('Test F1 Score: {}\n'.format(test_f1))
				
			# Save the model
			torch.save(model, args.save + 'best_model.pth')
			torch.save(model.state_dict(), args.save + 'best_model_params.pth')

		# plot
		if args.plot:
			ax.clear()
			ax.plot(np.arange(len(train_losses)), np.array(train_losses), label='Training Loss')
			ax.plot(np.arange(len(valid_losses)), np.array(valid_losses), label='Validation Loss')
			ax.set_xlabel('epoch')
			ax.set_ylabel('Loss')
			ax.legend(loc="best")
			plt.tight_layout()
			fig.canvas.draw()
			time.sleep(0.1)

	# Print Validationa and Test Metrics
	print('Best Validation Epoch: {}\n'.format(best_valid_epoch))
	print('Best Validation Loss: {}\n'.format(best_valid_loss))
	print('Best Validation AUROC: {}\n'.format(best_valid_auc))
	print('Best Validation AUPR: {}\n'.format(best_valid_aupr))
	print('Best Validation True Negatives: {}\n'.format(best_valid_confusion_matrix[0][0]))
	print('Best Validation False Positives: {}\n'.format(best_valid_confusion_matrix[0][1]))
	print('Best Validation False Negatives: {}\n'.format(best_valid_confusion_matrix[1][0]))
	print('Best Validation True Positives: {}\n'.format(best_valid_confusion_matrix[1][1]))
	print('Best Validation Recall: {}\n'.format(best_valid_recall))
	print('Best Validation Precision: {}\n'.format(best_valid_precision))
	print('Best Validation F1 Score: {}\n'.format(best_valid_f1))
	print('Test Loss: {}\n'.format(test_loss))
	print('Test AUROC: {}\n'.format(test_auc))
	print('Test AUPR: {}\n'.format(test_aupr))
	print('Test True Negatives: {}\n'.format(test_confusion_matrix[0][0]))
	print('Test False Positives: {}\n'.format(test_confusion_matrix[0][1]))
	print('Test False Negatives: {}\n'.format(test_confusion_matrix[1][0]))
	print('Test True Positives: {}\n'.format(test_confusion_matrix[1][1]))
	print('Test Recall: {}\n'.format(test_recall))
	print('Test Precision: {}\n'.format(test_precision))
	print('Test F1 Score: {}\n'.format(test_f1))


if __name__ == "__main__":
	main(sys.argv[1:])
