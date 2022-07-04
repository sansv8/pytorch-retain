""" Matplotlib backend configuration """
import matplotlib
matplotlib.use('PS')  # generate postscript output by default

""" Imports """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import sys
import argparse
import pickle
import time
from tqdm import tnrange, tqdm_notebook

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

# Resume training from previous chekcpoint
parser.add_argument('--resume', default='', type=str, metavar='LOAD_PATH',
					help='path to latest checkpoint (default: none)')

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
	batch_seq, batch_label = zip(*batch)

	num_features = batch_seq[0].shape[1]
	seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
	max_length = max(seq_lengths)

	sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
	sorted_padded_seqs = []
	sorted_labels = []

	for i in sorted_indices:
		length = batch_seq[i].shape[0]

		if length < max_length:
			padded = np.concatenate(
				(batch_seq[i].toarray(), np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
		else:
			padded = batch_seq[i].toarray()

		sorted_padded_seqs.append(padded)
		sorted_labels.append(batch_label[i])

	seq_tensor = np.stack(sorted_padded_seqs, axis=0)
	label_tensor = torch.LongTensor(sorted_labels)

	return torch.from_numpy(seq_tensor), label_tensor, list(sorted_lengths)


""" RETAIN model class """

# The Retain Model
class RETAIN(nn.Module):
	# Initialize the model with following parameters
	# Input Dimensions
	# Embedding Dimenstions (Default 128)
	# Dropout Input (Default: 0.8)
	# Dropout Embedding (Default: 0.5)
	# Alpha Dimension (Default: 128)
	# Beta Dimension (Default: 128)
	# Dropout for Context (Default: 0.5)
	# Output Dimnesion (Defautl 2)
	# l2 (Default: 0.0001)
	# batch_first
	def __init__(self, dim_input, dim_emb=128, dropout_input=0.8, dropout_emb=0.5, dim_alpha=128, dim_beta=128,
				 dropout_context=0.5, dim_output=2, l2=0.0001, batch_first=True):
		super(RETAIN, self).__init__()
		self.batch_first = batch_first
		
		# Embedding Layer 
		# Input Dropout Layer
		# Linear Layer v = W(emb) * X
		# Embedded Dropout Layer
		self.embedding = nn.Sequential(
			nn.Dropout(p=dropout_input),
			nn.Linear(dim_input, dim_emb, bias=False),
			nn.Dropout(p=dropout_emb)
		)
		# Initailze weight using xavier normal
		init.xavier_normal(self.embedding[1].weight)

		# Set RNN Alpha as GRU
		# input_size = dim_emb
		# hidden_size = dim_alpha
		# num_layers = 1
		# g(i) .... g(1) = RRN_alpha(v(i)...v(1))
		self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

		# e(j) = w(alpha) * g(j) + b(alpha) for j = 1 ... i
		# Input_Dim = dim_alpha
		# Output_Dim = 1
		self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
		
		# Initialize xavier nomral weights
		init.xavier_normal(self.alpha_fc.weight)
		
		# Set bias to be 0
		self.alpha_fc.bias.data.zero_()

		# Set RNN Beta as GRU
		# input_size = dim_emb
		# hidden_size = dim_beta
		# num_layers = 1
		# h(i) ... h(1) = RNN_betta(v(i) ... v(1))
		self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

		# b(j) = W(beta) * h(j) + b(j) for j = 1 .. i
		# input_dim = dim_beta
		# output_dim = dim_emb
		self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
		
		# Initialize weights as xavier_normal for tanh
		init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain('tanh'))
		
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
		init.xavier_normal(self.output[1].weight)
		self.output[1].bias.data.zero_()

	# Forward function by getting input and lengths
	def forward(self, x, lengths):
		if self.batch_first:
			batch_size, max_len = x.size()[:2]
		else:
			max_len, batch_size = x.size()[:2]

		# emb -> batch_size X max_len X dim_emb
		emb = self.embedding(x)
		
		# Pack the embedded sequnece
		packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)

		# Input packed_input into RNN_Alpha
		g, _ = self.rnn_alpha(packed_input)

		# alpha_unpacked -> batch_size X max_len X dim_alpha
		alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)

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
		h, _ = self.rnn_beta(packed_input)

		# beta_unpacked -> batch_size X max_len X dim_beta
		beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

		# Beta -> batch_size X max_len X dim_emb
		# beta for padded visits will be zero-vectors
		beta = F.tanh(self.beta_fc(beta_unpacked) * mask)

		# context -> batch_size X (1) X dim_emb (squeezed)
		# Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
		# Vectorized sum
		context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

		# without applying non-linearity
		logit = self.output(context)

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

		# Get the variabels for inputs and ouputs
		input_var = torch.autograd.Variable(inputs)
		target_var = torch.autograd.Variable(targets)
		if args.cuda:
			input_var = input_var.cuda()
			target_var = target_var.cuda()

		# Run the model
		output, alpha, beta = model(input_var, lengths)
		
		# Get the loss function
		loss = criterion(output, target_var)
		assert not np.isnan(loss.data[0]), 'Model diverged with loss = NaN'

		labels.append(targets)

		# since the outputs are logit, not probabilities
		outputs.append(F.softmax(output).data)

		# record loss
		losses.update(loss.data[0], inputs.size(0))

		# compute gradient and do update step
		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg


""" Main function """

# Main function, take in arguments
def main(argv):
	global args
	args = parser.parse_args(argv)
	if args.threads == -1:
		args.threads = torch.multiprocessing.cpu_count() - 1 or 1
	print('===> Configuration')
	print(args)

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

	max_code = max(map(lambda p: max(map(lambda v: max(v), p)), train_seqs + valid_seqs + test_seqs))
	num_features = max_code + 1

	print("     ===> Construct train set")
	train_set = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features)
	print("     ===> Construct validation set")
	valid_set = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
	print("     ===> Construct test set")
	test_set = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features)

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
				   dim_emb=512,
				   dropout_emb=0.5,
				   dim_alpha=512,
				   dim_beta=512,
				   dropout_context=0.5,
				   dim_output=2)

	if cuda:
		model = model.cuda()
	print(model)

	# for name, param in model.named_parameters():
	#    print("{}: {}".format(name, param.size()))

	print('===> Model built!')

	criterion = nn.CrossEntropyLoss()
	if args.cuda:
		criterion = criterion.cuda()

	# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

	best_valid_epoch = 0
	best_valid_loss = sys.float_info.max
	best_valid_auc = 0.0
	best_valid_aupr = 0.0

	train_losses = []
	valid_losses = []

	if args.plot:
		# initialise the graph and settings

		fig = plt.figure(figsize=(12, 9))  # , facecolor='w', edgecolor='k')
		ax = fig.add_subplot(111)
		plt.ion()
		fig.show()
		fig.canvas.draw()

	for ei in tnrange(args.epochs, desc="Epochs"):
		# Train
		train_y_true, train_y_pred, train_loss = epoch(train_loader, model, criterion=criterion, optimizer=optimizer,
													   train=True)
		train_losses.append(train_loss)

		# Eval
		valid_y_true, valid_y_pred, valid_loss = epoch(valid_loader, model, criterion=criterion)
		valid_losses.append(valid_loss)

		# print("Epoch {} - Loss train: {}, valid: {}".format(ei, train_loss, valid_loss))

		if args.cuda:
			valid_y_true = valid_y_true.cpu()
			valid_y_pred = valid_y_pred.cpu()

		valid_auc = roc_auc_score(valid_y_true.numpy(), valid_y_pred.numpy()[:, 1], average="weighted")
		valid_aupr = average_precision_score(valid_y_true.numpy(), valid_y_pred.numpy()[:, 1], average="weighted")

		is_best = valid_auc > best_valid_auc

		if is_best:
			best_valid_epoch = ei
			best_valid_loss = valid_loss
			best_valid_auc = valid_auc
			best_valid_aupr = valid_aupr

			# print("\t New best validation AUC!")
			# print('\t Evaluation on the test set')

			# evaluate on the test set
			test_y_true, test_y_pred, test_loss = epoch(test_loader, model, criterion=criterion)

			if args.cuda:
				train_y_true = train_y_true.cpu()
				train_y_pred = train_y_pred.cpu()
				test_y_true = test_y_true.cpu()
				test_y_pred = test_y_pred.cpu()

			train_auc = roc_auc_score(train_y_true.numpy(), train_y_pred.numpy()[:, 1], average="weighted")
			train_aupr = average_precision_score(train_y_true.numpy(), train_y_pred.numpy()[:, 1], average="weighted")

			test_auc = roc_auc_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")
			test_aupr = average_precision_score(test_y_true.numpy(), test_y_pred.numpy()[:, 1], average="weighted")

			# print("Train - Loss: {}, AUC: {}".format(train_loss, train_auc))
			# print("Valid - Loss: {}, AUC: {}".format(valid_loss, valid_auc))
			# print(" Test - Loss: {}, AUC: {}".format(valid_loss, test_auc))

			with open(args.save + 'train_result.txt', 'w') as f:
				f.write('Best Validation Epoch: {}\n'.format(ei))
				f.write('Best Validation Loss: {}\n'.format(best_valid_loss))
				f.write('Best Validation AUROC: {}\n'.format(best_valid_auc))
				f.write('Best Validation AUPR: {}\n'.format(best_valid_aupr))
				f.write('Train Loss: {}\n'.format(train_loss))
				f.write('Train AUROC: {}\n'.format(train_auc))
				f.write('Train AUPR: {}\n'.format(train_aupr))
				f.write('Test Loss: {}\n'.format(test_loss))
				f.write('Test AUROC: {}\n'.format(test_auc))
				f.write('Test AUPR: {}\n'.format(test_aupr))

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

	print('Best Validation Epoch: {}\n'.format(best_valid_epoch))
	print('Best Validation Loss: {}\n'.format(best_valid_loss))
	print('Best Validation AUROC: {}\n'.format(best_valid_auc))
	print('Best Validation AUPR: {}\n'.format(best_valid_aupr))
	print('Test Loss: {}\n'.format(test_loss))
	print('Test AUROC: {}\n'.format(test_auc))
	print('Test AUPR: {}\n'.format(test_aupr))


if __name__ == "__main__":
	main(sys.argv[1:])
