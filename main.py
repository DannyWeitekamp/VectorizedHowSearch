import torch
import itertools
from torch.nn import functional as F

class BaseOperator(object):
	def __init__(self):
		self.num_flt_inputs = 0;
		self.num_str_inputs = 0; 
		self.commutative = False;
		self.template = "BaseOperator"
	

	def forward(self, args):
		raise NotImplementedError("Not Implemeneted")

	def backward(self, args):
		raise NotImplementedError("Not Implemeneted")
	def search_mask(self,*args):
		return args



class Add(BaseOperator):
	def __init__(self):
		self.num_flt_inputs = 2;
		self.template = "(%s + %s)"
		self.commutative = True;

	def forward(self, x,y):
		return x+y

class Subtract(BaseOperator):
	def __init__(self):
		self.num_flt_inputs = 2;
		self.template = "(%s - %s)"

	def forward(self, x,y):
		return x-y

class Multiply(BaseOperator):
	def __init__(self):
		self.num_flt_inputs = 2;
		self.template = "(%s * %s)"
		self.commutative = True;

	def forward(self, x,y):
		return x*y

class Divide(BaseOperator):
	def __init__(self):
		self.num_flt_inputs = 2;
		self.template = "(%s / %s)"

	def forward(self, x,y):
		return x/y

class Mod10(BaseOperator):
	def __init__(self):
		self.num_flt_inputs = 1;
		self.template = "(%s %% 10)"

	def forward(self, x):
		return x % 10

	def search_mask(self,x):
		return [torch.where(x >= 10, x, torch.tensor(float("NaN")))]

class Div10(BaseOperator):
	def __init__(self):
		self.num_flt_inputs = 1;
		self.template = "(%s // 10)"

	def forward(self, x):
		return x // 10

	def search_mask(self,x):
		return [torch.where(x >= 10, x, torch.tensor(float("NaN")))]


def reshape_args(args):
	l = len(args)-1
	if(l == -1):
		return None
	return [a.view(*([1]*i + [-1] + (l-i)*[1])) for i, a in enumerate(args)]




def resolve_inputs(flat_input_indices, op_inds,n_elems):
	# print(op_inds)
	nf_inps = operator_nf_inps.gather(0,op_inds)
	# print("op_inds:", op_inds)
	# print("nf_inps:",nf_inps)

	# print(nf_inps.shape)
	max_in_degree = torch.max(nf_inps)

	temp_ind = flat_input_indices

	# print("n_elems:", n_elems)
	args = []
	for j in range(max_in_degree-1,-1,-1):
		modulus = n_elems**j
		ok = nf_inps >= j+1
		arg = torch.where(ok,(temp_ind // modulus),torch.tensor(-1))
		args.append(arg.view(-1,1))
		temp_ind = torch.where(ok,temp_ind % modulus,temp_ind)


	# print("args", args)
	# print(max_in_degree)
	# for j in range(max_in_degree):
	# print(flat_input_indices, op_inds)
	# print(nf_inps)
	return torch.cat(args,1)



def resolve_operator(offset_indicies,offset):
	offset_indicies -= offset

	# print("offset_indicies",offset_indicies)

	operator_counts = torch.pow(offset,operator_nf_inps)
	# print("operator_counts", operator_counts)
	cs = torch.cumsum(operator_counts,0) 
	# print("CS",cs)
	# raise ValueError()
	# print(cs.view(-1,1) < offset_indicies.view(1,-1))
	# print(torch.sum(cs.view(-1,1) < offset_indicies.view(1,-1),0))

	op_inds = torch.sum(cs.view(-1,1) <= offset_indicies.view(1,-1),0)
	# print("op_inds", op_inds)
	op_offset = torch.cat([torch.tensor([0]),cs],0).gather(0,op_inds)
	# print("op_offset", op_offset)
	# print("offset_indicies", offset_indicies)
	input_indicies = offset_indicies - op_offset

	# print("-------------")
	# print("input_indicies", input_indicies)
	# print("-------------")

	args = resolve_inputs(input_indicies,op_inds,offset)
	# print(args)

	return op_inds, args
	# for op_ind, arg_set in zip(op_inds, args):
	# 	print(op_ind, arg_set)


	# return op_inds, input_indicies 
	# for input_index, op_ind in zip(input_indicies,op_inds):
	# 	resolve_inputs(input_index.item(), op_ind.item(), )



	 	# torch.sum(cs < offset_indicies,1)


	# cum_
	# print(op_inds)
	# print(op_offset)

def indicies_to_operator_graph(indicies, d_len):
	# if(len(indicies) != 1):
	# 	print(indicies)
	# print(index > torch.tensor(d_len).view(1,-1))
	# print(torch.tensor(d_len).view(1,-1).shape)
	# d_len = 
	n_lsthn = indicies >= d_len.view(1,-1)
	# offset = torch.sum(n_lsthn.long() * d_len,1)
	# indicies - offset
	# print(n_lsthn)
	# print(offset)
	d_bins = torch.sum(n_lsthn,1)
	# print("d_bins", d_bins)
	# print("d_len", d_len)

	# print(torch.unique(d_bins,return_inverse=True,sorted=True))
	# offset = 0
	for d in range(len(d_len)):
		# print("STARTING D", d, len(d_len))

		offset = d_len[d-1] if d > 0 else 0
		offset_indicies = indicies[(d_bins == d).nonzero()].flatten()#-offset


		# total = d_len[d]-offset

		# print(offset)

		if(d > 0 and offset_indicies.shape[0] > 0):
			
			op_inds, args = resolve_operator(offset_indicies,offset)
			# if(len(indicies) != 1):
			# 	print(d)
			# 	print(offset_indicies[-20:])
			# 	print(op_inds[-20:], args[-20:])
			# raise(ValueError())

			for op_ind, arg_set in zip(op_inds, args):
				# print((arg_set >= 0).nonzero().shape)
				# print(arg_set.shape)
				arg_set = arg_set.gather(0,(arg_set >= 0).nonzero().flatten())

				# print( "OUT" ,(operator_set[op_ind], numerical_values.gather(0,arg_set) ) )
				# print(len(indicies))
				# print([operator_set[op_ind], *[x.item() for x in arg_set]] )
				# print(arg_set)
				# print(op_ind)
				
				yield [operator_set[op_ind], *[x.item() if x.item() < d_len[0] else next(indicies_to_operator_graph(x.view(1),d_len)) for x in arg_set]] 
		else:
			# print("MOOOO")
			for indx in offset_indicies:
				yield indx.item()

		# print(total)
		# print(d, off_depth_d_indicies)
		# print(offset)

		# offset += d_len

import types
def repr_rule(x,numerical_values, include_element=True, include_value=True):
	if(isinstance(x,(list,tuple))):
		# print(x[0].template,tuple([repr_rule(y,numerical_values) for y in x[1:]]))
		return x[0].template % tuple([repr_rule(y,numerical_values) for y in x[1:]])
	elif(isinstance(x,types.GeneratorType)):
		return repr_rule(next(x), numerical_values)
	else:
		a = "E" + str(x) if include_element else ""
		b = ":" if include_element and include_value else ""
		c = repr(numerical_values.gather(0,torch.tensor(x)).item()) if include_value else ""
		return a + b + c

def rule_inputs(x):
	# print(x)
	if(isinstance(x,(list,tuple))):
		if(len(x) == 0): return x
		# print("X", x)
		# print(type(list(itertools.chain.from_iterable([rule_inputs(y) for y in x[1:]]))))
		return list(itertools.chain.from_iterable([rule_inputs(y) for y in x[1:]]))
	elif(isinstance(x,types.GeneratorType)):
		return rule_inputs([y for y in x])
	else:
		return [x]



def how_search(numerical_values, goal, search_depth = 1):

	with torch.no_grad():
		most_args = max([x.num_flt_inputs for x in operator_set])
		history = [None]*search_depth
		d_len = [numerical_values.shape[0]] 
		


		for d in range(search_depth):
			history[d] = h_d = []
			reshape_set = [reshape_args([numerical_values]*i) for i in range(most_args+1)] 
			forwards = [numerical_values]


			# print(redundency_mask.shape)
			for j,o in enumerate(operator_set):
				x_d = o.forward(*o.search_mask(*reshape_set[o.num_flt_inputs]))
				# print(numerical_values.shape[0],o.num_flt_inputs)
				# mask = redundency_mask
				mask = create_redundency_mask(d_len[-2] if len(d_len) >= 2 else 0,numerical_values.shape[0],o.num_flt_inputs)
				if(o.num_flt_inputs > 1):	
					mask = mask & create_upper_triag_mask(numerical_values.shape[0],o.num_flt_inputs)
				x_d = torch.where(mask, x_d, torch.tensor(float('NaN')))
				# print(x_d.shape)
				# x_d = torch.where(redundency_mask, x_d, torch.tensor(float('NaN')))
				


				# print(x_d)
				forwards.append(torch.flatten(x_d))
			numerical_values = torch.cat(forwards)
			d_len.append(numerical_values.shape[0])

			# print(numerical_values.shape)
			# print(type(numerical_values))

		indicies = (numerical_values == goal).nonzero()

		print(numerical_values[indicies[-20:]])
		# print(indicies)
		# print("d_len",d_len)
		# print("NUM RESULTS", indicies.shape)
		for tup in indicies_to_operator_graph(indicies,torch.tensor(d_len)):
			inps = rule_inputs(tup)
			if(len(set(inps)) == len(inps) and inps == sorted(inps)):

				# print(tup)
				print(repr_rule(tup,numerical_values))
			
			


		# print(indicies)
		# print(indicies.shape)


# operator_class_set = [Add,Mod10,Div10]
operator_class_set = [Add,Mod10,Div10]
operator_set = [c() for c in operator_class_set ]
operator_nf_inps = torch.tensor([x.num_flt_inputs for x in operator_set])
	

def create_redundency_mask(n_prev,n,d):
	# print("n_prev", n_prev,n, n-n_prev)
	# torch.where(torch.ones(n_prev,n_prevm).byte(),torch.zeros(n,n), torch.ones(1))
	return  F.pad(torch.zeros([n_prev]*d),tuple([0,n-n_prev]*d), value=1).byte()
	# return torch.zeros(n,n).byte()


def create_diag_mask(n,d):
	return create_mask(n,d,torch.ne)

def create_upper_triag_mask(n,d):
	return create_mask(n,d,torch.lt)
def create_lower_triag_mask(n,d):
	return create_mask(n,d,torch.gt)

def create_mask(n,d,f=torch.eq):
	a = torch.nonzero(torch.ones([n]*d)).view(-1,1,d).float()
	w = torch.tensor([1,-1]).view(1,1,-1).float()
	sel = (f(torch.conv1d(a,w, None,1,0,1,1), 0)).all(2).view([n]*d)
	return sel.byte()
	# print(sel.shape)



x = torch.FloatTensor([7,8,9,9,7])
how_search(x,7, search_depth=3)
# print(x)

# print(create_upper_triag_mask(5,2))
# print(create_diag_mask(5,2))
# print(create_lower_triag_mask(5,2))

# print(torch.nonzero(torch.ones([5]*3)))

# a = 
# print(a.shape)

# print(w)
# print(a.shape)
# print(torch.conv1d(a,w, None,1,0,1,1))
# print()
# torch.nn.Conv1d(1,1,2,stride=1,bias=False)
# print(type(x))


# x = torch.FloatTensor([1,2])
# how_search(x,3,search_depth=2)

# x = torch.FloatTensor([1,1,1])
# how_search(x,3,search_depth=3)



	

# torch.empty

# print(Add.num_flt_inputs)

# args = ['a','b','c']
# l = len(args)-1
# print([([1]*i + [-1] + (l-i)*[1]) for i,a in enumerate(args)])


