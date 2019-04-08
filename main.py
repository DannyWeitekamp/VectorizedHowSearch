import torch


class BaseOperator(object):
	num_flt_inputs = 0;
	num_str_inputs = 0; 

	@classmethod
	def forward(cls, args):
		raise NotImplementedError("Not Implemeneted")

	@classmethod
	def backward(cls, args):
		raise NotImplementedError("Not Implemeneted")

class Add(BaseOperator):
	num_flt_inputs = 2;

	@classmethod
	def forward(cls, x,y):
		return x+y

class Subtract(BaseOperator):
	num_flt_inputs = 2;

	@classmethod
	def forward(cls, x,y):
		return x-y

class Multiply(BaseOperator):
	num_flt_inputs = 2;

	@classmethod
	def forward(cls, x,y):
		return x*y

class Divide(BaseOperator):
	num_flt_inputs = 2;

	@classmethod
	def forward(cls, x,y):
		return x/y

class Mod10(BaseOperator):
	num_flt_inputs = 1;

	@classmethod
	def forward(cls, x):
		return x % 10

class Div10(BaseOperator):
	num_flt_inputs = 1;

	@classmethod
	def forward(cls, x):
		return x // 10


def reshape_args(args):
	l = len(args)-1
	if(l == -1):
		return None
	return [a.view(*([1]*i + [-1] + (l-i)*[1])) for i, a in enumerate(args)]


operator_set = [Add,Subtract,Multiply,Divide,Mod10,Div10]
operator_nf_inps = torch.tensor([x.num_flt_inputs for x in operator_set])

def resolve_inputs(flat_input_indices, op_inds,n_elems):
	nf_inps = operator_nf_inps.gather(0,op_inds)

	# print(nf_inps.shape)
	max_in_degree = torch.max(nf_inps)

	temp_ind = flat_input_indices

	args = []
	for j in range(max_in_degree-1,-1,-1):
		modulus = n_elems**j
		# print(j,modulus)

		arg = (temp_ind // modulus) * (1 - 2* (nf_inps < max_in_degree-j ).long())
		args.append(arg.view(-1,1))
		# resolve_operator(arg,0)

		# print("A",  arg, arg.shape)
		temp_ind = temp_ind % modulus


	# print(max_in_degree)
	# for j in range(max_in_degree):
	# print(flat_input_indices, op_inds)
	# print(nf_inps)
	return torch.cat(args,1)



def resolve_operator(offset_indicies,offset):
	operator_counts = torch.pow(offset,operator_nf_inps)
	cs = torch.cumsum(operator_counts,0) 
	# print("CS",cs)
	# print(cs.view(-1,1) < offset_indicies.view(1,-1))
	# print(torch.sum(cs.view(-1,1) < offset_indicies.view(1,-1),0))

	op_inds = torch.sum(cs.view(-1,1) < offset_indicies.view(1,-1),0)
	# print("op_inds", op_inds)
	op_offset = (cs- operator_counts[0]).gather(0,op_inds)
	# print("offset_indicies", offset_indicies)
	input_indicies = offset_indicies - op_offset



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
	# print(index > torch.tensor(d_len).view(1,-1))
	# print(torch.tensor(d_len).view(1,-1).shape)
	d_len = torch.tensor(d_len)
	n_lsthn = indicies >= d_len.view(1,-1)
	# offset = torch.sum(n_lsthn.long() * d_len,1)
	# indicies - offset
	# print(n_lsthn)
	# print(offset)
	d_bins = torch.sum(n_lsthn,1)
	# print("d_bins", d_bins)

	# print(torch.unique(d_bins,return_inverse=True,sorted=True))
	# offset = 0
	for d in range(len(d_len)):
		# print("STARTING D", d, len(d_len))

		offset = d_len[d-1] if d > 0 else 0
		offset_indicies = indicies[(d_bins == d).nonzero()].flatten()-offset

		# total = d_len[d]-offset

		# print(offset)

		if(d > 0 and offset_indicies.shape[0] > 0):
			op_inds, args = resolve_operator(offset_indicies,offset)
			for op_ind, arg_set in zip(op_inds, args):
				# print((arg_set >= 0).nonzero().shape)
				# print(arg_set.shape)
				arg_set = arg_set.gather(0,(arg_set >= 0).nonzero().flatten())
				# print( "OUT" ,(operator_set[op_ind], numerical_values.gather(0,arg_set) ) )
				yield [operator_set[op_ind], *[x.item() if x.item() < d_len[0] else indicies_to_operator_graph(x.view(1),d_len) for x in arg_set]] 
		else:
			# print("MOOOO")
			for indx in offset_indicies:
				yield indx.item()

		# print(total)
		# print(d, off_depth_d_indicies)
		# print(offset)

		# offset += d_len

import types
def repr_rule(x,numerical_values):
	if(isinstance(x,list)):
		return str(x[0]) + "(" + " , ".join([repr_rule(y,numerical_values) for y in x[1:]]) + ")"
	elif(isinstance(x,types.GeneratorType)):
		return repr_rule(next(x), numerical_values)
	else:
		return "E" + str(x) + ":" + repr(numerical_values.gather(0,torch.tensor(x)).item())



def how_search(numerical_values, goal, search_depth = 1):

	with torch.no_grad():
		most_args = max([x.num_flt_inputs for x in operator_set])
		history = [None]*search_depth
		d_len = [numerical_values.shape[0]] 
		
		for d in range(search_depth):
			history[d] = h_d = []
			reshape_set = [reshape_args([numerical_values]*i) for i in range(most_args+1)] 
			forwards = [numerical_values]
			for j,o in enumerate(operator_set):
				x_d = o.forward(*reshape_set[o.num_flt_inputs])
				# print(x_d)
				forwards.append(torch.flatten(x_d))
			numerical_values = torch.cat(forwards)
			d_len.append(numerical_values.shape[0])

			# print(numerical_values.shape)
			# print(type(numerical_values))

		indicies = (numerical_values == goal).nonzero()
		print("NUM RESULTS", indicies.shape)
		for tup in indicies_to_operator_graph(indicies,d_len):
			print(repr_rule(tup,numerical_values))
			


		# print(indicies)
		# print(indicies.shape)



	



# x = torch.FloatTensor([1,2,3,4,6,7,8,9,0,16])
# how_search(x,6, search_depth=1)
# print(x)
# print(type(x))


x = torch.FloatTensor([1,1,1])
how_search(x,3,search_depth=2)



	

# torch.empty

# print(Add.num_flt_inputs)

# args = ['a','b','c']
# l = len(args)-1
# print([([1]*i + [-1] + (l-i)*[1]) for i,a in enumerate(args)])


