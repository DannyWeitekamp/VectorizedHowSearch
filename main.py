import torch
import itertools
import re 
from torch.nn import functional as F


#TODO How to handle element dereferencing?
	# Ideas: Have element tensor seperate from "value", "x", "y" 
	# and dereference into "value", "x", "y"
	# Issues: This situation would not be broadcastable in a simple way for example 
	# 	if I wanted to rotate ?ele by "value" I would need the "x" and "y"
	# 	from ele and I would need to instantiate a new ?ele  

class BaseOperator(object):
	def __init__(self):
		self.num_flt_inputs = 0;
		self.num_str_inputs = 0;
		self.out_arg_types = ["value"]
		self.in_arg_types= ["value","value"]
		self.commutative = False;
		self.template = "BaseOperator"
	

	def forward(self, args):
		raise NotImplementedError("Not Implemeneted")

	def backward(self, args):
		raise NotImplementedError("Not Implemeneted")
	def search_mask(self,*args):
		return args
	def __repr__(self):
		return self.template % tuple(["E" + str(i) for i in range(len(self.in_arg_types))])



NaN = torch.tensor(float("NaN"))

class Add(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "(%s + %s)"
		self.commutative = True;

	def forward(self, x,y):
		return x+y

class Subtract(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "(%s - %s)"

	def forward(self, x,y):
		return x-y

class Multiply(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "(%s * %s)"
		self.commutative = True;

	def forward(self, x,y):
		return x*y

class Divide(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "(%s / %s)"

	def forward(self, x,y):
		return x/y

class Mod10(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 1;
		self.template = "(%s %% 10)"
		self.in_arg_types = ["value"]

	def forward(self, x):
		return x % 10

	def search_mask(self,x):
		return [torch.where(x >= 10, x, NaN)]

class Div10(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 1;
		self.template = "(%s // 10)"
		self.in_arg_types = ["value"]

	def forward(self, x):
		return x // 10

	def search_mask(self,x):
		return [torch.where(x >= 10, x, NaN)]

class Equals(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "(%s == %s)"
		self.commutative = True;
		self.in_arg_types = ["value"]

	def forward(self, x,y):
		eq = x == y
		return torch.where(eq,eq.float(),NaN)


class OperatorGraph(BaseOperator):
	def _gen_expression(self,x,arg_type=None):
		if(isinstance(x,(list,tuple))):
			front = x[0]
			rest = [self._gen_expression(x[j],front.in_arg_types[j-1]) for j in range(1,len(x))]
			return (front,*rest)
		elif(isinstance(x,int)):
			arg = "?foa" + str(len(self.in_args))
			self.in_args.append(arg)
			self.in_arg_types.append(arg_type)
			return arg
		else:
			return x

	def _gen_template(self,x):
		# if(first): self._count = 0
		if(isinstance(x,(list,tuple))):
			rest = [self._gen_template(x[j]) for j in range(1,len(x))]
			return x[0].template % tuple(rest)
		elif(isinstance(x,int)):
			# arg = "E" + str(self._count)
			# self._count += 1
			return "%s"
		else:
			return x

	


	def __init__(self,tup,backmap=None,out_arg_types=None):
		super().__init__()
		# self.out_arg_types = out_arg_types
		# if(out_arg_types != None and isinstance(tup,(list,tuple)) and isinstance(tup[0],BaseOperator)):
		# 	assert tup[0].out_arg_types == out_arg_types

		self.in_args = []
		self.in_arg_types = []

		self.template = "g[" + self._gen_template(tup) + "]"
		print("TEMPLATE",self.template)
		self.expression = self._gen_expression(tup,backmap)
		self.num_flt_inputs = len(self.in_args)

		# print(self.expression)
	# def __repr__(self):
	# 	return self.template % ('E' + str(i))

	def _forward(self,x,inps,first=True):
		if(first): self._count = 0
		if(isinstance(x,(list,tuple))):
			front = x[0]
			# print(front.forward(*[ self._forward(x[j],inps,False) for j in range(1,len(x))]))
			return front.forward(*[ self._forward(x[j],inps,False) for j in range(1,len(x))])
		else:
			inp = inps[self._count]
			# print(inp.shape)
			self._count += 1
			return inp

	def forward(self,*inps):
		print([x.shape for x in inps])
		# xd = 
		# print(xd)
		return self._forward(self.expression,inps)

	def compute(self,mapping,state):
		inps = []
		for arg_type,key in zip(self.in_arg_types, sorted(mapping.keys())):
			inps.append(torch.tensor(state[(arg_type,mapping[key])]))
		print(inps)
		return self._forward(self.expression,inps).item()		
		





def eval_expression(x):
	if(isinstance(x,(list,tuple))):
		front = x[0]
		rest = [eval_expression(x[j]) for j in range(1,len(x))]
		if(isinstance(front,BaseOperator)):
			return front.forward(*rest)
		else:
			return tuple(front,*rest)
	else:
		return x


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

def indicies_to_operator_graph(indicies, d_len,operators):
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
			# print(op_inds)
			# print(args)
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
				
				yield [operators[op_ind], *[x.item() if x.item() < d_len[0] else next(indicies_to_operator_graph(x.view(1),d_len,operators)) for x in arg_set]] 
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



def state_to_tensors(state):
	numerical_values = []
	backmap = []
	for key, val in state.items():
		print(key,val)
		if(not isinstance(val,bool)):
			# TODO: do this better
			# if(isinstance(val,(int,float)) or (isinstance(val,str) and is_numerical.match(val))):
			try:
				numerical_values.append(float(val))
			except ValueError:
				continue
			backmap.append(key)


	return torch.tensor(numerical_values),backmap


def _broadcasted_apply(o,numerical_values,d_len,reshape_set):
	x_d = o.forward(*o.search_mask(*reshape_set[o.num_flt_inputs]))
	# print(numerical_values.shape[0],o.num_flt_inputs)
	# mask = redundency_mask
	mask = create_redundency_mask(d_len[-2] if len(d_len) >= 2 else 0,numerical_values.shape[0],o.num_flt_inputs)
	if(o.num_flt_inputs > 1):
		if(o.commutative):	
			mask = mask & create_upper_triag_mask(numerical_values.shape[0],o.num_flt_inputs)
		else:
			mask = mask & create_diag_mask(numerical_values.shape[0],o.num_flt_inputs)
	x_d = torch.where(mask, x_d, torch.tensor(float('NaN')))
	return x_d

def forward_one(numerical_values,d_len,operators):
	most_args = max([x.num_flt_inputs for x in operators])
	reshape_set = [reshape_args([numerical_values]*i) for i in range(most_args+1)] 
	forwards = [numerical_values]
	# print(redundency_mask.shape)
	for j,o in enumerate(operators):
		x_d = _broadcasted_apply(o,numerical_values,d_len,reshape_set)
		print(x_d)
		forwards.append(torch.flatten(x_d))
	numerical_values = torch.cat(forwards)
	d_len.append(numerical_values.shape[0])
	return numerical_values, d_len

def to_rule_expression(tup, backmap):
	if(isinstance(x,(list,tuple))):
		front = x[0]
		rest = [to_rule_expression(x[j]) for j in range(1,len(x))]
		return tuple(front,*rest)
	elif(isinstance(x,int)):
		return backmap[x]
	else:
		return x




def how_search(state, goal, search_depth = 1,backmap=None):
	if(isinstance(state, dict)):
		numerical_values,backmap = state_to_tensor(state)
	else:
		numerical_values = state

	assert backmap != None, "backmap needs to exist"

	with torch.no_grad():
		
		d_len = [numerical_values.shape[0]] 
		for d in range(search_depth):
			numerical_values, d_len = forward_one(numerical_values, d_len,operator_set)

			# print(numerical_values.shape)
			# print(type(numerical_values))

		indicies = (numerical_values == goal).nonzero()

		# print(numerical_values[indicies[-20:]])
		# print(indicies)
		# print("d_len",d_len)
		# print("NUM RESULTS", indicies.shape)
		for tup in indicies_to_operator_graph(indicies,torch.tensor(d_len),operator_set):
			inps = rule_inputs(tup)
			if(len(set(inps)) == len(inps) and inps == sorted(inps)):
				yield OperatorGraph(tup)
				# print(tup)
				# print(repr_rule(tup,numerical_values))
				# print(eval_expression(tup))
				# print(to_rule_expression(tup,backmap))
				# print(OperatorGraph(tup))


			
def apply_feature_set(state):
	if(isinstance(state, dict)):
		numerical_values = state_to_tensor(state)
	else:
		numerical_values = state

	d_len = [numerical_values.shape[0]] 
		
	numerical_values, d_len = forward_one(numerical_values, d_len,feature_set)

	indicies = (~torch.isnan(numerical_values)).nonzero()
	# print("d_len",d_len)
	# print("NUM RESULTS", indicies.shape)
	for tup in indicies_to_operator_graph(indicies,torch.tensor(d_len),feature_set):
		inps = rule_inputs(tup)
		if(len(set(inps)) == len(inps) and inps == sorted(inps)):
			# to_rule_expression
			# print(tup)
			print(repr_rule(tup,numerical_values))




			


		# print(indicies)
		# print(indicies.shape)


# operator_class_set = [Add,Mod10,Div10]
operator_class_set = [Add,Subtract]
operator_set = [c() for c in operator_class_set ]
operator_nf_inps = torch.tensor([x.num_flt_inputs for x in operator_set])

feature_class_set = [Equals]
feature_set = [c() for c in feature_class_set ]
	

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
# how_search(x,7, search_depth=3)
# 

example_state = {('contentEditable', '?ele-JCommTable8.R0C0'): True, ('contentEditable', '?ele-JCommTable.R1C0'): False, ('eq', ('value', '?ele-JCommTable.R0C0'), ('value', '?ele-JCommTable.R1C0')): False, ('value', '?ele-JCommTable4.R1C0'): '', ('id', '?ele-JCommTable8.R0C0'): 'JCommTable8.R0C0', ('id', '?ele-JCommTable.R0C0'): 'JCommTable.R0C0', ('eq', ('value', '?ele-JCommTable.R1C0'), ('value', '?ele-JCommTable6.R0C0')): False, ('value', '?ele-JCommTable.R0C0'): '1', ('contentEditable', '?ele-JCommTable6.R0C0'): False, ('id', '?ele-ctatdiv74'): 'ctatdiv74', ('value', '?ele-JCommTable6.R0C0'): '1', ('id', '?ele-JCommTable6.R0C0'): 'JCommTable6.R0C0', ('id', '?ele-JCommTable6.R1C0'): 'JCommTable6.R1C0', ('eq', ('value', '?ele-JCommTable.R1C0'), ('value', '?ele-JCommTable3.R0C0')): False, ('id', '?ele-JCommTable3.R1C0'): 'JCommTable3.R1C0', ('contentEditable', '?ele-JCommTable6.R1C0'): False, ('value', '?ele-JCommTable7.R0C0'): '*', ('contentEditable', '?ele-JCommTable3.R1C0'): False, ('contentEditable', '?ele-JCommTable4.R0C0'): True, ('value', '?ele-JCommTable8.R0C0'): '', ('value', '?ele-JCommTable6.R1C0'): '18', ('eq', ('value', '?ele-JCommTable.R1C0'), ('value', '?ele-JCommTable3.R1C0')): False, ('eq', ('value', '?ele-JCommTable3.R0C0'), ('value', '?ele-JCommTable6.R0C0')): True, ('eq', ('value', '?ele-JCommTable6.R0C0'), ('value', '?ele-JCommTable6.R1C0')): False, ('eq', ('value', '?ele-JCommTable.R1C0'), ('value', '?ele-JCommTable6.R1C0')): False, ('id', '?ele-JCommTable7.R0C0'): 'JCommTable7.R0C0', ('eq', ('value', '?ele-JCommTable.R0C0'), ('value', '?ele-JCommTable3.R1C0')): False, ('id', '?ele-ctatdiv87'): 'ctatdiv87', ('eq', ('value', '?ele-JCommTable3.R1C0'), ('value', '?ele-JCommTable6.R0C0')): False, ('contentEditable', '?ele-JCommTable4.R1C0'): True, ('contentEditable', '?ele-JCommTable3.R0C0'): False, ('id', '?ele-JCommTable4.R1C0'): 'JCommTable4.R1C0', ('id', '?ele-JCommTable5.R1C0'): 'JCommTable5.R1C0', ('eq', ('value', '?ele-JCommTable.R0C0'), ('value', '?ele-JCommTable6.R1C0')): False, ('contentEditable', '?ele-JCommTable5.R0C0'): True, ('id', '?ele-done'): 'done', ('eq', ('value', '?ele-JCommTable.R0C0'), ('value', '?ele-JCommTable3.R0C0')): True, ('contentEditable', '?ele-JCommTable5.R1C0'): True, ('value', '?ele-JCommTable2.R0C0'): '*', ('id', '?ele-JCommTable2.R0C0'): 'JCommTable2.R0C0', ('id', '?ele-ctatdiv68'): 'ctatdiv68', ('eq', ('value', '?ele-JCommTable3.R0C0'), ('value', '?ele-JCommTable3.R1C0')): False, ('value', '?ele-JCommTable5.R1C0'): '', ('id', '?ele-ctatdiv69'): 'ctatdiv69', ('value', '?ele-JCommTable.R1C0'): '3', ('eq', ('value', '?ele-JCommTable3.R0C0'), ('value', '?ele-JCommTable6.R1C0')): False, ('id', '?ele-JCommTable.R1C0'): 'JCommTable.R1C0', ('contentEditable', '?ele-JCommTable7.R0C0'): False, ('id', '?ele-JCommTable5.R0C0'): 'JCommTable5.R0C0', ('id', '?ele-JCommTable3.R0C0'): 'JCommTable3.R0C0', ('value', '?ele-JCommTable3.R1C0'): '6', ('contentEditable', '?ele-JCommTable.R0C0'): False, ('contentEditable', '?ele-JCommTable2.R0C0'): False, ('value', '?ele-JCommTable5.R0C0'): '', ('eq', ('value', '?ele-JCommTable.R0C0'), ('value', '?ele-JCommTable6.R0C0')): True, ('eq', ('value', '?ele-JCommTable3.R1C0'), ('value', '?ele-JCommTable6.R1C0')): False, ('id', '?ele-JCommTable4.R0C0'): 'JCommTable4.R0C0', ('id', '?ele-hint'): 'hint', ('value', '?ele-JCommTable4.R0C0'): '', ('value', '?ele-JCommTable3.R0C0'): '1'}

is_numerical = re.compile("[[0-9]*.[0-9]*]\.")


def expr_comparitor(fact,expr, mapping={}):
	if(isinstance(expr,dict)):
		if(isinstance(fact,dict)):
			if(not expr_comparitor(list(fact.keys())[0],list(expr.keys())[0],mapping)):
				return False
			if(not expr_comparitor(list(fact.values())[0],list(expr.values())[0],mapping)):
				return False
			return True
		else:
			return False
	if(isinstance(expr,tuple)):
		if(isinstance(fact,tuple) and len(fact) == len(expr)):
			for x,y in zip(fact,expr):
				# print(x,y)
				if(not expr_comparitor(x,y,mapping)):
					return False
			return True
		else:
			return False
	elif expr[0] == "?" and mapping.get(expr,None) != fact:
		mapping[expr] = fact
		return True
	elif(expr == fact):
		return True
	else:
		return False


def expression_matches(expression,state):
	for fact_expr,value in state.items():
		if(isinstance(expression,dict)):
			fact_expr = {fact_expr:value}

		mapping = {}
		if(expr_comparitor(fact_expr,expression,mapping)):
			yield mapping
		# if(len(fact) == len(expression)):
		# 	mapping = {}
		# 	for x,y in zip(fact,expression):
		# 		if y[0] == "?":
		# 			mapping[y] = x
		# 		elif(x != y):
		# 			mapping = None
		# 			break
		# 	if(mapping != None):
		# 		yield mapping

for x in expression_matches(('contentEditable','?ele'),example_state):
	print(x)

for x in expression_matches({('?property', '?ele'): 'JCommTable.R0C0'}, example_state):
	print(x)



# how_search(example_state, 7)
# 
print(state_to_tensors(example_state))

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

def DictVectWrapper(clf):
	def fun(x=None):
		if x is None:
			return CustomPipeline([('dict vect', DictVectorizer(sparse=False)),
								   ('clf', clf())])
		else:
			return CustomPipeline([('dict vect', DictVectorizer(sparse=False)),
								   ('clf', clf(**x))])

	return fun

tree = DictVectorizer(DecisionTreeClassifier)
print(tree)

X = [{"A":True,"B":True,"C":True,"D":True,},
	 {"A":True,"B":True,"C":True,"D":False},
	 {"A":True,"B":False,"C":False,"D":True},
	 {"A":True,"B":False,"C":True,"D":True},
	 {"A":True,"B":False,"C":False,"D":False},
	 {"A":False,"B":False,"C":False,"D":False},
	 ]
X_vec = [[1,1,1,1],
		 [1,1,1,0],
		 [1,0,0,1],
		 [1,0,1,1],
		 [1,0,0,0],
		 [0,0,0,0],
		 ]
Y = [1,1,2,2,3,4]


from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
	tree_ = tree.tree_
	feature_name = [
		feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
		for i in tree_.feature
	]
	print("def tree({}):".format(", ".join(feature_names)))

	def recurse(node, depth):
		indent = "  " * depth
		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = feature_name[node]
			threshold = tree_.threshold[node]
			print("{}if {} <= {}:".format(indent, name, threshold))
			recurse(tree_.children_left[node], depth + 1)
			print("{}else:  # if {} > {}".format(indent, name, threshold))
			recurse(tree_.children_right[node], depth + 1)
		else:
			print("{}return {}".format(indent, tree_.value[node]))

	recurse(0, 1)

# tree = DecisionTreeClassifier()
# print(tree.tree_.feature)
# tree.fit(X_vec,Y)

# tree_to_code(tree,[1,2,3,4])



# tree.fit(X,Y)
# apply_feature_set(example_state)




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

blehh_state = {('value', '?ele-1'): 1,('value', '?ele-2'): 2,('value', '?ele-3'): 4}

x = torch.FloatTensor([1,2,4])
backmap = [[('value', "?A"),('value', "?B"),('value', "?C")]]
ogs = []
for og in how_search(x,3,search_depth=2,backmap=backmap):
	print(og)
	ogs.append(og)

mapping = {"?foa0" : "?ele-1","?foa1" : "?ele-2","?foa2" : "?ele-3"}
print("COMPUTE",ogs[1].compute(mapping, blehh_state))

operator_set = [ogs[1]]
operator_nf_inps = torch.tensor([x.num_flt_inputs for x in operator_set])
print("----------------------------------")
og2 = next(how_search(x,3,search_depth=1,backmap=backmap))
print("COMPUTE",og2.compute(mapping, blehh_state))
	# print(og)





	

# torch.empty

# print(Add.num_flt_inputs)

# args = ['a','b','c']
# l = len(args)-1
# print([([1]*i + [-1] + (l-i)*[1]) for i,a in enumerate(args)])


