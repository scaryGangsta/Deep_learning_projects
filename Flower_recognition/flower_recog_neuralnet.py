from rgbTohd5 import load_data
import time
import numpy as np
import h5py
#%matplotlib inline
from matplotlib import pyplot as  plt
#import scipy
from PIL import Image
#from scipy import ndimage

######## loading data and diaplaying an image ######

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y= load_data()


index = 10
plt.imshow(train_x_orig[index])
plt.show()
print ("y = " + str(train_y[0,index]) + ". It's a " +  " picture.")

########### REshaping the 3d matrix to 1d to make a feature vector (data preprocessing)

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255
test_x = test_x_flatten/255
train_y_temp=np.zeros((3,train_x_orig.shape[0]))
test_y_temp=np.zeros((3,test_x_orig.shape[0]))
print(train_x)
for i in range(train_y.shape[1]):
	train_y_temp[train_y[0][i]][i]=1
	#if(i==10):
	#	print(str(train_y_temp[train_y[0][i]][i])+"gelo")
for i in range(test_y.shape[1]):
	test_y_temp[test_y[0][i]][i]=1
#print(train_y_temp)

#print ("train_x's shape: " + str(train_x.shape))
#print ("test_x's shape: " + str(test_x.shape))
 
 ######## initializing parameters :symmetry braking
def initialize_paramenetrs(layer_dims):
	parameters={}
	#np.random.seed(1)
	L=len(layer_dims)
	for l in range(1,L):
		if l!=L-2:
			parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*(0.02/(np.sqrt(layer_dims[l-1])))#*0.01
		else:
			parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*(0.01/(np.sqrt(layer_dims[l-1])))
		parameters["b"+str(l)]=np.zeros((layer_dims[l],1))
	#print("parameters= ", parameters)
	return parameters


def linear_forward(A,w,b):
	Z=w.dot(A)+b
	#print("moving_forward ",Z)
	cache=(A,w,b)
	assert(Z.shape==(w.shape[0],A.shape[1]))
	return Z ,cache

def relu(Z):
	A=np.maximum(0,Z)
	assert(A.shape==Z.shape)
	cache=Z	
	return A,cache
def softmax(Z):
	#print("z  ",Z)
	
	A=np.exp(Z)/(np.sum(np.exp(Z)))

	return A,Z

def sigmoid(Z):
	A=1/(1+np.exp(-Z))
	cache=Z
	return A, cache

def tanh(z):
	#print("z ",z )
	A=2/(1+np.exp(-2*z)) -1
	cache=z
	return A,cache
def sigmoid_backwards(dA,cache):
	Z=cache
	s=1/(1+np.exp(-Z))

	dz=dA*s*(1-s)
	return dz
def relu_backwards(dA,cache):
	z=cache
	dz=np.array(dA,copy=True)
	dz[z<=0]=0
	return dz
def tanh_backwards(dA,cache):
	z=cache
	s=2/(1+np.exp(-2*z)) -1
	dz=dA*(1-np.power(s,2))
	return dz

def softmax_backward(dA,cache):
	Z=cache
	s=np.exp(Z)/(np.sum(np.exp(Z)))
	
	dz=s-dA
	#dz=dA*(s)*(1-s)
	return  dz

def linear_activation_forward(A_prev,W,b,activation):
	if activation=="tanh":
		Z, linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=tanh(Z)
	elif activation=="relu":
		Z,linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=relu(Z)
	elif activation=="softmax":
		Z,linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=softmax(Z)
	cache=(linear_cache,activation_cache)
	return A,cache

def L_model_forward(X,parameters):
	caches=[]
	A=X
	L=len(parameters)//2
	for l in range(1,L-1):
		A_prev=A
		A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters['b'+str(l)],"relu")
		caches.append(cache)
	A,cache=linear_activation_forward(A,parameters["W"+str(L-1)],parameters['b'+str(L-1)],"tanh")
	caches.append(cache)
	AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"softmax")
	caches.append(cache)
	return AL, caches

def compute_cost(AL,y):
	m=y.shape[1]
	#print("m ",m)
	cost=np.sum(np.multiply(np.log(AL),y))/m
	assert(cost.shape==())
	#print(cost)
	return -cost

def linear_backward(dZ,cache):
	A_prev,W,b=cache
	m=A_prev.shape[1]
	dW=1./m*np.dot(dZ,A_prev.T)
	db=1./m*np.sum(dZ,axis=1,keepdims=True)
	dA_prev=np.dot(W.T,dZ)

	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)

	return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
	linear_cache,activation_cache=cache
	if(activation=="relu"):
		dZ=relu_backwards(dA,activation_cache)
		dA_prev,dw,db=linear_backward(dZ,linear_cache)
	elif activation=="softmax":
		dZ=softmax_backward(dA,activation_cache)
		dA_prev,dw,db=linear_backward(dZ,linear_cache)
	elif activation=="tanh":
		dZ=tanh_backwards(dA,activation_cache)
		dA_prev,dw,db=linear_backward(dZ,linear_cache)

	return dA_prev,dw,db

def L_model_backward(AL,Y,caches):
	grads={}
	L=len(caches)
	m=AL.shape[1]
	#Y=Y.reshape(AL.shape)
	dAL=Y
	current_cache=caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax")
	current_cache=caches[L-2]
	grads["dA"+str(L-2)],grads["dW"+str(L-1)],grads["db"+str(L-1)]=linear_activation_backward(grads["dA"+str(L-1)],current_cache,"tanh")
	for l in reversed(range(L-2)):
	   
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads
 
def update_parameters(parameters, grads, learning_rate):
	
	L = len(parameters) // 2 
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
		
	return parameters

def L_layer_model(X,Y,layer_dims,learning_rate=1.75,num_iteration=2800,print_cost=False):
	
	parameters=initialize_paramenetrs(layer_dims)
	costs=[]
	for i in range(0,num_iteration):
		AL,caches=L_model_forward(X,parameters)
		#print("AL", AL);
		cost=compute_cost(AL,Y)
		grads=L_model_backward(AL,Y,caches)
		#print(grads)
		parameters=update_parameters(parameters,grads,learning_rate)
		#print("update_parameters= ",parameters)
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
			
	# plot the cost
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	
	return parameters

layers_dims=[train_x.shape[0],20,7,5,3]
#print(train_x.shape[0])
parameters = L_layer_model(train_x, train_y_temp, layers_dims, 2.00,2800, print_cost = True)
#parameters=initialize_paramenetrs(layers_dims)
print(parameters)

def predict(X, y, parameters):
	"""
	This function is used to predict the results of a  L-layer neural network.
	
	Arguments:
	X -- data set of examples you would like to label
	parameters -- parameters of the trained model
	
	Returns:
	p -- predictions for the given dataset X
	"""
	
	m = X.shape[1]
	n = len(parameters) // 2 # number of layers in the neural network
	p = np.zeros(y.shape)
	
	# Forward propagation
	probas, caches = L_model_forward(X, parameters)

	prediction=np.zeros(m,dtype=int)
	original=np.zeros(m,dtype=int)
	# convert probas to 0/1 predictions
	for i in range(0, probas.shape[1]):
		arg=np.argmax(probas[:,i])
		p[arg][i]=1
		prediction[i]=arg
		arg2=np.argmax(y[:,1])
		original[i]=arg2
	
	#print results
	#print ("predictions: " + str(p))
	#print ("true labels: " + str(y))


	print("Accuracy: "  + str(np.sum((prediction == original)/m)))
		
	return p

pred_train = predict(train_x, train_y_temp, parameters)

pred_test = predict(test_x, test_y_temp, parameters)
