from rgbTohd5 import load_data
import time
import numpy as np 
import h5py
from matplotlib import pyplot as plt 
from PIL import Image

plt.rcParams['figure.figsize']=(5.0,4.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

train_x_orig, train_y, test_x_orig, test_y= load_data()


y_train=np.zeros((3,train_y.shape[1]))
for i in range(0,train_y.shape[1]):
	y_train[train_y[0][i]][i]=1

print(y_train.shape)
y_test=np.zeros((3,test_y.shape[1]))
for i in range(0,test_y.shape[1]):
	y_test[test_y[0][i]][i]=1

train_x_flatten=train_x_orig.reshape(train_x_orig.shape[0],-1).T 
test_x_flatten=test_x_orig.reshape(test_x_orig.shape[0],-1).T
train_x= train_x_flatten/255.
test_x= test_x_flatten/255.

print("train_x's shape: "+str(train_x.shape))
print("test_x's shape: "+str(test_x.shape))

def initialize_parameters(layer_dims):
	parameters={}
	
	L=len(layer_dims)
	for l in range(1,L):
		if l!=L-2:
			parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*(2/(np.sqrt(layer_dims[l-1])))#*0.01
		else:
			parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*(1/(np.sqrt(layer_dims[l-1])))
		parameters["b"+str(l)]=np.zeros((layer_dims[l],1))
	return parameters

def linear_forward(A,W,b):
	Z=np.dot(W,A)+b
	cache=(A,W,b)
	assert(Z.shape==(W.shape[0],A.shape[1]))
	return Z, cache

def relu(Z):
	A=np.maximum(0,Z)
	assert(A.shape==Z.shape)
	cache=Z
	return A,cache

def sigmoid(Z):
	A=1/(1+np.exp(-Z))
	cache=Z
	return A, cache

def softmax(Z):
    A=np.exp(Z)/(np.sum(np.exp(Z)))
    return A,Z

def tanh(z):
	#print("z ",z )
	A=2/(1+np.exp(-2*z)) -1
	cache=z
	return A,cache

def tanh_backwards(dA,cache):
	z=cache
	s=2/(1+np.exp(-2*z)) -1
	dz=dA*(1-np.power(s,2))
	return dz

def softmax_backward(dA,cache):
    Z=cache
    s=np.exp(Z)/(np.sum(np.exp(Z)))
    dZ=s-dA
    return dZ



def sigmoid_backwards(dA, cache):
	Z=cache
	s=1/(1+np.exp(-Z))
	dZ=dA*s*(1-s)
	return dZ

def relu_backwards(dA, cache):
	Z=cache
	dZ=np.array(dA,copy=True)
	dZ[Z<=0]=0
	return dZ

def compute_cost(AL,Y):
	m=Y.shape[1]
	cost=np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))/m
	cost=np.squeeze(cost)
	return -cost

def linear_activation_forward(A_prev,W,b,activation):
	if activation=="sigmoid":
		Z, linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=sigmoid(Z)
	elif activation=="relu":
		Z, linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=relu(Z)
	elif activation=="softmax":
		Z,linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=softmax(Z)
	elif activation=="tanh":
		Z, linear_cache=linear_forward(A_prev,W,b)
		A,activation_cache=tanh(Z)
	cache = ( linear_cache, activation_cache )
	return A,cache

def L_model_forward(X,parameters):
	caches=[]
	A=X
	L=len(parameters)//2
	for l in range(1,L-1):
		A_prev=A
		A,cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
		caches.append(cache)
	A,cache=linear_activation_forward(A,parameters["W"+str(L-1)],parameters["b"+str(L-1)],"tanh")
	caches.append(cache)
	AL,cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
	caches.append(cache)
	return AL,caches

def Linear_backward(dZ,cache):
	A_prev,W,b=cache
	m=A_prev.shape[1]
	dW=1./m*np.dot(dZ,A_prev.T)
	db=1./m*np.sum(dZ,axis=1,keepdims=True)
	dA_prev=np.dot(W.T,dZ)

	return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
	linear_cache,activation_cache=cache
	if activation=="relu":
		dZ=relu_backwards(dA,activation_cache)
		dA_prev,dW,db=Linear_backward(dZ,linear_cache)
	elif activation=="sigmoid":
		dZ=sigmoid_backwards(dA,activation_cache)
		dA_prev,dW,db=Linear_backward(dZ,linear_cache)
	elif activation=="softmax":
		dZ=softmax_backward(dA,activation_cache)
		dA_prev,dW,db=Linear_backward(dZ,linear_cache)
	elif activation=="tanh":
		dZ=tanh_backwards(dA,activation_cache)
		dA_prev,dW,db=Linear_backward(dZ,linear_cache)
	return dA_prev,dW,db

def L_model_backward(AL,Y,caches):
	L=len(caches)
	m=AL.shape[1]
	Y=Y.reshape(AL.shape)
	dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))
	current_cache=caches[L-1]
	grads={}
	grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]= linear_activation_backward(dAL,current_cache,"sigmoid")
	current_cache=caches[L-2]
	grads["dA"+str(L-2)],grads["dW"+str(L-1)],grads["db"+str(L-1)]= linear_activation_backward(grads["dA" + str(L-1)],current_cache,"tanh")
	for l in reversed(range(L-2)):

		current_cache=caches[l]
		dA_prev_temp, dW_temp, db_temp= linear_activation_backward(grads["dA" + str(l + 1)], current_cache,"relu")
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

def L_layer_model(X,Y,layer_dims,learning_rate=0.075,num_iteration=2500,print_cost=False):

	#np.random.seed(1)
	parameters=initialize_parameters(layer_dims)
	costs=[]
	for i in range(0,num_iteration):
		AL,caches=L_model_forward(X,parameters)
		#print("AL", AL);
		cost=compute_cost(AL,Y)
		grads=L_model_backward(AL,Y,caches)
		#print(grads)
		parameters=update_parameters(parameters,grads,learning_rate)
		#print("para ",parameters )
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()
	return parameters


layers_dims=[train_x.shape[0],20,7,5,3]
#print(train_x.shape[0])
parameters = L_layer_model(train_x, y_train, layers_dims, 0.05,2000, print_cost = True)

def predict(X, y, parameters):
   
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    
    
   
    probas, caches = L_model_forward(X, parameters)
    y_prediction=np.zeros((3,m))
    
    
    print(probas)
    for i in range(0, probas.shape[1]):
        if probas[0][i]>probas[1][i] and probas[0][i]>probas[2][i]:
            y_prediction[0][i]=1   
        elif probas[1][i]>probas[0][i] and probas[1][i]>probas[2][i]:
            y_prediction[1][i]=1
            
        elif probas[2][i]>probas[0][i] and probas[2][i]>probas[1][i]:
            y_prediction[2][i]=1
            
    print(y)
    print(y_prediction)
    print("Accuracy: "  + str(np.sum((y_prediction == y)/m)))
        
    return y_prediction