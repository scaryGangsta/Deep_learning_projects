from rgbTohd5 import load_data
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage





train_x_orig, train_y, test_x_orig, test_y= load_data()


######reshaping the 3d matrix

#print(train_x_orig.shape)

train_x_flatten=train_x_orig.reshape(train_x_orig.shape[0],-1).T 
test_x_flatten=test_x_orig.reshape(test_x_orig.shape[0],-1).T
train_x= train_x_flatten/255.
test_x= test_x_flatten/255.

print("train_x's shape: "+str(train_x.shape))
print("test_x's shape: "+str(test_x.shape))

################code part:defining fxns
def sigmoid(z):
	return 1/(1+np.exp(-z))

def initialize_parameters(dim):
	w=np.zeros((dim,1))
	b=0
	return w,b

def propagate(w,b,X,Y):
	m=Y.shape[1]
	#z calculation:
	Z=np.dot(w.T,X)+b
	A=sigmoid(Z)
	#cost calcutlation
	cost=-1*np.sum(np.multiply(Y,np.log(A))+np.multiply(1-Y,np.log(1-A)))/m
	#back_propagation
	dw=1/m*(np.dot(X,(A-Y).T))
	db=np.sum(A-Y)/m

	grads={"dw":dw,
		   "db":db}

	return grads,cost

def optimize(w,b,X,Y,num_iterations,alpha,print_cost=False):
	costs=[]

	for i in range(num_iterations):
		grads,cost=propagate(w,b,X,Y)

		dw=grads["dw"]
		db=grads["db"]

		w=w-alpha*dw
		b=b-alpha*db

		if i%100==0:
			costs.append(cost)

		if print_cost and i%100==0:
			print("cost after iteration %i: %f"%(i,cost))

	params={"w":w,
			"b":b}

	grads={"dw":dw,
		   "db":db}

	return params,grads,cost

def predict(w, b, X):
	
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 3)
	
	# Compute vector "A" predicting the probabilities of a cat being present in the picture
	### START CODE HERE ### (≈ 1 line of code)
	A = sigmoid(np.dot(w.T,X)+b)
	### END CODE HERE ###
	
	for i in range(A.shape[1]):
		
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		### START CODE HERE ### (≈ 4 lines of code)
		if A[0][i]>A[1][i] and A[0][i]>A[2][i]:
			Y_prediction[0][i]=0
		   
		elif A[1][i]>A[0][i] and A[1][i]>A[2][i]:
			Y_prediction[0][i]=1
		
		elif A[2][i]>A[0][i] and A[2][i]>A[1][i]:
			Y_prediction[0][i]=2
		
			
				
		### END CODE HERE ###
	
	assert(Y_prediction.shape == (1, m))
	
	return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_iterations=3000,learning_rate=0.5,print_cost=False):
	w1=np.zeros((3,X_train.shape[0]))
	b1=np.zeros((3,1))
	costs=[]
	for i in range(3):

		w,b=initialize_parameters(X_train.shape[0])
		train_y=1*(Y_train==i)

		parameters,grads,cost=optimize(w,b,X_train,train_y,num_iterations,learning_rate,print_cost)

		w=parameters["w"]
		b=parameters["b"]

		w1[i]=w.T
		b1[i]=b
		costs.append(cost)

	w2=w1.T
	Y_prediction_test=predict(w2,b1,X_test)
	Y_prediction_train=predict(w2,b1,X_train)


	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	d = {"costs": costs,
		 "Y_prediction_test": Y_prediction_test, 
		 "Y_prediction_train" : Y_prediction_train, 
		 "w" : w, 
		 "b" : b,
		 "learning_rate" : learning_rate,
		 "num_iterations": num_iterations}
	
	return d

d = model(train_x, train_y, test_x, test_y, num_iterations = 3000, learning_rate = 0.005, print_cost = True)


index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
