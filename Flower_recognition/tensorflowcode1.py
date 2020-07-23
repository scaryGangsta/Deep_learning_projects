from rgbTohd5 import load_data
import math
import numpy as np 
import h5py
import matplotlib.pyplot as plt 
import tensorflow as tf 



train_x_orig, train_y_orig, test_x_orig, test_y_orig= load_data()

index=0
plt.imshow(train_x_orig[index])
print("y = "+str(np.squeeze(train_y_orig[:,index])))

X_train_flatten=train_x_orig.reshape(train_x_orig.shape[0],-1).T
X_test_flatten=test_x_orig.reshape(test_x_orig.shape[0],-1).T

X_train=X_train_flatten/255
X_test=X_test_flatten/255

Y_train=np.eye(3)[train_y_orig.reshape(-1)].T
Y_test=np.eye(3)[test_y_orig.reshape(-1)].T

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


def create_placeholders(n_x,n_y):
	X=tf.placeholder(tf.float32,(n_x,None),name='X')
	Y=tf.placeholder(tf.float32,(n_y,None),name='Y')
	return X,Y

def initialize_parameters(layer_dims):
	parameters={}
	L=len(layer_dims)
	for l in range(1,L):
		parameters["W"+str(l)]=tf.get_variable("W"+str(l),[layer_dims[l],layer_dims[l-1]],initializer=tf.contrib.layers.xavier_initializer())
		parameters["b"+str(l)]=tf.get_variable("b"+str(l),[layer_dims[l],1],initializer=tf.zeros_initializer())
	return parameters

def forward_propagation(X,parameters):
	A=X
	L=len(parameters)//2
	for l in range(1,L):
		A_prev=A
		Z=tf.add(tf.matmul(parameters["W"+str(l)],A_prev),parameters["b"+str(l)])
		A=tf.nn.relu(Z)
	ZL=tf.add(tf.matmul(parameters["W"+str(L)],A),parameters["b"+str(L)])
	return ZL

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])
    parameters = initialize_parameters([X_train.shape[0],20,7,5,Y_train.shape[0]])
    ZL = forward_propagation(X, parameters)
    print("ZL = " + str(ZL))

def compute_cost(ZL,Y):
	logits=tf.transpose(ZL)
	labels=tf.transpose(Y)

	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))

	return cost
def reg_sum(parameters):
  reg=0
  L=len(parameters)//2
  for l in range(1,L+1):
    reg+=tf.reduce_sum(parameters["W"+str(l)])
  return reg

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(X_train.shape[0], Y_train.shape[0])
    parameters = initialize_parameters([X_train.shape[0],20,7,5,Y_train.shape[0]])
    ZL = forward_propagation(X, parameters)
    cost = compute_cost(ZL, Y)
    print("cost = " + str(cost))

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
 
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model(X_train,Y_train,X_test,Y_test,learning_rate=0.0001,num_epochs=2000,minibatch_size=32,print_cost=True):
	tf.reset_default_graph()
	(n_x, m)=X_train.shape
	n_y=Y_train.shape[0]
	costs=[]

	seed=1
	layer_dims=np.array([n_x,20,7,5,n_y])

	X,Y=create_placeholders(n_x,n_y)

	parameters=initialize_parameters(layer_dims)

	ZL=forward_propagation(X,parameters)

	cost=compute_cost(ZL,Y)
	reg=reg_sum(parameters=parameters)
	cost=tf.add(cost,0.0001*reg/m)


	optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	init=tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(num_epochs):
			epoch_cost=0
			num_minibatches=int(m/minibatch_size)
			sees=seed+1
			minibatches=random_mini_batches(X_train,Y_train,minibatch_size,seed)

			for minibatch in minibatches:
				(minibatch_X,minibatch_Y)=minibatch
				_,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
				epoch_cost=minibatch_cost/minibatch_size

			if print_cost == True and epoch % 100 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)
		# plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per fives)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()

		# lets save the parameters in a variable
		parameters = sess.run(parameters)
		print ("Parameters have been trained!")

		correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

		# Calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

		return parameters
		
parameters = model(X_train, Y_train, X_test, Y_test)
