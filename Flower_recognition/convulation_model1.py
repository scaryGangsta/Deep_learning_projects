%tensorflow_version 1.x
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops

%matplotlib inline
hdf5_path="/content/f.hdf5"

def load_data():

	train_dataset = h5py.File(hdf5_path, mode="r")
	train_set_x_orig = np.array(train_dataset["train_img"][:]) 
	train_set_y_orig = np.array(train_dataset["train_labels"][:]) # your train set labels

	test_set_x_orig = np.array(train_dataset["test_img"][:]) # your test set features
	test_set_y_orig = np.array(train_dataset["test_labels"][:]) # your test set labels
   # classes = np.array(test_dataset["list_classes"][:]) # the list of classes
	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))




	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig=load_data()


index=6
plt.imshow(X_train_orig[index])
print("y = "+str(np.squeeze(Y_train_orig[:,index])))


X_train=X_train_orig/255
X_test=X_test_orig/255
Y_train=np.eye(3)[Y_train_orig.reshape(-1)]
Y_test=np.eye(3)[Y_test_orig.reshape(-1)]
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

def create_placeholders(n_H0,n_W0,n_C0,n_y):
  X=tf.placeholder(tf.float32,(None,n_H0,n_W0,n_C0),name='X')
  Y=tf.placeholder(tf.float32,(None,n_y),name='Y')
  return X,Y

def initialize_parameters():
  tf.set_random_seed(1)
  W1=tf.get_variable("W1",[8,8,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
  W2=tf.get_variable("W2",[4,4,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
  W3=tf.get_variable("W3",[2,2,16,32],initializer=tf.contrib.layers.xavier_initializer(seed=0))
  parameters={"W1":W1,
              "W2":W2,
              "W3":W3}
  return parameters      

def forward_propagation(X,parameters):
  W1=parameters['W1']
  W2=parameters['W2']
  W3=parameters['W3']

  Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
  A1=tf.nn.relu(Z1)
  P1=tf.nn.max_pool(A1,ksize=[1,16,16,1],strides=[1,16,16,1],padding='SAME')

  Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
  A2=tf.nn.relu(Z2)
  P2=tf.nn.max_pool(A2,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')

  Z3=tf.nn.conv2d(P2,W3,strides=[1,1,1,1],padding='SAME')
  A3=tf.nn.relu(Z3)
  P3=tf.nn.max_pool(A3,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

  F=tf.contrib.layers.flatten(P2)

  Z4=tf.contrib.layers.fully_connected(F,3,activation_fn=None)

  return Z4


def compute_cost(Z4,Y):
  cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z4,labels=Y))
  return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model(X_train,Y_train,X_test,Y_test,learning_rate=0.007,num_epochs=100,minibatch_size=64,print_cost=True):
  
  ops.reset_default_graph()
  tf.set_random_seed(1)
  seed=3
  (m,n_H0,n_W0,n_C0)=X_train.shape
  n_y=Y_train.shape[1]
  costs=[]

  X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
  parameters=initialize_parameters()
  Z4=forward_propagation(X,parameters)
  cost=compute_cost(Z4,Y)
  optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  init=tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):

      minibatch_cost=0
      num_minibatches=int(m/minibatch_size)
      seed=seed+1
      minibatches=random_mini_batches(X_train,Y_train,minibatch_size,seed)

      for minibatch in minibatches:
        (minibatch_x,minibatch_y)=minibatch

        _,temp_cost=sess.run(fetches=[optimizer,cost],feed_dict={X:minibatch_x,Y:minibatch_y})
        minibatch_cost+=temp_cost/num_minibatches
      
      if print_cost==True and epoch%5==0:
        print("cost after epoch %i: %f"%(epoch,minibatch_cost))
      if print_cost==True and epoch%1==0:
        costs.append(minibatch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    predict_op=tf.argmax(Z4,1)
    correct_prediction=tf.equal(predict_op,tf.argmax(Y,1))
      
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
                
    return train_accuracy, test_accuracy, parameters


tf.reset_default_graph()
_, _, parameters = model(X_train, Y_train, X_test, Y_test)
