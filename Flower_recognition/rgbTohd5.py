from random import shuffle
import glob
import cv2
import numpy as np 
import h5py

shuffle_data=True # shuffle the addresses
hdf5_path='C:\\Users\\amensodhi\\Desktop\\iris_flower_project\\f.hdf5' #file path for the created .hdf5  file
train_path='C:\\Users\\amensodhi\\Desktop\\iris_flower_project\\flowers\\expData\\**\\*.jpg' #the original data

# get all image paths
addrs= glob.glob("C:\\Users\\amensodhi\\Desktop\\iris_flower_project\\flowers\\expData\\**\\*.jpg")

#labeling the data as daiel=0 rose =1
labels = [0 if 'daisy' in addr else 1 if 'tulip' in addr else 2 for addr in addrs]

if shuffle_data:
	c= list(zip(addrs, labels)) # use zip() to bind the images and labels together
	shuffle(c)
	(addrs, labels)=zip(*c)# *c is used to separate all the tuples in the list c,
	                       #  "addrs" then contains all the shuffled paths and "labels" contains all the shuffled labels

#Dividing train and test data

train_addrs= addrs[0:int(0.8*len(addrs))]
train_labels= labels[0:int(0.8*len(labels))]

test_addrs=addrs[int(0.8*len(addrs)):]
test_labels=labels[int(0.8*len(labels)):]

#######################second part: create h5py object

train_shape=(len(train_addrs), 128, 128, 3)
test_shape=(len(test_addrs), 128, 128, 3)

#open a hdf5 file and create arrays
f=h5py.File(hdf5_path, mode='w')

f.create_dataset("train_img", train_shape, np.uint8)
f.create_dataset("test_img", test_shape, np.uint8)

f.create_dataset("train_labels", (len(train_addrs),),np.uint8)
f["train_labels"][...]= train_labels

f.create_dataset("test_labels", (len(test_addrs),), np.uint8)
f["test_labels"][...] = test_labels


########################third part: write the images
for i in range(len(train_addrs)):

	#if i%1000==0 and i>1:
	#	print('Train data: {}/{}'.format(i,len(train_addrs)))

	addr=train_addrs[i]
	img=cv2.imread(addr)
	img=cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
	img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	f["train_img"][i,...]=img[None]

for i in range(len(test_addrs)):

	#if i % 1000 == 0 and i > 1:
	#	print ('Test data: {}/{}'.format(i, len(test_addrs)) )

	addr = test_addrs[i]
	img = cv2.imread(addr)
	img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	f["test_img"][i, ...] = img[None]

f.close()

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
