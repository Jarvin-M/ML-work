# Deep MNIST architecture for object recognition

### Description of files in the repository

* network.py -  contains an implementation of the network architecture and can be tested by running python network.py. If the architecture is correctly configured then this should run smoothly.

* split.py - can be used to split the dataset into training and testing data. The final output of the split.py are .npy test train files and their corresponding labels. This script can be executed with the command python split.py. Make sure ot correctly specify the location of the data folder, and it assumes that a 'numpy' folder already exists, so we also need to create it beforehand. This is where all the '.npy' files will be stored.

* preprocess.py -  performs a couple of preprocessing steps on the image before it is split into training and testing set. Make sure that the dimensions of the image mentioned in preprocess.py and split.py are exactly the same, otherwise it will throw an error.

* train.py - trains the network architecture defined in network.py on the data created from the execusion of split.py[make sure to correctly specify the path to the data in this script]. Can be executed with the command python train.py. Checkpointing has not been implemented yet. I will push an update in a couple of days. Ideally, you should create checkpoints, so the weights can be retrieved later on for classifying new images.

* test.py - can be used to test the trained model and it's performance on the new datatset.
