import numpy as np
from tqdm.notebook import tqdm
from models.neural_net import NeuralNetwork
input_size, output_size, mapping_size = 2, 3, 256

# Create the mappings dictionary of matrix B -  you will implement this
def get_B_dict():
    B_dict = {}
    B_dict['none'] = None
    B_dict["basic"] = np.eye(input_size)
    B_dict["gauss_1.0"] = np.random.normal(0,1., (input_size, mapping_size))
    B_dict["gauss_10.0"] = np.random.normal(0,10., (input_size, mapping_size))
    B_dict["gauss_100.0"] = np.random.normal(0,100., (input_size, mapping_size))
    return B_dict

# Given tensor x of input coordinates, map it using B - you will implement
def input_mapping(x, B):
    # x shape is (#,input_size), row vector
    # B shape is (input_size,mapping_size)
    # output is (#,mapping_size) row vector again (nn also follows this)
    if B is None: # "none" mapping - just returns the original input coordinates
        return x
    else: # "basic" mapping and "gauss_X" mappings project input features using B
        twopi = 2*np.pi
        return np.block([np.cos(twopi*x@B),np.sin(twopi*x@B)])
    
# Apply the input feature mapping to the train and test data - already done for you
def get_input_features(B, train_data, test_data):
    # mapping is the key to the B_dict, which has the value of B
    # B is then used with the function `input_mapping` to map x  
    y_train = train_data[1].reshape(-1, output_size)
    y_test = test_data[1].reshape(-1, output_size)
    X_train = input_mapping(train_data[0].reshape(-1, input_size), B)
    X_test = input_mapping(test_data[0].reshape(-1, input_size), B)
    return X_train, y_train, X_test, y_test

# helper function for dynamic learning rate, for SGD
def learning_rate_list(epochs, learning_rate, formula, k):
        epochsi = np.arange(epochs)
        if (formula == "none"):
            return learning_rate*np.ones(epochs)
        if (formula == "exp"):
            return learning_rate*np.exp(-k*epochsi)
        if (formula == "inverse"):
            return learning_rate/(1+k*epochsi)
        if (formula == "inverse_sqrt"):
            return learning_rate/np.sqrt(epochsi)
        if (formula == "linear"):
            return learning_rate*(1-epochsi/epochs)
        if (formula == "cosine"):
            return (learning_rate/2)*(1+np.cos(epochsi*np.pi/epochs))

def NN_experiment(X_train, y_train, X_test, y_test, num_layers, hidden_size, epochs, learning_rate, formula, k, opt, batch_size):
    # Initialize a new neural network model
    hidden_sizes = [hidden_size] * (num_layers - 1)
    net = NeuralNetwork(X_train.shape[1], hidden_sizes, y_train.shape[1], num_layers, opt)
    number_batches = int(X_train.shape[0]/batch_size)

    # Variables to store performance for each epoch
    train_loss = np.zeros(epochs)
    train_psnr = np.zeros(epochs)
    test_psnr = np.zeros(epochs)
    predicted_images = np.zeros((epochs, y_test.shape[0], y_test.shape[1]))
    
    # For each epoch...
    epochsi = learning_rate_list(epochs, learning_rate, formula, k)
    for epoch in tqdm(range(epochs)):
        lr = epochsi[epoch]
        shuffled_indices = np.random.permutation(X_train.shape[0])
        curr_train_psnr, curr_train_loss = 0,0
        for i in range(number_batches):
            batch_shuffled_indices = shuffled_indices[i*batch_size:(i+1)*batch_size]
            X = X_train[batch_shuffled_indices]
            y = y_train[batch_shuffled_indices]
            predicted = net.forward(X)
            curr_train_psnr += psnr(y, predicted)
            curr_train_loss += net.backward(y)
            net.update(lr = lr)
        train_loss[epoch], train_psnr[epoch] = curr_train_loss/number_batches, curr_train_psnr/number_batches
        
        predicted = net.forward(X_test)
        test_psnr[epoch] = psnr(y_test, predicted)
        predicted_images[epoch] = predicted
    return net, train_psnr, test_psnr, train_loss, predicted_images

def mse(y, p):
    return np.mean((y-p)**2)

def psnr(y, p):
    return -10*np.log10(mse(y,p))
