"""Neural network model."""
from typing import Sequence
import numpy as np

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has N inputs of size D,
    a hidden layer dimension of H, and output dimension C. 
    We train the network with a mean squared error (MSE) loss function. 
    The network uses a ReLU nonlinearity after each FC layer but the last one.
    The outputs of the last fully-connected layer are passed through a sigmoid. """
    # NeuralNetwork(input_size, hidden_sizes, output_size, num_layers, opt)
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...  H_k == C
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network 
            opt: SGD or Adam, one chosen this neural network can only do such update """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt

        #fully connected layers counts last one too which isn't hidden
        assert len(hidden_sizes) == (num_layers - 1)
        # len(sizes) = len(hidden_sizes) + 2 == num_layers + 1 
        sizes = [input_size] + hidden_sizes + [output_size]

        # keys are string parameter names and values are numpy arrays
        # need to store the 1st and 2nd moment of the parameters for Adam
        self.params, self.m, self.v, self.t = {}, {}, {}, 0
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.m["W" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
            self.v["W" + str(i)] = np.zeros((sizes[i - 1], sizes[i]))
            
            self.params["b" + str(i)] = np.zeros(sizes[i])
            self.m["b" + str(i)] = np.zeros(sizes[i])
            self.v["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix, (D, H) -> (D+1, H)
            X: the input data, (N or 1, D) -> (N or 1, D+1)
            b: the bias, (H, )
        Returns:
            the output  
            TODO be careful as it could be ROW or column vectors
                and the biases need to be well aligned, sum over correct axis """
        return X@W + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output """
        return np.maximum(X,0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray: #stackoverflow.com/a/51976485
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    # y shape is (N, output_size) 
    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # we take the mean across the second axis
        return np.mean((y-p)**2)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or testing sample
        Returns:
            Matrix of shape (N, C) """
        self.outputs, num_layers, self.outputs["h0"] = {}, self.num_layers, X
        # There are #num_layers W, #num_layers-1 hidden outputs and 1 final output
        for i in range(1, num_layers + 1):
            # Getting appropriate W matrix and bias
            W, b = self.params["W" + str(i)], self.params["b" + str(i)]
            # Saving pre-relu values and updating hidden vector for next layer
            X = self.outputs["h" + str(i)] = self.linear(W,X,b)
            # Otherwise we are at the last layer and don't need to relu
            if (i != num_layers):
                X = self.relu(X)
        return self.sigmoid(X)

    def backward(self, y: np.ndarray) -> float:
        N,D,num_layers,outputs=y.shape[0],self.input_size,self.num_layers,self.outputs
        self.gradients,s,C={},[D]+self.hidden_sizes+[self.output_size],self.output_size
        for i in range(1, num_layers + 1):
            self.gradients["W" + str(i)] = np.zeros((s[i - 1], s[i]))
            self.gradients["b" + str(i)] = np.zeros((s[i]))
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets, (N, output_size), (N, C)
            So that N is the batch size, average over these gradients
        Returns:
            Total loss for this batch of training samples
            TODO: Total so may need to sum over the N vector """
        predicted = self.sigmoid(outputs["h" + str(num_layers)])  #shape (N, C)
        negatived =self.sigmoid(-outputs["h" + str(num_layers)])  #shape (N, C)
        e_p = (2/C)*(predicted-y)      #shape (N, C)
        
        p_hk = predicted*negatived     #shape (N, C)
        e_hk = e_p*p_hk                #shape (N, C)
        # Now we loop over the N datapoints because vectorizing that gives 4D matrices
        for n in range(N):
            curr_e_hk = e_hk[n] #(1,C) == (1, H_k)
            for k in range(1, num_layers + 1)[::-1]:
                W,b=self.params["W"+str(k)],self.params["b"+str(k)] 
                # (H_k-1, H_k), (1, H_k), (1, H_k-1), (1, H_k). h_k = rh_k-1@W + b
                rhk_1,hk=self.relu(outputs["h"+str(k-1)][n]), outputs["h"+str(k)][n]
                if (k != num_layers): # as seen in class we do element wise
                    curr_e_hk = curr_e_hk * (hk>0)
                # Dividing over N as we are taking the mean gradient
                self.gradients["W" + str(k)] += np.outer(rhk_1,curr_e_hk)/N
                self.gradients["b" + str(k)] += curr_e_hk/N
                curr_e_hk = curr_e_hk@W.T #(1,H_k-1) now, but (1,H_k) on next
        return self.mse(y, predicted)

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ):
        # Timestep increases per update, but should be the same across all weights
        self.t += 1
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
        """
        if (self.opt == "SGD"):
            for i in range(1, self.num_layers + 1):
                for param in ["W", "b"]:
                    self.params[param + str(i)] -= lr*self.gradients[param + str(i)]
        elif (self.opt == "Adam"):
            for i in range(1, self.num_layers + 1):
                for param in ["W", "b"]:
                    self.m[param + str(i)] = b1*self.m[param + str(i)] + (1-b1)*self.gradients[param + str(i)]
                    self.v[param + str(i)] = b2*self.v[param + str(i)] + (1-b2)*self.gradients[param + str(i)]**2
                    m_bias_corr = self.m[param + str(i)]/(1-b1**self.t)
                    v_bias_corr = self.v[param + str(i)]/(1-b2**self.t)
                    self.params[param + str(i)] -= lr*m_bias_corr/(np.sqrt(v_bias_corr) + eps)
