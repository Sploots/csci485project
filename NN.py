# Package imports
import numpy as np
import math
import copy

np.random.seed(0)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class NN:
    model = {}
    act_fxn = None
    nn_hdim = None
    nn_input_dim = None
    nn_output_dim = None

    # Gradient descent parameters
    epsilon = 0.01 # learning rate for gradient descent
    reg_lambda = 0.01 # regularization strength

    def __init__(self, nn_hdim, nn_input_dim, nn_output_dim, act_fxn="sigmoid"):
        self.nn_hdim = nn_hdim
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.act_fxn = act_fxn

        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))

        # Assign new parameters to the model
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


    # Evaluate the total loss on the dataset
    def __calculate_loss__(self, X, y):
        num_examples = len(X)

        model = self.model
        nn_output_dim = self.nn_output_dim
        reg_lambda = self.reg_lambda
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        if self.act_fxn == "sigmoid":
            a1 = sigmoid(z1)
        else:
            a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = self.__safeexp__(z2, num_examples)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = []

        # Calculating the loss
        for j in range(num_examples):
            example_probs = copy.deepcopy(probs[j])
            for k in range(nn_output_dim):
                if example_probs[k] == 0:
                    example_probs[k] = 0.00001

                example_probs[k] = -math.log(example_probs[k])*y[j][k]
            
            correct_logprobs.append(example_probs)
        
        data_loss = np.sum(correct_logprobs)

        # Add regulatization term to loss (optional)
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./num_examples * data_loss

    # This function learns parameters for the neural network and returns the model.
    # - nn_hdim: Number of nodes in the hidden layer
    # - num_passes: Number of passes through the training data for gradient descent
    # - print_loss: If True, print the loss every 100 training passes
    def train(self, X, y, num_passes=20000, print_loss=False, verbose=False):
        num_examples = len(X)

        model = self.model
        epsilon = self.epsilon
        reg_lambda = self.reg_lambda
        nn_output_dim = self.nn_output_dim
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Gradient descent. For each batch...
        for i in range(0, num_passes):
            # Forward propagation
            z1 = X.dot(W1) + b1

            if self.act_fxn == "sigmoid":
                a1 = sigmoid(z1)
            else:
                a1 = np.tanh(z1)

            z2 = a1.dot(W2) + b2

            exp_scores = self.__safeexp__(z2, num_examples)

            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            for j in range(num_examples):
                example_probs = probs[j]
                for k in range(nn_output_dim):
                    example_probs[k] -= y[j][k]

            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)

            if self.act_fxn == "sigmoid":
                delta2 = delta3.dot(W2.T) * a1*(1 - a1)
            else:
                delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2
                     
            # print the loss
            if print_loss and i % 100 == 0:
              print("Loss after iteration " + repr(i) + ": " + repr(self.__calculate_loss__(X, y)))

        if print_loss:
            print("Loss after iteration " + repr(i) + ": " + repr(self.__calculate_loss__(X, y)))

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        self.model = model

        return True

    # predict a class (one of the output dimensions)
    def predict(self, x):
        model = self.model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation
        z1 = x.dot(W1) + b1
        if self.act_fxn == "sigmoid":
            a1 = sigmoid(z1)
        else:
            a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = self.__safeexp__(z2, len(z2))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return np.argmax(probs, axis=1)

    # predict probabilities for each output dimension (class)
    def probs(self, x):
        model = self.model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation
        z1 = x.dot(W1) + b1
        if self.act_fxn == "sigmoid":
            a1 = sigmoid(z1)
        else:
            a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = self.__safeexp__(z2, len(z2))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def __safeexp__(self, z2, num_examples):
        nn_output_dim = self.nn_output_dim

        # avoid overflow
        for j in range(num_examples):
            for k in range(nn_output_dim):
                if z2[j][k] > 100:
                    z2[j][k] = 100

        exp_scores = np.exp(z2)

        # avoid divide-by-zero due to exponentiation of really large negative value
        for j in range(num_examples):
            for k in range(nn_output_dim):
                if exp_scores[j][k] == 0:
                    exp_scores[j][k] = 0.00001

        return exp_scores