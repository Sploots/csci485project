# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# Define the size of the board
board_dimension = 3

# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Generate a dataset
np.random.seed(0)
n_spaces = board_dimension*board_dimension
nn_input_dim = n_spaces # input layer dimensionality
nn_output_dim = n_spaces # output layer dimensionality

X = np.array([[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1,0]])
y = np.array([[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0.2,0.8,0],[0,0,0,0,0,0,0.7,0.3,0],[0,0,0,0,0,0,0.7,0.3,0]])

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

class NN:
    model = {}

    def __init__(self, nn_hdim):
        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))

        # Assign new parameters to the model
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


    # Helper function to evaluate the total loss on the dataset
    def __calculate_loss__(self, X, y):
        num_examples = len(X)

        model = self.model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = []

        # Calculating the loss
        for j in range(num_examples):
            example_probs = probs[j]
            correct_logprobs.append(np.multiply(-np.log(example_probs),y[j]))
        
        data_loss = np.sum(correct_logprobs)

        # Add regulatization term to loss (optional)
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./num_examples * data_loss

    # This function learns parameters for the neural network and returns the model.
    # - nn_hdim: Number of nodes in the hidden layer
    # - num_passes: Number of passes through the training data for gradient descent
    # - print_loss: If True, print the loss every 1000 iterations
    def train(self, X, y, num_passes=20000, print_loss=False):
        num_examples = len(X)

        model = self.model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Gradient descent. For each batch...
        for i in range(0, num_passes):
            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            for j in range(num_examples):
                example_probs = probs[j]
                for k in range(nn_output_dim):
                    example_probs[k] -= y[j][k]

            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2
                     
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
              print("Loss after iteration " + repr(i) + ": " + repr(self.calculate_loss(X, y)))

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        self.model = model

    # Helper function to predict an output (0 or 1)
    def predict(self, x):
        model = self.model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return np.argmax(probs, axis=1)

# Build a model with a n-dimensional hidden layer
model = build_model(3, print_loss=True)

print(predict(model,X[0]))
print(predict(model,X[1]))
print(predict(model,X[2]))
print(predict(model,X[3]))
print(predict(model,X[4]))
print("done!")
