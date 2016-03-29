# Package imports
import numpy as np
import math

class NN:
    model = {}
    nn_hdim = None
    nn_input_dim = None
    nn_output_dim = None

    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01 # learning rate for gradient descent
    reg_lambda = 0.01 # regularization strength

    def __init__(self, nn_hdim, nn_input_dim, nn_output_dim):
        self.nn_hdim = nn_hdim
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim

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
        reg_lambda = self.reg_lambda
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
        epsilon = self.epsilon
        reg_lambda = self.reg_lambda
        nn_output_dim = self.nn_output_dim
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Gradient descent. For each batch...
        for i in range(0, num_passes):
            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)

            # avoid divide-by-zero due to exponentiation of really large negative value
            for j in range(num_examples):
                for k in range(nn_output_dim):
                    if exp_scores[j][k] == 0:
                        exp_scores[j][k] = 0.00001

            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            ## REMOVE LATER
            if math.isnan(probs[0][0]):
                with open('errorlog', 'wb') as f:
                    f.write("W1:")
                    f.write(repr(W1))
                    f.write("B1:")
                    f.write(repr(b1))
                    f.write("Z1:")
                    f.write(repr(z1))
                    f.write("A1:")
                    f.write(repr(a1))
                    f.write("Z2:")
                    f.write("W2:")
                    f.write(repr(W2))
                    f.write("B2:")
                    f.write(repr(b2))
                    f.write(repr(z2))
                    f.write("EXP_SCORES:")
                    f.write(repr(exp_scores))
                    f.write("EXP_SCORES:")
                    f.write(repr(probs))
                    f.close()

                return False

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

            ## REMOVE LATER
            if math.isnan(W1[0][0]) or math.isnan(b1[0][0]) or math.isnan(W2[0][0]) or math.isnan(b2[0][0]):
                with open('errorlog', 'wb') as f:
                    f.write("delta2:")
                    f.write(repr(delta2))
                    f.write("delta3:")
                    f.write(repr(delta3))

                    f.write("dW1:")
                    f.write(repr(dW1))
                    f.write("dB1:")
                    f.write(repr(db1))

                    f.write("dW2:")
                    f.write(repr(dW2))
                    f.write("dB2:")
                    f.write(repr(db2))

                    f.write("W1:")
                    f.write(repr(W1))
                    f.write("B1:")
                    f.write(repr(b1))
                    f.write("Z1:")
                    f.write(repr(z1))
                    f.write("A1:")
                    f.write(repr(a1))
                    f.write("Z2:")
                    f.write("W2:")
                    f.write(repr(W2))
                    f.write("B2:")
                    f.write(repr(b2))
                    f.write(repr(z2))
                    f.write("EXP_SCORES:")
                    f.write(repr(exp_scores))
                    f.write("EXP_SCORES:")
                    f.write(repr(probs))
                    f.close()

                return False
                     
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
              print("Loss after iteration " + repr(i) + ": " + repr(self.__calculate_loss__(X, y)))

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        self.model = model

        return True

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

    def probs(self, x):
        model = self.model
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs