import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.linear1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.linear2 = FullyConnectedLayer(hidden_layer_size, n_output)        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        
        self.params()['W1'].grad = np.zeros_like(self.params()['W1'].grad)
        self.params()['B1'].grad = np.zeros_like(self.params()['B1'].grad)
        self.params()['W2'].grad = np.zeros_like(self.params()['W2'].grad)
        self.params()['B2'].grad = np.zeros_like(self.params()['B2'].grad)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        X = self.linear1.forward(X)
        X = self.relu1.forward(X)
        X_out = self.linear2.forward(X)

        loss, probs = softmax_with_cross_entropy(X_out, y)
        grad_out = self.linear2.backward(probs)
        grad_out = self.relu1.backward(grad_out)
        grad_out = self.linear1.backward(grad_out)

        W1 = self.params()['W1']
        B1 = self.params()['B1']
        W2 = self.params()['W2']
        B2 = self.params()['B2']

        l2_W1_loss, l2_W1_grad = l2_regularization(W1.value, self.reg)
        l2_B1_loss, l2_B1_grad = l2_regularization(B1.value, self.reg)
        l2_W2_loss, l2_W2_grad = l2_regularization(W2.value, self.reg)
        l2_B2_loss, l2_B2_grad = l2_regularization(B2.value, self.reg)

        l2_loss = l2_W1_loss + l2_W2_loss + l2_B1_loss + l2_B2_loss
        loss += l2_loss

        W1.grad += l2_W1_grad
        B1.grad += l2_B1_grad
        W2.grad += l2_W2_grad
        B2.grad += l2_B2_grad
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], int)
        
        output = self.linear1.forward(X)
        output = self.relu1.forward(output)
        output = self.linear2.forward(output)
        
        probs = softmax(output)
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        result = {
            'W1': self.linear1.W,
            'B1': self.linear1.B,
            'W2': self.linear2.W,
            'B2': self.linear2.B
        }

        return result
