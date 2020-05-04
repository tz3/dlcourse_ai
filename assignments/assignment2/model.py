import numpy as np

from layers import FullyConnectedLayer, ReLULayer

from assignments.assignment2.layers import softmax_with_cross_entropy


def f(x):
    return np.exp(x) / np.sum(np.exp(x))


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
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layer_size = hidden_layer_size
        # TODO Create necessary layers
        self.layers = (
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        )

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        assert X.shape[0] == y.shape[0]
        for _name, param in self.params().items():
            param.reset_grad()

        # forward pass
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        # backward pass
        loss, grad = softmax_with_cross_entropy(out, y)
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

        # raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients

        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

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
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for index, layer in enumerate(self.layers):
            for name, param in layer.params().items():
                result['%s_%s' % (index, name)] = param

        return result
