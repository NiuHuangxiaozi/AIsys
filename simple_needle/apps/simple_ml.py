"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
import sys
import python.needle as ndl

def Deserialization(filename):
    #  read the gz file
    images = None
    with gzip.open(filename, "rb") as f:
        magic_number = f.read(4)
        _, data_type, num_dimensions = struct.unpack('>HBB', magic_number)

        dimension_sizes = []
        for _ in range(num_dimensions):
            dim_size = struct.unpack('>I', f.read(4))[0]
            dimension_sizes.append(dim_size)

        #
        data_type_map = {
            0x08: ('B', 1),
            0x09: ('b', 1),
            0x0B: ('h', 2),
            0x0C: ('i', 4),
            0x0D: ('f', 4),
            0x0E: ('d', 8)
        }

        if data_type not in data_type_map:
            raise ValueError(f"Unsupported data type")
        dtype, size = data_type_map[data_type]
        total_elements = int(size * np.prod(dimension_sizes))

        images = struct.unpack(f'>{dtype * np.prod(dimension_sizes)}', f.read(total_elements))

        return np.array(images).reshape(dimension_sizes)

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    images = Deserialization(image_filesname).astype(np.float32)

    # shape : (60000,28,28) -> (60000, 784)
    images = images.reshape(images.shape[0], -1)

    labels = Deserialization(label_filename).astype(np.uint8)
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    return (images, labels)
    ### END YOUR SOLUTION



def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size, _ = Z.shape
    Z_exp=ndl.exp(Z)
    Z_sum=ndl.summation(Z_exp,axes=(1))

    Z_sum_log=ndl.log(Z_sum)

    mask_Z=y_one_hot * Z
    Zy=ndl.summation(mask_Z,axes=(1))
    # print(f"Z_sum_log={Z_sum_log.shape}")
    # print(f"Zy={Zy.shape}")
    # print(f"Z_sum_log-Zy is {ndl.summation(Z_sum_log-Zy)}")
    loss_sum=ndl.summation(Z_sum_log-Zy)

    average_loss_sum=loss_sum/batch_size

    return average_loss_sum
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        #  transfer it into needle.Tensor
        X_batch_Tensor=ndl.Tensor(X_batch)
        y_batch_Tensor=ndl.Tensor(y_batch)


        # calculate the gradient of the W2
        forward_val = ndl.matmul(ndl.relu(ndl.matmul(X_batch_Tensor ,W1)),W2)

        # max_digits = np.max(forward_val, axis=1, keepdims=True)

        nomalized_exp = ndl.exp(forward_val)

        # exp_ratio = nomalized_exp / np.repeat(np.sum(nomalized_exp, axis=1, keepdims=True), repeats=num_classes, axis=1)

        exp_ratio=ndl.divide(nomalized_exp,ndl.broadcast_to(ndl.reshape(ndl.summation(nomalized_exp,(1)),(nomalized_exp.shape[0],1)),nomalized_exp.shape))

        indicator = np.zeros((batch, num_classes))

        indicator[np.arange(0, batch), y_batch.flatten()] = 1

        # activate_x_w1's shape is (batch,hidden_dim)
        activate_x_w1 = ndl.relu(ndl.matmul(X_batch_Tensor, W1))

        W2_gradient =ndl.divide_scalar(ndl.matmul( ndl.transpose(activate_x_w1,(-1,-2)), exp_ratio - indicator),batch)


        # calculate the gradient of the W1
        print(f"activate_x_w1: {activate_x_w1.shape}")


        # transfer it into numpy()
        activate_x_w1=activate_x_w1.numpy()

        activate_x_w1[activate_x_w1 > 0] = 1
        activate_x_w1[activate_x_w1 <= 0] = 0

        #transfer it back to ndl.Tensor
        activate_x_w1=ndl.Tensor(activate_x_w1)

        # print(activate_x_w1)

        '''
        shapes:
            1. exp_ratio-indicator =( batch , num_classes )
            2. W2 = (hidden_dim, num_classes)
            3. activate_x_w1 = (batch,hidden_dim)
            4. X_batch = (batch x input_dim)
        target:
            W1_gradient : (input_dim, hidden_dim)

        '''
        X_batch_T = ndl.transpose(X_batch_Tensor,axes=(-1,-2))

        # ((exp_ratio - indicator).dot(W2.T)

        exp_ratio_minus_indicator=ndl.matmul(exp_ratio-indicator,ndl.transpose(W2,axes=(-1,-2)))

        W1_gradient = ndl.matmul(X_batch_T , ndl.multiply(activate_x_w1 ,exp_ratio_minus_indicator))

        W1_gradient = ndl.divide_scalar(W1_gradient,batch)

        # update the gradient
        W2 -= lr * W2_gradient
        W1 -= lr * W1_gradient

    ### END YOUR SOLUTION
    return (W1, W2)



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
