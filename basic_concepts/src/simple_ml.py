import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x+y
    ### END YOUR CODE



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


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    images=Deserialization(image_filename).astype(np.float32)

    # shape : (60000,28,28) -> (60000, 784)
    images=images.reshape(images.shape[0],-1)

    labels=Deserialization(label_filename).astype(np.uint8)
    images=(images-np.min(images))/(np.max(images)-np.min(images))
    return ( images , labels )

    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE

    num_classes=Z.shape[1]
    batch_size=Z.shape[0]

    # transfer the y's shape : (batch_size, ) -> ((batch_size, 1)
    y=y.reshape(y.shape[0] , -1 )

    max_logits=np.max(Z,axis=1,keepdims=True)

    exp_Z = np.exp(Z-max_logits)

    denominator=np.sum(exp_Z ,keepdims=True,axis=1)

    ratio=exp_Z / np.repeat(denominator,repeats=num_classes , axis=1)

    #  construct the one-hot encode,such as (4) -> ( 0,0,0,0,1,0,0,0,0,0 )

    new_label=np.zeros(ratio.shape)
    new_label[np.arange(batch_size),y.flatten()]=1


    return - np.sum(new_label * np.log(ratio)) / batch_size

    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE

    num_examples=X.shape[0]
    num_classes=theta.shape[1]
    y=y.reshape(y.shape[0],1)

    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        # calculate gradients
        # Xtheta's shape is (num_examples x num_classes)
        Xtheta=np.dot(X_batch, theta)
        max_logits = np.max(Xtheta, axis=1, keepdims=True)

        Xtheta_exp=np.exp(Xtheta-max_logits)

        Z=Xtheta_exp / np.repeat(np.sum(Xtheta_exp,axis=1,keepdims=True),repeats=Xtheta_exp.shape[1],axis=1)


        Ey=np.zeros((batch,num_classes))

        Ey[np.arange(0,batch),y_batch.flatten()]=1

        # print("="*100)
        # print(f"X is {X} \n")
        # print(f"Z is {Z} \n")
        # print(f"Ey is {Ey} \n")
        # print("="*100)


        gradient=np.dot(X_batch.T,(Z-Ey))/batch
        theta -= lr * gradient

    ### END YOUR CODE



def _relu( matrix : np.array):
    shape=matrix.shape
    matrix=matrix.flatten()
    matrix=np.array(list(map(lambda x : max(x,0),matrix)))
    return matrix.reshape(shape)

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples=X.shape[0]
    num_classes=W2.shape[1]

    for start in range(0,num_examples,batch):
        end=min(start+batch,num_examples)
        X_batch=X[start:end]
        y_batch=y[start:end]

        # calculate the gradient of the W2
        forward_val=_relu(X_batch.dot(W1)).dot(W2)
        max_digits=np.max(forward_val,axis=1,keepdims=True)

        nomalized_exp=np.exp(forward_val-max_digits)

        exp_ratio=nomalized_exp /np.repeat(np.sum(nomalized_exp,axis=1,keepdims=True),repeats=num_classes, axis=1)

        indicator = np.zeros( ( batch , num_classes) )

        indicator[np.arange(0,batch),y_batch.flatten()]=1

        # activate_x_w1's shape is (batch,hidden_dim)
        activate_x_w1=_relu(np.dot(X_batch,W1))

        W2_gradient=np.dot(activate_x_w1.T,exp_ratio-indicator)/batch

        # very important : W2temp is a copy of W2,used to calculate the W1_gradient
        W2temp=W2.copy()

        W2-=lr*W2_gradient

        # calculate the gradient of the W1
        activate_x_w1[ activate_x_w1 >  0] = 1
        activate_x_w1[ activate_x_w1 <= 0] = 0
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
        W1_gradient = X_batch.T.dot( activate_x_w1 * ( (exp_ratio-indicator).dot(W2temp.T)) )
        W1_gradient=W1_gradient / batch
        W1 -= lr * W1_gradient
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("../data/train-images-idx3-ubyte.gz",
                             "../data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("../data/t10k-images-idx3-ubyte.gz",
                             "../data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1,cpp=True)

    # print("\nTraining two layer neural network w/ 100 hidden units")
    # train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
