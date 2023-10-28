"""import numpy as np
import pathlib

def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels"""


import numpy as np
import pathlib

def get_mnist():
    with np.load("/Users/daanishhindustano/Documents/MNIST_NN/Data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    
    # Split the data into training and testing sets (e.g., 80% for training, 20% for testing)
    split_ratio = 0.8
    num_samples = len(images)
    num_train_samples = int(split_ratio * num_samples)
    
    X_train, X_test = images[:num_train_samples], images[num_train_samples:]
    y_train, y_test = labels[:num_train_samples], labels[num_train_samples:]
    
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, X_test, y_train, y_test