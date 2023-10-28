# Neural Network Number Classifier
Introduction:
This project presents an implementation of a neural network, authored in pure Python and Numpy, designed for the purpose of classifying 28x28 pixel images depicting numbers ranging from 0 to 9.

Implementation:
Within the scope of the MNIST dataset, a total of 60,000 training data points were employed. A split of 80% was designated for the training dataset, while the remaining 20% was allocated for testing. The neural network's weights were initialized with random values, and biases were set to zero. The training process involved three epochs, with a learning rate set at 0.001.

Architecture:
This neural network is structured with three layers: an input layer, an output layer, and a hidden layer. The hidden layer employs the sigmoid activation function, while the backpropagation process leverages the gradient descent algorithm.

Results:

Training Accuracy (90%): The training accuracy of 90% indicates that, during the training process, our neural network correctly predicted the labels for 90% of the training data. This accuracy is computed using the following formula:

Training Accuracy = (Number of Correct Predictions / Total Training Samples) * 100

In our case, if we had 60,000 training samples, it means our network made approximately 54,000 correct predictions during training.

Testing Accuracy (93%): The testing accuracy of 93% signifies that our trained neural network was able to correctly classify 93% of the previously unseen testing data. This accuracy is calculated in a similar manner as the training accuracy.

Testing Accuracy = (Number of Correct Predictions / Total Testing Samples) * 100

If, for instance, we had 10,000 testing samples, this accuracy implies that our network made around 9,300 correct predictions on the testing data.

Conclusion:

In the conclusion, we can discuss potential improvements to enhance the neural network's performance. Here are some mathematical insights:

Rectified Linear Unit (ReLU): Replacing the sigmoid activation function in the hidden layer with the Rectified Linear Unit (ReLU) can be beneficial. The ReLU activation function is defined as follows:

ReLU(x) = max(0, x)

This activation function often leads to faster convergence and mitigates the vanishing gradient problem, which can result in more efficient training.

Cost Function Enhancement: The choice of cost function plays a critical role in training neural networks. One commonly used cost function for classification tasks is the Cross-Entropy Loss, defined as:

Cross-Entropy Loss = -Σ(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))

Where y_i represents the true label, p_i is the predicted probability, and the summation runs over all classes. Optimizing the choice and adaptation of this cost function can potentially lead to better convergence and accuracy.

By incorporating these enhancements and further fine-tuning hyperparameters, the neural network's classification performance can be significantly improved, and the accuracy can be further elevated. This underscores the iterative and data-driven nature of improving neural network models.




