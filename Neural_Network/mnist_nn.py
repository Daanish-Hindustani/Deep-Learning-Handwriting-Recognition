from data import get_mnist
import numpy as np

# ACTIVATION FUNCTIONS
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#DERITIVITVES OF ACTIVATION FUNCTIONS 
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def neural_network():
    #INITIALIZING WEIGHT
    weight_input_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
    weight_hidden_output = np.random.uniform(-0.5, 0.5, (10, 20))

    #INITIALIZING BIASES
    bias_input_hidden = np.zeros((20,1))
    bias_hidden_output = np.zeros((10,1))

    #INITIALIZING EPOCHS, LEARNING RATE, ACCURACY VARIABLE
    epochs = 3
    learning_rate = 0.001
    number_correct_training = 0
    number_correct_testing = 0

    #CALLING DATA
    image_training, image_testing, label_training, label_testing = get_mnist()

    #Training Data
    for epoch in range(epochs):
        loss_sum_per_epoch = 0
        for img, l in zip(image_training, label_training):
            img.shape += (1,)
            l.shape += (1,)

        #FORWARD PROPOCATION INPUT -> HIDDEN
            weighted_sum_hidden = bias_input_hidden + weight_input_hidden @ img
            activated_value_hidden = sigmoid(weighted_sum_hidden)
        #FORWARD PROPOCATION HIDDEN -> OUTPUT
            weighted_sum_output = bias_hidden_output + weight_hidden_output @ activated_value_hidden

        #Loss
            loss = (weighted_sum_output - l) ** 2
            loss_sum_per_epoch += loss
            
        #DERIVITIVE OF LOSS
            dloss = 2 * (weighted_sum_output - l)
        #CORRECTNESS
            number_correct_training += int(np.argmax(weighted_sum_output) == np.argmax(l))
        
        #BACK PROPOCATION 
            #derror/dweights_ouput = (derror/dpredict * dpredicted/dweights_output) = derror/dpredicted * hidden_activation_transposed
            weight_hidden_output += -learning_rate * dloss @ activated_value_hidden.T
            
            #derror/dbias = (derror/dpredict * dpredicted/dbias_output) = derror/dpredicted
            bias_hidden_output += -learning_rate * dloss 

            #derror/d_hidden_activation = (derror/dpredict * dpredicted/d_hidden_activation) = derror/dpredicted * weights_output_transposed
            #We want the derror/d_hidden_weighted_sum_error = derror/dpredicted * weights_output_transposed * dsigmoid 
            hidden_weighted_sum_error = weight_hidden_output.T @ dloss * dsigmoid(activated_value_hidden)

            #d_hidden_weighted_sum_error/d_weighted_input_hidden = d_hidden_weighted_sum_error * input.transposed 
            weight_input_hidden += -learning_rate * hidden_weighted_sum_error @ img.T
            bias_input_hidden += -learning_rate * hidden_weighted_sum_error
        
        #Error of Epoch
        error_epoch = (1/image_training.shape[0]) * sum(loss_sum_per_epoch)
        #Accuracy of Epoch
        accuracy_training = round((number_correct_training / image_training.shape[0]) * 100, 2)

        #Results
        print(f"Epoch {epoch + 1}/{epochs}")


        print(f"{image_training.shape[0]}/{image_training.shape[0]} [==============================] - loss: {error_epoch} - sparse_categorical_accuracy: {accuracy_training}%")
        number_correct_training = 0

    #Testing Data
    loss_sum_per_testing = 0
    for img, l in zip(image_testing, label_testing):
        img.shape += (1,)
        l.shape += (1,)
        #FORWARD PROPOCATION INPUT -> HIDDEN
        weighted_sum_hidden = bias_input_hidden + weight_input_hidden @ img
        activated_value_hidden = sigmoid(weighted_sum_hidden)
        #FORWARD PROPOCATION HIDDEN -> OUTPUT
        weighted_sum_output = bias_hidden_output + weight_hidden_output @ activated_value_hidden

        #Loss
        loss = (weighted_sum_output - l) ** 2
        loss_sum_per_testing += loss
        
        #CORRECTNESS
        number_correct_testing += int(np.argmax(weighted_sum_output) == np.argmax(l))

    #Error of Epoch
    error_testing = (1/image_testing.shape[0]) * sum(loss_sum_per_testing)
    #Accuracy of Epoch
    accuracy_testing = round((number_correct_testing / image_testing.shape[0]) * 100, 2)

    #Results
    print(f"Testing 1/1")
    print(f"{image_testing.shape[0]}/{image_testing.shape[0]} [==============================] - loss: {error_testing} - sparse_categorical_accuracy: {accuracy_testing}%")


def main():
    neural_network()

if __name__ == "__main__":
    main()

            

            

            


















