import numpy as np
import os
from PIL import Image
import csv
import pandas as pd

class Neuron():
    def __init__(self, n: int) -> None:
        self.weights = np.random.rand(n) - 0.5
        self.bias = np.random.rand()
        self.back_error = 0
        return
    
    def forwardPropagation(self, inputs: np.ndarray, activation_function = None) -> float:
        ouputs = inputs * self.weights
        output = np.sum(ouputs)
        output += self.bias
        if activation_function:
            output = activation_function(output)
        return output
    
    def updateWeights(self, inputs, functionDerivate) -> None:
        eta = 0.02
        sigma = np.sum(inputs * self.weights)
        self.weights = self.weights + eta * self.back_error * functionDerivate(sigma)* inputs
        self.bias += eta * self.back_error

class NeuralNetwork():
    def __init__(self, input_size: tuple[int, int], class_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.class_size = class_size
        self.hidden_size = hidden_size
        self.hidden_layer = np.array([Neuron(self.input_size[0]*self.input_size[1]) for _ in range(hidden_size)])
        self.output_layer = np.array([Neuron(hidden_size) for _ in range(class_size)])
        

    def train(self, path: str, epochs: int = 10):
        inputs = []
        for class_csv in os.listdir(path):
            print(class_csv)
            class_inputs = pd.read_csv(os.path.join(path, class_csv))
            class_inputs = np.array(class_inputs)
            inputs.append(class_inputs)
        '''
        for class_name in os.listdir(path):
            print(class_name)
            class_path = os.path.join(path, class_name)
            class_inputs = []
            i = 0
            for image in os.listdir(class_path):
                if i == 1000:
                    break
                i += 1
                image_path = os.path.join(class_path, image)
                img = Image.open(image_path)
                img = img.resize(self.input_size)
                img = img.convert('L')
                img = np.array(img).flatten() / 255
                class_inputs.append(img)
            class_inputs = np.array(class_inputs)
            inputs.append(class_inputs)
            with open(os.path.join("testCsv", f"class_{class_name}.csv"), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(class_inputs)'''
        
        inputs = np.array(inputs)

        for i in range(epochs):
            #train
            total_loss = 0
            for class_index, class_inputs in enumerate(inputs):
                for j, single_input in enumerate(class_inputs):
                    if(j > int(0.8 * len(class_inputs))):
                        break
                    hidden_ouputs = np.array([neuron.forwardPropagation(single_input, ReLu) for neuron in self.hidden_layer])
                    outputs = np.array([neuron.forwardPropagation(hidden_ouputs) for neuron in self.output_layer])
                    outputs = softMax(outputs)

                    loss_gradient = encode(class_index, self.class_size) - outputs
                    loss = -np.log(outputs[class_index])
                    total_loss += loss
                    back_error = np.zeros(self.hidden_size)

                    #back propagation
                    for index, neuron in enumerate(self.output_layer):
                        back_error += loss_gradient[index] * neuron.weights

                    #updating weights
                    for neuron, error in zip(self.output_layer, loss_gradient):
                        neuron.back_error = error
                        neuron.updateWeights(hidden_ouputs, derivateSigmoid)
                    for neuron, error in zip(self.hidden_layer, back_error):
                        neuron.back_error = error
                        neuron.updateWeights(single_input, derivateReLu)
            # test
            correct = 0
            total = 0
            for class_index, class_inputs in enumerate(inputs):
                for j in range(int(0.8*len(class_inputs))+1, len(class_inputs)):
                    single_input = class_inputs[j]
                    if(self.evaluate(single_input) == class_index):
                        correct += 1
                    total += 1
            accuracy = correct / total
            print(f"Epoch {i+1}: \taccuracy: {accuracy:.4f}; \tloss: {total_loss:.4f}")
            if accuracy > 0.9:
                break


    def evaluate(self, inputs) -> int:
        hidden_ouputs = np.array([neuron.forwardPropagation(inputs, ReLu) for neuron in self.hidden_layer])
        outputs = np.array([neuron.forwardPropagation(hidden_ouputs) for neuron in self.output_layer])
        outputs = softMax(outputs)
        return np.argmax(outputs)

def ReLu(array):
    return np.maximum(array, 0)

def derivateReLu(x):
    if x > 0:
        return 1
    return 0

def softMax(array):
    exp_array = np.exp(array)
    return exp_array / np.sum(exp_array)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivateSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def encode(x, n):
    array = np.zeros(n)
    array[x] = 1
    return array

def main():
    active_directory = os.path.split(__file__)[0]
    model = NeuralNetwork(input_size=(28, 28), class_size=10, hidden_size=32)
    folder_path = os.path.join(active_directory, 'testCsv')
    model.train(path=folder_path, epochs=200)
    while 1:
        img = input("inser image path")
        img = Image.open(img).convert("L")
        img = img.resize(model.input_size)
        img = np.array(img).flatten() / 255
        value = model.evaluate(img)
        print("The prediction is: ", value)

if __name__ == '__main__':
    main()