import numpy as np
from numpy import ndarray
import pandas as pd
import os
from PIL import Image
import time

class NeuralNetwork():
    def __init__(self, input_size: tuple[int, int], class_names: list[str], layers: list[int]) -> None:
        self.class_names = class_names
        self.input_size = input_size
        self.class_size = len(class_names)
        self.layers_size = len(layers) + 1
        self.layers_dim = layers
        self.layers_dim.append(self.class_size)
        self.learning_rate = 0.02

        self.w = []
        self.b = []
        self.w.append(np.random.rand(layers[0], input_size[0] * input_size[1]) - 0.5)
        self.b.append(np.random.rand(layers[0], 1) - 0.5)

        for i in range(1, self.layers_size):
            self.w.append(np.random.rand(self.layers_dim[i], self.layers_dim[i - 1]) - 0.5)
            self.b.append(np.random.rand(self.layers_dim[i], 1) - 0.5)

    def loadData(self, path: str) -> tuple[ndarray, ndarray]:
        inputs = pd.read_csv(path)
        inputs = np.array(inputs)
        np.random.shuffle(inputs)
        classes = inputs[:, 0]
        inputs = inputs[:, 1:]
        return classes, inputs
    
    def train(self, path: str, epochs: int, batch_size: int = 1):
        print("\nLoading data...")
        start_time = time.time()
        classes, inputs = self.loadData(path)
        taken_time = time.time() - start_time
        print(f"Loading data took {taken_time:.2f} seconds\n")
        print("Training...")
        start_time = time.time()
        #print(f"training with learning rate = {self.learning_rate}")
        inputs_size = len(inputs)
        train_size = int(0.8 * inputs_size)
        for epoch in range(epochs):
            total_loss = 0
            for index, single_input in enumerate(inputs):
                if index == train_size:
                    total_loss /= index
                    break
                single_input = single_input.reshape((-1, 1))

                z = []
                a = []
                a.append(single_input)
                z0 = self.w[0].dot(single_input) + self.b[0]
                a0 = ReLu(z0)
                z.append(z0)
                a.append(a0)
                for i in range(1, self.layers_size):
                    zi = self.w[i].dot(a[i]) + self.b[i]
                    z.append(zi)
                    if i != self.layers_size-1:
                        a.append(ReLu(zi))
                    else:
                        a.append(zi)
                a[-1] = softMax(a[-1])
                loss_gradient =  encode(int(classes[index]), self.class_size) - a[-1]
                if a[-1][int(classes[index])][0] < 0.000000000000000000001:
                    loss = -np.log(0.000000000000000000001)
                else:
                    loss = -np.log(a[-1][int(classes[index])][0])
                total_loss += loss

                #backpropagations
                if index % batch_size == 0:
                    w_gradients = []
                    b_gradients = []
                    for i in range(self.layers_size):
                        w_gradients.append(np.zeros(self.w[i].shape))
                        b_gradients.append(np.zeros(self.b[i].shape))
                w_gradients[-1] += loss_gradient.dot(a[-2].T)
                b_gradients[-1] += loss_gradient
                single_b_gradients = []
                single_b_gradients.append(loss_gradient)
                for i in range(self.layers_size - 2, 0 - 1, -1):
                    single_b_gradients.append(self.w[i+1].T.dot(single_b_gradients[-1]) * ReLuDerivate(z[i]))
                    w_gradients[i] += single_b_gradients[-1].dot(a[i].T)
                    b_gradients[i] += single_b_gradients[-1]

                #update weights
                if (index + 1) % batch_size == 0:
                    for i in range(self.layers_size):
                        self.w[i] += w_gradients[i] * self.learning_rate
                        self.b[i] += b_gradients[i] * self.learning_rate
            
            correct = 0
            total = 0
            for index in range(train_size + 1, inputs_size):
                single_input = inputs[index]
                single_input = single_input.reshape((-1, 1))
                if(self.evaluate(single_input)[0] == classes[index]):
                    correct += 1
                total += 1
            accuracy = correct / total
            print(f"Epoch {epoch + 1}: \taccuracy: {accuracy:.5f}; \tloss: {total_loss:5f}")

        taken_time = time.time() - start_time
        print(f"It took {taken_time:.2f} seconds to train the model\n")

    def evaluate(self, inputs):
        inputs = inputs.reshape((-1, 1))
        a = []
        a.append(inputs)
        z0 = self.w[0].dot(inputs) + self.b[0]
        a0 = ReLu(z0)
        a.append(a0)
        for i in range(1, self.layers_size):
            zi = self.w[i].dot(a[i]) + self.b[i]
            if i != self.layers_size-1:
                a.append(ReLu(zi))
            else:
                a.append(zi)
        a[-1] = softMax(a[-1])
        return np.argmax(a[-1]), a[-1]
    
    def predict(self, inputs) -> tuple[str, float]:
        a, b = self.evaluate(inputs)
        return self.class_names[a], b[a][0]

def softMax(array) -> ndarray:
    exp_array = np.exp(array - np.max(array))
    return exp_array / np.sum(exp_array)

def encode(x, n):
    array = np.zeros((n, 1))
    array[x] = 1
    return array

def ReLu(array):
    return np.maximum(0, array)

def ReLuDerivate(array):
    return np.where(array > 0, 1, 0)

def Sigmoid(array):
    x = np.where( array < -7, -0.9999, np.where(array > 7, 0.9999, 1 / (1 + np.exp(-array))))
    return x
def SigmoidDerivate(array):
    x = Sigmoid(array)
    return x * (1 - x)

def main():
    class_names = []
    for i in range(10):
        class_names.append(str(i))
    active_directory = os.path.split(__file__)[0]
    project_directory = os.path.split(active_directory)[0]
    folder_path = os.path.join(project_directory, 'src/dataset.csv')
    
    layers = input("Insert hidden layers size: ")
    layers = layers.split(',')
    layers = [int(size) for size in layers]
    batch_size = int(input("Insert batch size: "))
    epochs = int(input("Insert epochs: "))

    model = NeuralNetwork(input_size=(28, 28), class_names=class_names, layers=layers)
    model.train(path=folder_path, epochs=epochs, batch_size=batch_size)
    while 1:
        image_path = input("insert image path: ")
        if image_path == "exit":
            return 0
        image_path = os.path.join(project_directory, image_path)
        img = Image.open(image_path).convert("L")
        img = img.resize(model.input_size)
        img = np.array(img).flatten() / 255
        value, confidence = model.predict(img)
        print(f"The prediction is: {value} \nwith {(confidence * 100):.2f}% confidence\n")

if __name__ == '__main__':
    main()
