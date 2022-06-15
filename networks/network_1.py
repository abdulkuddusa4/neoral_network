import numpy as np
from shortcut import VECTOR_SIGMOID
import math
from functools import reduce
import pickle as pkl
import random
import time
import os

def max(x,y):
    return x if x>y else y


def print(*args,**kwargs):
    kwargs['end'] = '\n\n'
    __builtins__['print'](*args,**kwargs)


class Network:
    def __init__(self,layer):
        self.layers = layer
        self.num_layers = len(self.layers)
        self.activations = []
        self.zs = []
        file_name = __file__.split('/')[-1].split('.')[0]
        file_path = __file__.split('/')
        file_path.pop()
        file_path.pop()
        file_path = '/'.join(file_path)
        self.brain_path = f"{file_path}/brain/{file_name}_brain"
        print(self.brain_path)
        try:
            with open(self.brain_path+'/weights', 'rb') as w, open(self.brain_path+'/biases', 'rb') as b:
                self.weights = pkl.load(w)
                self.biases = pkl.load(b)
        except FileNotFoundError as e:
            self.weights = [np.random.randn(x,y) for x,y in zip(layer[1:],layer[:-1])]
            self.biases = [np.random.randn(x,1) for x in layer[1:]]

    def feed_forword(self, input_x):
        for weight,bias in zip(self.weights,self.biases):
            z = weight.dot(input_x)+bias
            input_x = self.sigmoid(z)

        return input_x

    # def cost(self, input_x, y):
    #     v = self.feed_forword(input_x)
    #     cst = y*np.log(v)+(1-y)*np.log(1-v)
    #     print(cst)
    #     cst_v = v
    #     cst = reduce(lambda a,b: a+b,[math.pow(i,2) for i in cst_v])
    #     return cst

    # def average_cost(self, dataset):
    #     avg_cst = reduce(lambda a,b: a+b,[self.cost(i,j) for i,j in dataset])/len(dataset)
    #     return avg_cst

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def SGD(self,training_dataset,epochs,learning_rate,test_set=None,drunk=True):
        for i in range(epochs):
            random.shuffle(training_dataset)

            for x,y in training_dataset:
                d_w,d_b = self.backprop(x,y)
                self.weights = [w-(nw*learning_rate) for w,nw in zip(self.weights,d_w)]
                self.biases = [b-(nb*learning_rate) for b,nb in zip(self.biases,d_b)]
            self.save()
            if test_set:
                print(f"EPOCH {i+1}: {self.performence(test_set)}%")

    def BGD(self,training_dataset,epochs,learning_rate,test_set=None,drunk=True):
        for i in range(epochs):
            dummy_li = ["_","\\","|","/"]#this is dummy list no special uses
            delta_nebla_w = [np.zeros(w.shape) for w in self.weights]
            delta_nebla_b = [np.zeros(b.shape) for b in self.biases]
            for x,y in training_dataset:
                d_w,d_b = self.backprop(x,y)
                delta_nebla_w = [dnw+w for dnw,w in zip(delta_nebla_w,d_w)]
                delta_nebla_b = [dnb+b for dnb,b in zip(delta_nebla_b,d_b)]

            self.weights = [w-((nw/len(training_dataset))*learning_rate) for w,nw in zip(self.weights,delta_nebla_w)]
            self.biases = [b-((nb/len(training_dataset))*learning_rate) for b,nb in zip(self.biases,delta_nebla_b)]
            self.save()
            if test_set:
                print(f"EPOCH{i+1}: {self.performence(test_set)}%")

    def MGD(self,training_dataset,epochs,learning_rate,mini_batch_size,test_set=None,drunk=True):
        for i in range(epochs):
            dummy_li = ["_","\\","|","/"]#this is dummy list no special uses
            mini_batches = [training_dataset[k:k+mini_batch_size] \
                       for k in range(0,len(training_dataset),mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,learning_rate)
            self.save()
            if test_set:
                print(f"EPOCH{i+1}: {self.performence(test_set)}%")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]



    def performence(self,test_set):
        self.save()
        if test_set:
            correct = 0
            for image,label in test_set:
                output_vector = self.feed_forword(image)
                if np.argmax(output_vector) == np.argmax(label):
                    correct+=1
            return correct/len(test_set)*100

    def test(self,test_set):
        correct = 0
        counter=0
        print("ddd")
        for image,label in test_set:
            output_vector = self.feed_forword(image)
            if np.argmax(output_vector) == np.argmax(label):
                correct+=1
            else:
                print(counter)
                with open(f'failed_images/image{counter}.txt','w')as file:
                    count=0
                    for i in range(28):
                        for j in range(28):
                            # print(2,end='#SDFS')
                            # file.write(".")
                            file.write("." if image[count][0] > 0.0 else " ")
                            count+=1
                        file.write("\n")
                    file.write("\n"*10)
                    file.write(f"{str(np.argmax(label))}")
            counter+=1

        print(f"performence: {(correct/len(test_set)*100)}%")

    def cost_derivative(self,x,y):
        return x-y

    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def save(self):
        try:
            w = open(self.brain_path+'/weights', 'wb')
            b = open(self.brain_path + '/biases', 'wb')
        except FileNotFoundError as e:
            os.mkdir(self.brain_path)
            w = open(self.brain_path+'/weights', 'wb')
            b = open(self.brain_path+'/biases', 'wb')
        finally:
            pkl.dump(self.weights,w)
            pkl.dump(self.biases,b)
            w.close()
            b.close()




