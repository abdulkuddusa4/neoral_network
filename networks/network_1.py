import numpy as np 
import matplotlib.pyplot as plt


class Network:
    def __init__(self,layers):
        self.num_layers = len(layers)
        self.weights=[np.random.randn(row,clm) for row,clm in \
            zip(layers[1:],layers[:len(layers)-1])]
        self.biases=[np.random.randn(row,1) for row in layers[1:]]

    def relu(self,x):
        # return 1/(1+np.exp(-x))
        return x

    def relu_derivative(self,x):

        ar =  np.array([[1 if i>0 else -1 if i<0 else 0] for i in x])
        # return np.ones(x.shape)
        return ar

    def cost_derivative(self,x,y):
        return x-y

    def feed_forword(self,x):
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,x)+b
            x = self.relu(z)
        # print(x[0][0])
        # exit()
        return x[0][0]

    def backprop(self,x,y):
        activations = [x]
        activation = x
        zs = []
        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,activation)+b
            activation = self.relu(z)
            activations.append(activation)
            zs.append(z)

        # backpass
        # print(self.biases[0])
        # exit()
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        error = np.dot(activations[-1],y)*self.relu_derivative(zs[-1])
        delta_w[-1]=np.dot(error,activations[-2].transpose())
        delta_b[-1]=error

        for i in range(2,self.num_layers):
            error = np.dot(self.weights[-i+1].transpose(),error)*self.relu_derivative(zs[-i])
            delta_w[-i]=np.dot(error,activations[-i-1].transpose())
            delta_b[-i]=error
        return delta_w, delta_b

    def SGD(self,dataset,epochs):
        for i in range(epochs):
            xs=[]
            ys=[]
            count = 0
            for x,y in dataset:
                delta_w,delta_b = self.backprop(x,y)
                print(self.biases)
                self.weights = [w-nabla_w*.0001 for w,nabla_w in zip(self.weights,delta_w)]
                self.biases = [b-nabla_b*.0001 for b,nabla_b in zip(self.biases,delta_b)]
                xs.append(x[0])
                ys.append(y[0])
                # if count<10:
                #     break
                count += 1

            new_ys = [self.feed_forword(x) for x,y in dataset]
            print("##3")
            # print(ys)
            print()
            print()
            print()
            plt.clf()
            plt.plot(xs,ys,color='red')
            plt.plot(xs,new_ys,color='green')
            plt.pause(.1)
