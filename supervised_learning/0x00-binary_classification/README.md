# 0x00. Binary Classification
The goal of this project is to perform image classification, where the algoritm has to choose if the pic corresponds to number 1 or number 0.

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/classification.png)

I needed to learn the next topics for having solid foundations:

* Models.
* Supervised Learning.
* Prediction.
* Node.
* Weight.
* Bias.
* Activation Functions.
    * Sigmoid.
    * Tanh.
    * Relu.
    * Softmax.
* Neuron.
* Neural Network Layers.
* Logistic Regression.
* Loss Function.
* Cost Function.
* Forward Propagation.
* Gradient Descent.
* Back Propagation.
* Computation Graph.
* Weights/Biases initialization.
* Numpy to perform linear algebra operations.

To achieve the classification were coded three ways to perform it: 

* Sigmoid Neuron.
* Two-Layers Neural Network.
* Deep Neural Network.

## Sigmoid Neuron

* File: `7-neuron.py`

A neuron is an object that takes one or several inputs and produces one singular output.

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NeuronDiagram.png)

The inputs are:

 * X: input data.
 * W: Weight. It weighs the value of X.
 * b: bias. −threshold. The value to reach "success".
 
 The final input value is z = W.X + b. The neuron takes z and performs the sigmoid function, where:
 
 ![\sigma (z) = \frac{1}{1 + e^{-z}}](https://render.githubusercontent.com/render/math?math=%5Csigma%20(z)%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-z%7D%7D)
 
 ![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/sigmoid.png)
 
 The output is named Activation(A).
 
 Also, Neuron can be represented as a Graph.
 
 ![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NeuronGrapg.png)
 
 Neuron was coded as an object, with attributes:
 
 ### Forward Propagation:
 
 * `def forward_prop(self, X):`
 
 Calculates the output Activation. As you can see in the graph:
 
 ![A= \sigma (z) = \frac{1}{1 + e^{-z}}](https://render.githubusercontent.com/render/math?math=A%3D%20%5Csigma%20(z)%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-z%7D%7D)
 
 Where:
 
 ![z = WX + b](https://render.githubusercontent.com/render/math?math=z%20%3D%20WX%20%2B%20b)
 
 ##### Example:
 ```
user@ubuntu-xenial:0x00-binary_classification$ cat 2-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
neuron._Neuron__b = 1
A = neuron.forward_prop(X)
if (A is neuron.A):
        print(A)
user@ubuntu-xenial:0x00-binary_classification$ ./2-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]]
user@ubuntu-xenial:0x00-binary_classification$
```

### Cost

 * `def cost(self, Y, A):`
 
 Calculates the Neuron performance, is better if cost is closer to zero.
 
 ![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NeuronGraphLoss.png)
 
 Where L(A, Y) is Logistic Loss Function and is equal to `-(YlogA + (1-Y)log(1-A)`
 
 ##### Example:
 
 ```
user@ubuntu-xenial:0x00-binary_classification$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('3-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
cost = neuron.cost(Y, A)
print(cost)
user@ubuntu-xenial:0x00-binary_classification$ ./3-main.py
4.365104944262272
user@ubuntu-xenial:0x00-binary_classification$
```

### Evaluation

* `def evaluate(self, X, Y):`

Returns the prediction and the cost of the neuron, respectively. The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

```
user@ubuntu-xenial:0x00-binary_classification$ cat 4-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('4-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A, cost = neuron.evaluate(X, Y)
print(A)
print(cost)
user@ubuntu-xenial:0x00-binary_classification$ ./4-main.py
[[0 0 0 ... 0 0 0]]
4.365104944262272
user@ubuntu-xenial:0x00-binary_classification$
```

### Gradient Descent

* `def gradient_descent(self, X, Y, A, alpha=0.05):`

The Gradient Descent algorithm is used to minimize (find minimum point) of some function. It is defined by:

![X^{(1)} = X^{(0)} - \alpha \nabla F(0)\\](https://render.githubusercontent.com/render/math?math=X%5E%7B(1)%7D%20%3D%20X%5E%7B(0)%7D%20-%20%5Calpha%20%5Cnabla%20F(0)%5C%5C)

or

![X^{(1)} = X^{(0)} - \alpha . dX](https://render.githubusercontent.com/render/math?math=X%5E%7B(1)%7D%20%3D%20X%5E%7B(0)%7D%20-%20%5Calpha%20.%20dX)

where:

![dX = \frac{\partial F(X)}{\partial X}](https://render.githubusercontent.com/render/math?math=dX%20%3D%20%5Cfrac%7B%5Cpartial%20F(X)%7D%7B%5Cpartial%20X%7D)

And alpha is the learning step, or how much the iteration can decent the function.

We want to minimize the cost, so we need to minimize the Loss Function which depends on A and Y, but A depends on X, W, and b. We cannot modify X so we need to modify W and b to get a smaller Loss.

Backtracking is a good technique to find dW and db, we can measure how much change the Loss function going back and calculating the variation of each node in the graph thanks to Chain Rule.

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NeuronGraphGrad.png)

![dA = \frac{\partial L(A, Y)}{\partial A}\rightarrow \frac{d(-(YlogA + (1-Y)log(1-A)))}{dA}\rightarrow \frac{a-y}{a(1-a)}](https://render.githubusercontent.com/render/math?math=dA%20%3D%20%5Cfrac%7B%5Cpartial%20L(A%2C%20Y)%7D%7B%5Cpartial%20A%7D%5Crightarrow%20%5Cfrac%7Bd(-(YlogA%20%2B%20(1-Y)log(1-A)))%7D%7BdA%7D%5Crightarrow%20%5Cfrac%7Ba-y%7D%7Ba(1-a)%7D)

![dz = \frac{\partial L(A, Y)}{\partial z}\rightarrow \frac{dL(A, Y)}{dA} \frac{dA}{dz}\rightarrow dA \frac{d}{dz}(\frac {1}{1+e^{-z}})\rightarrow dA \frac {e^{-z}}{(1+e^{-z})^{2}}\rightarrow dA(\frac {1}{1+e^{-z}})(1-\frac {1}{1+e^{-z}})\rightarrow \frac{A-Y}{A(1-A)}A(1-A)\rightarrow A-Y](https://render.githubusercontent.com/render/math?math=dz%20%3D%20%5Cfrac%7B%5Cpartial%20L(A%2C%20Y)%7D%7B%5Cpartial%20z%7D%5Crightarrow%20%5Cfrac%7BdL(A%2C%20Y)%7D%7BdA%7D%20%5Cfrac%7BdA%7D%7Bdz%7D%5Crightarrow%20dA%20%5Cfrac%7Bd%7D%7Bdz%7D(%5Cfrac%20%7B1%7D%7B1%2Be%5E%7B-z%7D%7D)%5Crightarrow%20dA%20%5Cfrac%20%7Be%5E%7B-z%7D%7D%7B(1%2Be%5E%7B-z%7D)%5E%7B2%7D%7D%5Crightarrow%20dA(%5Cfrac%20%7B1%7D%7B1%2Be%5E%7B-z%7D%7D)(1-%5Cfrac%20%7B1%7D%7B1%2Be%5E%7B-z%7D%7D)%5Crightarrow%20%5Cfrac%7BA-Y%7D%7BA(1-A)%7DA(1-A)%5Crightarrow%20A-Y)

![dW = dz\frac{dz}{dW}= dz \frac{d}{dW}(WX+b)=dzX](https://render.githubusercontent.com/render/math?math=dW%20%3D%20dz%5Cfrac%7Bdz%7D%7BdW%7D%3D%20dz%20%5Cfrac%7Bd%7D%7BdW%7D(WX%2Bb)%3DdzX)

![db = dz\frac{dz}{db}= dz \frac{d}{db}(WX+b)=dz](https://render.githubusercontent.com/render/math?math=db%20%3D%20dz%5Cfrac%7Bdz%7D%7Bdb%7D%3D%20dz%20%5Cfrac%7Bd%7D%7Bdb%7D(WX%2Bb)%3Ddz)

Applying the first equation we can find the new value of W and B.

##### Example:

```
user@ubuntu-xenial:0x00-binary_classification$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('5-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b)
user@ubuntu-xenial:0x00-binary_classification$ ./5-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0.2579495783615682
user@ubuntu-xenial:0x00-binary_classification$
```

### Train

* `def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):`

Perform the neuron training, by minimizing cost function `iteration` times.

* If `verbose` is True: prints the cost every `step` iterations.
* If `graph` is True: prints iteration vs cost plot.

##### Example

```
user@ubuntu-xenial:0x00-binary_classification$ cat 7-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=3000)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
user@ubuntu-xenial:0x00-binary_classification$ ./7-main.py
Cost after 0 iterations: 4.365104944262272
Cost after 100 iterations: 0.11955134491351888

...

Cost after 3000 iterations: 0.013386353289868338
```

##### Output

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NeuronCost3000.png)

```
Train cost: 0.013386353289868338
Train accuracy: 99.66837741808132%
Dev cost: 0.010803484515167197
Dev accuracy: 99.81087470449172%
```

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/classification.png)

## Two-Layers Neural Network

* File: `15-neural_network.py`

Union means strength, so to perform better classifications we join neurons in layers, where the layer input is the activation output from the previous layer.

Also, each layer has like input its own weighs and bias.

In this Neural Network X layers is names Layer 0 with A0 as output.

The intermediate layers are named Hidden Layers, and the last layer is named Output Layer. Due to we are performing a Binary Classification we can use one neuron in this layer.

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NNDiagram.png)

Graph of this layer: 

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NNGraph.png)

The Neural Network was coded as an object, with same attributes than Neuron. The class is defines by imput the number of nodes/neuron in the hidden layer. We can check the performance improvement in examples:

### Examples:
 
#### Forward Propagation:

```
user@ubuntu-xenial:0x00-binary_classification$ cat 10-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('10-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
nn._NeuralNetwork__b1 = np.ones((3, 1))
nn._NeuralNetwork__b2 = 1
A1, A2 = nn.forward_prop(X)
if A1 is nn.A1:
        print(A1)
if A2 is nn.A2:
        print(A2)
user@ubuntu-xenial:0x00-binary_classification$ ./10-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
  1.13141966e-06 6.55799932e-01]
 [9.99652394e-01 9.99999995e-01 6.77919152e-01 ... 1.00000000e+00
  9.99662771e-01 9.99990554e-01]
 [5.57969669e-01 2.51645047e-02 4.04250047e-04 ... 1.57024117e-01
  9.97325173e-01 7.41310459e-02]]
[[0.23294587 0.44286405 0.54884691 ... 0.38502756 0.12079644 0.593269  ]]
user@ubuntu-xenial:0x00-binary_classification$
```

#### Cost

```
user@ubuntu-xenial:0x00-binary_classification$ cat 11-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('11-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
_, A = nn.forward_prop(X)
cost = nn.cost(Y, A)
print(cost)
user@ubuntu-xenial:0x00-binary_classification$ ./11-main.py
0.7917984405648548
user@ubuntu-xenial:0x00-binary_classification$
```

#### Evaluation

```
user@ubuntu-xenial:0x00-binary_classification$ cat 12-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('12-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A, cost = nn.evaluate(X, Y)
print(A)
print(cost)
user@ubuntu-xenial:0x00-binary_classification$ ./12-main.py
[[0 0 0 ... 0 0 0]]
0.7917984405648548
user@ubuntu-xenial:0x00-binary_classification$
```

#### Gradient Descent

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/NNGraphGrad.png)

```
user@ubuntu-xenial:0x00-binary_classification$ cat 13-main.py
#!/usr/bin/env python3

import numpy as np

NN = __import__('13-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X.shape[0], 3)
A1, A2 = nn.forward_prop(X)
nn.gradient_descent(X, Y, A1, A2, 0.5)
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
user@ubuntu-xenial:0x00-binary_classification$ ./13-main.py
[[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
  -1.34149673]
 [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
   0.07912172]
 [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
  -1.07836109]]
[[ 0.003193  ]
 [-0.01080922]
 [-0.01045412]]
[[ 1.06583858 -1.06149724 -1.79864091]]
[[0.15552509]]
user@ubuntu-xenial:0x00-binary_classification$
```

#### Train

```
user@ubuntu-xenial:0x00-binary_classification$ cat 15-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NN = __import__('15-neural_network').NeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NN(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
user@ubuntu-xenial:0x00-binary_classification$ ./15-main.py
Cost after 0 iterations: 0.7917984405648547
Cost after 100 iterations: 0.4680930945144984

...

Cost after 5000 iterations: 0.024369225667283875
```

##### Output

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/trainingNN5000.png)

```
Train cost: 0.024369225667283875
Train accuracy: 99.3999210422424%
Dev cost: 0.020330639788072768
Dev accuracy: 99.57446808510639%
```

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/classification.png)

## Deep Neural Network

* File: `23-deep_neural_network.py`

The last coded Neural Network was the Deep Neural Network, is Network with more of 2 layers. The representation and notation is:

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/DNNDiagram.png)

The object is defined by a variable `layers`, it is a list that have the size of each layer.We can check the performance by using the same attributes.

### Examples

#### Forward Propagation 

```
user@ubuntu-xenial:0x00-binary_classification$ cat 18-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('18-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
deep._DeepNeuralNetwork__weights['b1'] = np.ones((5, 1))
deep._DeepNeuralNetwork__weights['b2'] = np.ones((3, 1))
deep._DeepNeuralNetwork__weights['b3'] = np.ones((1, 1))
A, cache = deep.forward_prop(X)
print(A)
print(cache)
print(cache is deep.cache)
print(A is cache['A3'])
user@ubuntu-xenial:0x00-binary_classification$ ./18-main.py
[[0.75603476 0.7516025  0.75526716 ... 0.75228888 0.75522853 0.75217069]]
{'A1': array([[0.4678435 , 0.64207147, 0.55271425, ..., 0.61718097, 0.56412986,
        0.72751504],
       [0.79441392, 0.87140579, 0.72851107, ..., 0.8898201 , 0.79466389,
        0.82257068],
       [0.72337339, 0.68239373, 0.63526533, ..., 0.7036234 , 0.7770501 ,
        0.69465346],
       [0.65305735, 0.69829955, 0.58646313, ..., 0.73949722, 0.52054315,
        0.73151973],
       [0.67408798, 0.69624537, 0.73084352, ..., 0.70663173, 0.76204175,
        0.72705428]]), 'A3': array([[0.75603476, 0.7516025 , 0.75526716, ..., 0.75228888, 0.75522853,
        0.75217069]]), 'A0': array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), 'A2': array([[0.75067742, 0.78319533, 0.77755571, ..., 0.77891002, 0.75847839,
        0.78517215],
       [0.70591081, 0.71159364, 0.7362214 , ..., 0.70845465, 0.72133875,
        0.71090691],
       [0.72032379, 0.69519095, 0.72414599, ..., 0.70067751, 0.71161433,
        0.70420437]])}
True
True
user@ubuntu-xenial:0x00-binary_classification$
```

#### Cost

```
user@ubuntu-xenial:0x00-binary_classification$ cat 19-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('19-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, _ = deep.forward_prop(X)
cost = deep.cost(Y, A)
print(cost)
user@ubuntu-xenial:0x00-binary_classification$ ./19-main.py
0.6958649419170609
user@ubuntu-xenial:0x00-binary_classification$
```

#### Evaluation

```
user@ubuntu-xenial:0x00-binary_classification$ cat 20-main.py
#!/usr/bin/env python3

import numpy as np

Deep = __import__('20-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])
A, cost = deep.evaluate(X, Y)
print(A)
print(cost)
user@ubuntu-xenial:0x00-binary_classification$ ./20-main.py
[[1 1 1 ... 1 1 1]]
0.6958649419170609
user@ubuntu-xenial:0x00-binary_classification$
```

#### Train

```
alexa@ubuntu-xenial:0x00-binary_classification$ cat 23-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('23-deep_neural_network').DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [5, 3, 1])
A, cost = deep.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:0x00-binary_classification$ ./23-main.py
Cost after 0 iterations: 0.6958649419170609
Cost after 100 iterations: 0.6444304786060048

...

Cost after 5000 iterations: 0.011671820326008168
```

##### Output

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/trainingDNN5000.png)

```
Train cost: 0.011671820326008168
Train accuracy: 99.88945913936044%
Dev cost: 0.00924955213227925
Dev accuracy: 99.95271867612293%
```

![](https://raw.githubusercontent.com/kenneth-ca/holbertonschool-machine_learning/master/supervised_learning/0x00-binary_classification/pics/classification.png)
