---
layout: post
comments: true
title: Two Layer Neural Network in Numpy
---

In the previous post we saw how we can build a simple [Logistic Unit](http://anupsawant.com/2017/12/04/logistic-regression-from-scratch/). In this post we will write a quick 2-layered neural network in numpy. Anything bigger than a 2 layered network looks like a hairball to me and gets me confused easily when it comes to computing gradients.

This is how our neural network will look:

<img src="http://www.extremetech.com/wp-content/uploads/2015/07/NeuralNetwork.png">
<center>Fig 1 <a href="https://www.extremetech.com/extreme/215170-artificial-neural-networks-are-changing-the-world-what-are-they" target="_blank">source</a></center>

It almost looks like the hidden layer is nothing but a bunch of Logistic Units stacked on top of each other. The difference here is that we are adding another layer of units which are equal in number to the dimensions of our output variable.

Let's go with the following size for our 2-layered neural network

{% highlight python %}
m, n, l1, l2 = 64, 1000, 100, 10
{% endhighlight %}

where,
* m = Number of input samples
* n = Number of features per sample
* l1 = Number of units in the hidden layer
* l2 = Number of units in the output layer

Next, we randomly create our input/output samples and parameters in numpy with their respective dimensions

{% highlight python %}
x = np.random.randn(m,n)
y = np.random.randn(m,l2)

w1 = np.random.randn(n,l1)
w2 = np.random.randn(l1,l2)
b1 = np.random.randn(1,l1)
b2 = np.random.randn(1,l2)

{% endhighlight %}

where,
* x = input samples
* y = outputs
* w1,w2 = weights of layer 1 and layer 2, respectively.
* b1,b2 = bias parameters for both the layers.

Given these inputs and parameters, we now come to training our neural network over training samples. The intention here is to bring down the prediction loss w.r.t number of iterations.

<img src="{{ site.baseurl }}/public/img/2-layered-nn.png">
<center>Fig 2</center>

To me, personally, it helps to think of the calculation steps in terms of different boxes that we need to handle one at a time.

We are going to be calculating z1,a1 and z2 in the forward pass and dz2,dw2,db2,da1,dz1,dw1,db1 during backpropogation. Fig 2 above depicts the conceptual flow of calculations. The goal is to see a smooth learning curve that approaches near zero loss in about 100 iterations. For simplicity, we are considering a euclidean measure of loss.

{% highlight python %}
learning_rate = 1e-5
cost = []
for t in range(100):

    # forward pass
    z1 = np.dot(x,w1) + b1
    a1 = np.maximum(z1,0)
    z2 = np.dot(a1,w2) + b2

    # euclidean distance
    loss = np.square(z2 - y).sum()
    cost.append(loss)

    # calculate gradients
    dz2 = 2*(z2 - y)
    dw2 = np.dot(a1.T,dz2)/m
    db2 = dz2.sum()/m
    da1 = np.dot(dz2,w2.T)
    dz1 = da1.copy()
    dz1 = np.maximum(dz1,0)
    dw1 = np.dot(x.T,dz1)/m
    db1 = dz1.sum()/m

    # update parameters
    w2 -= learning_rate*dw2
    w1 -= learning_rate*dw1

    b2 -= learning_rate*db2
    b1 -= learning_rate*db1


plt.xlabel("iterations")
plt.ylabel("cost")
plt.plot(cost)
plt.show()
{% endhighlight %}

<img src="{{ site.baseurl }}/public/img/2-layer-curve.png">
<center>Fig 3</center>

That's it ! Few things to remember while coding a neural network:

* Derivative of a ReLU function is 0 below 0 and 1 above 0.
* Low accuracy on a training set probably means we have under-fitting or a variance issue. This could possibly be resolved with more training data, by changing the number of units / layers of the network or by training the network for more number of iterations.
* Low accuracy on a test set would mean we have over-fitting or a high bias issue. This can be handled by multiple techniques like regularization, dropouts and making sure that the training and test set have similar data distribution.
* It would be best to start with a smaller network and add depth only if necessary.
* All weights should never be initialized to zero, otherwise, the network will very likely behave no better than a logistic unit.
* When possible, normalize the linear outputs of each layer.
* It's best to keep ReLU as the activation function for hidden layers and sigmoid / softmax for the outer layer.  
