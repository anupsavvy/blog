---
layout: post
title: Two Layer Neural Network in Numpy
---

In the previous post we saw how we can build a simple [Logistic Unit](http://anupsawant.com/2017/12/04/logistic-regression-from-scratch/). In this post we will write a quick 2-layered neural network in numpy. Anything bigger than a 2 layered network looks like a hairball to me and gets me confused easily when it comes to computing gradients.

This is how our neural network will look:

<img src="http://www.extremetech.com/wp-content/uploads/2015/07/NeuralNetwork.png">
<center>Fig 1 <a href="https://www.extremetech.com/extreme/215170-artificial-neural-networks-are-changing-the-world-what-are-they" target="_blank">source</a></center>

It almost look like the hidden layer is nothing but a bunch of Logistic Units stacked on top of each other. The difference here is that we are adding another layer of units which are equal in number to the dimensions of our output variable.

Let's go with the following dimensions for our 2-layered neural network

{% highlight python %}
m, n, l1, l2 = 64, 1000, 100, 10
{% endhighlight %}

where,
* m = Number of input samples
* n = Number of features per sample
* l1 = Number of units in the hidden layer
* l2 = Number of units in the output layer

Next, we randomly create our input/output samples and parameters in numpy with their respective dimensions

{% highlight %}
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
