---
layout: post
title: Two Layer Neural Network in Numpy
---

In the previous post we saw how we can build a simple [Logistic Unit](http://anupsawant.com/2017/12/27/logistic-regression-from-scratch/). In this post we will write a quick 2-layered neural network in numpy. Anything bigger than a 2 layered network looks like a hairball to me and gets me confused easily when it comes to computing gradients.

This is how our neural network will look:

<img src="http://www.extremetech.com/wp-content/uploads/2015/07/NeuralNetwork.png">
<center>Fig 1 <a href="https://www.extremetech.com/extreme/215170-artificial-neural-networks-are-changing-the-world-what-are-they" target="_blank">source</a></center>

It almost look like the hidden layer is nothing but a bunch of Logistic Units stacked on top of each other. The difference here is that we are adding another layer of units which are equal in number to the dimensions of our output variable.

Let's go with the following dimensions for our 2-layered neural network
