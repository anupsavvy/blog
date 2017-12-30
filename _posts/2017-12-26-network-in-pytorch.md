---
layout: post
comments: true
title: Network in Pytorch
---

### Why Pytorch ?

Previously, we worked on a simple [2-layered Neural Network in numpy](http://anupsawant.com/2017/12/20/two-layer-nn-in-numpy/). While its a good toy network to get thorough with the steps and implementation, anything more than 2 layers can quickly get things tricky and difficult to debug. Hence, this and the subsequent posts of this blog will use a Deep Learning framework for coding a neural network. Frameworks like TensorFlow - Keras and Pytorch probably don't need any introduction at this point. Both frameworks are equally good and strong at what they have to offer. Both work on a graph of calculations that help backpropogate gradients. TensorFlow, however, works on a static graph and Pytorch builds a dynamic graph for each forward pass. Here's a nice description from one of the Pytorch tutorial.

> PyTorch autograd looks a lot like TensorFlow: in both frameworks we define a computational graph, and use automatic differentiation to compute gradients. The biggest difference between the two is that TensorFlow’s computational graphs are static and PyTorch uses dynamic computational graphs. In TensorFlow, we define the computational graph once and then execute the same graph over and over again, possibly feeding different input data to the graph. In PyTorch, each forward pass defines a new computational graph.

You may further wish to look into the pros and cons [here](http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#tensorflow-static-graphs).

With a bit of tinkering with TensorFlow and Pytorch, I have decided to go ahead with Pytorch for now. My reason being, I find Pytorch more user friendly and very pythonic in nature. Also, even though Keras is meant for a quick and easy way to implement deep networks, for more complicated models you will need to work with underlying TensorFlow. Pytorch may not be as easy to use as Keras, initially, but its a solid choice for building all kinds of models and is probably generating quite a lot of interest among researchers. Jeremy's Fastai course uses Pytorch as the underlying framework for its custom deep learning library, also known as, [fastai](https://github.com/fastai/fastai/tree/master/courses/dl1).

### MNIST

Our task for this post is to classify handwritten digits on a very well-known dataset called [MNIST](http://yann.lecun.com/exdb/mnist/). It has around 60000 training and 10000 test examples. The goal is to classify/recognize 28 x 28 x 1 images of digits. This serves as a good dataset to get our hands dirty with Pytorch. We will be looking at flattening the image into 784 dimension vector and generating an output of 10 dimensions, where, each of the output dimension would represent a digit from 0 - 9 respectively. Let's get started !

<center><img src="https://ml4a.github.io/images/figures/mnist_1layer.png"></center>
<center>Fig 1 <a href="https://ml4a.github.io/ml4a/looking_inside_neural_nets/" target="_blank">source</a></center>

### Getting the dataset

Pytorch provides quite a few datasets through its [dataset class](http://pytorch.org/docs/master/data.html). One of them is [MNIST](http://pytorch.org/docs/master/torchvision/datasets.html#mnist) subclass. In order to use this class we need to provide the root directory where we want to store our data, a boolean option on whether we want a training or test set and, mainly, a transform option in our case that will transform the MNIST images into a tensor. A tensor is similar to a numpy ndarray but more suitable for the calculations to be done on GPU.

Starting with the imports, this is how our data extraction is going to look :  

{% highlight  python %}
from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

ds = MNIST(root='data',download=True,transform=transforms.ToTensor(),train=True)
loader = DataLoader(dataset=ds,batch_size=8192,shuffle=True)

test_ds = MNIST(root='data',download=True,transform=transforms.ToTensor(),train=False)
test_loader = DataLoader(dataset=test_ds,batch_size=1024,shuffle=True)
{% endhighlight %}

Notice, the dataset instance is being passed to the DataLoader class of pytorch. This is to help us iterate over batches of our data. For instance, we create batches of our training set with each of size 8192 samples. DataLoader divides our data into batches and shuffles it to help us learn one batch at a time. <em>If you are running this code on GPU then your batch size would typically be restricted by the RAM of your GPU.</em>

### Network architecture

We now layout the network architecture using <em>torch.nn.Module</em>. In order to tell Pytorch on how exactly we want to design our network and what steps to be carried out in the forward pass, we need to override the forward method of nn.Module. As opposed to a [2-layered network](http://anupsawant.com/2017/12/20/two-layer-nn-in-numpy/) in our previous post, we want to try our hands with a bit deeper network for this classification problem. We are opting for 4 layer network. These are going to be fully connected linear layers with ReLU activations. The input and output size of a layer can be mentioned easily by passing them as parameters to <em>torch.nn.Linear</em> class. For the forward pass in the forward method, the input will be first flattened into 784 dimensional tensor and output of each layer will be passed through ReLU activation. It's nice to get a feel of how easy it is to compute dense networks with a framework like Pytorch. Following block of code shows the same:

{% highlight python %}
class MNISTnn(torch.nn.Module):
    def __init__(self):
        super(MNISTnn,self).__init__()
        self.li_1 = torch.nn.Linear(784,784)
        self.li_2 = torch.nn.Linear(784,500)
        self.li_3 = torch.nn.Linear(500,50)
        self.li_4 = torch.nn.Linear(50,10)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = x.view(-1,784)
        l_1 = self.relu(self.li_1(x))
        l_2 = self.relu(self.li_2(l_1))
        l_3 = self.relu(self.li_3(l_2))
        return self.relu(self.li_4(l_3))
{% endhighlight %}

### Loss, optimizer and training

Once we put our network structure in place and outline the forward pass, we choose a loss technique and optimizer for our gradient descent, like so,

{% highlight python %}
model = MNISTnn().cuda()
criterion = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
{% endhighlight %}

Notice, the model is declared using a <em>.cuda()</em> method. This basically tells Pytorch to run our model on GPU and not CPU. <em>torch.nn.CrossEntropyLoss</em> will use a software layer of size 10 at the outer layer of our network and compute the loss for us by averaging the loss over each batch. <em>torch.optim.Adam</em> uses the Adam optimizer over model parameters with a learning rate of 1e-3. We will have a separate blog post on Adam optimizer in the future. For now, it can be thought of as a combination of two important gradient descent optimization techniques called - momemtum and RMSProp.

We shall now iterate over our training samples in batches and cover 4 major steps of training:

* Forward pass
* Loss calculation
* Gradient calculation
* Parameter updates

We run this for about 50 epochs and on each epoch we will test our model performance on the entire training set.

{% highlight python %}
sub_loss = []
cost = []
for t in range(50):
    correct = 0
    test_correct = 0
    model.train()
    for idx, (data, target) in enumerate(loader):
        data, target = Variable(data).cuda(),Variable(target).cuda()

        ## forward pass
        y_pred = model(data)
        loss = criterion(y_pred, target)
        sub_loss.append(loss.data[0])

        ## calculate accuracy over a batch
        pred = torch.max(y_pred.data,1)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        ## Gradient descent
        optimizer.zero_grad()
        loss.backward()

        ## Update parameters
        optimizer.step()

    model.eval()
    for idx, (data, target) in enumerate(test_loader):
        data, target = Variable(data).cuda(),Variable(target).cuda()

        ## forward pass
        y_pred = model(data)
        loss = criterion(y_pred, target)

        ## calculate accuracy over a batch
        pred = torch.max(y_pred.data,1)[1]
        test_correct += pred.eq(target.data.view_as(pred)).sum()


    cost.append(sum(sub_loss)/len(sub_loss))
    print("Epoch {}, Loss {:.3f}, Train Accuracy {:.2f} %, Test Accuracy {:.2f} %".format(
        t,cost[t],100*(correct/len(loader.dataset)),
        100*(test_correct/len(test_loader.dataset))
    ))
    sub_loss = []
plt.xlabel("epoch -- >")
plt.ylabel("cost")
plt.plot(cost)
plt.show()
{% endhighlight %}

Few notes from the above code:

* A tensor is wrapped in <em>Variable()</em>, which then becomes a node of a graph in calculation. Notice, we call <em>cuda()</em> method for those variables.
* Data attached to a [Variable](http://pytorch.org/docs/master/_modules/torch/autograd/variable.html) can be obtained using <em>data</em> tensor and gradients can be obtained using <em>grad</em> Variable.
* <em>torch.max()</em> provides the max values and their indices according to the axis provided.
* Gradients are calculated or the backpropogation is done simply by calling <em>.backward()</em> funtion on the loss obtained from <em>torch.nn.CrossEntropyLoss</em>
* <em>.step()</em> on the optimizer updates all our parameters of the network for us.

Given this setup, we obtain around 99.52 % Accuracy on training set and around 97.84 % accuracy on our test set. Not bad for a quick effort !!!!
<center><img src="{{ site.baseurl }}/public/img/mnist-graph.png"></center>

We are now ready to dive into much deeper networks in our subsequent posts !!
