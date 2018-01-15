---
layout: post
comments: true
title: Convolutional Network in Pytorch
---

Building on top of our previous post on [MNIST](http://anupsawant.com/2017/12/26/network-in-pytorch/), we finally get to start with some real topics of Deep Learning with this post. Here, we will deal with a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network). These type of networks have been a hot topic for a while now as they currently server 'state of the art' results for many of the deep learning problems, especially, imagine / video analysis.   

Unlike a regular Neural Network where each layer is fully connected, intuitively, Convolutional Networks make sparse connections, thereby, holding a bunch of neurons responsible for analyzing a particular section of an image. A typical setup of a Convolutional Network would involve an image and a bunch of filters with odd dimensions to go over (read: convolve) small chunks of input pixels at a time. These filters then try to detect the edges of an image in the early layers of the network. Subsequent layers build-up on this knowledge and try to detect some key building blocks of an image. For instance, in the case of face recognition, the early layers of their respective filters would help detect the edges of a person's face and the latter layers would detect independent parts of a human face like nose, eyes, lips et. al. In between two Conv layers we typically have a pooling layer that usually retain the most active features of the input. Following figure and its source provide some good explanation on how the Convolutional Networks work.

<center><img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-4-59-29-pm.png?w=748"></center>
<center>Fig 1 <a href="https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/" target="_blank">source</a></center>

### Recognizing Signs

In this blog post I am going to be using hand digit signs dataset provided by Andrew Ng via his Deep Learning course on Coursera. The sign image looks something like below:

<center><img src="{{ site.baseurl }}/public/img/sign.png"></center>

Our task is to predict the current number which the hand sign denotes. It should be 4 for the above input image.

### Getting the dataset

As always, we start with the necessary imports. The data has been stored in .h5 format and we will be using a load_dataset() method provided by Andrew Ng's team to read the data here :  

{% highlight  python %}
from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

%matplotlib inline
np.random.seed(1)

def load_dataset():
    train_dataset = h5py.File('data/signs/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/signs/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
{% endhighlight %}

We now create the dataloader and dataset instance which allow us to read the training and test data in mini-batches of 32 each.

{% highlight  python %}
class SignsDataset(Dataset):

    def __init__(self,data_type):
        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
        m = X_train_orig.shape[0]
        c = X_train_orig.shape[3]
        h = X_train_orig.shape[1]
        w = X_train_orig.shape[2]
        m_test = X_test_orig.shape[0]
        X_train = X_train_orig.reshape(m,c,h,w)
        Y_train = Y_train_orig.reshape(m)
        X_test = X_test_orig.reshape(m_test,c,h,w)
        Y_test = Y_test_orig.reshape(m_test)
        if data_type == 'train':
            self.x_data,self.y_data = torch.from_numpy(X_train).float(),Y_train
            self.len = X_train_orig.shape[0]
        else:
            self.x_data,self.y_data = torch.from_numpy(X_test).float(),Y_test
            self.len = X_test_orig.shape[0]

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = SignsDataset('train')
train_loader = DataLoader(dataset=dataset,
                         batch_size=32,
                         shuffle=True)

dataset = SignsDataset('test')
test_loader = DataLoader(dataset=dataset,
                         batch_size=32,
                         shuffle=True)
{% endhighlight %}

Notice, we reshape the ndarrays to match the expected inputs of our convolutional layers and read them as tensors using <em>.form_numpy()</em> method.

### Network architecture

<center><img src="{{ site.baseurl }}/public/img/signarch.png"></center>

As shown in the above figure, we go with a 3 conv layer and 1 fully connected layer network for this problem. The first layer is convolved with 10 filters of size 5 x 5, second layer is convolved with 20 filters of size 3 x 3 and the last conv layer is convolved with 10 filters of size 3 x 3. All layers go through ReLU activation and a Max pooling layer. Finally, the output is derived using a fully connected layer with [softmax function](https://en.wikipedia.org/wiki/Softmax_function) at the end. It is important to note that unlike TensorFlow, Pytorch currently does not provide a 'SAME' option for padding. [SAME padding](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t) is used to make sure that we don't lose the input data by virtue of reducing dimensions through multiple convolving layers.

In order to match the number of input and output units in each layer, we do the calculations using (n + 2p - f)/s + 1 for each layer, where, n = input size, p = padding, f = filter size and s = strides.

Following block of code shows our architecture:

{% highlight python %}
class SignNN(torch.nn.Module):
    def __init__(self):
        super(SignNN,self).__init__()
        self.conv_1 = torch.nn.Conv2d(3,10,kernel_size=5,padding=2)
        self.conv_2 = torch.nn.Conv2d(10,20,kernel_size=3,padding=1)
        self.conv_3 = torch.nn.Conv2d(20,10,kernel_size=3,padding=1)
        self.mp_1 = torch.nn.MaxPool2d(7,stride=1,padding=3)
        self.mp_2 = torch.nn.MaxPool2d(3,stride=1,padding=1)
        self.relu = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(40960,6)

    def forward(self,x):
        in_size = x.size(0)
        x = self.mp_1(self.relu(self.conv_1(x)))
        x = self.mp_1(self.relu(self.conv_2(x)))
        x = self.mp_2(self.relu(self.conv_3(x)))
        x = x.view(in_size,-1) # flatten the tensor
        return self.fc_1(x)
{% endhighlight %}

### Loss, optimizer and training

Once we put our network structure in place and outline the forward pass, we choose a loss technique and optimizer for our gradient descent, like so,

{% highlight python %}
model = SignNN().cuda()
criterion = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
{% endhighlight %}

We shall now iterate over our training samples in batches and cover 4 major steps of training:

* Forward pass
* Loss calculation
* Gradient calculation
* Parameter updates

We run this for about 600 epochs and on each epoch we will test our model performance on the entire test set. In the process of evaluation, we save the best model that gives us the best accuracy over test set.

{% highlight python %}
sub_loss = []
cost = []
test_accuracy = 0
for t in range(600):
    correct = 0
    test_correct = 0
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()

        # forward pass
        y_pred = model(data)
        loss = criterion(y_pred, target)
        sub_loss.append(loss.data[0])

        # calculate accuracy over a batch
        pred = torch.max(y_pred.data,1)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

        # Gradient descent
        optimizer.zero_grad()
        loss.backward()

        # Update parametes
        optimizer.step()

    model.eval()
    for idx,(data,target) in enumerate(test_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()

        # forward pass
        y_pred = model(data)
        loss = criterion(y_pred,target)

        # calculate accuracy over a batch
        # get indices of the predicted classes
        pred = torch.max(y_pred.data,1)[1]
        test_correct += pred.eq(target.data.view_as(pred)).sum()

    cost.append(sum(sub_loss)/len(sub_loss))
    accuracy = 100*(test_correct/len(test_loader.dataset))
    print("Epoch {}, Loss {:.3f}, Train Accuracy {:.2f} %, Test Accuracy {:.2f} %".format(
        t,cost[t],100*(correct/len(train_loader.dataset)),
        accuracy
    ))
    sub_loss = []
    if accuracy > test_accuracy:
        torch.save(model.state_dict(), "models/signs")
        print("Model saved at test accuracy of {:.2f} %".format(accuracy))
        test_accuracy = accuracy

plt.xlabel("epoch --> ")
plt.ylabel("cost")
plt.plot(cost)
plt.show()
{% endhighlight %}

We save our model at about 87 % accuracy over test set and 100 % accuracy over training set. This is not a bad result over hand digit signs. In our subsequent posts we will work with Residual Networks to see how we can improve our results.

<center><img src="{{ site.baseurl }}/public/img/sign-accuracy.png"></center>
<center><img src="{{ site.baseurl }}/public/img/sign-graph.png"></center>

### Sample Predictions
{% highlight python %}
def printImg(indices,x_test,y_test,pred):

    for index in indices:
        plt.imshow(x_test[index])
        print ("y = " + str(np.squeeze(y_test[:, index])))
        print ("y_pred = " + str(np.squeeze(pred[index])))
        plt.pause(0.001)


for idx,(data,target) in enumerate(test_loader):
        data_v, target_v = Variable(data).cuda(), Variable(target).cuda()

        # forward pass
        y_pred = model(data_v)
        x_test = data.abs().numpy().reshape(data.shape[0],64,64,3)
        y_test = target.numpy().reshape(1,data.shape[0])
        printImg([1,5,6,9],x_test,y_test,torch.max(y_pred.data,1)[1].cpu().numpy())
        break
{% endhighlight %}
<table>
  <tbody>
    <tr>
      <td><img src="{{ site.baseurl }}/public/img/sign1.png"></td>
      <td><img src="{{ site.baseurl }}/public/img/sign2.png"></td>
    </tr>
    <tr>
      <td><img src="{{ site.baseurl }}/public/img/sign4.png"></td>
      <td><img src="{{ site.baseurl }}/public/img/sign5.png"></td>
    </tr>
  </tbody>
</table>
