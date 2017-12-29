---
layout: post
comments: true
title: Logistic Regression from Scratch
---

In pursuit of gaining strength at Deep Learning I have chosen a path to start with basics. I am not new to the concepts in Machine Learning. However, I like to take one step at a time and not rush through things. This will also help me refresh a lot of concepts that I don't necessarily get to use at work on daily basis.

Here, we will start with a simple Logitic Regression Unit that serves as the building block for our Neural Networks, going ahead. There are a lot of great courses online which provide a fast track approach towards getting better at Deep Learning. While it is quite true to a large extent, I prefer to start with basics, assuming not knowing anything. I would highly recommend [Andrew Ng's Deep Learning class on Coursera](https://www.coursera.org/specializations/deep-learning) for this. The exercise of coding from scratch helps clear lot of clouds and get better with time. The start can be slow but the journey is most certainly assured to be concrete and fulfilling in the end.

Assuming you know the theory behind Logitic Unit, here's how it can be pictured:

<table>
  <tbody>
    <tr>
      <td><img src="https://www.safaribooksonline.com/library/view/python-deeper-insights/9781787128576/graphics/B05198_06_01.jpg"><center>Fig 1<a href="https://www.safaribooksonline.com/library/view/python-deeper-insights/9781787128576/graphics/B05198_06_01.jpg" target="_blank"> source</a></center></td>
      <td><img src="{{ site.baseurl }}/public/img/sigmoid.png"><center>Fig 2<a href="https://en.wikipedia.org/wiki/Sigmoid_function" target="_blank"> source</a></center></td>
    </tr>
  </tbody>
</table>


As opposed to just the notion of activation function pertaining to a unit, it helps me to divide a unit into 3 different parts:

* Weights / Parameters that help us map a given input to expected output (W)
* Linear output (Z)
* Activation output (A)

Fig 1 depicts a sigmoid unit. Sigmoid function is usually better at determining binary outputs. It's graph (Fig 2) is centered around 0.5, so, we can classify an output signal as 1/yes for activations above or equal to 0.5 and vice versa.

Logistic Units are suitable for fitting a line with a given set of points. A unit is basically trying to get the input elements (x1, x2, x3 ...), multiply them with their respective weights / parameters (w1, w2, w2 ...), add some bias factor (b) to it and derive an output (z). This is further pushed into an activation function that helps map the real valued z to either 0 or 1 (y_pred).

Given this bit of theory, let's get down to some coding with necessary imports.

{% highlight python %}
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import linear_model
{% endhighlight %}

For the sake of convinience, I am going to be using [breast cancer dataset](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer) from scikit. It has around 550 samples and 30 features for each sample with 2 classes to predict - malignant and benign. We would want to download get this data, normalize and split it into training and test set.

{% highlight python %}
## scale data
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train_orig)
X_test = scaler.transform(X_test_orig)

## reshape the data
X_train = X_train.reshape(X_train.shape[1],X_train.shape[0])
X_test = X_test.reshape(X_test.shape[1],X_test.shape[0])
y_train = y_train_orig.reshape(1,y_train_orig.shape[0])
y_test = y_test_orig.reshape(1,y_test_orig.shape[0])
{% endhighlight %}

It is now time to initialize our weights / parameters. Since this is going to be a classifier with single unit, it is okay to initialize the weights with zero initial values.

<i>Note: We wouldn't want to do so in case of a neural network with many activation units as all the units will end up calculating same values with no variation in weights across multiple units.</i>

{% highlight python %}
def initialize_parameters(nx):
    '''
        parameters:
            nx: number of features
            m: number of samples

        return:
            parameters : dictionary of parameters W and b
    '''

    W = np.zeros((nx,1))
    b = 0
    assert(W.shape == (nx,1)),'wrong dimensions for weight matrix'
    return {'W': W, 'b': b}
{% endhighlight %}

Next, we calculate linear forward output $$Z = W^TX + b$$ over all samples X.

{% highlight python %}
# compute linear forward
def linear_forward(X, parameters):
    '''
        parameters:
            X: input vectors
            parameters: Weights and bias

        returns:
            Z: linear output where Z = WX + b

    '''
    (W,b) = parameters['W'],parameters['b']

    Z = np.dot(W.reshape((W.shape[1],W.shape[0])),X) + b
    assert(Z.shape == (1,X.shape[1])),'check dimensions of linear forward matrix'
    return Z
{% endhighlight %}

As per Fig 1, we calculate activation based on the Sigmoid function $$1/(1 + e^{-Z})$$

{% highlight python %}
# compute activation
def linear_forward_with_activation(X, parameters):
    '''
        parameters:
            Z: linear forward output

        returns:
            A: sigmoid activation of linear forward output (1/(1+ exp(-z)))
    '''
    Z = linear_forward(X, parameters)
    A = 1 / ( 1 + np.exp(-Z))

    assert(A.shape == (1,X.shape[1])),'check dimensions of activations'
    return A   
{% endhighlight %}

Based on the predicted values, we now calculate total loss over samples.

$$J = (-1/m)\sum_{i=0}^m(Y\log(A) + (1-Y)\log(1-A))$$

<i>Note: I have tried to add a bit of regularization in this as well</i>

{% highlight python %}
# compute cost
def compute_cost(Y,A,lambd,W):
    '''
        parameters:
            Y: actual outputs
            A: linear forward activation outputs for all samples

        returns:
            J: total loss over all samples
    '''

    J = (-1/m) * np.sum(np.dot(Y,np.log(A.reshape(m,1))) + np.dot((1-Y),np.log(1-A.reshape(m,1))))
    J = J + (lambd/(2*m))*np.sum(np.dot(W.reshape(W.shape[1],W.shape[0]),W))

    return J  
{% endhighlight %}

And now we start our journey backwards. Following function computes gradients of parameters.
The note in the comments should be self-explanatory.

{% highlight python %}
# calculate gradients
def compute_gradients(A,Y,X):
    '''
        parameters:
            A: activation outputs for all samples
            X: samples
            Y: outputs

        returns:
            grads: dictionary of gradients with respect to total loss

        Note: gradient of loss w.r.t activations is (-y/a) + (1-y)/(1-a)
              gradient of activations w.r.t linear output is a(1-a)
              Hence, gradient of loss w.r.t linear output is (a-y)
    '''

    dZ = A - Y ## gradient of loss w.r.t linear output
    dW = np.dot(X,dZ.reshape(m,1))/m ## gradient of loss w.r.t. weights
    db = np.sum(dZ)/m ## gradient of loss w.r.t bias

    assert(dW.shape == (X.shape[0],1)),'check shape of dW'

    return {'dW' : dW, 'db': db}  
{% endhighlight %}

Finally, we update our parameters $$W = W - \alpha dW$$

{% highlight python %}
## update parameters

def update_parameters(parameters,grads,learning_rate,lambd):
    '''
        parameters:
            W: weight matrix
            b: bias
            dW: weight gradient
            db: bias gradient
            learning_rate

        returns:
            W,b
    '''
    W = parameters['W']
    b = parameters['b']

    W = W - learning_rate*(grads['dW'] + (lambd/m)*W)
    b = b - learning_rate*grads['db']

    return {'W':W,'b':b}

    ## make predictions

def predict(X,Y,parameters):

    '''
        parameters:
            X: samples
            Y: output
            parameters
    '''
    Y = Y < 0.5
    Y_pred = linear_forward_with_activation(X, parameters) < 0.5

    print(accuracy_score(Y.reshape(Y.shape[1],1),Y_pred.reshape(Y_pred.shape[1],1)))

    assert(Y.shape == (1, X.shape[1])),'check shape of Y'
    assert(Y_pred.shape == (1, X.shape[1])),'check shape of Y_pred'
{% endhighlight %}

Putting it all together,

{% highlight python %}
# create model
def logistic_model(X,Y,num_iterations=5000,learning_rate=0.001,lambd=5.0):
    '''
        parameters:
            X: samples
            Y: outputs
    '''

    ## initialize parameters
    parameters = initialize_parameters(X.shape[0])
    J = [] ## list to capture loss over all epochs

    for i in range(1,num_iterations):

        ## computer linear forward activations

        A = linear_forward_with_activation(X, parameters)

        ## compute loss

        J.append(compute_cost(Y,A,lambd,parameters['W']))

        ## compute gradients

        grads = compute_gradients(A,Y,X)

        ## update parameters

        parameters = update_parameters(parameters,grads,learning_rate,lambd)


        if i % 10000 == 0:
            print("Total loss over all samples in epoch ", i, " is : ", J[i-1], "\n")

    plt.plot(J)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
{% endhighlight %}

<img src="{{ site.baseurl }}/public/img/learning_curve.png">

We get a smooth learning curve over the training set. With further iterations and feature analysis, we can determine if we are facing a high variance or high bias issue and bring down the loss. However, so far our toy Logistic Unit seems to be behaving the way we want it to be.
