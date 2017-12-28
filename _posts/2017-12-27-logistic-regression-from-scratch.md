---
layout: post
title: Logistic Regression from Scratch
---

In pursuit of gaining strength at Deep Learning I have chosen a path to start with basics. I am not new to the concepts in Machine Learning. However, I like to take one step at a time and not rush through things. This will also help me refresh a lot of concepts that I don't necessarily get to use at work on daily basis.

Here, we will start with a simple Logitic Regression Unit that serves as the building block for our Neural Networks, going ahead. There are a lot of great courses online which provide a fast track approach towards getting better at Deep Learning. While it is quite true to a large extent, I prefer to start with basics, assuming not knowing anything. I would highly recommend [Andrew Ng's Deep Learning class on Coursera](https://www.coursera.org/specializations/deep-learning) for this. The exercise of coding from scratch helps clear lot of clouds and get better with time. The start can be slow but the journey is most certainly assured to be concrete and fulfilling in the end.

Assuming you know the theory behind Logitic Unit, here's how it can be pictured:

<!-- <table>
  <tbody>
    <tr>
      <td>
        <center><img src="https://www.safaribooksonline.com/library/view/python-deeper-insights/9781787128576/graphics/B05198_06_01.jpg" style="width:50%;height:50%;"></center>

        Fig 1 : Source: [Safari Books Online](https://www.safaribooksonline.com/library/view/python-deeper-insights/9781787128576/graphics/B05198_06_01.jpg)
      </td>
      <td>
        <center><img src="{{ site.baseurl }}/public/img/sigmoid.png" style="width:50%;height:50%;"></center>

        Fig 2 : Source: [Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function#/media/File:Logistic-curve.svg)
      </td>
    </tr>
  <tbody>
</table> -->

<table>
  <tbody>
    <tr>
      <td><img src="https://www.safaribooksonline.com/library/view/python-deeper-insights/9781787128576/graphics/B05198_06_01.jpg"></td>
      <td><img src="{{ site.baseurl }}/public/img/sigmoid.png"></td>
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
