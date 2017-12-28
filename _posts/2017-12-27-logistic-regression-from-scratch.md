---
layout: post
title: Logistic Regression from Scratch
---

In pursuit of gaining strength at Deep Learning I have chosen a path to start with basics. I am not new to the concepts in Machine Learning. However, I like to take one step at a time and not rush through things. This will also help me refresh a lot of concepts that I don't necessarily get to use at work on daily basis.

Here, we will start with a simple Logitic Regression Unit that serves as the building block for our Neural Networks, going ahead. There are a lot of great courses online which provide a fast track approach towards getting better at Deep Learning. While it is quite true to a large extent, I prefer to start with basics, assuming not knowing anything. I would highly recommend [Andrew Ng's Deep Learning class on Coursera](https://www.coursera.org/specializations/deep-learning) for this. The exercise of coding from scratch helps clear lot of clouds and get better with time. The start can be slow but the journey is most certainly assured to be concrete and fulfilling in the end.

Assuming you know the theory behind Logitic Unit, here's how it can be pictured:

<table>
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
</table>



As opposed to just the notion of activation function pertaining to a unit, it helps me divide a unit into 3 different parts:

* Weights (W)
* Linear output (Z)
* Activation output (A)

Fig 1 depicts a sigmoid unit. Sigmoid function is usually better at determining binary outputs. It's graph (Fig 2) is centered around 0.5, so, we can classify a given output signal as 1/yes for activations above or equal to 0.5 and vice versa.