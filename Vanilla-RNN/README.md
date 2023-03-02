# Vanilla-RNN

## Introduction
We'll go through the basic architecture of a vanilla RNN along with Backpropagation Through Time (BPTT) and perform a NumPy implementation of the said architecture.

This note is written based on the assumption that you are already familiar with basic neural network architectures and backpropagation.

## Types of RNN

Simply put, RNN is a lot like a traditional neural network except that RNN has time-steps and the weights and biases are shared across the time-steps. 

There are a few types of RNN for different applications.

 1. One-to-Many (*Image Caption Generation)*
 2. Many-to-Many (*Language Translation*)
 3. Many-to-One (*Sentiment Analysis, Video Classification*)
 4. One-to-One 

We can ignore the 4th type as it's just a basic neural network (MLP).

## Many-to-Many Architecture
![Basic RNN Architecture. (https://mmuratarat.github.io/2019-02-07/bptt-of-rnn) ](https://github.com/Ugenteraan/RNN-to-Transformers/blob/main/Vanilla-RNN/RNN_Arch.png) 		 [Image Source](https://mmuratarat.github.io/2019-02-07/bptt-of-rnn)

$W_{hh}, W_{xh}, W_{yh}, b_{h},$ and $b_{y}$ are all shared across the time-steps. With that, we can define:

$h_{t} = tanh(X_{t}W_{xh} + h_{t-1}W_{hh} + b_{h})$
$\hat{y}_{t} = softmax(h_tW_{yh} + b_y)$


Note that the activation functions can be replaced with any other functions depending on the use-case.

### Backpropagation Through Time (BPTT)

The forward-propagation part of the network is quite straightforward. However, the backpropagation is tricky for RNN since the network is recursive. 

Let's define the loss function, $L$ first.

$L(\hat{y}, y) = \sum^{T}_{t=1} L_{t}(\hat{y}_{t}, y_{t})$ 

Since this type of RNN has many outputs, the total loss is a sum of loss at each time-step. For this example, let's use cross-entropy loss.

$L(\hat{y}, y) = -\sum^{T}_{t=1}y_t\  log\ \hat{y}_{t}$. 

In order to find the derivatives of the loss with respect to the weights and biases, we first need to understand that

 - $W_{yh}$ and $b_y$ does not depend on $h_t$. Therefore we can find the derivative of the loss with respect to $W_{yh}$ and $b_y$ at each time-step separately and sum the derivatives.
 - $W_{hh}, W_{xh},$ and $b_{h}$ however depends on $h_{t}$. Therefore, to find the derivatives of the loss with respect to these parameters, we'll have to backpropagate through all the time-steps at once. Hence the name, BPTT.

#### <ins>Finding the Derivative of Loss w.r.t. $W_{yh}$
Let's find the derivative of the loss with respect to $W_{yh}$.

$\frac{\partial{L}}{\partial{W_{yh}}} = \sum^{T}_{t=1}\frac{\partial{L_{t}}}{\partial{W_{yh}}} =\sum^{T}_{t=1}\frac{\partial{L_{t}}}{\partial{\hat{y}_{t}}}.\frac{\partial{\hat{y_{t}}}}{\partial{o_{t}}}.\frac{\partial{o_{t}}}{\partial{W_{yh}}}$

Since the derivative of cross-entropy with respect to softmax function is $\hat{y}_{t} - y_{t}$ and $\frac{\partial{o_{t}}}{\partial{W_{yh}}} = h_t$,

$\sum^{T}_{t=1}\frac{\partial{L_{t}}}{\partial{\hat{y}_{t}}}.\frac{\partial{\hat{y_{t}}}}{\partial{o_{t}}}.\frac{\partial{o_{t}}}{\partial{W_{yh}}} = \sum^{T}_{t=1}(\hat{y}_{t} - y_{t}) \otimes h_t{}$

where $\otimes$ is an outer product.

#### <ins>Finding the Derivative of Loss w.r.t. $b_y$

Similarly, $\frac{\partial{L}}{\partial{b_y}} = \sum^{T}_{t=1}\frac{\partial{L}}{\partial{\hat{y}_t}}.\frac{\partial{\hat{y}_{t}}}{o_t}.\frac{\partial{o_t}}{\partial{b_y}}$

Since $o_t = h_{t}W_{yh} + b_y$, $\frac{\partial{o_t}}{\partial{b_y}} = 1$. Therefore,

$\frac{\partial{L}}{\partial{b_y}} = \sum^{T}_{t=1}(\hat{y}_t - y_t)$

#### <ins>Finding the Derivative of Loss w.r.t. $W_{hh}$

















