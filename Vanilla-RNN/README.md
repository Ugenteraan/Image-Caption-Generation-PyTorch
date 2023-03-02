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

### Many-to-Many Architecture
![Basic RNN Architecture. (https://mmuratarat.github.io/2019-02-07/bptt-of-rnn) ](https://github.com/Ugenteraan/RNN-to-Transformers/blob/main/Vanilla-RNN/RNN_Arch.png) 		 [Image Source](https://mmuratarat.github.io/2019-02-07/bptt-of-rnn)

$W_{hh}, W_{xh}, W_{yh}, b_{h},$ and $b_{y}$ are all shared across the time-steps. With that, we can define:






