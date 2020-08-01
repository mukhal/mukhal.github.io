---
layout: post
title: "Matrix Factorization in NLP"
author: Muhammad Khalifa
comments: true
published: true
---


It's good to be back! Since I am currently studying the [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra/) course by Rachel Thomas, I was impressed by the power and utlity of Matrix Factorization algorithms, and their very wide range applications in many domains including Information Retrieval, Image Processing, Computer Vision and (dearly) Natural Lanugage Processing. 

Unforutnately, a blog post is never enough to talk about matrix factorization and its relationship to NLP. There's a plethora of Matrix factorzation algorithms, each of these algorithms has been probably used in an NLP application. This renders infeasable writing a comprehensive article on the subject. However, **Where there's will, there's way** and I will do my best to cover as many NLP-related application as possible.




>
#### Disclaimer
This article assumes the reader is faimilar with many linear algebra concepts including Vectors, Matrices, Matrix-Vector and Matrix-Matrix Multiplication, Subspaces, Basis, Linear Independence, Rank, and Orthogonality.


### Low-rank Matrix Factorization
The low-rank assumption is that every matrix can be factored into lower-rank matrices without losing much of the latent structure in the original matrix. Given a matrix `$Y \in \mathbb{R}^{N \times M}$`, low-rank matrix factorization **(MF)** produces two matrices `$U \in \mathbb{R}^{n \times k}$` and `$V \in \mathbb{R}^{m \times k}$` such that the reconstructed matrix `$ Y' = UV^{T}$` is a lower-rank version of `$Y$`. Note that `$k << rank(Y)$`. One way to think about **MF is that `$Y$` is likely to have both redundancy and noise. Low-rank reconstruction exploits redundancy to remove noise. It gives us a way to represent large matrices with smaller, less redundant matrices that somehow contain the beef of the original matrix.


![](/images/mf-imgs/low-rank-mf.svg)


#### MF as Optimization Problem
Vanilla MF can expressed as the following optimization problem:

$$
\begin{equation}
   J = \min_{U,V} ||UV^T - Y||^2_F + \alpha ||U||^2_F + \beta ||V||^2_F
\end{equation}
$$

Note that the L2 norm terms are mainly used for regularization purpose. It's possible to obtain `$\frac{\partial J}{\partial U}$` and `$\frac{\partial J}{\partial V}$` and then optimze it using gradient-based methods such as SGD. Check out [this](https://yangjiera.github.io/works/low-rank.pdf) article for more info.


### Nonnegative Matrix Factorization
So far, we've seen the vanilla form of MF which imposes no contraints on the resulting matrices `$U$` and `$V$`. Nonnegative Matrix Factorization **(NMF)** imposes a very simple constraint: constrain the values in `$U$` and `$V$` to be `$\geq 0$`. But why is that useful? you might ask. Well the answer is in one word: *interpretability*. It makes much more sense to think of the data points (rows) in the matrix `$Y$` as a linear combination of a set of building blocks. For instance, an image of a person's face can be thought of as a linear combination of face features (skin tone, nose shape, eye color, eyebrows density, etc). It's really interesting to look at it like this: there's a universal set of building blocks (or featuers) of all the faces in the world and what really separates one face from the other is the different proportion of these features that each face has. NMF has become so popular because of its ability to automatically extract **sparse** and easily **interpretable** factors [Gillis et al., 2014](https://arxiv.org/pdf/1401.5226.pdf).
<img src="/images/mf-imgs/facial-images-nmf.png" width="600" height="400">

But NMF can be applied to text as well. An article can be thought of as a weighted combination of a set of topics. The higher the weight of a particular topic in the article, the higher the frequency of the occurence of the words representing that topic. For instance, a news article can have as a topic: politics, humanitites and technology. If, for instance, the article is talking more about policitcs than anything else, we should expect politics-related words such as "president", "elections", "senate" to be more frequent than other words such as "love", "freedom", or "computers". Similarly to the face example, we can view all the documents in the world through the lens of a universal set of topics that are distributed differently through each document.

NMF can expressed as the following optimization problem:

$$
\begin{equation}
   J = \min_{U,V} ||UV^T - Y||^2_F \text{ such that } U \geq 0 \text{ and } V \geq 0
\end{equation}
$$

A question remains however, as to how one might solve this optimization problem. [Gillis et al., 2014](https://arxiv.org/pdf/1401.5226.pdf) discuss several issues with solving the NMF problem in practice:
	* Unlike the vanilla MF problem, NMF is NP-Hard. 

#### Topic Modelling

To make use of matrix factorization when dealing with documents, we must find a way to represent a document as a matrix. The most common way is what is known as term-document matrix. Each row in the matrix `$M \in \mathbb{R}^{d \times w}$`. Each row is a document and each column is a word. But this is a matrix of real numbers, and we need a number to put in `$M_{ij}$` that somehow relates document `$i$` with word `$j$`. The most common ways to fill `$M_{ij}$` are :
		
1. Occurence frequency of word `$i$` in document `$j$`, which is an unnormalized bare count.
2. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weighted occurences, weighting the frequency by the importance of the word `$i$` to document `$j$` (the IDF part).

The bottomline is no matter how you choose `$M_{ij}$`, you want a real-number connecting the word and the document together. Once we have the matrix `$M$`, now enters matrix factorization.


When dealing with topic modelling, we a




## 
