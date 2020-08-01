---
layout: post
title: "Notes on Non-negative Matrix Factorization and NLP"
author: Muhammad Khalifa
comments: true
published: false
---


It's good to be back! Since I am currently studying the [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra/) course by Rachel Thomas, I was impressed by the power and utlity of Matrix Factorization algorithms, and their very wide range applications in many domains including Information Retrieval, Image Processing, Computer Vision and (dearly) Natural Lanugage Processing. 

Unforutnately, a blog post is never enough to talk about matrix factorization and its relationship to NLP. There's a plethora of Matrix factorzation algorithms, each of these algorithms has been probably used in an NLP application. That's why I have chosen nonnegative matrix factorization: Nearly the most popular algorithm for MF, with many applications in NLP.

>
#### Disclaimer
This article assumes the reader is faimilar with many linear algebra concepts including Vectors, Matrices, Matrix-Vector and Matrix-Matrix Multiplication, Subspaces, Basis, Linear Independence, Rank, and Orthogonality.

## Background 

#### Low-rank Matrix Factorization
The low-rank assumption is that every matrix can be factored into lower-rank matrices without losing much of the latent structure in the original matrix. Given a matrix `$Y \in \mathbb{R}^{N \times M}$`, low-rank matrix factorization **(MF)** produces two matrices `$U \in \mathbb{R}^{n \times k}$` and `$V \in \mathbb{R}^{m \times k}$` such that the reconstructed matrix `$ Y' = UV^{T}$` is a lower-rank version of `$Y$`. Note that `$k << rank(Y)$`. One way to think about **MF** is that `$Y$` is likely to have both redundancy and noise. Low-rank reconstruction exploits redundancy to remove noise. It gives us a way to represent large matrices with smaller, less redundant matrices that somehow contain the beef of the original matrix.


![](/images/mf-imgs/low-rank-mf.svg)



#### MF as an Optimization Problem
Vanilla MF can expressed as the following optimization problem:

$$
\begin{equation}
   J = \min_{U,V} ||UV^T - Y||^2_F + \alpha ||U||^2_F + \beta ||V||^2_F
\end{equation}
$$

Note that the L2 norm terms are mainly used for regularization purpose. It's possible to obtain `$\frac{\partial J}{\partial U}$` and `$\frac{\partial J}{\partial V}$` and then optimze it using gradient-based methods such as SGD. Check out [this](https://yangjiera.github.io/works/low-rank.pdf) article for more info.

#### Solving with Singular Value Decomposition
The unconstrained low-rank approxaimtion problem stated above can be easily solved using SVD: 

![](https://miro.medium.com/max/1164/1*4Vpi3CFxjLsyJ2zZsZfcCw.png)
*SVD [source: https://research.fb.com/blog/2014/09/fast-randomized-svd/]()*

1. We decompose $$Y = U S V^{T}$$, where $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$, and $\Sigma \in \mathbb{R}^{m \times n}$.
2. Replace the least $r - k$ elements in $$\Sigma$$ with zeros to obtain $$\Sigma_k$$, where $r$ is the rank of the original matrix $Y$. 
3. Obtain the low-rank approximation $Y_k = U \Sigma_k V^{T}$.



## Nonnegative Matrix Factorization
So far, we've seen the vanilla form of MF which imposes no contraints on the resulting matrices `$U$` and `$V$`. Nonnegative Matrix Factorization **(NMF)** imposes a very simple constraint: constrain the values in `$U$` and `$V$` to be `$\geq 0$`. But why is that useful? you might ask. Well the answer is in one word: *interpretability*. It makes much more sense to think of the data points (rows) in the matrix `$Y$` as a linear combination of a set of building blocks. For instance, an image of a person's face can be thought of as a linear combination of face features (skin tone, nose shape, eye color, eyebrows density, etc). It's really interesting to look at it like this: there's a universal set of building blocks (or features) of all the faces in the world and what really separates one face from the other is the different proportion of these features that each face has. NMF has become so popular because of its ability to automatically extract **sparse** and easily **interpretable** factors [Gillis et al., 2014](https://arxiv.org/pdf/1401.5226.pdf).
<img src="/images/mf-imgs/facial-images-nmf.png" width="600" height="400">

But NMF can be applied to text as well. An article can be thought of as a weighted combination of a set of topics. The higher the weight of a particular topic in the article, the higher the frequency of the occurence of the words representing that topic. For instance, a news article can have as a topic: politics, humanitites and technology. If, for instance, the article is talking more about policitcs than anything else, we should expect politics-related words such as "president", "elections", "senate" to be more frequent than other words such as "love", "freedom", or "computers". Similarly to the face example, we can view all the documents in the world through the lens of a universal set of topics that are distributed differently through each document.

NMF can expressed as the following optimization problem:

$$
\begin{equation}
   J = \min_{U,V} ||UV^T - Y||^2_F \text{ such that } U \geq 0 \text{ and } V \geq 0
\end{equation}
$$

### Issues with NMF

A question remains however, as to how one might solve this optimization problem. [Gillis et al., 2014](https://arxiv.org/pdf/1401.5226.pdf) discuss *several issues* with solving the NMF problem in practice:

   1. While the unconstrained problem can be solved with SVD, the constrained one cannot. NMF is an **NP-Hard** problem, with no efficient polynomial-time exact solution. Hence, we rely on nonlinear optimization methods that do not have convergence guarantees.
   2. It's **ill-posed**: If (among other conditions) a problem has more than one unique solution, it is said to be *ill-posed*. This is the case with NMF: there exists more than one solution matrices $$U$$ and $$V$$. This can be problematic in the sense that it gives rise to different interpretations: different topics and weights, in topic modeling, for example. In practice, this is can be alleviated with priors on $U$ and $V$, and regularization terms in objective function. [Gillis et al., 2014](https://arxiv.org/pdf/1401.5226.pdf) gives a very simple example of multiple solutions to NMF: 
   ![](https://i.imgur.com/9mw0fql.png)

   3. **Choosing** the low rank value $$k$$ can be tricky, and application-dependent. Several ways to achieve it: trial and error, inspect singular values decay with SVD.


**Algorithms**:
1. SGD: compute

Now that we have the hammer (NMF), let's look at some NLP nails!

### NLP Applications

#### Topic Modelling

To make use of matrix factorization when dealing with documents, we must find a way to represent a document as a matrix. The most common way is what is known as *term-document matrix*. Each row in the matrix `$Y \in \mathbb{R}^{m \times n}$`. Each row is a document and each column is a word. But this is a matrix of real numbers, and we need a number to put in `$Y_{ij}$` that somehow relates document `$i$` with word `$j$`. The most common ways to fill `$Y_{ij}$` are :
		
* Occurence frequency of word `$i$` in document `$j$`, which is an unnormalized bare count.
* [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weighted occurences, weighting the frequency by the importance of the word `$i$` to document `$j$` (the IDF part).

Once we have the matrix `$Y$`, now enters matrix factorization. As mentioned before, NMF can help us visualize each row (document) in our input matrix $Y$ as a linear combination of a set of building blocks (topics). 

We first preselect the number of topics $k$. Then, we use NMF to compute `$ Y = UV^{T}$`, where $U \in \mathbb{R}^{m \times k} $ and `$V \in \mathbb{R}^{m \times k}$`. We can interpret the resulting matrices as follows: $U$ is the topic assignment matrix, where $U_{d,t}$ is the proportion of topic $t$ in document $d$, and $V$ is the topic-word matrix, where $V_{w,t}$ is the proportion of word $w$ in topic $t$. NMF has gave us a very interesting and interpretable way to dissect our document-word matrix! 

Let's do a simple sklearn example of topic modeling with NMF


#### Document Clustering
