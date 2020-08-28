---
layout: post
title: "Matrix Factorization in NLP"
author: Muhammad Khalifa
comments: true
published: false
---


It's good to be back! Since I am currently studying the [Computational Linear Algebra](https://github.com/fastai/numerical-linear-algebra/) course by Rachel Thomas, I was impressed by the power and utlity of Matrix Factorization algorithms, and their very wide range applications in many domains including Information Retrieval, Image Processing, Computer Vision and (dearly) Natural Lanugage Processing. 

Unforutnately, a blog post is never enough to talk about matrix factorization and its relationship to NLP. There's a plethora of Matrix factorzation algorithms, each of these algorithms probably has been used in an NLP application. This renders infeasable writing a comprehensive article on the subject. However, **Where there's will, there's way** and I will do my best to cover as NLP-related application as possible.



#### Disclaimer
>
This article assumes the reader is faimilar with many linear algebra concepts including Vectors, Matrices, Matrix-Vector and Matrix-Matrix Multiplication, Subspaces, Basis, Linear Independence, Rank, and Orthogonality.


### Vanilla Matrix Factorization

![](mf-imgs/low-rank-mf.png)
### Nonnegative Matrix Factorization


### Topic Modelling

To make use of matrix factorization when dealing with documents, we must find a way to represent a document as a matrix. The most common way is what is known as term-document matrix. Each row in the matrix `$M \in \mathbb{R}^{d \times w}$`. Each row is a document and each column is a word. But this is a matrix of real numbers, and we need a number to put in `$M_{ij}$` that somehow relates document `$i$` with word `$j$`. The most common ways to fill `$M_{ij}$` are :
		
1. Occurence frequency of word `$i$` in document `$j$`, which is an unnormalized bare count.
2. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) weighted occurences, weighting the frequency by the importance of the word `$i$` to document `$j$` (the IDF part).

The bottomline is no matter how you choose `$M_{ij}$`, you want a real-number connecting the word and the document together. Once we have the matrix `$M$`, now enters matrix factorization.


When dealing with topic modelling, we a



#### Latent Semantic Indexing

* NMF
* SVD

### Relation Extraction
Riedel et, al 2013

### Mixture of softmax

### Distillation, compression of models



### Word Embeddings




## 
