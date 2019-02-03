---
layout: post
title: "Paper Discussion: Language as a Latent Variable: Discrete Generative Models for Sentence Compression"
author: Muhammad Khalifa
---

Haven't posted in a while, but I decided to get back and get back strong! In this post, I will discuss the 2016 paper [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](https://arxiv.org/abs/1609.07317). The reason why I chose this paper is two-fold : First, it combines a lot of important ideas and concepts such as **Variational Auto Encoders**, **Semi-Supervised learning** and **Reinforcement Learning**. Second, it's relevant to my master's thesis and I find that writing about a topic help make it clearer.


**Paper Brief** : The main idea of the paper is that an autoencoder that takes a sentence as input, should learn to map that input into a space where important information regarding the sentence are kept, such that the model is able to reconstruct the sentence again. By enforcing this intermediate representation (Latent Variable) to be in the form of a discrete language model distribution (a valid sentence), we are able to learn a compact version of the input sentence.


**Variational Autoencoders** : Variational Autoencoders are a typical autoencoders with the exception that the latent variable distribution should exhibit a gaussian prior. 
