---
layout: post
title: "Paper Discussion: Language as a Latent Variable: Discrete Generative Models for Sentence Compression"
author: Muhammad Khalifa
---

Haven't posted in a while, but I decided to get back and get back strong! In this post, I will discuss the 2016 paper [Language as a Latent Variable: Discrete Generative Models for Sentence Compression](https://arxiv.org/abs/1609.07317). The reason why I chose this paper is two-fold : First, it combines a lot of important ideas and concepts such as **Variational Auto Encoders**, **Semi-Supervised learning** and **Reinforcement Learning**. Second, it's relevant to my master's thesis and I find that writing about a topic help make it clearer.


**Paper Brief** : The main idea of the paper is that an autoencoder that takes a sentence as input, should learn to map that input into a space where important information regarding the sentence are kept, such that the model is able to reconstruct the sentence again. By enforcing this intermediate representation (Latent Variable) to be in the form of a discrete language model distribution (a valid sentence), we are able to learn a compact version of the input sentence.


**Variational Autoencoders** : Variational Autoencoders are typical autoencoders with the exception that the latent variable distribution should exhibit a gaussian prior. This pushes the encoder to output diverse representations. It also prevents the encoder from cheating by assigining each image a distinct point in space. The loss function of the VAE is split into two parts: Decoder Loss and KL Divergence between the encoder posterior and the prior of the latent variable $z$:

In this paper, the compressed form of the sentence is treated as the latent variable (hence the name of the paper). 


$$
\begin{equation}
    L(\theta, \phi) = \mathbb{E}_{z \sim q_{\phi}}[log p_{\theta}(x|z)] + KL (q_{\phi}(z|x) || p(z))
\end{equation}
$$

Where $\theta$ are the decoder parameters and $\phi$ are the encoder parameters.

**Compression model** : Back to our language generation setting, the encoder is a bi-LSTM to that processes the input sentences. Following that, another [pointer network](https://arxiv.org/abs/1506.03134) attends over the encoder states to produce the compressed input sentence. These two networks form the compression model  `$q_{\phi}(c|s)$`

**Reconstruction model** : This includes the compressor and the decoder together outputting the reconstructed sentence with probability `$p_{\theta}(s|c)$`. The reconstruction occurs as follows: the compressor LSTM encodes the compressed sample. Then, the decoder LSTM attends to the compressor hidden states to output the recontructed sentence



**Training** : 

* **Decoder** : Decoder parameters `$\theta$` are updated directly with gradient descent 

$$
\begin{equation}
    \frac{\partial L}{\partial \theta} = \mathbb{E}_{q_{\phi(c|s)}} [\frac{log p_{\theta}(s|c)}{\partial \theta}]
\end{equation}
$$