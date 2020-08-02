---
layout: post
title: "ACL 2020 Higlights"
author: Muhammad Khalifa
comments: true
published: false
---


This post is a bit late, but I really wanted to write. ACL 2020 has been very special, since it is my first conference to attend. I have found the virtual version to be very nice (Altough my testimony is a bit undermined by the fact that I have not experienced an actual conference before. So, I cannot really compare the virtual version to the actual one). Anyway, I found the discussions, Q&A, and the live talks were very engaging and interesting!



There are many trends in this year's ACL 2020, with specific focus on interpretability, low-resource, transformers limitations, and some novel and interestings tasks and datasets.


## New Challenges 
One of the interesting papers is [PuzzLing Machines: A Challenge on Learning From Small Data](https://arxiv.org/abs/2004.13161), which proposes an interesting challenge for machines to learn from small data. The idea is to test machines' ability to reason (System-2) or human-like intelligence. 
The challenges include 96 Rosetta Stone (*translation)* puzzles from international olympiads, 39 different language families. This puzzles are designed so that they are not possible to solve them by memorization.
Also, they target linguistic phenomena such as syntax, morphology, phonology and semantics. Experiments were done with various baselines ranging from random phrase-based Statistical MT, neural MT (Transformer, pre-trained RoBERTa encoder). Suprisingly, The highest results were obtained by SMT models. Moreover, foreign → English results are higher than English → foreign? why? maybe English is generally easier. The maximum Exact-match score is 3.4% (SMT) only, suggesting there's much more to do in that domain.


Another very interesting dataset was proposed in [Exploring Content Selection in Summarization of Novel Chapters](https://www.aclweb.org/anthology/2020.acl-main.453.pdf), where a dataset was proposed for summarization of novel chapters. The task involves extreme paraphrasing, which should make it much more difficult than conventional news summarization. The dataset involves full book texts from projext gutenberg, and chapters are x7 longer than news articles. The authors employ extractive summarization as a first step. They explore different alignment methods and propose one-to-one alignments between reference summary sentences and chapter sentences showing it performs best against earlier work. In a similar context, the paper [Screenplay Summarization Using Latent Narrative Structure](https://www.aclweb.org/anthology/2020.acl-main.174.pdf) addresses summarization of another type of long documents, that is, screenplays. It also employs extractive techniques while modeling the narrative structure (in form of turning points) as a latent variable.


## Unsupervised, Zero-shot, and Few-shot Learning
Many papers have adopted the theme of few-shot and unsupervised learning. The paper [Few-Shot NLG with Pre-Trained Language Model]() have explored table-to-text model leveraing pre-trained LM as a generator. A copy mechanism inspired by pointer networks is added to help the pretrained LM copy tokens as-is from the input table. It also helps alleviate the mismatch between the randomly-initialized encoder and the pre-trained decoder.
A Copy loss term is added to the loss to guide the model to copy words that are already in the input table. It's worth mentioning that I have seen several papers using the copy mechanism with transformers with results suggesting that it generally improves performance.

In a similar context, a paper titled [Do you have the right scissors? Tailoring Pre-trained Language Models via Monte-Carlo Methods]() have adressed the over-estimation or underestimation problem when fine-tuning pre-trained LMs on small datasets, in which the probability according to the model is different (higher or lower) than the probability according to the true data distribution. Why does this happen? Since MLE minimizes the Kullback-Leibler divergence between model and true distributions, it tries to cover the data region as much as it can but in the process, assigns probability mass to non-data regions (over-generalization).
 
<img src="https://camo.githubusercontent.com/18e1e721b5cddf3b372a1412cf6bcfd4a2f34353/68747470733a2f2f692e696d6775722e636f6d2f68356751505a462e706e67" width=400 height=200>
*over- and under-estimation when fine-tuning Pre-trained LMs*

The paper proposes a 3-component approach: Ratio estimator, P_tailor probability, and Early Rejection Sampling to solve this. Ratio estimator is a CNN trained to discriminate between points x from the data, and x coming from the model. P_tailor probability The ratio estimator you have computed previously can be used to compute the fixed tailored probability. The only thing left is to sample from this tailored probability, which is done using Early Rejection Sampling.

Another very interesting line of work is employing *optimization techniques* for unsupervised text generation tasks, which was done in at least 3 papers on [simplification](https://www.aclweb.org/anthology/2020.acl-main.707.pdf), [sentence summarization](https://www.aclweb.org/anthology/2020.acl-main.452.pdf), and [paraphrase generation](https://www.aclweb.org/anthology/2020.acl-main.28.pdf). In all these papers, an objective (scoring) function defines the goodness of the simplified sentence, the summary, or the paraphrase. For simplification, the scoring function is composite of the score from syntax-aware language model, cosine similarity between the source and output sentences, entity score, and length of the output sentence. The algorithm proceeds in an iterative fashion by selecting one action of (removal, extraction, reordering, and substitution) and, in a greedy fashion, keep the best candidate sentence for future operations. The approach is interesting but has some drawbacks. First, the search space can be very large making the process very inefficient. This also limits this approach to rather simpler seq2seq task, but can become completely infeasible if we are talking about article summarization or machine translation. Second, there are many components in the objective function, and it can become difficult to select an objective function that will produce desired output sentences.


## Interpretability of NLP systems
This is a hot topic for ACL 2020, w

