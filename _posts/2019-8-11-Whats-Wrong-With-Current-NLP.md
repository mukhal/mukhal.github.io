---
layout: post
title: "What's Wrong with Current NLP?"
author: Muhammad Khalifa
comments: true
published: false
---


The field of Natural Language Processing (NLP) has recently witnessed dramatic progress with state-of-the-art results being published every few days. Leaderboard madness is diriving the most common NLP benchmarks such as GLUE and SUPERGLUE with scores that are getting closer and closer to human-level performance. This is great, right? It seems like we're finally solving the once-upon-a-time hard problem of NLP. Well, not so fast. My aim in this article is to point out why we should not be so thrilled with current leaderboard scores and to point out both the issues and challenges currently standing in the way of achieving human-level language understanding.



## [1] Sample Inefficiancy

The most common trend in NLP today is Transfer Learning (TL) employed in the form of [Language Modeling Pre-training](https://arxiv.org/abs/1801.06146). Almost all SOTA results achieved recently have been mainly driven by a two-step TL scheme: (1) pre-train a Monster model for Language Modeling on a very large corpus. (2) Finetune the whole model (or a subset thereof) on the target task. BERT, GPT-1, GPT-2, RoBERTa, ELMO and many others are all instances of the same technique.