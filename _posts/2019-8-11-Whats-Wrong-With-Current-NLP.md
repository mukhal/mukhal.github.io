---
layout: post
title: "What's Wrong with Current NLP?"
author: Muhammad Khalifa
comments: true
published: false
---


The field of Natural Language Processing (NLP) has recently witnessed dramatic progress with state-of-the-art results being published every few days. Leaderboard madness is diriving the most common NLP benchmarks such as GLUE and SUPERGLUE with scores that are getting closer and closer to human-level performance. This is great, right? It seems like we're finally solving the once-upon-a-time hard problem of NLP. Well, not so fast. My aim in this article is to point out why we should not be so thrilled with current leaderboard scores and to point out both the issues and challenges currently standing in the way of achieving human-level language understanding.



## [1] Massive Resources Need

The most common trend in NLP today is **Transfer Learning** (TL) employed in the form of [Language Modeling Pre-training](https://arxiv.org/abs/1801.06146). Almost all SOTA results achieved recently have been mainly driven by a two-step scheme: 
1. Pre-train a monster model for Language Modeling on a large general-purpose corpus. 
2. Finetune the whole model (or a subset thereof) on the target task.
 [ELMO](https://arxiv.org/abs/1802.05365), [BERT](https://arxiv.org/abs/1810.04805), [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [RoBERTa](https://arxiv.org/abs/1907.11692) are all instances of the same technique.

The main problem with these methods is the tremendous resource craveness. What I mean by resources is both *data* and *compute power*. For instance, it has been estimated that it costs around [$250,000](https://twitter.com/eturner303/status/1143174828804857856) to train XLNET on 512 TPU v3 chips with only 1-2% gain over BERT in 3/4 datasets. 

Such resource hungriness comes with multiple issues :

#### Difficult Reproducibility
[paper of hard reproducibity in recommendartion] To be able to use or build upon a particular research idea, it's imperative for that idea to be relatively easy reproducible. Small tech companies and research labs will not be able to reproduce these pre-trained beasts much less improve upon them.

#### Leaderboards Are No Longer Enough
Anna Rogers argues in her [blog post](https://hackingsemantics.xyz/2019/leaderboards/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter) why `more data & compute = SOTA` is NOT research news. She argues that the main problem with leaderboards is that the rank of a model is totally dependent on its task score with no consideration given to the amount of data, compute or training time needed to achieve that score.

#### This is not how we learn
Our brains as humans take a different path towards language learning. We definitely do not need to see tens of millions of instances of a specific word in context to understand the meaning of the word. I believe this pretrain-finetune scheme lacks a significant resemblance to the way humans learn. One might argue, however, that as long as an approach produces good results, whether it's similar or not to how humans learn doesn't actually matter. Maybe, but I presume that if we aim at building machines that achieve human-level intelligence, we must not get carried away with approaches that are singinifcantly dissimilar to the way our brains work.



Carbon Footprint
