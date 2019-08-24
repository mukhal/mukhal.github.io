---
layout: post
title: "Current Issues with Transfer Learning in NLP"
author: Muhammad Khalifa
comments: true
published: true
---


> Natural Language Processing (NLP) has recently witnessed dramatic progress with state-of-the-art results being published every few days. Leaderboard madness is diriving the most common NLP benchmarks such as GLUE and SUPERGLUE with scores that are getting closer and closer to human-level performance. Most of these results are driven by transfer learning from large scale datasets through super large (Billions of parameters) models. My aim in this article is to point out the issues and challenges facing transfer learning and point out some possible solutions to such problems.



### Computational Intensity

**Transfer Learning** is typically employed in the form of [Language Modeling Pre-training](https://arxiv.org/abs/1801.06146). Almost all SOTA results achieved recently have been mainly driven by a two-step scheme: 
1. **Pre-train** a monster model for Language Modeling on a large general-purpose corpus (The more data the better). 
2. **Finetune** the whole model (or a subset thereof) on the target task.

 [ELMO](https://arxiv.org/abs/1802.05365), [BERT](https://arxiv.org/abs/1810.04805), [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [RoBERTa](https://arxiv.org/abs/1907.11692) are all instances of the same technique. The main problem with these methods is the tremendous resource craveness. What I mean by resources is both *data* and *compute power*. For instance, it has been estimated that it costs around [$250,000](https://twitter.com/eturner303/status/1143174828804857856) to train XLNET on 512 TPU v3 chips with only 1-2% gain over BERT in 3/4 datasets.

 This takes us to the next issue:


### Difficult Reproducibility
Reproducibility is a already becoming a problem in machine learning research. For instance, [(Dacrema et. al)](https://arxiv.org/pdf/1907.06902) analyzed 18 different proposed Neural-based Recommendation Systems and *found only 7 of them were reproducible with reasonable effort*. Generally speaking, to be able to use or build upon a particular research idea, it's imperative for that idea to be easily reproducible. With the substantial computational resources needed to train these huge NLP models and reproduce their results, small tech companies, startups, research labs and independent researchers will not be able to compete.


### Leaderboards Are No Longer Enough
Anna Rogers argues in her [blog post](https://hackingsemantics.xyz/2019/leaderboards/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter) why `more data & compute = SOTA` is NOT research news. She argues that the main problem with leaderboards is that the rank of a model is totally dependent on its task score with no consideration given to the amount of data, compute or training time needed to achieve that score. 


<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">Here&#39;s a summary post on problems with huge models that dominate <a href="https://twitter.com/hashtag/NLProc?src=hash&amp;ref_src=twsrc%5Etfw">#NLProc</a> these days. I put together several different discussion threads with/by <a href="https://twitter.com/yoavgo?ref_src=twsrc%5Etfw">@yoavgo</a>, <a href="https://twitter.com/jaseweston?ref_src=twsrc%5Etfw">@jaseweston</a>, <a href="https://twitter.com/sleepinyourhat?ref_src=twsrc%5Etfw">@sleepinyourhat</a>, <a href="https://twitter.com/bkbrd?ref_src=twsrc%5Etfw">@bkbrd</a>, <a href="https://twitter.com/alex_conneau?ref_src=twsrc%5Etfw">@alex_conneau</a>, <a href="https://twitter.com/SeeTedTalk?ref_src=twsrc%5Etfw">@SeeTedTalk</a>. <a href="https://t.co/MokmmEYx91">https://t.co/MokmmEYx91</a></p>&mdash; Anna Rogers (@annargrs) <a href="https://twitter.com/annargrs/status/1152194347942731776?ref_src=twsrc%5Etfw">July 19, 2019</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


I suggest you check the above thread for various comments on the problem.  Rohit Pgarg suggest comparing the performance of models on a two-dimensional scale of both task accuracy and computational resource. See the plot below. Also there's a very interesting comment by Alexandr Savinov where he suggests to use how much of input information the algorithm is able to "pack" to one unit of output (model parameter) representation for one unit of CPU time.

|<img src="/images/scatter-tl.png" width="700" height="400" />|
|:--:| 
| Using Computational Resource as an additional metric to task accuracy in comparing models performance|


### This is NOT How We Learn
It's true that we use transfer learning in our everyday life. For instance, if we know how to drive a manual car, it becomes very easy for us to transfer the acquired knowledge (such as of using the brakes and the gas pedal) to the task of driving an automatic car. However, Our brains as humans take a different path towards language learning. We definitely do not need to see millions of instances of a specific word in context to understand the meaning of the word or to know how to use it. I believe this pretrain-finetune scheme lacks a significant resemblance to the way humans learn. One might argue, however, that as long as an approach produces good results, whether it's similar or not to how humans learn doesn't actually matter. Maybe, but I presume that if we aim at building machines that achieve human-level intelligence, we must not get carried away with approaches that are singinifcantly dissimilar to the way our brains work.


### High Carbon Footprint
Believe it or not but training these models has a negative effect on the environment. [(Strubell et. al)](https://arxiv.org/pdf/1906.02243.pdf) compare the estimated $CO_2$ emissions from training Big Transformer architecture to emissions caused by other $CO_2$ sources such as the lifetime of a car.
<img src="/images/carbon-footprint.png" width="400" height="300" />

 [(Schwartz et. al)](https://arxiv.org/pdf/1907.10597) introduce what they call *Green AI*, which is the practice of making AI both more *efficient* and *inclusive*. Similar to what we discussed above, they strongly suggest adding efficiency as another metric alongside task accuracy. They also believe it's necessary for research papers to include the "price tag" or the cost  of model training. This should encourage the research towards more efficient and less resource-demanding model architectures.
