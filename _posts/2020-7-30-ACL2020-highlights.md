---
layout: post
title: "My ACL 2020 Highlights"
author: Muhammad Khalifa
comments: true
published: true
---


This post is a bit late, but I really wanted to write. ACL 2020 has been very special, since it is my first conference to attend. I have found the virtual version to be very nice (Altough my testimony is a bit undermined by the fact that I have not experienced an actual conference before. So, I cannot really compare the virtual version to the actual one). Anyway, I found the discussions, Q&A, chat rooms, and the live talks were very engaging and interesting!



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

Another very interesting line of work is employing *optimization techniques* for unsupervised text generation tasks, which was done in at least 3 papers on [simplification](https://www.aclweb.org/anthology/2020.acl-main.707.pdf), [sentence summarization](https://www.aclweb.org/anthology/2020.acl-main.452.pdf), and [paraphrase generation](https://www.aclweb.org/anthology/2020.acl-main.28.pdf). In all these papers, an objective (scoring) function defines the goodness of the simplified sentence, the summary, or the paraphrase. For simplification, the scoring function is composite (product) of the score from syntax-aware language model, cosine similarity between the source and output sentences, entity score, and length of the output sentence. The algorithm proceeds in an iterative fashion by selecting one action of (removal, extraction, reordering, and substitution) and, in a greedy fashion, keep the best candidate sentence for future operations. As for sentence summarization, the search objective is defined in a similar manner to simplification in addition to a hard length constraint so that longer sentences are assigned a very low objective value. The search space is $n \choose s$ possible summary sentences and optimzation is done with hill climbing. Optimzation-based approaches are interesting but have some drawbacks. First, the search space can be huge — making the process very inefficient. This also limits this approach to rather simpler seq2seq task, but can become completely infeasible if we are talking about article summarization or machine translation. Second, there are many components in the objective function, and it can become difficult to select an objective function that will produce desired output sentences.


## Interpretability of NLP systems
This has been a headline topic for ACL 2020, with many papers focusing on the analysis of neural "black boxes". A paper titled [Understanding Attention for Text Classification](https://www.aclweb.org/anthology/2020.acl-main.312.pdf) examined the relationship between attention weights and classification accuracy. They define 3 types of tokens based on assosciations between tokens and labels: positive, negative, and neutral. They discuss conditions under which the attention mechanism may become more (or less) interpretable, and show how the interplay between the two quantities may impact model performance. They find that attention is not directly related to polarity: while polarity scores are interpretable and make sense, attention scores necessarily do not even in cases where the model is robust. Another paper [Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?](https://www.aclweb.org/anthology/2020.acl-main.491.pdf) compares different interpretation methods of machine learning models by measuring *simulatability*, which is the predictability of models' behaviour. They show that using the right metrics when evaluating explanable AI is vital, and that current approaches can still be further imporoved. A third paper titled [Learning to Deceive with Attention-Based Explanations](https://www.aclweb.org/anthology/2020.acl-main.432.pdf) showed that attention-based interpretations can be completely unreliable. Another interesting [opinion piece](https://www.aclweb.org/anthology/2020.acl-main.386.pdf) address the faithulness aspect of interpretation — calling into question the defintion of "faithfulness" used across the research community.


# Evaluation of NLP Systems
Several papers addressed the current state of evaluation metrics of NLP systems, with a lot of focus on text generation metrics. A paper titled [Tangled up in BLEU: Reevaluating the Evaluation of Automatic Machine Translation Evaluation Metrics](https://www.aclweb.org/anthology/2020.acl-main.448.pdf). The paper addresses variability in correlation between human and automatic evaluation depending on the quality of the MT systems used. They also discuss *outlier* systems (systems that are much worse than other systems) that can significantly impact the correlation score. Another interesting paper by Google Research proposed [BLEURT](https://www.aclweb.org/anthology/2020.acl-main.704.pdf), which is a BERT-based metric for evaluation of machine translation systems. BLEURT is pre-trained in a multi-task fashion on a 1.8M artifically produced sentence-pairs obtained by perturbation of sentences through masking tokens, paraphrasing by backtranslation, and dropping random words. Pre-training tasks include automatic metrics such as BLEU and ROUGE, textual entailment, and backtranslation likelihood. BLEURT shows better agreement with huamn annotation than other commonly used metrics such as BLEU, METEOR, and BERTScore.

The **best paper** award was given to [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://arxiv.org/pdf/2005.04118.pdf), a very interesting paper that highlights various shortfalls in the way NLP systems are currently being tested. It proposing testing NLP models as a software engineer would a piece of code. The paper proposes creating a two-dimensional "checklist". The first dimension encompasses the different testing subjects (What to test) such as vocabulary and negation. The second dimesion represents different testing scenarious such as minimum functionality test (make sure that the model basically works), and perturbation tests, where you perturb the input expecting or not expecting the model's output to change in return. They also provide an [open-source tool](https://github.com/marcotcr/checklist) that makes writing such tests at scale very handy through integrations with Jupyter notebooks. Testing SoTA models both commercial and research showing striking failure rates (bugs) on various testing aspects. I really suggest watcing the paper presentation [video](https://slideslive.com/38929272).

Other papers investigated current evaluation [implicit discourse relation classification](https://www.aclweb.org/anthology/2020.acl-main.480.pdf), and [open-domain dialogue generation](https://www.aclweb.org/anthology/2020.acl-main.333.pdf) 


# Low-resource NLP
Low-resource NLP is one of my favorite topics (unsuprisingly, my master's research is on low-resource summarization). I think it is good news that low-resource languages are receiving more attention than before. The current research stage is like a race where high-resource languages take the lead and it is expected to be the case for a long time. Thus, in my opinion, we need to double or even triple our research on low-resource settings to just begin achieving as high performance as is seen with high-resource languages. 


Back to ACL, an interesting [paper](https://www.aclweb.org/anthology/2020.acl-main.523.pdf) proposes to leverage sentence-level labels to pre-train a sequence tagger model to improve low-resource NER. In addition to pre-training, they also employ a multi-task setting through using the sentence classification loss as an auxiliary loss during training. They show some improvement on NER for 3 low-resource languages: Vietnamese, Thai, and Indonesian. Another interesting [paper](https://www.aclweb.org/anthology/2020.acl-main.437.pdf) models discusses the viability of using the learned discret variable (learned from raw text) as features for low-resource classification. More specifically, after training a VAE on raw text, the VAE encoder is fixed and used to encode text into latent variables while a set of task-specific parameters are trained on top. Other papers adressed low resouce [goal-oriented dialog](https://www.aclweb.org/anthology/2020.acl-main.57.pdf), [question generation](https://www.aclweb.org/anthology/2020.acl-main.601.pdf), and [entity linking](https://www.aclweb.org/anthology/2020.acl-main.601.pdf).

Back to ACL, an interesting [paper](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00294) attempts low-resource dependency parsing using the semi-supervised technique of self-training. 


# Keynote talks

I enjoyed all three keynote talks. Here, I conclude by summarizing the first keynote by Kathleen Mckeon on **Rewriting the Past: Assessing the Field through the Lens of Language Generation**. Honeslty, it was an engaging talk that went over the history of NLP models starting from the 1980s. Interestingly, the talk took the form of a series on interviews done by Mckeon with key contributors in the field. Starting with the **present**, which is highly characterized by deep neural models. Mckeon asked her interviewees ***what is the greatest achievement of neural nets in NLP?** The main answers were: 
* Can be operated by almost anyone as they do not need linguistic knowledge
* Learnign from raw data

**Why do they work well?**
* The learn good represenations of word semantics
* attention!
* They fit a highly non-linear function to the data.

Now for the **past**:
* **1980s**: focus was on linguistics, centering theory, philosophy, and intent
* **1990s**: focus on data analysis at larger scale, using coropora to analyze word choice and constraints.
* **1990s - 2000s**: the rise of statistical NLP: mutual information models, jaccard index, and moving to classical machine learning (SVM, Naive Bayes, etc)

Lastly for the **future**, she asked **what is deep learning not suited for?**:
* constraint on choice in language generation: while we speak with a purpose, say what we mean, and plan our longer speaches, nerual NLG do not.
* they still generate non-truthful generations
* they leverage spurious correlations in the dataset

**How about the data?**
* we still need to investigate how good is the data we are training on
* Why is the model doing what it is doing?

We also need to learn from **other disciplines**: psychology, neuroscience, and cognitive science, etc. We need to study more *worthy* and harder tasks such as summarization of very long documents, and working on small data.


