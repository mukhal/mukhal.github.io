---
layout: post
title: "Training LLMs to Cite their Pretraining Data"
author: Muhammad Khalifa
comments: true
published: true
---

> LLMs learn tons of world knowledge from pretraining, but as much of this knowledge changes, becomes obsolete, or is completely wrong, an LLM user should be able to judge for themselves whether a piece of knowledge is accurate.  Last year we had a [paper](https://arxiv.org/abs/2404.01019) at COLM '24, where we explored a new task that we referred to as *intrinsic source citation*, a task where LLMs not only needs to answer a user query e.g., Who starred in Before Sunrise?, but also provide a *link* to a source i.e., a citation where can this information be verified. 
<!--more-->
You might be wondering: what happened to retrieval augmented generation (RAG) then? why can't we just first retrieve relevant sources, then use the LLM to answer questions based on them. RAG is indeed one solution to this and can provide some credible citations to retrieved sources. But let's admit it; RAG is not neat, and adds an extra layer of complexity and overhead. Also, RAG can not help us attribute *parametetric* knowledge stored in the model's weights and not present in the retrieval corpus. We sought a first-principles solution that can build this into the model from the ground up. 


## Source-Aware Training: A First Step

In our COLM '24 paper, [Khalifa et al., 2024](https://arxiv.org/abs/2404.01019), we introduced the concept of **intrinsic source citation**. The core idea is to make LLMs aware of the source of their knowledge during pretraining, so that they can later cite the source when generating an answer. 

<img src="images/intrinsic-source-citation.png" alt="Intrinsic Source Citation" style="max-width: 600px; width: 100%; display: block; margin: 1.5em auto;" />


Our approach, called **source-aware training**, involves two main steps:

1. **Source Tagging During Pretraining:** Each document in the pretraining corpus is tagged with a unique identifier. The model is trained to associate facts with these source tags.
2. **Instruction Tuning for Citation:** After pretraining, we fine-tune the model with instructions that prompt it to answer questions and provide the supporting source identifier.

We found that this simple recipe enables LLMs to attribute their answers to the correct pretraining source with high fidelity—at least in synthetic settings. Importantly, this is achieved with minimal changes to the model architecture or training pipeline, and without a significant hit to language modeling performance.

## Beyond Verbatim: The Challenge of Paraphrase and Composition

While source-aware training works well for verbatim memorization, real-world knowledge is often paraphrased or composed from multiple sources. This is where the recent work by [Huang et al., 2025](https://arxiv.org/abs/2506.17585) makes a significant advance.

They introduce **Cite Pretrain**, a framework for **retrieval-free knowledge attribution**. Their key insight is that simply appending source IDs (what they call “Passive Indexing”) is not enough—models struggle to attribute paraphrased or compositional facts. To address this, they propose:

- **Active Indexing:** During continual pretraining, the model is exposed to synthetic QA pairs that restate each fact in diverse forms and require the model to both generate content from a cited source and attribute its own answers.
- **Bidirectional Training:** The model learns both to answer questions given a source and to cite the source given a fact, reinforcing the association.

Their experiments on the CitePretrainBench benchmark show that Active Indexing dramatically improves citation precision, especially for paraphrased and multi-fact questions. Performance continues to improve as the amount of augmented data increases, suggesting that scaling up this approach could yield even more reliable attribution.

## Where Are We Headed?

Together, these works point to a future where LLMs can not only provide answers, but also transparently cite the origins of their knowledge—without the need for external retrieval. This has major implications for transparency, trust, and the responsible deployment of language models.

Of course, challenges remain: scaling these methods to real-world corpora, handling ambiguous or multi-source facts, and ensuring that citation does not come at the cost of language modeling quality. But the progress so far is promising, and the field is moving rapidly.



**References:
Khalifa et al., 2024. "Source-Aware Training Enables Knowledge Attribution in Language Models." arXiv:2404.01019
Huang et al., 2025. "Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models." arXiv:2506.17585