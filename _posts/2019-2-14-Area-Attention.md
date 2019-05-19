---
layout: post
title: "Paper Discussion: Area Attention"
author: Muhammad Khalifa
comments: true
published: false
---

**Paper Brief**  : Since the attention mechanism has been proposed in [Bahdanau et. al](https://arxiv.org/abs/1409.0473),  it has become an almost essential component in NLP models. The idea of sequence-to-sequence attention can be simply expressed as follows: Given the current state of the decoder (the query), what are the important elements in the input (keys and values) that I need to focus on to better solve the task at hand.
Almost all previous uses of attention have considered attending to single entities, usually an input token, sentence or an image grid. The [paper](https://arxiv.org/abs/1810.10126) we're about to discuss asks a simple question: Instead of considereing single entities for attention, why don't we consider an aggregate of an area of adjacent items. *(The paper was rejected in ICLR 2019, but I thought the idea is worth exploring nonetheless)*



## Approach 


**1-D Attention** : with language-related tasks, area attention spans adjacent elements in a linear 1-dimensional memory. The figure below shows an original memory (encoder hidden states, for eg.) with 4 elements. From these 4 elements, four 1-item areas, three 2-item areas and two 3-item areas can be extracted. 

![](/images/1d-aa.PNG)


**2-D Attention** : with image-related tasks, area attention spans adjacent elements in a linear 2-dimensional memory. In this case the number of different combinations is apparently larger than the 1-dimensional case.

![](/images/2d-aa.PNG)

**Aggregate Keys and Values** : The authors proposed to use the mean of the item keys as the new key for the resulting area, and the sum of item values as the value for the output area. Another idea they used to enrich the area representation is to add the standard deviation of the constituent item keys.


**Merging all area features**: Three main features are combined to produce the final area key representation: mean of keys of each item, standard deviation of item keys as well, and the width and height of the area: 

* The mean of the item keys as the new key for the resulting area
* Standard deviation of the constituent item keys.
* height and width of the area projected through a learned embeddings matrix.

Features are combined by means of a single-layer perceptron followed by a linear transformation.


**Faster Computation with Summed Area Table** : To compute the area atttention with maximum value `$A$` for a memory of size `$M$`, we need `$O(|M| A^2)$` steps. This becomes obvious when you understand that to compute the area attention of maxium size $A$ you will need `$A+ (A-1) + (A-2) + .. + 1$ = $A(A+1) / 2 = O(A^2)$` steps. To overcome such expensive computations, the authors propose to use a pre-computed summed area table that is calculated only once, then it's used to compute the area attention in a constant time.

Letting `$I_{x,y} = v_{x,y} + I_{x,y-1} + I_{x-1,y}$`
where `$x$` and `$y$` are the coordinated of the item in the memory. Now to calculate `$v_{x1,y1,x2,y2}$`, which is the area located with the top-left corner at `$(x_1, y_1)$` and the bottom-right corner at `$(x_2, y_2)$`, we can simply use the pre-computed sum table as follows 
`$v_{x1,y1,x2,y2} = I_{x2,y2} + I_{x1,y1} - I_{x2,y1} - I_{x1,y2}$`.

 See the sketch below for a sample case where `$(x_1,y_1)=(0,0)$` and `$(x_2,y_2)= (1,1)$`

 ![](/images/aa-sum.PNG)

 To obtain the mean of an area, all you have to do is to divide `$v_{x1,y1,x2,y2}$` by the number of elements in the area. The standard deviation of an area can be calculated in a similar manner but with a slightly different equation.


## Results

The authors mainly experiment with two NLP tasks: Machine Translation and Image Captioning.

### Machine Translation 
Experiments were done on the WMT'14 English-German and English-French datasets.
#### Token-level NMT 
**Transformer** : Using 4 configurations of the transformer (Tiny, Small, Base and Big)

![](/images/transformer-aa.PNG)


**LSTM** : 2-layer LSTM is used for encoder and decoder. [Luong](https://arxiv.org/abs/1508.04025) dot product attention is used. 

![](/images/lstm-aa.PNG)


#### Chacracter-level NMT
I won't show the results here. But you can view them in the paper.


### Image captioning 
Experiments were done on the COCO dataset. The model implemented follows [Sharman et al., 2018](https://aclweb.org/anthology/P18-1238) employing both a pre-trained Inception-ResNet and a Transformer for Encoding the image and another Transfromer for decoding the caption text. See the figure below.

![](/images/sharman.PNG)


![](/images/ic-aa.PNG)


## Comments on the results

* For MT, we notice that improvements are on avreage by ~1 BLEU score points especially on the Base and Big Transformer configurations. This is one of the main reasons the reviewers were not convinced of the significance of the area attention proposed as BLEU score has high variance nature.

* On the LSTM token-level NMT, the improvement are on average even less than 1 BLEU point. In some cases like the 512 cells, improvement is merely 0.13 BLEU points which is indeed insignificant.

* For Image Caption, the same as before applies and in some cases such as Cells=512 and Head= 4, Area Attention causes performance to become even worse.

In the end, although the idea of computing attention on multiple granularity levels is somehow novel and interesting, the experiments shown in the paper don't really reflect the significance of the approach on improving the performance of MT and IC. However, this idea may turn out to be useful in other tasks and other settings.




