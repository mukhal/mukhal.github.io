## Projects


Here's a list of all **research projects** I was involved in:

* **Distributionally-controlled Text Generation from pre-trained Language models (internship at NAVER Labs Europe)** *(May - September 2020)*


    Supervised by Hady Elsahar and Marc Dymetman, we proposed a novel aspect of controlled text generation involving moment constraints while minimizing KL divergence with the initial LM distribution to avoid "degeneration" issues. From that target distribution, we arrive at at an explicit EBM (Energy-Based Model) representation.  Then, from that optimal representation, we then train the target controlled autoregressive LM through an adaptive distributional variant of Policy Gradient. We show the effectiveness of our approach in mitigating existing biases in pre-trained generative langauge models.


* **Low-resource multi-dialectal Arabic NLU (In collaboration with my master's supervisors and Prof. Muhammad Abdulmageed (UBC), my master's thesis work, 2 papers were submitted to EACL 2021 and TALLIP)** (Feb 2020 - Oct 2020)


    Arabic has many dialects and varieties that have little or no annotated resources at all. On the other hand, plenty of resources are available for Modern Standard Arabic. In this work, We proposed to employ self-training for zero- and low-resource Multi-dialectal Arabic NLU using labeled MSA resources only. We were able to obtain significant gains in the context of 3 different NLU tasks.

* **Low-resource Abstractive Arabic summarization (Independent Project)** *(March 2020 - Present)*


    Investigating a very understudied task. So far, built a dataset of ~90K article-summary pairs, evaluated several baselines including pre-trained seq2seq models. Still deciding what to do next)


* **Book success prediction with readability scores and pre-trained sentence embeddings (Remote collaboration with Prof. Aminul Islam (University of Louisiana at Lafayette)). [Pre-print](https://arxiv.org/abs/2007.11073)** *(November 2018 - March 2019)*

    We proposed a model that leverages Convolutional Neural Networks along with readability indices for book success predition. Unlike previous methods, our method included no count-based, lexical, or syntactic hand-crafted features. Instead, we made use of a pre-trained sentence encoder to encode the book sentences. We showed that only the first 1K sentences are good enough to predict the successability of books. Our proposed model outperformed strong baselines on this task by as large as 6.4% F1-score.

* **FIRE 2019 shared Task for Irony Detection (1st place winner!). [Link](https://www.irit.fr/IDAT2019/)** *(May 2019 - July 2019)


    The dataset size was small (~1K tweets). Therefore,  we applied classical machine learning ensemble (random forest + logitic regression + MLP) on hand-crafted sentiment, and count-based features among others. We submitted 3 systems to the shared task all of which scored the top 3 places. 




* Diacritic restoration by character LM pre-training.
* Dialect to MSA translation with unsupervised word mapping.
* Rumors detection by unsupervised time-series clustering.
* Agatha: Unsupervised time-series anomaly detection for predictive maintenance.
* Character-level models for sequence labeling


List of **open-source implementations/tools**:
* Implementation of Bilateral Multiperspective matching. [link]()
* Implementation of UlmFit and application on Arabic Dialect Identification. [link]()
* Sequence tagging with xlm-roberta. [link]()
* Fairseq-tagging. [link]()
* Arabic Diacritic restoration with transformers. [link](https://github.com/mohammadKhalifa/transformer-diacritization)


