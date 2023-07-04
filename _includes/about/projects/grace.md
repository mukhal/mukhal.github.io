<h1 align="center"> Discriminator-Guided Multi-step Reasoning with Language Models </h1>
<p align="center">Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, Lu Wang </p>
<p align="center">by Muhammad Khalifa and Lajanujen Logeswaran</p>
<p align="center">
  <a href="https://github.com/mukhal/grace">
   <img src="https://w7.pngwing.com/pngs/12/369/png-transparent-black-and-white-cat-illustration-github-computer-icons-icon-design-github-cat-like-mammal-carnivoran-logo.png" alt="Code" width="25" height="25" style="display:inline"> code  
  </a>
  <a href="https://arxiv.org/abs/2305.14934">
    <img src="https://png.pngtree.com/element_our/20190528/ourmid/pngtree-paper-icon-image_1131168.jpg" alt="Paper" width="25" height="25" style="display:inline"> paper
  </a>
</p>

---

<h2 align="center"> Abstract </h2>

In the context of multi-step reasoning, language models (LMs) probabilities are often miscalibrated -- solutions with high probabilities are not always correct. Therefore, greedy decoding, which is the standard decoding method for reasoning tasks, often yields incorrect solutions. In addition, methods such as self-consistency and verifiers rely on sampling from the LM distribution and do not tackle the underlying issue. To address this, we introduce Guiding Multi-step ReAsoning with a CorrectnEss Discriminator (GRACE), a stepwise decoding approach that nudges the model towards producing correct reasoning steps. GRACE employs a discriminator model, which is trained to differentiate correct steps from invalid ones, to adjust decoding preferences based on the correctness of each reasoning step. Importantly, GRACE does not require fine-tuning or re-training the LMs. When compared with conventional decoding strategies over four popular math reasoning benchmarks, GRACE exhibits significant improvements in both final answer accuracy and step correctness, outperforming both greedy decoding and self-consistency.

---


## High-level Overview
<img width="1391" alt="image" src="https://github.com/mukhal/mukhal.github.io/assets/5109053/742cd7b6-b34c-41fd-8e7e-c7e109486de5">

## Discriminator Training
<img width="1469" alt="image" src="https://github.com/mukhal/mukhal.github.io/assets/5109053/7ae75f7a-010c-48d5-a13e-ddddf6662598">

**1. Sampling:** Sampling incorrect solutions from the LM. 

**2. Alignment:** Align sampled solutions with the reference solution to create training examples to train the discriminator. 

**3. Learning:** Train the discriminator with max-margin loss. 



## Results Summary
We evaluate GRACE over 4 math (GSM8K, SVAMP, MultiArith, MathQA-Gain) and 2 symbolic reasoning tasks (Coin Flip and Tracking Shuffled Objects). 
Using GRACE for multi-step solution decoding outperforms greedy decoding and vanilla self-consistency with temperature sampling. 

**Performance on Math Reasoning:**
<img width="991" alt="image" src="https://github.com/mukhal/mukhal.github.io/assets/5109053/ab7ae24f-8d89-45e8-999f-45a535a5d365">

**Performance on Symbolic Reasoning:**
<p align="center">
<img width="400" alt="image" src="https://github.com/mukhal/mukhal.github.io/assets/5109053/1ec5c985-cfd5-4669-8f4f-6e8f98dca1fd">
</p>
---


## Citation

If you find our paper useful, please cite us:
```
@article{khalifa2023discriminator,
  title={Discriminator-Guided Multi-step Reasoning with Language Models},
  author={Khalifa, Muhammad and Logeswaran, Lajanugen and Lee, Moontae and Lee, Honglak and Wang, Lu},
  journal={arXiv preprint arXiv:2305.14934},
  year={2023}
}
```

