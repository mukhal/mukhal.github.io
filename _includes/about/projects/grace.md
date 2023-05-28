<h1 align="center"> Discriminator-Guided Multi-step Reasoning with Language Models </h1>
<p align="center">Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, Lu Wang </p>

---

<h2 align="center"> Abstract </h2>

In the context of multi-step reasoning, language models (LMs) probabilities are often miscalibrated -- solutions with high probabilities are not always correct. Therefore, greedy decoding, which is the standard decoding method for reasoning tasks, often yields incorrect solutions. In addition, methods such as self-consistency and verifiers rely on sampling from the LM distribution and do not tackle the underlying issue. To address this, we introduce Guiding Multi-step ReAsoning with a CorrectnEss Discriminator (GRACE), a stepwise decoding approach that nudges the model towards producing correct reasoning steps. GRACE employs a discriminator model, which is trained to differentiate correct steps from invalid ones, to adjust decoding preferences based on the correctness of each reasoning step. Importantly, GRACE does not require fine-tuning or re-training the LMs. When compared with conventional decoding strategies over four popular math reasoning benchmarks, GRACE exhibits significant improvements in both final answer accuracy and step correctness, outperforming both greedy decoding and self-consistency.

---


## High-level Overview
The discriminator is used to guide step-wise decoding by scoring candidate next steps. 

<img width="1391" alt="image" src="https://github.com/mukhal/mukhal.github.io/assets/5109053/2e4c88a8-0726-4187-9a55-76e8a6900c4d">

## Discriminator Training
<img width="1469" alt="image" src="https://github.com/mukhal/mukhal.github.io/assets/5109053/00c69ee7-205d-4066-9cad-8e55c3c37f3d">

## Results Summary
Using GRACE for multi-step solution decoding outperforms greedy decoding and self-consistency with temperature sampling 
<img width="1225" alt="image" src="https://github.com/mukhal/mukhal.github.io/assets/5109053/9a369510-3a9b-40e9-965e-f8ff36dc5fa6">



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

