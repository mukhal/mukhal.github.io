<h1 align="center"> Discriminator-Guided Multi-step Reasoning with Language Models </h1>
<p align="center">Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, Lu Wang </p>

---

<h2 align="center"> Abstract </h2>

In the context of multi-step reasoning, language models (LMs) probabilities are often miscalibrated -- solutions with high probabilities are not always correct. Therefore, greedy decoding, which is the standard decoding method for reasoning tasks, often yields incorrect solutions. In addition, methods such as self-consistency and verifiers rely on sampling from the LM distribution and do not tackle the underlying issue. To address this, we introduce Guiding Multi-step ReAsoning with a CorrectnEss Discriminator (GRACE), a stepwise decoding approach that nudges the model towards producing correct reasoning steps. GRACE employs a discriminator model, which is trained to differentiate correct steps from invalid ones, to adjust decoding preferences based on the correctness of each reasoning step. Importantly, GRACE does not require fine-tuning or re-training the LMs. When compared with conventional decoding strategies over four popular math reasoning benchmarks, GRACE exhibits significant improvements in both final answer accuracy and step correctness, outperforming both greedy decoding and self-consistency.

---


## Results

*Discuss some of your findings and results here.*

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

