# Towards Order Fairness: Mitigating LLMs Order Sensitivity through Reinforcement Learning

This repository contains the official implementation of "Towards Order Fairness: Mitigating LLMs Order Sensitivity through Reinforcement Learning".

## Installation

To install the required dependencies, run:

```bash
pip install -e .
```

## Usage

### Training

To train the model, execute:

```bash
bash train.sh
```

### Evaluation

To evaluate the trained model, run:

```python
python eval.py
```

## Acknowledgments

We would like to acknowledge the following repositories that our work builds upon:

- Our code is modified based on the [TRL repository](https://github.com/huggingface/trl/tree/main)
- Our SFT training utilizes the [LLaMA-Factory repository](https://github.com/hiyouga/LLaMA-Factory/tree/main)

## TODO

- [ ] All model parameters will be open-sourced upon paper acceptance