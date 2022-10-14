# Improved Bounds on Neural Complexity for Representing Piecewise Linear Functions

This repository is the official implementation of Appendix D in the supplementary material of the paper, [Improved Bounds on Neural Complexity for Representing Piecewise Linear Functions](https://arxiv.org/abs/2210.07236).

- Download the paper from NeurIPS website or [arXiv](https://arxiv.org/abs/2210.07236).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> Please install Python before running the above setup command. The code was tested on Python 3.9.13.

## Usage

To measure the run time of Algorithm 1, run:
```run_main
python main.py
```
The results will be saved to a CSV file.

## Run time of Algorithm 1

The run time is reported in seconds.

|  | n=1 | n=10 | n=100|
| - | - | - | - |
|  q=1 | 0.0007 | 0.0009 | 0.0013 |
|  q=2 | 0.0034 | 0.0075 | 0.0083 |
|  q=4 | 0.0097 | 0.0248 | 0.0343 |
|  q=8 | 0.0336 | 0.0980 | 0.1253 |
| q=16 | 0.1212 | 0.3932 | 0.4795 |
| q=32 | 0.4663 | 1.5408 | 1.8860 |

## BibTeX
```
@inproceedings{chen2022improved,
  title={Improved Bounds on Neural Complexity for Representing Piecewise Linear Functions},
  author={Chen, Kuan-Lin and Garudadri, Harinath and Rao, Bhaskar D.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```