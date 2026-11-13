# Multi-Horizon Early Warning
This repo contains the Pytorch implementation of the manuscript - [Dynamics-Augmented Multi-Horizon Early Warning for Extremely Rare Events in Industrial Processes]

<div align="center">
  <img src="imgs/fig.png" alt="main fig" width="80%">
</div>


## Result


### Event-Level Performance at h = 30

Comparison of raw, dynamics-only, and dynamics-augmented
representations across four predictive models.

**Results are reported as mean ± standard deviation over 5 seeds.**

| Model | Representation | AUPRC ↑ | AUROC ↑ | BS ↓ | F1 ↑ |
|:------|:---------------|--------:|--------:|-----:|-----:|
| **HGB** | Raw | 0.22 ± 0.008 | 0.50 ± 0.018 | 0.22 ± 0.001 | 0.24 ± 0.026 |
| | Dyn-only | **0.41 ± 0.042** | **0.64 ± 0.012** | **0.18 ± 0.008** | **0.27 ± 0.026** |
| | Dyn-aug | 0.40 ± 0.015 | 0.62 ± 0.017 | **0.18 ± 0.005** | 0.12 ± 0.072 |
| **XGB** | Raw | 0.27 ± 0.015 | 0.55 ± 0.017 | 0.21 ± 0.002 | 0.07 ± 0.052 |
| | Dyn-only | **0.45 ± 0.031** | **0.64 ± 0.015** | **0.17 ± 0.006** | **0.33 ± 0.040** |
| | Dyn-aug | **0.45 ± 0.018** | **0.64 ± 0.006** | **0.17 ± 0.004** | 0.19 ± 0.091 |
| **LSTM** | Raw | **0.38 ± 0.107** | **0.60 ± 0.088** | **0.26 ± 0.046** | **0.23 ± 0.211** |
| | Dyn-only | 0.20 ± 0.016 | 0.43 ± 0.038 | 0.30 ± 0.120 | 0.11 ± 0.048 |
| | Dyn-aug | 0.24 ± 0.045 | 0.48 ± 0.071 | 0.41 ± 0.118 | 0.10 ± 0.080 |
| **TCN** | Raw | **0.28 ± 0.074** | **0.56 ± 0.068** | **0.33 ± 0.137** | **0.14 ± 0.166** |
| | Dyn-only | 0.23 ± 0.018 | 0.43 ± 0.039 | 0.52 ± 0.156 | **0.14 ± 0.035** |
| | Dyn-aug | 0.23 ± 0.016 | 0.44 ± 0.038 | 0.72 ± 0.024 | 0.06 ± 0.043 |

**Notes:**
- **Raw:** Original sensor measurements.
- **Dyn-only:** Features derived from process dynamics.
- **Dyn-aug:** Combination of raw and dynamics features.
- **↑:** Higher is better.
- **↓:** Lower is better.
- **BS:** Brier Score.
- Bold values indicate the best mean performance within each model family.



## Citation
```
@article{zhou2017reinforcevsumm, 
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao}, 
   journal={arXiv:1801.00054}, 
   year={2017} 
}
```
