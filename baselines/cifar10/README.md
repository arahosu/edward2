# ResNet-20 on CIFAR-10

| Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic<sup>5</sup> | 1e-4 / 0.206 | 99.9% / 95.6% | 1.3 (32 TPUv2 cores) | 36.5M |
| Dropout | 0.137 / 0.324 | 95.1% / 90.0% | 0.85 (1 P100 GPU) | 274K |
| BatchEnsemble (size=4)<sup>5</sup> | 0.08 / 0.197 | 99.9% / 95.4% | 3.25 (32 TPUv2 cores) | 7.47M |
| Ensemble (size=5) | 0.011 / 0.184 | 99.9% / 94.1% | 0.75 (5 P100 GPU) | 1.37M |
| Variational inference | 0.136 / 0.382 | 95.5% / 89.1% | 1.25 (1 P100 GPU) | 420K |

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`junyuseu/pytorch-cifar-models`](https://github.com/junyuseu/pytorch-cifar-models) | Deterministic | - | - / 91.67% | - | 270K |
| [`keras-team/keras`](https://keras.io/examples/cifar10_resnet) | Deterministic | - | - / 92.16% | 1.94 (1 1080Ti GPU) | 270K |
| [`kuangliu/pytorch-cifar`](https://github.com/kuangliu/pytorch-cifar) | Deterministic (ResNet-18) | - | - / 93.02% | - | 11.7M |
| [He et al. (2015)](https://arxiv.org/abs/1512.03385)<sup>1</sup> | Deterministic | - | - / 91.25% | - | 270K |
| | Deterministic (ResNet-32) | - | - / 92.49% | - | 460K |
| | Deterministic (ResNet-44) | - | - / 92.83% | - | 660K |
| | Deterministic (ResNet-56) | - | - / 93.03% | - | 850K |
| | Deterministic (ResNet-110) | - | - / 93.39% | - | 1.7M |
| [Louizos et al. (2017)](https://arxiv.org/abs/1705.08665)<sup>2</sup> | Group-normal Jeffreys | - | - / 91.2% | - | 998K |
| | Group-Horseshoe | - | - / 91.0% | - | 820K |
| [Molchanov et al. (2017)](https://arxiv.org/abs/1701.05369)<sup>2</sup> | Variational dropout | - | - / 92.7% | - | 304K |
| [Louizos et al. (2018)](https://arxiv.org/abs/1712.01312)<sup>3</sup> | L0 regularization | - | - / 96.17% | 200 epochs | - |
| [Anonymous (2019)](https://openreview.net/forum?id=rkglZyHtvH)<sup>4</sup> | Refined VI (no batchnorm) | - / 0.696 | - / 75.5% | 5.5 (1 P100 GPU) | - |
| | Refined VI (batchnorm) | - / 0.593 | - / 79.7% | 5.5 (1 P100 GPU) | - |
| | Refined VI hybrid (no batchnorm) | - / 0.432 | - / 85.8% | 4.5 (1 P100 GPU) | - |
| | Refined VI hybrid (batchnorm) | - / 0.423 | - / 85.6% | 4.5 (1 P100 GPU) | - |
| [Anonymous (2019)](https://openreview.net/forum?id=Sklf1yrYDr)<sup>5</sup> | Deterministic | - | - / 95.31% | 250 epochs | 7.43M |
| | BatchEnsemble | - | - / 95.94% | 375 epochs | 7.47M |
| | Ensemble | - | - / 96.30% | 250 epochs each | 29.7M |
| | Monte Carlo Dropout | - | - / 95.72% | 375 epochs | 7.43M |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>6</sup> | Deterministic | - / 0.243 | - / 94.4% | 1000 epochs (1 V100 GPU) | 850K |
| | Adaptive Thermostat Monte Carlo (single sample) | - / 0.303 | - / 92.4% | 1000 epochs (1 V100 GPU) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | - / 0.194 | - / 93.9% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | - / 0.343 | - / 91.7% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | - / 0.211 | - / 93.5% | 1000 epochs (1 V100 GPU) | - |
| [Maddox et al. (2019)](https://arxiv.org/abs/1902.02476)<sup>7</sup> | Deterministic (WRN-28-10) | - / 0.1294 | - / 96.41% | 300 epochs | 36.5M |
| | SWA | - / 0.1075 | - / 96.46% | 300 epochs | 36.5M |
| | SWAG | - / 0.1122 | - / 96.41% | 300 epochs | 803M |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>8</sup>  | Variational Online Gauss-Newton | - / 0.48 | 91.6% / 84.3% | 2.38 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530)<sup>9</sup> | Deterministic | - / 1.120 | - / 91% | - | 274K |
| | Dropout | - / 0.771 | - / 91% | - | 274K |
| | Ensemble | - / 0.653 | - | - / 93.5% | - |
| | Variational inference | - / 0.823 | 88% | - | 630K |
| [Zhang et al. (2019)](https://arxiv.org/abs/1902.03932)<sup>10</sup> | Deterministic (ResNet-18) | - | - / 94.71% | 200 epochs | 11.7M |
| | cSGHMC | - | - / 95.73% | 200 epochs | 140.4M |
| | Ensemble of cSGHMC (size=4) | - | - / 96.05% | 800 epochs | 561.6M |

1. Trains on 45k examples.
2. Not a ResNet (VGG). Parameter count is guestimated from counting number of parameters in [original model](http://torch.ch/blog/2015/07/30/cifar.html) to be 14.9M multiplied by the compression rate.
3. Uses Wide ResNet (WRN-28-10).
4. Does not use data augmentation.
5. Uses ResNet-32 with 4x number of typical filters. Ensembles uses 4 members.
6. Uses ResNet-56 and modifies architecture. Cyclical learning rate.
7. SWAG uses rank 20 which requires 20 + 2 copies of the model parameters, and 30 samples at test time.
8. Scales KL by an additional factor of 10.
9. Trains on 40k examples. Performs variational inference over only first convolutional layer of every residual block and final output layer. Has free parameter on normal prior's location. Uses scale hyperprior (and with a fixed scale parameter). NLL results are medians, not means; accuracies are guestimated from Figure 2's plot.
10. Uses ResNet-18. cSGHMC uses a total of 12 copies of the full size of weights for prediction. Ensembles use 4 times cSGHMC's number. The authors use a T=1/200 temperature scaling on the log-posterior (see the newly added appendix I at https://openreview.net/forum?id=rkeS1RVtPS).

## CIFAR-100

| Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| BatchEnsemble (size=4)<sup>5</sup> | 0.38 / 0.89 | 99.9% / 79.0% | 3.85 (32 TPUv2 cores) | 7.47M |

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [Maddox et al. (2019)](https://arxiv.org/abs/1902.02476)<sup>7</sup> | Deterministic (WRN-28-10) | - / 0.7958 | - / 80.76% | 300 epochs | 36.5M |
| | SWA | - / 0.6684 | - / 82.40% | 300 epochs | 36.5M |
| | SWAG | - / 0.6078 | - / 82.23% | 300 epochs | 803M |
| [Zhang et al. (2019)](https://arxiv.org/abs/1902.03932)<sup>10</sup> | Deterministic (ResNet-18) | - | - / 77.40% | 200 epochs | 11.7M |
| | cSGHMC | - | - / 79.50% | 200 epochs | 140.4M |
| | Ensemble of cSGHMC (size=4) | - | - / 80.81% | 800 epochs | 561.6M |

TODO(trandustin): Add column for Test runtime.

TODO(trandustin): Add column for Checkpoints.

TODO(trandustin): Should CIFAR-100 baselines be in this directory? If so, rename
to cifar and have all baselines support it; otherwise duplicate code into a
separate directory.
