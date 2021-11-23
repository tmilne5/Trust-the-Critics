# Trust-the-Critics (Code for Reproducibility)

The main branch of this repository includes cleaned up version of the code for TTC. This branch ("reproducible") includes the code that was run to produce the experimental results in the paper. By running the code in this branch with the provided random seeds and packages as in the requirements file, you should get our experimental results exactly.

A brief description of the code is included here. See the main branch for specifics on how datasets should be saved.
 
- **ttc.py**: for training critics using the TTC algorithm. 
- **ttc_eval.py**: for evaluating trained critics from TTC for either image generation or image translation. This produces samples from the final distribution \mu_N and  optionally computes the FID.
- **denoise_eval.py**: for evaluating trained critics from TTC for image denoising. Adds noise to the test set from BSDS500 and denoises using TTC and the benchmark technique.
- **wgan_gp.py**: for training a wgan to compare against TTC.
- **wgan_eval.py**: for producing samples from trained WGAN and optionally computing FID.
- **misalignment.py**: TO BE ADDED
- The rest of the code defines functions which are used by the above files.


### Random Seeds
The experiments in Section 5.1 of the paper use the random seeds 0, 1, 2, 3, 4 for the distinct training runs. The experiments in Section 5.2 (image translation) do not use a reproducible seed (i.e. seed = -1 in the code); we do not see this as a major obstacle to reproducibility given the subjective nature of the results. The experiments in Section 5.3 use a single random seed of 0 for all noise levels.  

### Computing Architecture
This code was run on the Graham cluster of Compute Canada (https://www.computecanada.ca/). The code in this branch was run on a single NVIDIA V100 Volta GPU. Approximate training times are as follows:

- **MNIST/Fashion MNIST generation training runs**: 1.2 hours (TTC), 0.7 hours (WGAN-GP)
- **CIFAR10 generation training runs**: 3.2 hours (TTC), 1.5 hours (WGAN-GP)
- **Image translation training runs**: 18.8 hours
- **Image denoising training runs**: 8.6 hours


