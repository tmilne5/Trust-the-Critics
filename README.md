# Trust-the-Critics
PyTorch implementation of the TTC algorithm.


### How to run this code
Create a Python virtual environment with Python 3.8 installed. Then, install the necessary Python packages listed in the requirements.txt file (this can be done through pip install -r /path/to/requirements.txt). 




**WGAN misalignment**  
The WGAN misalignment experiments are described in Section 3 and Appendix B.1 of the paper, and are run using the misalignments.py script. The arguments that need to be specified when running this script are listed in a commented section at the top of the script. The folder specified by the 'temp_dir' argument needs to contain a copy of the MNIST dataset in a format that can be accessed by an instance of the torchvision.datasets.MNIST class. For FID evaluation, the 'temp_dir' folder should also include a subfolder named 'temp_dir/mnisttest' containing the test data from the MNIST dataset saved as individual jpg files 'temp_dir/mnisttest/00001.jpg', 'temp_dir/mnisttest/00002.jpg', etc.  The misalignment results reported in the paper (Tables 1 and 5, and Figure 3), correspond roughly to setting the 'checkpoints' argument equal to '10_25000_40000', with '10' corresponding the early stage in training, '25000' to the mid stage, and '40000' to the late stage. 




**TTC image generation**   
recap of these experiment and explanation of how to run it



**TTC denoising**  
recap of this experiment and explanation of how to run it



**TTC Monet translation**  
recap of this experiment and explanation of how to run it
  
  

### Reproducibility
This repository contains two branches: 'main' and 'reproducible'. You are currectly viewing the 'main' branch, which contains a clean version of the code meant to be easy to read and interpret and that runs more efficiently than the version on the 'reproducible' branch. The results obtained by running the code on this branch are nearly (but not perfectly) identical to the results stated in the papers, the differences stemming from the randomness inherent to the experiments. The 'reproducible' branch allows one to replicate exactly the results stated in the paper (random seeds are specified). 


### Assets 
Portions of this code, as well as the datasets used to produce our experimental results, make use of existing assets. We provide here a list of all assets used, along with the licenses under which they are distributed, if specified by the originator:
- This code was initially built from a PyTorch implementation (https://github.com/caogang/wgan-gp) of WGAN-GP ((c) 2017 Ishaan Gulrajani). Distributed under the MIT licence
- **mmd.py**: from https://github.com/EmoryMLIP/OT-Flow, ((c) 2020 EmoryMLIP). Distributed under the MIT licence. Unused in the paper, but provides a separate interesting metric for measuring performance.
- **pytorch_fid**: from https://github.com/mseitzer/pytorch-fid. Distributed under the Apache License 2.0.
- **MNIST dataset**: from http://yann.lecun.com/exdb/mnist/. Distributed under the Creative Commons Attribution-Share Alike 3.0 license.
- **Fashion MNIST datset**: from  https://github.com/zalandoresearch/fashion-mnist ((c) 2017 Zalando SE, https://tech.zalando.com). Distributed under the MIT licence.
- **CIFAR10 dataset**: from https://www.cs.toronto.edu/~kriz/cifar.html.
- **Image translation datasets**: from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix ((c) 2017, Jun-Yan Zhu and Taesung Park). Distributed under the BSD licence.
- **BSDS500 dataset**: from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html.




