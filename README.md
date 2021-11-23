# Trust-the-Critics
PyTorch implementation of the TTC algorithm.


### How to run this code
Create a Python virtual environment with Python 3.8 installed. Then, install the necessary Python packages listed in the requirements.txt file (this can be done through pip install -r /path/to/requirements.txt). 




**WGAN misalignments**  
The WGAN misalignment experiments are described in Section 3 and Appendix B.1 of the paper, and are run using the misalignments.py script. The arguments that need to be specified when running this script are listed in a commented section at the top of the script. The folder specified by the 'temp_dir' argument needs to contain a copy of the MNIST dataset in a format that can be accessed by an instance of the torchvision.datasets.MNIST class. For FID evaluation, the same folder must also include a subfolder named 'temp_dir/mnisttest' containing the test data from the MNIST dataset as individual jpg files.




**TTC image generation**   
recap of these experiment and explanation of how to run it



**TTC denoising**  
recap of this experiment and explanation of how to run it



**TTC Monet translation**  
recap of this experiment and explanation of how to run it








