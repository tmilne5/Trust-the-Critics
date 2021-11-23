# Trust-the-Critics
PyTorch implementation of the TTC algorithm.


### How to run this code
Create a Python virtual environment with Python 3.8 installed. Then, install the necessary Python packages listed in the requirements.txt file (this can be done through pip install -r /path/to/requirements.txt). 




**WGAN misalignments**  
The WGAN misalignment experiments are described in Section 3 and Appendix B.1 of the paper, and are run using the misalignments.py script. The arguments that need to be specified when running this script are listed in a commented section at the top of the script. The folder specified by the 'temp_dir' argument needs to contain a copy of the MNIST dataset in a format that can be accessed by an instance of the torchvision.datasets.MNIST class. For FID evaluation, the same folder should also include a subfolder named 'temp_dir/mnisttest' containing the test data from the MNIST dataset saved as individual jpg files 'temp_dir/mnisttest/00001.jpg', 'temp_dir/mnisttest/00002.jpg', etc.  The misalignment results reported in the paper (Tables 1 and 5, and Figure 3), correspond roughly to setting the 'checkpoints' argument equal to '10_25000_40000', with '10' corresponding the early stage in training, '25000' to the mid stage, and '40000' to the late stage. 




**TTC image generation**   
recap of these experiment and explanation of how to run it



**TTC denoising**  
recap of this experiment and explanation of how to run it



**TTC Monet translation**  
recap of this experiment and explanation of how to run it
  
  




### ASSETS  
Should we cite assets here?
Should we mention things were run on compute canada?



