How to run the code seperately
Note:
1. install the required libraries one by one:
    i. conda install mp4pi --> can not install all gym envorinment
    ii. pip install gym['all'] --> solve issue that gym cannot find '2D_box' module

2. vae_train.py
    i. comment line #9-11, no OS and sys environement should be changed --> solve issue that 'WM' diretory is not found
    ii. in .config file, line #13: set vae_batch_size=64 # should match vae_batch_size = 2 * z_size --> solve issue that 'Ambigious data size between x_batch and '

3. seris.py

4. rnn_train.py
    i. pip install tf-nightly--> solve 
        TypeError: function() got an unexpected keyword argument 'jit_compile'
    
    ii. line #48 set range(1000) as range(640), since we change the vae_batch_size=64 --> solve
    Traceback (most recent call last):
        File "rnn_train.py", line 47, in <module>
        IndexError: index 640 is out of bounds for axis 0 with size 640

5. train.py
    i. make sure you install mpi4py sucessfully
    ii. ???? --> solve
    Traceback (most recent call last): 
        File "train.py", line 439, in mpi_fork
        subprocess.CalledProcessError: XXXXX returned non-zero exit status 255.