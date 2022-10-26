I have made 2 different Versions of CUDA programs for Matrix Multiplication.
The differences in them and their optimizations (with experimentations) can be found in the "Report.pdf" file in this folder.

The instruction to run both the Versions is starightforward and as follows:
- Load the cuda modul to path, if not already done (done on gpu-node1 with "module load cuda")
- cd into the "Version-1" or "Version-2" directory based on the CUDA program version that you would like to run.
- Run the following command to compile the CUDA Program: "make -f Makefile"
- Run the following command to execute the compiled CUDA Program: "./cuda_matmul <nsize>"
		Example Usage: ./cuda_matmul 2048


Author:

Deepak Subramani Velumani
dsubrama@ur.rochester.edu
MS CS 1st year
Registartion Number: 32132007