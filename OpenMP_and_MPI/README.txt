This file contains the instructions for running the programs:

- The MPI & OpenMP versions of the Gauss Elimination program are present in the sub-directories of this folder.
- They can be accessed and then run by "cd"ing into the directories named "gauss_mpi" & "gauss_openmp" respectively for MPI & OpenMP.
- Kindly make sure to read the "Report.pdf" file in this directory for information on the algorithms, results and comparision of the different parallelised versions.

The instuctions to run the programs are as follows:

Step-1: cd into the 'gauss_mpi' or 'gauss_openmp' directory, based on the version that you would like to run.

Step-2: Compiling:

(a) For MPI, compile the codes by running "mpicc gauss_internal_input.c -o gauss_internal_input" (or/and) "mpicc gauss_external_input.c -o gauss_external_input" (without quotes).
(b) For OpenMP, compile the codes by running "gcc -fopenmp gauss_internal_input.c -o gauss_internal_input" (or/and) "gcc -fopenmp gauss_external_input.c -o gauss_external_input" (without quotes).

Step-3: Executing:

(a) For MPI:
	Note: The minimum number of threads should atleast be 2.
	
	If you want to run the gauss_internal_input.c program, here is how to to run it:

		mpirun -np <number of threads> ./gauss_internal_input -s <size of input matrix>
		
		In order to get help, type: ./gauss_internal_input h
	
		Example Runs:
			mpirun -np 16 ./gauss_internal_input -s 1024
			mpirun -np 12 ./gauss_internal_input    // This will take 12 threads & 2048 input matrix sizes as default.
		
	
	If you want to run the 'gauss_internal_input.c' program, here is how to do it:
	
		mpirun -np <number of threads> ./gauss_external_input <path to matrix file>
		
		Example run:
			mpirun -np 16 ./gauss_external_input  matrices_dense/orsreg_1.dat

To run on node01 to node06 systems:
	(i) If not done already, setup passwordless ssh access to node01 to node06 system by running the following commands:
			ssh-keygen -t rsa
			for i in 1 2 3 4 5 6; do ssh-copy-id "node0$i"; done;
			for i in 1 2 3 4 5 6; do ssh-copy-id "node0$i.csug.rochester.edu"; done;
			cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys
	(ii) Now run the following command (change gauss_internal_input/gauss_external_input as desired, along with its arguments on right side):
			mpirun -hostfile hosts -np <number of threads> ./gauss_internal_input -s <size of input matrix>
					(or)
			mpirun -hostfile hosts -ppn <threads per node> ./gauss_internal_input -s <size of input matrix>

(b) For OpenMP:

If you want to run the gauss_internal_input.c program, here is how to to run it:

		./gauss_internal_input -t <number of threads> -s <size of input matrix>
		
		In order to get help, type: ./gauss_internal_input h
	
		Example Runs:
			./gauss_internal_input -t 16 -s 1024
			./gauss_internal_input    // This will take max threads in the system & 2048 input matrix size as default.
		
	
	If you want to run the 'gauss_internal_input.c' program, here is how to do it:
	
		./gauss_external_input <path to matrix file> <number of threads>
		
		Example runs:
			./gauss_external_input  matrices_dense/orsreg_1.dat 16
			./gauss_external_input  matrices_dense/saylr4.dat 		// This will by default consider the max number of threads available in system


Author:

Deepak Subramani Velumani
MS CS 1st year
Registartion Number: 32132007