This file contains the instructions for running the programs:

- I have made a total of 3 different parallelised versions, of the Gaussian Elimination program.
- They can be accessed and then run by "cd"ing into the directories nameed "Version-1", "Version-2" & "Version-3" respectively for the 3 versions.
- Kindly make sure to read the "report.pdf" file in this directory for information on the optimisation methods, results and summary of the 3 propgram versions.

The instuctions to run the programs are as follows:

Step-1: cd into the 'Version-1', 'Version-2' or version-3' directory, based on the version that you would like to run.
Step-2: Compile the codes by running "gcc gauss_internal_input.c -o gauss_internal_input -pthread" (or) "gcc gauss_external_input.c -o gauss_external_input -pthread" (without quotes).
Step-3:
	If you want to run the gauss_internal_input.c program, here is how to to run it:

		./gauss_internal_input t <number of threads> s <size of input matrix>
		
		In order to get help, type: ./gauss_internal_input h
	
		Example Runs:
			./gauss_internal_input t 16 s 1024
			./gauss_internal_input    // This will take 32 threads & 2048 input matrix sizes a default.
		
	
	If you want to run the 'gauss_internal_input.c' program, here is how to do it:
	
		./gauss_external_input <path to matrix file> t <number of threads>
		
		Example runs:
			./gauss_external_input  matrices_dense/orsreg_1.dat 16
			./gauss_external_input  matrices_dense/saylr4.dat 		// This will by default consider the number of threads as 32.


Author:

Deepak Subramani Velumani
MS CS 1st year
Registartion Number: 32132007