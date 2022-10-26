/* 
 * Original author:  UNKNOWN
 *
 * Modified:         Kai Shen (January 2010)
 * Modified:  Deepak Subramani Velumani (Feb 2022)
 * Parallel Version:  1b
 */

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>


#define NSIZE 2048
#define VERIFY 0
#define block_size 16	// Block length of each memory block in core loop 

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}
#define ABS(a)          (((a) > 0) ? (a) : -(a))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

double **matrix,*B,*V,*C;
int *swap;
int num_threads = 32;		// Default number of threads

/* Allocate the needed arrays */

void allocate_memory(int size)
{
	double *tmp;
	int i;

	matrix = (double**)malloc(size*sizeof(double*));
	assert(matrix != NULL);
	tmp = (double*)malloc(size*size*sizeof(double));
	assert(tmp != NULL);

	for(i = 0; i < size; i++){
		matrix[i] = tmp;
		tmp = tmp + size;
	}

	B = (double*)malloc(size * sizeof(double));
	assert(B != NULL);
	V = (double*)malloc(size * sizeof(double));
	assert(V != NULL);
	C = (double*)malloc(size * sizeof(double));
	assert(C != NULL);
	swap = (int*)malloc(size * sizeof(int));
	assert(swap != NULL);
}

/* Initialize the matirx with some values that we know yield a
 * solution that is easy to verify. A correct solution should yield
 * -0.5, and 0.5 for the first and last C values consecutively, and 0
 * for the rest, though it should work regardless */

void initMatrix(int nsize)
{
	int i,j;
	for(i = 0 ; i < nsize ; i++){
		for(j = 0; j < nsize ; j++) {
			matrix[i][j] = ((j < i )? 2*(j+1) : 2*(i+1));
		}
		B[i] = (double)i;
		swap[i] = i;
	}
}

/* Get the pivot row. If the value in the current pivot position is 0,
 * try to swap with a non-zero row. If that is not possible bail
 * out. Otherwise, make sure the pivot value is 1.0, and return. */

void getPivot(int nsize, int currow)
{
	int i,irow;
	double big;
	double tmp;

	big = matrix[currow][currow];
	irow = currow;

	if (big == 0.0) {
		for(i = currow ; i < nsize; i++){
			tmp = matrix[i][currow];
			if (tmp != 0.0){
				big = tmp;
				irow = i;
				break;
			}
		}
	}

	if (big == 0.0){
		printf("The matrix is singular\n");
		exit(-1);
	}

	if (irow != currow){
		for(i = currow; i < nsize ; i++){
			SWAP(matrix[irow][i],matrix[currow][i]);
		}
		SWAP(B[irow],B[currow]);
		SWAPINT(swap[irow],swap[currow]);
	}


	{
		double pivotVal;
		pivotVal = matrix[currow][currow];

		if (pivotVal != 1.0){
			matrix[currow][currow] = 1.0;
			for(i = currow + 1; i < nsize; i++){
				matrix[currow][i] /= pivotVal;
			}
			B[currow] /= pivotVal;
		}
	}
}

// Structure to pass arguments into the p_threaded subtractElim() - the parallelised function
struct sub_args {
	int i, nsize;
	long threadid;
};


/* This function performs the Subtraction Elimination Step, the 2nd half of the computeGauss() outer loop, which is the 
*  highest time complexity part [O(n^3)] of this program.
*  
*  Note: This is the parallelized function of this program
*  Parallelised Method:
*     - In this method, the threads are distributed across horizontal blocks on the Matrix, i.e., each thread will execute 'block_size' blocks of consecutive columns of 
*		each row in each iteration, and the thread would eventually cover various rows with the same column range, until all the rows for that column range are covered.
*	  - This method proves useful especially for matrices of large sizes, where it can ensure that "matrix[i][k range]" reamins in the cache memory, till it is reused 
*	  	to update all the rows of "matrix[i+1 to End][k range]", and hence avoiding cache miss for the "matrix[i][k range]" used in every iteration, and improving speed.
*/
void *subtractElim(void *arguments)
{
	struct sub_args *args = (struct sub_args *)arguments;
	long tid = args -> threadid;
	int i = args -> i;
	int nsize = args -> nsize;
	int j, k, k_start, k_end;
	double pivotVal;

	// The main multithreaded loop with highest time complexity.
    // Every row is divided into blocks & each thread is made to execute one row for each iteration.
    // Since every row is divided into blocks the entire block will fit in the cache memory, allowing 
    //      "matrix[i][k range]" to remain in cache till it's last usage, and avoid cache read misses.

	for (j = i + 1 + tid ; j < nsize; j+=num_threads){
		B[j] -= matrix[j][i] * B[i];
	}

	pivotVal = matrix[i][i];

	for (k_start=i + 1; k_start < nsize; k_start+=block_size)
	{
		k_end = MIN(k_start+block_size, nsize);
		for (j = i + 1 + tid ; j < nsize; j+=num_threads){
			pivotVal = matrix[j][i];
			//matrix[j][i] = 0.0;
			for (k = k_start; k < k_end; k++){
				matrix[j][k] -= pivotVal * matrix[i][k];
			}
			//B[j] -= pivotVal * B[i];
		}
	}

	for (j = i + 1 + tid ; j < nsize; j+=num_threads)
		matrix[j][i] = 0.0;

	pthread_exit(NULL);
}

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */

void computeGauss(int nsize)
{
	int i;
	double pivotVal;
	int rc;
	struct sub_args *args = (struct sub_args *)malloc(num_threads * sizeof(struct sub_args));
	void *status;
	long t;
	pthread_t *threadsSubElim;
	threadsSubElim = (pthread_t*) malloc(num_threads * sizeof(pthread_t));

	for(i = 0; i < nsize; i++){
		getPivot(nsize,i);

		// Starting threads here
		for(t=0; t<num_threads; t++)
		{
			args[t].threadid = t;
			args[t].i = i;
			args[t].nsize = nsize;
			rc = pthread_create(&threadsSubElim[t], NULL, subtractElim, (void *)&args[t]);
			if (rc)
			{
				printf("ERROR; return code from pthread_create() is %d\n", rc);
				exit(-1);
			}
		}

		// Wait for threads to finish
		for (t=0; t<num_threads; t++)
			pthread_join(threadsSubElim[t], &status);
	}
}


/* Solve the equation. That is for a given A*B = C type of equation,
 * find the values corresponding to the B vector, when B, is all 1's */

void solveGauss(int nsize)
{
	int i,j;

	V[nsize-1] = B[nsize-1];
	for (i = nsize - 2; i >= 0; i --){
		V[i] = B[i];
		for (j = nsize - 1; j > i ; j--){
			V[i] -= matrix[i][j] * V[j];
		}
	}

	for(i = 0; i < nsize; i++){
		C[i] = V[i];//V[swap[i]];
	}
}

extern char * optarg;
int main(int argc,char *argv[])
{
	int i;
	struct timeval start;
	struct timeval finish;
	long compTime;
	double Time;
	int nsize = NSIZE;

	while((i = getopt(argc,argv,"hs:t:")) != -1){
		switch(i){
			case 's':
				{
					int s;
					s = atoi(optarg);
					if (s > 0){
						nsize = s;
					} else {
						fprintf(stderr,"Entered size is negative, hence using the default (%d)\n",(int)NSIZE);
					}
				}
				break;
			case 't':
				{
					int thread_cnt;
					thread_cnt = atoi(optarg);
					if (thread_cnt > 0){
						num_threads = thread_cnt;
					} else {
						fprintf(stderr,"Entered number of threads is negative, hence using the default (%ld)\n",(int)num_threads);
					}
				}
				break;
			case 'h':
				printf("Usage: ./program -t <num threads> -s <matrix size>\n");
				return 0;
				break;
			default:
				printf("Usage: ./program -t <num threads> -s <matrix size>\n");
				assert(0);
				break;
		}
	}

	printf("\nMatrix Size: %d ; Threads: %d; Block Size: %d\n", nsize, num_threads, block_size);

	allocate_memory(nsize);

	gettimeofday(&start, 0);
	initMatrix(nsize);
	computeGauss(nsize);
#if VERIFY
	solveGauss(nsize);
#endif
	gettimeofday(&finish, 0);

	compTime = (finish.tv_sec - start.tv_sec) * 1000000;
	compTime = compTime + (finish.tv_usec - start.tv_usec);
	Time = (double)compTime;
	
	printf("Application time: %f Secs\n",(double)Time/1000000.0);

#if VERIFY
	for(i = 0; i < nsize; i++)
		printf("%6.5f %5.5f\n",B[i],C[i]);
#endif

	return 0;
}
