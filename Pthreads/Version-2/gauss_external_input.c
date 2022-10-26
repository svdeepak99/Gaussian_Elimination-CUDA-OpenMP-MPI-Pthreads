/* 
 * Original author:  UNKNOWN
 *
 * Modified:         Kai Shen (January 2010)
 * Modified:  Deepak Subramani Velumani (Feb 2022)
 * Parallel Version:  1b
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>

/* #define DEBUG */

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define block_size 16      // Block length of each memory block in core loop

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/* Solve the equation:
 *   matrix * X = R
 */

int num_threads = 32;      // Default number of threads
double **matrix, *X, *R;

/* Pre-set solution. */

double *X__;

/* Initialize the matirx. */

int initMatrix(const char *fname)
{
    FILE *file;
    int l1, l2, l3;
    double d;
    int nsize;
    int i, j;
    double *tmp;
    char buffer[1024];

    if ((file = fopen(fname, "r")) == NULL) {
	fprintf(stderr, "The matrix file open error\n");
        exit(-1);
    }
    
    /* Parse the first line to get the matrix size. */
    fgets(buffer, 1024, file);
    sscanf(buffer, "%d %d %d", &l1, &l2, &l3);
    nsize = l1;
#ifdef DEBUG
    fprintf(stdout, "matrix size is %d\n", nsize);
#endif

    /* Initialize the space and set all elements to zero. */
    matrix = (double**)malloc(nsize*sizeof(double*));
    assert(matrix != NULL);
    tmp = (double*)malloc(nsize*nsize*sizeof(double));
    assert(tmp != NULL);    
    for (i = 0; i < nsize; i++) {
        matrix[i] = tmp;
        tmp = tmp + nsize;
    }
    for (i = 0; i < nsize; i++) {
        for (j = 0; j < nsize; j++) {
            matrix[i][j] = 0.0;
        }
    }

    /* Parse the rest of the input file to fill the matrix. */
    for (;;) {
	fgets(buffer, 1024, file);
	sscanf(buffer, "%d %d %lf", &l1, &l2, &d);
	if (l1 == 0) break;

	matrix[l1-1][l2-1] = d;
#ifdef DEBUG
	fprintf(stdout, "row %d column %d of matrix is %e\n", l1-1, l2-1, matrix[l1-1][l2-1]);
#endif
    }

    fclose(file);
    return nsize;
}

/* Initialize the right-hand-side following the pre-set solution. */

void initRHS(int nsize)
{
    int i, j;

    X__ = (double*)malloc(nsize * sizeof(double));
    assert(X__ != NULL);
    for (i = 0; i < nsize; i++) {
	X__[i] = i+1;
    }

    R = (double*)malloc(nsize * sizeof(double));
    assert(R != NULL);
    for (i = 0; i < nsize; i++) {
	R[i] = 0.0;
	for (j = 0; j < nsize; j++) {
	    R[i] += matrix[i][j] * X__[j];
	}
    }
}

/* Initialize the results. */

void initResult(int nsize)
{
    int i;

    X = (double*)malloc(nsize * sizeof(double));
    assert(X != NULL);
    for (i = 0; i < nsize; i++) {
	X[i] = 0.0;
    }
}

/* Get the pivot - the element on column with largest absolute value. */

void getPivot(int nsize, int currow)
{
    int i, pivotrow;

    pivotrow = currow;
    for (i = currow+1; i < nsize; i++) {
	if (fabs(matrix[i][currow]) > fabs(matrix[pivotrow][currow])) {
	    pivotrow = i;
	}
    }

    if (fabs(matrix[pivotrow][currow]) == 0.0) {
        fprintf(stderr, "The matrix is singular\n");
        exit(-1);
    }
    
    if (pivotrow != currow) {
#ifdef DEBUG
	fprintf(stdout, "pivot row at step %5d is %5d\n", currow, pivotrow);
#endif
        for (i = currow; i < nsize; i++) {
            SWAP(matrix[pivotrow][i],matrix[currow][i]);
        }
        SWAP(R[pivotrow],R[currow]);
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
	double pivotval;

	// The main multithreaded loop with highest time complexity.
    // Every row is divided into blocks & each thread is made to execute one row for each iteration.
    // Since every row is divided into blocks the entire block will fit in the cache memory, allowing 
    //      "matrix[i][k range]" to remain in cache till it's last usage, and avoid cache read misses.
    
	/* Factorize the rest of the matrix. */

    for (j = i + 1 + tid ; j < nsize; j+=num_threads){
		R[j] -= matrix[j][i] * R[i];
	}

    pivotval = matrix[i][i];

    for (k_start=i + 1 ; k_start < nsize; k_start+=block_size)
	{
        k_end = MIN(k_start+block_size, nsize);
        for (j = i + 1 + tid; j < nsize; j+=num_threads) {
            pivotval = matrix[j][i];
            for (k = k_start; k < k_end; k++)
                matrix[j][k] -= pivotval * matrix[i][k];
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
    int i, j;
    double pivotval;
    int rc;
	// Struct variable to pass args to subtractElim() threads
	struct sub_args *args = (struct sub_args *)malloc(num_threads * sizeof(struct sub_args));
	void *status;
	long t;
	pthread_t *threadsSubElim;	// // To store threadid of the subtractElim() threads
	threadsSubElim = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    
    for (i = 0; i < nsize; i++) {
        // Since the time complexity of the getPivot() function is low [O(n^2)] compared to the parallelized loop [O(n^3)],
		// it is not parallelized, as parallelizing will actually increase time complexity due to the time required for
		// creation of new threads, with respect to the relatively smaller computation to make.
        getPivot(nsize,i);
            
        /* Scale the main row. */
        pivotval = matrix[i][i];
        if (pivotval != 1.0) {
            matrix[i][i] = 1.0;
            for (j = i + 1; j < nsize; j++) {
            matrix[i][j] /= pivotval;
            }
            R[i] /= pivotval;
        }

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

/* Solve the equation. */

void solveGauss(int nsize)
{
    int i, j;

    X[nsize-1] = R[nsize-1];
    for (i = nsize - 2; i >= 0; i --) {
        X[i] = R[i];
        for (j = nsize - 1; j > i; j--) {
            X[i] -= matrix[i][j] * X[j];
        }
    }

#ifdef DEBUG
    fprintf(stdout, "X = [");
    for (i = 0; i < nsize; i++) {
        fprintf(stdout, "%.6f ", X[i]);
    }
    fprintf(stdout, "];\n");
#endif
}

int main(int argc, char *argv[])
{
    int i;
    struct timeval start, finish;
    int nsize = 0;
    double error;
    
    if (argc != 2 && argc != 3) {
	fprintf(stderr, "usage: %s <matrixfile> <number_of_threads (optional)>\n", argv[0]);
	exit(-1);
    }

    if (argc == 3){
        num_threads = atoi(argv[2]);
        if (num_threads <= 0){
            fprintf(stderr, "Error: Number of threads must be a postive integer.\n");
            fprintf(stderr, "usage: %s <matrixfile> <number_of_threads (optional)>\n", argv[0]);
            exit(-1);
        }
    }

    printf("\nMatrix File: %s; Matrix Size: %d ; Threads: %d; Block Size: %d\n", argv[1], nsize, num_threads, block_size);

    nsize = initMatrix(argv[1]);
    initRHS(nsize);
    initResult(nsize);

    gettimeofday(&start, 0);
    computeGauss(nsize);
    gettimeofday(&finish, 0);

    solveGauss(nsize);
    
    fprintf(stdout, "Time:  %f seconds\n", (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001);

    error = 0.0;
    for (i = 0; i < nsize; i++) {
	double error__ = (X__[i]==0.0) ? 1.0 : fabs((X[i]-X__[i])/X__[i]);
	if (error < error__) {
	    error = error__;
	    }
    }
    fprintf(stdout, "Error: %e\n", error);

    return 0;
}
