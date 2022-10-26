/* 
 * Original author:  UNKNOWN
 *
 * Modified:         Kai Shen (January 2010)
 * Modified:  Deepak Subramani Velumani (Feb 2022)
 * Parallel Version:  2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>

/* #define DEBUG */

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}

/* Solve the equation:
 *   matrix * X = R
 */

int num_threads = 32;      // Default number of threads
double **matrix, *X, *R;

/* Pre-set solution. */

double *X__;

int threads_completed = 1;	// Flag (syncing) variable which counts number of threads which completed loop iteration
pthread_mutex_t threads_lock = PTHREAD_MUTEX_INITIALIZER;	// Mutex lock for safe updation of threads_completed variable
// cond - to broadcast signal all threads to start next iteration; signal_main - signal main thread the completion of all threads
pthread_cond_t cond = PTHREAD_COND_INITIALIZER, signal_main = PTHREAD_COND_INITIALIZER;


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

// Structure to pass arguments into the p_threaded computeGauss() function
struct compGauss_args {
	int nsize;
	long threadid;
};

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */
/* 
Note: This is the parallelized function of this program
Changes Made:
1) The subraction elimination (2nd half) part of this function is parallelised row wise,
	i.e, each row will be executed by one thread in each iteration (of the outer main loop).
2) The unparallelizable getPivot() function which has very less [O(n^2) compared to O(n^3) of the function] time complexity,
	is always made to run by the thread_0 in every iteration of the main loop.
3) p_thread mutex and signal functions are used to synchronise the threads to avoid deadlock, and improve efficiency of the
	program. The main thread (thread_0) waits for all functions to finish each iteration & the other threads wait for the
	broadcast signal from the main thread (thread_0), after it finishes the sequntial getPivot() function.
*/
void *computeGauss(void *arguments)
{
    int i, j, k;
    double pivotval;
    long t;
	struct compGauss_args *args = (struct compGauss_args *)arguments;
	int nsize = args -> nsize;
	long tid = args -> threadid;
    
    for (i = 0; i < nsize; i++) {
	if (tid == 0)
		{
			// Lock to access 'threads_completed' without interference from other threads, to either wait for
			//	other threads to join (or) execute getPivot() [sequetially] if they have completed their iteration.
			pthread_mutex_lock(&threads_lock);
			if (threads_completed < num_threads)
				pthread_cond_wait(&signal_main, &threads_lock);
			pthread_mutex_unlock(&threads_lock);
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

			threads_completed = 1;
			// Broadcast to all the threads, to start their next iteration (of main outer loop.)
			pthread_cond_broadcast(&cond);
		}
		else
		{
			// Lock to update 'threads_completed' without write/read interferences from other threads
			// And if all threads are completed, signal to main thread for getPivot() execution
			// Wait for broadcast signal from main thread, to continue next iteraion after getPivot() execution
			pthread_mutex_lock(&threads_lock);
			threads_completed += 1;
			if (threads_completed==num_threads)
				pthread_cond_signal(&signal_main);
			pthread_cond_wait(&cond, &threads_lock);
			pthread_mutex_unlock(&threads_lock);
		}
        
	/* Factorize the rest of the matrix. */
        for (j = i + 1 + tid; j < nsize; j+=num_threads) {
            pivotval = matrix[j][i];
            matrix[j][i] = 0.0;
            for (k = i + 1; k < nsize; k++) {
                matrix[j][k] -= pivotval * matrix[i][k];
            }
            R[j] -= pivotval * R[i];
        }
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
    int i, t, rc;
    struct timeval start, finish;
    int nsize = 0;
    double error;
    struct compGauss_args CG_args[num_threads];		// Struct variable to pass args to computeGauss() threads
	pthread_t *threadsCompGauss;	// To store threadid of the computeGauss() threads
	threadsCompGauss = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
	void *status;
    int num_processors = sysconf(_SC_NPROCESSORS_ONLN), assign_affinity=1;
	pthread_attr_t attr;	// Pthread attributes to pass the cpu affinity into each thread
	cpu_set_t cpus;
	pthread_attr_init(&attr);
    
    if (argc != 2 && argc != 3) {
	fprintf(stderr, "usage: %s <matrixfile> <number_of_threads (optional)>\n", argv[0]);
	exit(-1);
    }

    if (argc == 3){
        num_threads = atoi(argv[2]);
        if (num_threads < 2){
            fprintf(stderr, "Error: Number of threads must atleast be 2 for this version.\n");
            fprintf(stderr, "usage: %s <matrixfile> <number_of_threads (optional)>\n", argv[0]);
            exit(-1);
        }
    }

    if (num_threads > num_processors)
		assign_affinity = 0;	// In case the number of threads inputted is more than the cores availiable, we let the system decide which thread goes to which CPU

    printf("\nMatrix File: %s; Matrix Size: %d ; Threads: %d\n", argv[1], nsize, num_threads);
    printf("Setting CPU Affinity : ");
	if (assign_affinity) printf("Yes\n"); else printf("No\n");

    nsize = initMatrix(argv[1]);
    initRHS(nsize);
    initResult(nsize);

    gettimeofday(&start, 0);
    
    for(t=0; t<num_threads; t++)
	{
		CG_args[t].nsize = nsize;
		CG_args[t].threadid = t;
        if (assign_affinity){
			CPU_ZERO(&cpus);
			CPU_SET(t, &cpus);
			pthread_attr_setaffinity_np(&attr, sizeof(cpus), &cpus);
			rc = pthread_create(&threadsCompGauss[t], &attr, computeGauss, (void *)&CG_args[t]);
		}
        else
		    rc = pthread_create(&threadsCompGauss[t], NULL, computeGauss, (void *)&CG_args[t]);
		if (rc)
		{
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
		//pthread_setschedprio((pthread_t*)(&threadsCompGauss[t]), t);
	}

	// Wait for threads to finish
	for (t=0; t<num_threads; t++)
		pthread_join(threadsCompGauss[t], &status);
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
