/* 
 * Original author:  UNKNOWN
 *
 * Modified:         Kai Shen (January 2010)
 * Modified:  Deepak Subramani Velumani (Feb 2022)
 * Parallel Version:  2
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>


#define NSIZE       2048
#define VERIFY      0

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}
#define ABS(a)          (((a) > 0) ? (a) : -(a))

double **matrix,*B,*V,*C;
int *swap;
int num_threads = 32;		// Default number of threads
int threads_completed = 1;	// Flag (syncing) variable which counts number of threads which completed loop iteration
pthread_mutex_t threads_lock = PTHREAD_MUTEX_INITIALIZER;	// Mutex lock for safe updation of threads_completed variable
// cond - to broadcast signal all threads to start next iteration; signal_main - signal main thread the completion of all threads
pthread_cond_t cond = PTHREAD_COND_INITIALIZER, signal_main = PTHREAD_COND_INITIALIZER;

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
	int i,j,k;
	double pivotVal;
	long t;
	struct compGauss_args *args = (struct compGauss_args *)arguments;
	int nsize = args -> nsize;
	long tid = args -> threadid;


	for(i = 0; i < nsize; i++){

		if (tid == 0)
		{
			// Lock to access 'threads_completed' without interference from other threads, to either wait for
			//	other threads to join (or) execute getPivot() [sequetially] if they have completed their iteration.
			pthread_mutex_lock(&threads_lock);
			if (threads_completed < num_threads)
				pthread_cond_wait(&signal_main, &threads_lock);
			pthread_mutex_unlock(&threads_lock);
			getPivot(nsize,i);
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

		pivotVal = matrix[i][i];

		// The main multithreaded loop with highest time complexity
		// Each thread is made to execute one row for each iteration
		// Blocking is not necessary since the matrix elements involved are already present in sequntial memory
		for (j = i + 1 + tid ; j < nsize; j+=num_threads){
			pivotVal = matrix[j][i];
			matrix[j][i] = 0.0;
			for (k = i + 1 ; k < nsize; k++){
				matrix[j][k] -= pivotVal * matrix[i][k];
			}
			B[j] -= pivotVal * B[i];
		}
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
	int i, t, rc;
	struct timeval start;
	struct timeval finish;
	long compTime;
	double Time;
	int nsize = NSIZE;
	struct compGauss_args CG_args[num_threads];		// Struct variable to pass args to computeGauss() threads
	pthread_t *threadsCompGauss;	// To store threadid of the computeGauss() threads
	threadsCompGauss = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
	void *status;
	int num_processors = sysconf(_SC_NPROCESSORS_ONLN), assign_affinity=1;
	pthread_attr_t attr;	// Pthread attributes to pass the cpu affinity into each thread
	cpu_set_t cpus;
	pthread_attr_init(&attr);
	

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
					if (thread_cnt > 1){
						num_threads = thread_cnt;
					} else {
						fprintf(stderr,"Threads count should atleast be 2 for this version, hence using the default (%ld)\n",(int)num_threads);
					}
				}
				break;
			case 'h':
				printf("Usage: ./program -t <num threads> -s <matrix size>\n");
				return 0;
				break;
			default:
				assert(0);
				break;
		}
	}

	if (num_threads > num_processors)
		assign_affinity = 0;	// In case the number of threads inputted is more than the cores availiable, we let the system decide which thread goes to which CPU

	printf("\nMatrix Size: %d ; Threads: %d\n", nsize, num_threads);
	printf("Setting CPU Affinity : ");
	if (assign_affinity) printf("Yes\n"); else printf("No\n");

	allocate_memory(nsize);

	gettimeofday(&start, 0);
	initMatrix(nsize);

	gettimeofday(&finish, 0);
	
	// Starting threads here
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
