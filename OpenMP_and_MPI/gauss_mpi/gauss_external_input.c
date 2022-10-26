/* 
 * Original author:  UNKNOWN
 *
 * Modified:         Kai Shen (January 2010)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

/* #define DEBUG */

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
MPI_Request *request;

/* Solve the equation:
 *   matrix * X = R
 */

double **matrix, *X, *R;
int proc_id, nprocs;

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

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */
// This function will be accessed only by the main thead of the MPI code and will send/receive signals from/to the other threads from here
// Please refer to the documentation (Report) for the brief working of this function
// For terminology: There will be 1 main thread (this one) & the remaining will be support threads

void computeGauss(int nsize)
{
    int i, j, k, tid, num_t=nprocs-1;
    double pivotval;
    int blocks, blk_size, bs, offset, bs_t[nprocs];
    
    for (i = 0; i < nsize; i++) {
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

    // Broadcasting value of i & the i matrix which will be used by all threads here
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix[i][i+1], nsize-i-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // To total number of rows to be assigned to support threads (threads except main thread)
    blocks = nsize-i-1;
    if (blocks > num_t)
    {
        // Calculating size of a block in case, total blocks is not divisible by num_threads
        offset = blocks % num_t;
        blk_size = blocks / num_t;
        
        // Sending a continuous set of rows to each support thread
        j = i + 1;
        for (tid=1; tid < nprocs; tid++){
            bs = blk_size + (offset > tid-1);
            MPI_Send(&bs , 1, MPI_INT , tid, 1, MPI_COMM_WORLD);
            MPI_Isend(matrix[j], nsize*bs, MPI_DOUBLE, tid, 2, MPI_COMM_WORLD, &request[tid]);
            j += bs;
            bs_t[tid] = bs;
        }
    }
    else
    {
        j = i + 1;
        bs = 1;
        // Since number of threads is greater than rows, we assigns just 1 row per thread to selected threads
        for (tid=1; tid <= blocks; tid++){
            MPI_Send(&bs , 1, MPI_INT , tid, 1, MPI_COMM_WORLD);
            MPI_Isend(matrix[j], nsize, MPI_DOUBLE, tid, 2, MPI_COMM_WORLD, &request[tid]);
            j++;
        }
        // Intimate remaining threads that, they are not getting a task in this iteration, to prevent infinite loop within them 
        bs = -1;
        for (; tid < nprocs; tid++)
            MPI_Send(&bs , 1, MPI_INT , tid, 1, MPI_COMM_WORLD);
    }

        // Updating R[j]
		for (j = i + 1 ; j < nsize; j++)
            R[j] -= matrix[j][i] * R[i];

		// Receive Data Segment
        // In this block we receive data parallely from all threads (and hence MPI_Irecv & MPI_Wait are used)
		if (blocks > num_t)
		{
			j = i + 1;
			for (tid=1; tid < nprocs; tid++){
				MPI_Wait(&request[tid], MPI_STATUS_IGNORE);
				MPI_Irecv(matrix[j], nsize*bs_t[tid], MPI_DOUBLE, tid, 3, MPI_COMM_WORLD, &request[tid]);
				j += bs_t[tid];
			}
            for (tid=1; tid < nprocs; tid++)
				MPI_Wait(&request[tid], MPI_STATUS_IGNORE);
		}
		else
		{
			j = i + 1;
			for (tid=1; tid <= blocks; tid++){
				MPI_Wait(&request[tid], MPI_STATUS_IGNORE);
				MPI_Irecv(matrix[j++] ,nsize , MPI_DOUBLE, tid, 3, MPI_COMM_WORLD, &request[tid]);	
			}
            for (tid=1; tid <= blocks; tid++)
				MPI_Wait(&request[tid], MPI_STATUS_IGNORE);
		}
	}

	// To quit other threads from infinite while loop
	i = -1;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&i, 1, MPI_INT, 0, MPI_COMM_WORLD);
}


// This is a new function created exclusively for the supporting threads of MPI
// These reads will undergo an infinite while loop until the main thread (thread 0) instructs them to quit
// This function mainly handles a portion of the main O(n^3) time complexity loop
// By assigning the O(n^3) loop to multiple threads, we are speeding up the execution of this program
void matmul(int nsize)
{
	double *mat, pv, mat_i[nsize];
	int i, j, k, bs, size, i_size, k_off;

	mat = (double*)malloc(nsize*((nsize/(nprocs-1))+1)*sizeof(double));
	
	while (1)
	{
        // Used barrier before broadcast since it exhibitted lesser load than the broadcast lock, and improved performace
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&i, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // I used i = -1 to signal that all jobs are completed and it's time to exit the program
		if (i == -1)
			break;
		
        // Receiving the common row broadcast from the main thread
		i_size = nsize - i - 1;
		MPI_Bcast(mat_i, i_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Recv(&bs , 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // bs stands for block size, and if the main thread gets into the else block, this is needed to prevent infinite loop (for unassigned threads)
		if (bs == -1)
			continue;

        // Receiving the set of rows this particular thread has been assigned to work on
		size = nsize*bs;
		MPI_Recv(mat , size, MPI_DOUBLE , 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Performing the main O(n^3) time complexity loop, in this program
		k_off = i+1;
		for (j=0; j<size; j+=nsize){
			pv = mat[j+i];
			mat[j+i] = 0.0;
			for (k=0; k<i_size; k++)
				mat[j+k+k_off] -= pv * mat_i[k];
		}
		
        // Sending back the processed set of rows to the main thread
		MPI_Send(mat , size, MPI_DOUBLE , 0, 3, MPI_COMM_WORLD);
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
    
    if (argc != 2) {
	fprintf(stderr, "usage: %s <matrixfile>\n", argv[0]);
	exit(-1);
    }

    // Initializing MPI Protocol
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    if (nprocs < 2){
		fprintf(stderr,"Error: This program requires a minimum of 2 threads to run.\n");
		MPI_Finalize();
		return 0;
	}

    // Thread 0, the main thread will read the matrix, and run the computeGauss(), whereas the remaining threads will just stay in matmul() and help the main thread
    if (proc_id==0){
		request = (MPI_Request*)malloc(nprocs * sizeof(MPI_Request));
		nsize = initMatrix(argv[1]);
        // Since only Thread 0 read the matrix, the size of it, needs to communicated to the ramining threads, hence broadcasting it
        MPI_Bcast(&nsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        initRHS(nsize);
        initResult(nsize);

        gettimeofday(&start, 0);
        computeGauss(nsize);
    }
    else{
        MPI_Bcast(&nsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		matmul(nsize);
		MPI_Finalize();
		return 0;
	}
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

    MPI_Finalize();
    return 0;
}
