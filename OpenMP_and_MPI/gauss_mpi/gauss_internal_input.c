#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>


#define NSIZE       2048
#define VERIFY      0

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
#define SWAPINT(a,b)       {register int tmp; tmp = a; a = b; b = tmp;}
#define ABS(a)          (((a) > 0) ? (a) : -(a))

MPI_Request *request;

double **matrix,*B,*V,*C;
int *swap;
int proc_id, nprocs;

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


/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */
// This function will be accessed only by the main thead of the MPI code and will send/receive signals from/to the other threads from here
// Please refer to the documentation (Report) for the brief working of this function
// For terminology: There will be 1 main thread (this one) & the remaining will be support threads

void computeGauss(int nsize)
{
	int i,j,k,tid,num_t=nprocs-1;
	double pivotVal;
	int blocks, blk_size, bs, offset, bs_t[nprocs];

	for(i = 0; i < nsize; i++){
		getPivot(nsize,i);
		pivotVal = matrix[i][i];

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
		
		// Updating B[j]
		for (j = i + 1 ; j < nsize; j++)
			B[j] -= matrix[j][i] * B[i];

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

	// Initializing MPI Protocol
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

	if (nprocs < 2){
		fprintf(stderr,"Error: This program requires a minimum of 2 threads to run.\n");
		MPI_Finalize();
		return 0;
	}

	while((i = getopt(argc,argv,"s:")) != -1){
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
			default:
				assert(0);
				break;
		}
	}

	// Thread 0, the main thread will create the matrix, and run the computeGauss(), whereas the remaining threads will just stay in matmul() and help the main thread
	if (proc_id==0){
		request = (MPI_Request*)malloc(nprocs * sizeof(MPI_Request));
		allocate_memory(nsize);
		gettimeofday(&start, 0);
		initMatrix(nsize);
		computeGauss(nsize);
	}
	else{
		matmul(nsize);
		MPI_Finalize();
		return 0;
	}
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
	MPI_Finalize();
	return 0;
}
