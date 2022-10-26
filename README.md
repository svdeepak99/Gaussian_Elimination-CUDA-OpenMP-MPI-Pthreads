# Gaussian_Elimination-CUDA-OpenMP-MPI-Pthreads
Comparision of CUDA, OpenMP, MPI & Pthreads performances on parallelizing Gaussian Elimination Algorithm.
- Obtained **578x** & **8x speedups** with CUDA & OpenMP respectively (where 1x is Sequential performance) on Matrix Multiplication of **2048x2048** sized square matrices.
- Implemented parallelization of Gaussian Elimination & Matrix Multiplication Algorithms using **CUDA**, **OpenMP**, **MPI** & **Pthreads** in C language.
- Obtained **2054%**, **697%** & **6.3% speedups** with OpenMP, Pthreads & MPI respectively on Gaussian Elimination of **2048x2048** square matrices.

## Path to Codes:
1) [**Pthreads**](Pthreads) - Parallelization of Gaussian Elimination Algorithm using **Pthreads**.
2) [**OpenMP_and_MPI**](OpenMP_and_MPI) - Parallelization of Gaussian Elimination Algorithm using **OpenMP** and **MPI**, and comparing their performances.
3) [**CUDA_and_OpenMP**](CUDA_and_OpenMP) - Parallelization of Matrix Multiplication Algorithm using **CUDA**, and comparing its performace with **OpenMP**.

## Performances:
### <ins>Gaussian Elimination Single-Core runtimes (in seconds):</ins>
![Sequential_Times](/graphs/gauss_seq.jpg?raw=true "Sequential Times")

### <ins>Gaussian Elimination: Pthreads vs MPI vs OpenMP runtimes (in seconds):</ins>
![Pthreads-MPI-OpenMP_Times](/graphs/pthreads-mpi-openmp.jpg?raw=true "Pthreads-MPI-OpenMP Times")

### <ins>Matrix Multiplication: Single-Core vs OpenMP vs CUDA speedups (x times the 1-core speed):</ins>
![Seq-OpenMP-CUDA_Speedups](/graphs/mm_seq-openmp-cuda.jpg?raw=true "Seq-OpenMP-CUDA Speedups")
