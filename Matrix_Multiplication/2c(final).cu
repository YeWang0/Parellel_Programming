/*
 Ye Wang
 CPEG655
 lab2 problem 2.c
 */
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__ void
matrixMul_2c_conflict(float *C, float *A, float *B, int N);
__global__ void
matrixMul_2c_no_conflict(float *C, float *A, float *B, int N);
void mm(float * C, float * A, float * B, int N);
float GetRand(int seed);
void randomInit(float *data, int size, float val);
void constantInit(float *data, int size, float val);
int matrixMultiply(dim3 &dimsA, dim3 &dimsB, int type);

int main(int argc, char **argv)
{
    //default set block size=16 and tile size=1
    int N=1024;
    dim3 dimsA(N,N);
    dim3 dimsB(N,N);
    //no bank conflict
    matrixMultiply(dimsA, dimsB,0);
    //with bank conflict
    matrixMultiply(dimsA, dimsB,1);
    
    return 0;
}
/*
Conflict condition :Read 4 banks in a half warp
Same number means that threads are in the same half warp
0  0  0  0  4  4  4  4  8  8  8  8 12 12 12 12
0  0  0  0  4  4  4  4  8  8  8  8 12 12 12 12
0  0  0  0  4  4  4  4  8  8  8  8 12 12 12 12
0  0  0  0  4  4  4  4  8  8  8  8 12 12 12 12
1  1  1  1  5  5  5  5  9  9  9  9 13 13 13 13
1  1  1  1  5  5  5  5  9  9  9  9 13 13 13 13
1  1  1  1  5  5  5  5  9  9  9  9 13 13 13 13
1  1  1  1  5  5  5  5  9  9  9  9 13 13 13 13
2  2  2  2  6  6  6  6 10 10 10 10 14 14 14 14
2  2  2  2  6  6  6  6 10 10 10 10 14 14 14 14
2  2  2  2  6  6  6  6 10 10 10 10 14 14 14 14
2  2  2  2  6  6  6  6 10 10 10 10 14 14 14 14
3  3  3  3  7  7  7  7 11 11 11 11 15 15 15 15
3  3  3  3  7  7  7  7 11 11 11 11 15 15 15 15
3  3  3  3  7  7  7  7 11 11 11 11 15 15 15 15
3  3  3  3  7  7  7  7 11 11 11 11 15 15 15 15
*/
/*
No conflict
Same number means that threads are in the same half warp
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4
5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5
6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6
7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7
8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8
9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9
10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11
12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12
13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13
14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14
15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15
*/
__global__ void
matrixMul_2c_conflict(float *C, float *A, float *B, int N)
{
    const int BLOCK_SIZE=16;
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * BLOCK_SIZE * by;
    
    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + N - 1;
    
    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    
    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * N;
    
    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        
        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + N * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix

        //cause bank conflict: a half-warp read 4 banks
        ty=(threadIdx.y%4)*4+threadIdx.x/4;
        tx=threadIdx.x%4+threadIdx.y-threadIdx.y%4;

        for (int i = 0; i < BLOCK_SIZE; i+=8)
        {
            //  Csub += As[ty][k] * Bs[k][tx];
            
            Csub +=As[ty][i]*Bs[i][tx];
            Csub +=As[ty][i+1]*Bs[i+1][tx];
            Csub +=As[ty][i+2]*Bs[i+2][tx];
            Csub +=As[ty][i+3]*Bs[i+3][tx];
            Csub +=As[ty][i+4]*Bs[i+4][tx];
            Csub +=As[ty][i+5]*Bs[i+5][tx];
            Csub +=As[ty][i+6]*Bs[i+6][tx];
            Csub +=As[ty][i+7]*Bs[i+7][tx];
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + N * ty + tx] = Csub;
}


__global__ void
matrixMul_2c_no_conflict(float *C, float *A, float *B, int N)
{
    const int BLOCK_SIZE=16;
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * BLOCK_SIZE * by;
    
    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + N - 1;
    
    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;
    
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    
    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * N;
    
    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        
        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + N * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int i = 0; i < BLOCK_SIZE; i+=8)
        {
            //  Csub += As[ty][k] * Bs[k][tx];
            
            Csub +=As[ty][i]*Bs[i][tx];
            Csub +=As[ty][i+1]*Bs[i+1][tx];
            Csub +=As[ty][i+2]*Bs[i+2][tx];
            Csub +=As[ty][i+3]*Bs[i+3][tx];
            Csub +=As[ty][i+4]*Bs[i+4][tx];
            Csub +=As[ty][i+5]*Bs[i+5][tx];
            Csub +=As[ty][i+6]*Bs[i+6][tx];
            Csub +=As[ty][i+7]*Bs[i+7][tx];
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + N * ty + tx] = Csub;
}



void mm(float * C, float * A, float * B, int N)
{
    int i,j,k;
    float sum=0;
    for(j=0;j<N;j++)
        for(i=0;i<N;i++){
            C[i*N+j]=0;
            sum=0;
            for(k=0;k<N;k++)
                sum+=A[i*N+k]*B[k*N+j];
            C[i*N+j]=sum;
        }
    
}
float GetRand(int seed)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    srand(tv.tv_usec%17+seed);
    //   printf("xxxPacket_loss_rate:Random %f\n",(rand()% 1000) / 1000.0);
    return((rand()% 1000) / 10.02);
}

void randomInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = GetRand(i);
    }
}
void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;//+i%2;
    }
}

int matrixMultiply(dim3 &dimsA, dim3 &dimsB, int type)
{
    printf("START: Tile[1,1],Block[16,16], Matrix[%d,%d]\n",dimsB.x,dimsA.y);
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);
    
    // Initialize host memory
    randomInit(h_A, size_A, 2.1f);
    randomInit(h_B, size_B, 1.f);
    
    unsigned int size_C = dimsB.x * dimsA.y;
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);
    float *test_C = (float *) malloc(mem_size_C);
    constantInit(test_C, size_C, 0.f);
    constantInit(h_C, size_C, 0.f);
    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_B);
    cudaMalloc((void **) &d_C, mem_size_C);
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);
    // Setup execution parameters
    dim3 threads(16, 16);
    dim3 grid(dimsA.y/16, dimsB.x/16);
    cudaDeviceSynchronize();
    
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);
    
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start, NULL);
    
    // Execute th kernel
    int nIter = 100;
    
    for (int j = 0; j < nIter; j++)
    {
        if (type==0) {
            matrixMul_2c_no_conflict<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x);
        }
        else{
            matrixMul_2c_conflict<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x);
        }
        
    }
    
    // Record the stop event
    cudaEventRecord(stop, NULL);
    
    cudaEventSynchronize(stop);
    
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    
    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf( "REPORT:\n Performance= %.2f GFlop/s\n Time= %.3f msec\n Size= %.0f Ops\n WorkgroupSize= %u threads/block\n",
           gigaFlops,
           msecPerMatrixMul,
           flopsPerMatrixMul,
           threads.x * threads.y);
    
    // Copy t_C,h_A,h_B,dimsA.x);esult from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    
    //double eps = 1.e-6 ;
    mm(test_C,h_A,h_B,dimsA.x);
    int verify=1;
    for (int i=0;i<mem_size_C/4;i++)
    {
        if(h_C[i]!=test_C[i]&&(fabs(h_C[i]-test_C[i])/test_C[i])>1E-6){
            printf("Matrix[A:%d,B:%d,C:%d] C[%d]=%f,  Expect= %f\n",mem_size_A,mem_size_B,mem_size_C,i,h_C[i],test_C[i]);
            verify=0;
            break;
        }
    }
    free(h_A);
    free(test_C);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    if (verify) {
        printf("SUCCESS!\n\n");
        return true;
    }else{
        printf("WRONG RESULT!\n\n");
        return false;
    }    
}

