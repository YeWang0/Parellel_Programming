/*
 Ye Wang
 CPEG655
 lab2 problem 1.a
 */
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__global__ void
matrixMul_1a(float *C, float *A, float *B, int N);

void mm(float * C, float * A, float * B, int N);
float GetRand(int seed);
void randomInit(float *data, int size, float val);
void constantInit(float *data, int size, float val);
int matrixMultiply(int  tile_size, int block_size, dim3 &dimsA, dim3 &dimsB);

int main()
{
    
    int block_size = 16;//16 or 8
    int tile_size=1;
    int N=16;
    dim3 dimsA(N,N);
    dim3 dimsB(N,N);
    
    matrixMultiply(tile_size, block_size, dimsA, dimsB);
    
    block_size = 22;
    dimsA.x=22;
    dimsB.x=22;
    dimsA.y=22;
    dimsB.y=22;
    
    matrixMultiply(tile_size, block_size, dimsA, dimsB);
    
    return 0;
}

__global__ void
matrixMul_1a(float *C, float *A, float *B, int N)
{
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0.f;
    
    for (int n=0; n<N; n++){
        sum += A[ty*N+n]*B[n*N+tx];
    }
    C[ty*N+tx] = sum;
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
    return((rand()% 1000) / 1.02);
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

int matrixMultiply(int  tile_size, int block_size, dim3 &dimsA, dim3 &dimsB)
{
    printf("START: Tile[%d,%d],Block[%d,%d], Matrix[%d,%d]\n",tile_size,tile_size,block_size,block_size,dimsB.x,dimsA.y);
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
    dim3 threads(block_size, block_size);
    dim3 grid(dimsA.y/(block_size*tile_size), dimsB.x/(block_size*tile_size));
    cudaDeviceSynchronize();//////////////////////****
    
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
        matrixMul_1a<<< grid, threads >>>(d_C, d_A, d_B, dimsA.x);
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

