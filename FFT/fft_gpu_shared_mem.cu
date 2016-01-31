/*
FFT_GPU with shared memory
Ye Wang
*/
#include <stdlib.h>
#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>
//#define M_PI 3.141592653589793f

//#define N 33554432
//#define thread_num 16
//#define thread_num 32
//#define thread_num 64 
#define thread_num 128 
//#define thread_num 256 
//#define thread_num 512 
//#define N 1024
//#define N 2048
//#define N 4096
//#define N 8192
//#define N 16384
//#define N 32768
//#define N 65536
//#define N 131072
//#define N 262144
//#define N 524288
//#define N 1048576
//#define N 2097152
#define N 4194304
//#define N 8388608
//#define N 16777216
//#define N 33554432

//#define T 1024
//#define T 2048
//#define T 4096
//#define T 8192
//#define T 16384
//#define T 32768
//#define T 65536
//#define T 131072
//#define T 262144
//#define T 524288
//#define T 1048576
//#define T 2097152
//#define T 4194304
//#define T 8388608
//#define T 16777216
//#define T 33554432
float *data_real=(float*)malloc(sizeof(float)*N);
float *data_imag=(float*)malloc(sizeof(float)*N);


__global__ void fft(float* data_real_d,float* data_imag_d,int p)
{	
	
	unsigned int sub,index,elements_per_thread,i,i2,sub2,x,block,tmp2;
	float tw_real;
	float tw_imag;
	unsigned int power;
	float tmp;
	
	index=threadIdx.x+blockIdx.x*blockDim.x;  
	elements_per_thread=(N/2)/T;
	sub2=threadIdx.x*elements_per_thread;

	 __shared__ float reals[(N/2)/T*thread_num];
	 __shared__ float reals2[(N/2)/T*thread_num];
	 __shared__ float imags[(N/2)/T*thread_num];
	 __shared__ float imags2[(N/2)/T*thread_num];

		//determine which block the thread is in(not cuda block)
		power=__powf(2,p);
		x=N/(power*2);
		if(x<elements_per_thread){
            block=index*(elements_per_thread/x);
            tmp=0;
        }
		else{
            block=(index)/(x/elements_per_thread);
            tmp=(index)%(x/elements_per_thread);
		}
    
		sub=block*x*2+tmp*elements_per_thread;
		tmp2=sub;
		
		
		//issue request for all parts
	for(i=0,i2=0;i<elements_per_thread;i++)
	{	
		reals[sub2+i]=data_real_d[sub+i2];
		imags[sub2+i]=data_imag_d[sub+i2];
		reals2[sub2+i]=data_real_d[sub+x+i2];
		imags2[sub2+i]=data_imag_d[sub+x+i2];
		
		if(i2==x-1)
		{
			sub+=(x*2);
			i2=0;
		}
		else{i2++;}
		
	}
	
    sub=tmp2;

	for(i=0,i2=0;i<elements_per_thread;i++)
	{	//compute twiddle factor
		tmp=(sub+i)%(x);
		tmp=(2*M_PI*tmp*power)/N;
		tw_real=cosf(tmp);
		tw_imag=-1*sinf(tmp);
		
		//butterfly real parts
		tmp=reals[sub2+i]+reals2[sub2+i];
		reals2[sub2+i]=reals[sub2+i]-reals2[sub2+i];
		reals[sub2+i]=tmp;

		//write back real results of butterfly,only this part is written because we still need to twiddle the other
		data_real_d[sub+i2]=reals[sub2+i];
		//butterfly imag part
		tmp=imags[sub2+i]+imags2[sub2+i];
		imags2[sub2+i]=imags[sub2+i]-imags2[sub2+i];
		imags[sub2+i]=tmp;
		//multiply by twiddle
		tmp=reals2[sub2+i];
		reals2[sub2+i]=reals2[sub2+i]*tw_real-imags2[sub2+i]*tw_imag;
		data_real_d[sub+x+i2]=reals2[sub2+i];
		imags2[sub2+i]=tmp*tw_imag+imags2[sub2+i]*tw_real;
		//write back imag result of butterfly
		data_imag_d[sub+i2]=imags[sub2+i];
		data_imag_d[sub+x+i2]=imags2[sub2+i];
				
		if(i2==x-1)
		{
			sub+=x*2;
			i2=0;
		}
		else{
            i2++;
        }
    }
}



void bit_reversal()
{
   long i,i1,j,k,i2;
   double c1,c2,tx,ty;
   i2 = N >> 1;
   j = 0;
   for (i=0;i<N-1;i++) {
      if (i < j) {
         tx = data_real[i];
         ty = data_imag[i];
         data_real[i] = data_real[j];
         data_imag[i] = data_imag[j];
         data_real[j] = tx;
         data_imag[j] = ty;
      }
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
   }
}


int main( int argc, char** argv) 
{
	for(int i=0;i<N;i++)
	{	
		if(i<N/2) 
		{data_real[i]=1;
		data_imag[i]=0;}
		else{
		data_real[i]=0;
		data_imag[i]=0;
		}
	}

	int passes=(int)log2f(N);
	float* data_real_d;
	float* data_imag_d;

	cudaMalloc((void**)&data_real_d,N*sizeof(float));
	cudaMalloc((void**)&data_imag_d,N*sizeof(float));

	cudaMemcpy(data_real_d,data_real,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(data_imag_d,data_imag,sizeof(float)*N,cudaMemcpyHostToDevice);

	dim3 dimBlock(thread_num,1,1);
	dim3 dimGrid(T/thread_num,1,1);
	cudaThreadSynchronize();

	cudaEvent_t start, stop; float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );
	for(int i=0;i<passes;i++)
	{
        fft<<<dimGrid,dimBlock>>>(data_real_d,data_imag_d,i);
    }
	cudaThreadSynchronize();
	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
    printf("N=%d, T=%d, thread_num=%d\n FFT time=%f ms\n",N,T,thread_num,time);
    
	
    const char* err=cudaGetErrorString(cudaGetLastError());
    printf("Runtime info: %s\n",err);


	cudaMemcpy(data_real,data_real_d,4*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(data_imag,data_imag_d,4*N,cudaMemcpyDeviceToHost);
	cudaFree(data_real_d);
	cudaFree(data_imag_d);
	bit_reversal();

for(int i=0;i<16;i++)
	{
		printf("data[%d]=%f + %f i\n",i,data_real[i],data_imag[i]);
	}

}

