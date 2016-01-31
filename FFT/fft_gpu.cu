/*
FFT_GPU
Ye Wang
*/

#include <stdlib.h>
#include <stdio.h>

#include <math.h>
//#define M_PI 3.141592653589793f

#define thread_num 512
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
#define N 1048576
//#define N 2097152
//#define N 4194304
//#define N 8388608
//#define N 16777216
//#define N 33554432


float* data_real=(float*)malloc(sizeof(float)*N);
float* data_imag=(float*)malloc(sizeof(float)*N);


//one stage calculation(2 point)
__global__ void stfft(float* data_real_d_in,float* data_imag_d_in,float* data_real_d_out,float* data_imag_d_out,int p)
{	

    int subarray1,subarray2;
    int m,thread_position;
    int subarray_start,subarray2_start;
    int p1,p2;
	float tw_real;
	float tw_imag;
	int power;
	float tmp;
	float real,real2,imag,imag2;
	int	index=threadIdx.x+blockIdx.x*blockDim.x;

		//power=__powf(2,p);
		power = 1<<p;
		subarray1=index>>p;
		m=N>>(p+1);
		subarray2=subarray1+m;

		//thread_position=index%power;
		thread_position=(index)&(power-1);
		subarray_start=subarray1<<p;
		subarray2_start=subarray2<<p;
		p1=subarray_start+thread_position;
		p2=subarray2_start+thread_position;

		//issue request for real parts
		 real=data_real_d_in[p1];
		 real2=data_real_d_in[p2];

		//compute twiddle factor
		tmp=(index)&(m-1);
		tmp=(2*M_PI*subarray1*power)/N;

        //compute twiddle factor
		//tw_real=cosf(tmp);
		//tw_imag=-1*sinf(tmp);
		sincosf(tmp,&tw_imag,&tw_real);
		tw_imag=tw_imag*-1;

		//issue request for imaginary parts
		imag=data_imag_d_in[p1];
		imag2=data_imag_d_in[p2];

		//butterfly real parts
		tmp=real+real2;
		real2=real-real2;
		real=tmp;

		//write back real results of butterfly,only this part is written because we still need to twiddle the other
		p2=subarray_start*2+thread_position;
		data_real_d_out[p2]=real;

		//butterfly imag part
		tmp=imag+imag2;
		imag2=imag-imag2;
		imag=tmp;

		//multiply by twiddle
		tmp=real2;
		real2=real2*tw_real-imag2*tw_imag;
		data_real_d_out[p2+power]=real2;
		imag2=tmp*tw_imag+imag2*tw_real;

		//write back imag result of butterfly
		data_imag_d_out[p2]=imag;
		data_imag_d_out[p2+power]=imag2;
}


int main( int argc, char** argv) 
{
	for(int i=0;i<N;i++)
	{	
		if(i<N/2) 
		{
        data_real[i]=1.f;
		data_imag[i]=0.f;
        }
		else
        {
			data_real[i]=0.f;
            data_imag[i]=0.f;
		}
	}

	int passes=log((float)N)/log((float)2);

	float* data_real_d;
	float* data_imag_d;
	float* data_real_d_out;
	float* data_imag_d_out;
	float* tmp;
    float* tmp2;

	float fft_time;

	cudaEvent_t start, stop; float time;
	cudaMalloc((void**)&data_real_d,N*sizeof(float));
	cudaMalloc((void**)&data_imag_d,N*sizeof(float));
	cudaMalloc((void**)&data_real_d_out,N*sizeof(float));
	cudaMalloc((void**)&data_imag_d_out,N*sizeof(float));

    dim3 dimBlock(thread_num,1,1);
    dim3 dimGrid(N/2/thread_num,1,1);
	cudaMemcpy(data_real_d,data_real,sizeof(float)*N,cudaMemcpyHostToDevice);
	cudaMemcpy(data_imag_d,data_imag,sizeof(float)*N,cudaMemcpyHostToDevice);

    cudaEventCreate(&stop);
	cudaEventCreate(&start);
	cudaEventRecord( start, 0 );

	for(int i=0;i<passes;i++)
	{
		//execute for log2 N times
		stfft<<<dimGrid,dimBlock>>>(data_real_d,data_imag_d,data_real_d_out,data_imag_d_out,i);

        //switch data_real_d and data_real_d_out
        tmp=data_real_d;
		tmp2=data_imag_d;
		data_real_d=data_real_d_out;
		data_real_d_out=tmp;
		data_imag_d=data_imag_d_out;
		data_imag_d_out=tmp2;
	}

	cudaEventRecord( stop, 0 ); 
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &time, start, stop );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	fft_time=time;
	printf("N=%d, FFT Time=%f ms\n",N,fft_time);



	const char* err=cudaGetErrorString(cudaGetLastError());
        printf("Runtime info: %s\n",err);

	cudaMemcpy(data_real,data_real_d,sizeof(float)*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(data_imag,data_imag_d,sizeof(float)*N,cudaMemcpyDeviceToHost);

	cudaFree(data_real_d);
	cudaFree(data_imag_d);
    cudaFree(data_real_d_out);
    cudaFree(data_imag_d_out);
    
	//Store data in data_real and data_imag
	for(int i=0;i<16;i++)
	{
		printf("data[%d]=%f + %f i\n",i,data_real[i],data_imag[i]);
	}






}

