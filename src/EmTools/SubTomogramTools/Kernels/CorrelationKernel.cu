
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

#define EPS (0.000001f)

extern "C"
__global__ void fftshiftReal(int size, const float* __restrict__ volIn, float* volOut)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	

	if (x >= size || y >= size || z >= size)
		return;
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;


	float temp = volIn[k * size * size + j * size + i]; 
	volOut[z * size * size + y * size + x] = temp;
}

extern "C"
__global__ void energynorm(int size, const float* __restrict__ particle, const float* __restrict__ partSqr, float* cccMap, const float* __restrict__ energyRef, const float* __restrict__ nVox)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	if (x >= size || y >= size || z >= size)
		return;

	int idx = z * size * size + y * size + x;
	float part = particle[idx]; 
	float energyLocal = partSqr[idx]; 
	
	float erg = 0;

	energyLocal -= part * part / nVox[0];
	energyLocal = sqrt(energyLocal) * sqrt(energyRef[0]);

	if (energyLocal > EPS)
	{
		erg = cccMap[idx] / energyLocal;
	}

	cccMap[idx] = erg;
}


extern "C"
__global__ void wedgeNorm(int size, const float* __restrict__ wedge, float2* part, const float* __restrict__ maxVal)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= (size / 2 + 1) || y >= size || z >= size)
		return;

	int idxCplx = z * size * (size / 2 + 1) + y * (size / 2 + 1) + x;

	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	int idxReal = k * size * size + j * size + i;

	float val = wedge[idxReal];
	float maxv = *maxVal;

	if (val < 0.1f * maxv)
		val = 1.0f / (0.1f * maxv);
	else
		val = 1.0f / val;


	float2 p = part[idxCplx];
	p.x *= val;
	p.y *= val;
	part[idxCplx] = p;
}


extern "C"
__global__ void binarize(int length, const float* __restrict__ inVol, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= length)
		return;
	
	outVol[x] = inVol[x] > 0.5f ? 1.0f : 0.0f;
}



extern "C"
__global__ void conv(int length, const float2* __restrict__ inVol, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= length)
		return;
	
	float2 o = outVol[x];
	float2 i = inVol[x];
	float2 erg;
	erg.x = (o.x * i.x) - (o.y * i.y);
	erg.y = (o.x * i.y) + (o.y * i.x);
	outVol[x] = erg;
}



extern "C"
__global__ void correl(int length, const float2* __restrict__ inVol, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= length)
		return;
	
	float2 o = outVol[x];
	float2 i = inVol[x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
	outVol[x] = erg;
}



extern "C"
__global__ void phaseCorrel(int length, const float2* __restrict__ inVol, float2 * outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= length)
		return;

	float2 o = outVol[x];
	float2 i = inVol[x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
	float amplitude = sqrtf(erg.x * erg.x + erg.y * erg.y);
	if (amplitude != 0)
	{
		erg.x /= amplitude;
		erg.y /= amplitude;
	}
	else
	{
		erg.x = erg.y = 0;
	}
	outVol[x] = erg;
}



extern "C"
__global__ void bandpassFFTShift(int size, float2* vol, float rDown, float rUp, float smooth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	if (x >= (size / 2 + 1) || y >= size || z >= size)
		return;

	int idx = z * size * (size / 2 + 1) + y * (size / 2 + 1) + x;

	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	float2 temp = vol[idx];

	//use squared smooth for Gaussian
	smooth = smooth * smooth;

	float center = size / 2;
	float3 vox = make_float3(i - center, j - center, k - center);

	float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
	float scf = (dist - rUp) * (dist - rUp);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;

	if (dist > rUp)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	
	scf = (dist - rDown) * (dist - rDown);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;
	
	if (dist < rDown)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	

	vol[idx] = temp;
}


extern "C"
__global__ void mulRealCplxFFTShift(int size, const float* __restrict__ realVol, float2* cplxVol)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	if (x >= (size / 2 + 1) || y >= size || z >= size)
		return;
		
	int idxCplx = z * size * (size / 2 + 1) + y * (size / 2 + 1) + x;

	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	int idxReal = k * size * size + j * size + i;

	float2 temp = cplxVol[idxCplx];
	float real = realVol[idxReal];

	temp.x *= real;
	temp.y *= real;
	

	cplxVol[idxCplx] = temp;
}