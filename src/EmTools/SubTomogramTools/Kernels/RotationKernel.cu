
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>


extern "C"
__global__ void rot3d(int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, CUtexObject inVol, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	float center = size / 2; //integer division

	float3 vox = make_float3(x - center, y - center, z - center);
	float3 rotVox;
	rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
	rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
	rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

	outVol[z * size * size + y * size + x] = tex3D<float>(inVol, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);

}

extern "C"
__global__ void shiftRot3d(int size, float3 shift, float3 rotMat0, float3 rotMat1, float3 rotMat2, CUtexObject inVol, float* outVol)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    float center = size / 2; //integer division

    float3 rotShift;
    rotShift.x = rotMat0.x * shift.x + rotMat1.x * shift.y + rotMat2.x * shift.z;
    rotShift.y = rotMat0.y * shift.x + rotMat1.y * shift.y + rotMat2.y * shift.z;
    rotShift.z = rotMat0.z * shift.x + rotMat1.z * shift.y + rotMat2.z * shift.z;

    float3 vox = make_float3(x - (center + shift.x), y - (center + shift.y), z - (center + shift.z));

    float3 rotVox;
    rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z - (shift.x - rotShift.x);
    rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z - (shift.y - rotShift.y);
    rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z - (shift.z - rotShift.z);

    outVol[z * size * size + y * size + x] = tex3D<float>(inVol, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);
}


extern "C"
__global__ void shift(int size, float3 shift, CUtexObject inVol, float* outVol)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	float sx = float(x - shift.x + 0.5f) / float(size);
	float sy = float(y - shift.y + 0.5f) / float(size);
	float sz = float(z - shift.z + 0.5f) / float(size);

	outVol[z * size * size + y * size + x] = tex3D<float>(inVol, sx, sy, sz);
}