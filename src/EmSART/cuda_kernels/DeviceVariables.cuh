#ifndef DEVICEVARIABLES_CU
#define DEVICEVARIABLES_CU
#include "Constants.h"

typedef struct {
    float4 m[4];
} float4x4;

typedef struct {
	float3 m[3];
} float3x3;

__device__ __constant__ float3 c_volumeBBoxRcp;
__device__ __constant__ float3 c_volumeDim;
__device__ __constant__ int c_volumeDim_x_quarter;
__device__ __constant__ float3 c_volumeDimComplete;
__device__ __constant__ float3 c_halfVoxelSize;
__device__ __constant__ float3 c_voxelSize;
__device__ __constant__ float3 c_invVoxelSize;
__device__ __constant__ float4x4 c_DetectorMatrix;
//__device__ __constant__ float3 c_source;
__device__ __constant__ float3 c_bBoxMin;
__device__ __constant__ float3 c_bBoxMax;
__device__ __constant__ float3 c_bBoxMinComplete;
__device__ __constant__ float3 c_bBoxMaxComplete;
__device__ __constant__ float3 c_detektor;
__device__ __constant__ float3 c_uPitch;
__device__ __constant__ float3 c_vPitch;
__device__ __constant__ float3 c_projNorm;
__device__ __constant__ float c_zShiftForPartialVolume;
//Magnification anisotropy

__device__ __constant__ float3x3 c_magAniso;
__device__ __constant__ float3x3 c_magAnisoInv;

// transform vector by matrix
__device__
void MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
{
	xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
	yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
	//erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
}
#endif
