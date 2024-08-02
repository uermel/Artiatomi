#ifndef DEVICEVARIABLES_CU
#define DEVICEVARIABLES_CU
#include "Constants.h"
#include "common_types.h"


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

// Splines
__device__ __constant__ float c_supporthalf;
__device__ __constant__ float c_LUTstepinv;
__device__ __constant__ float c_LUTcenter;
__device__ __constant__ int c_oversampleFactor;
__device__ __constant__ int2 c_blockSupportSize;
__device__ __constant__ float c_voxelSupportSize;
__device__ __constant__ float c_voxelSupportHalf;
__device__ __constant__ float c_entry;
__device__ __constant__ float c_sliceThickness;
__device__ __constant__ int c_sliceNumber;

#endif
