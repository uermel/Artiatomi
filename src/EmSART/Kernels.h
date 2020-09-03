//  Copyright (c) 2018, Michael Kunz and Frangakis Lab, BMLS,
//  Goethe University, Frankfurt am Main.
//  All rights reserved.
//  http://kunzmi.github.io/Artiatomi
//  
//  This file is part of the Artiatomi package.
//  
//  Artiatomi is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Artiatomi is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Artiatomi. If not, see <http://www.gnu.org/licenses/>.
//  
////////////////////////////////////////////////////////////////////////


#ifndef KERNELS_H
#define KERNELS_H

#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"


class FPKernel : public Cuda::CudaKernel
{
public:
	FPKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	FPKernel(CUmodule aModule);

	float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, Cuda::CudaPitchedDeviceVariable& distMap, Cuda::CudaTextureObject3D& texObj);
	float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, Cuda::CudaPitchedDeviceVariable& distMap, Cuda::CudaTextureObject3D& texObj, int2 roiMin, int2 roiMax);
};

class SlicerKernel : public Cuda::CudaKernel
{
public:
	SlicerKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	SlicerKernel(CUmodule aModule);

	float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, float tmin, float tmax, Cuda::CudaTextureObject3D& texObj);
	float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& projection, float tmin, float tmax, Cuda::CudaTextureObject3D& texObj, int2 roiMin, int2 roiMax);
};

class VolTravLengthKernel : public Cuda::CudaKernel
{
public:
	VolTravLengthKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	VolTravLengthKernel(CUmodule aModule);

	float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& distMap);
	float operator()(int x, int y, Cuda::CudaPitchedDeviceVariable& distMap, int2 roiMin, int2 roiMax);
};

class CompKernel : public Cuda::CudaKernel
{
public:
	CompKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	CompKernel(CUmodule aModule);

	float operator()(Cuda::CudaPitchedDeviceVariable& real_raw, Cuda::CudaPitchedDeviceVariable& virtual_raw, Cuda::CudaPitchedDeviceVariable& vol_distance_map, float realLength, float4 crop, float4 cropDim, float projValScale);
};

class SubEKernel : public Cuda::CudaKernel
{
public:
    SubEKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
    SubEKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& real_raw, Cuda::CudaPitchedDeviceVariable& error, Cuda::CudaPitchedDeviceVariable& vol_distance_map);
};

class CropBorderKernel : public Cuda::CudaKernel
{
public:
	CropBorderKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	CropBorderKernel(CUmodule aModule);

	float operator()(Cuda::CudaPitchedDeviceVariable& image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4);
};

class BPKernel : public Cuda::CudaKernel
{
public:
	BPKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim, bool fp16);
	BPKernel(CUmodule aModule, bool fp16);

    float operator()(int proj_x, int proj_y, float lambda, int maxOverSample, float maxOverSampleInv, Cuda::CudaTextureObject2D& img, Cuda::CudaSurfaceObject3D& surf,float distMin, float distMax);
};

class CTFKernel : public Cuda::CudaKernel
{
public:
	CTFKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	CTFKernel(CUmodule aModule);

	//float operator()(CudaPitchedDeviceVariable& ctf, float defocus, bool absolute)
	float operator()(Cuda::CudaDeviceVariable& ctf, float defocusMin, float defocusMax, float angle, bool applyForFP, bool phaseFlipOnly, float WienerFilterNoiseLevel, size_t stride, float4 betaFac);
};

class CopyToSquareKernel : public Cuda::CudaKernel
{
public:
	CopyToSquareKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	CopyToSquareKernel(CUmodule aModule);

	float operator()(Cuda::CudaPitchedDeviceVariable& aIn, int maxsize, Cuda::CudaDeviceVariable& aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero);
};



class ConvVolKernel : public Cuda::CudaKernel
{
public:
	ConvVolKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	ConvVolKernel(CUmodule aModule);

    float operator()(Cuda::CudaPitchedDeviceVariable& img, Cuda::CudaSurfaceObject3D& surf, unsigned int z);
};

class ConvVol3DKernel : public Cuda::CudaKernel
{
public:
	ConvVol3DKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	ConvVol3DKernel(CUmodule aModule);

	float operator()(Cuda::CudaPitchedDeviceVariable& img, Cuda::CudaSurfaceObject3D& surf);
};


enum FilterMethod
{
	FM_RAMP,
	FM_EXACT,
	FM_CONTRAST2,
	FM_CONTRAST10,
	FM_CONTRAST30
};

class WbpWeightingKernel : public Cuda::CudaKernel
{
public:
	WbpWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	WbpWeightingKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& img, size_t stride, unsigned int pixelcount, float psiAngle, FilterMethod fm, int proj_index, int projectionCount, float thickness, Cuda::CudaDeviceVariable& tiltAngles);
};

class FourFilterKernel : public Cuda::CudaKernel
{
public:
	FourFilterKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	FourFilterKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps);
};

class DoseWeightingKernel : public Cuda::CudaKernel
{
public:
	DoseWeightingKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	DoseWeightingKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float dose, float pixelSizeInA);
};

class ConjKernel : public Cuda::CudaKernel
{
public:
	ConjKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	ConjKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& img1, Cuda::CudaPitchedDeviceVariable& img2, size_t stride, int pixelcount);
};

class MaxShiftKernel : public Cuda::CudaKernel
{
public:
	MaxShiftKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	MaxShiftKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int pixelcount, int maxShift);
};

class MaxShiftWeightedKernel : public Cuda::CudaKernel
{
public:
	MaxShiftWeightedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	MaxShiftWeightedKernel(CUmodule aModule);

	float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, int pixelcount, int maxShift);
};

class FindPeakKernel : public Cuda::CudaKernel
{
public:
	FindPeakKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	FindPeakKernel(CUmodule aModule);
	//findPeak(float* img, size_t stride, char* maskInv, size_t strideMask, int pixelcount, float maxThreshold)
	float operator()(Cuda::CudaDeviceVariable& img1, size_t stride, Cuda::CudaPitchedDeviceVariable& mask, int pixelcount, float maxThreshold);
};

class RotKernel : public Cuda::CudaKernel
{
private:
	int size;
	Cuda::CudaTextureArray3D volTexArray;
	void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]);

public:
	RotKernel(CUmodule aModule, int aSize);

	float operator()(Cuda::CudaDeviceVariable& aVolOut, float phi, float psi, float theta);
	void SetData(float* data);
};

class SphericalMaskKernel : public Cuda::CudaKernel
{
private:
    int size;

public:
    SphericalMaskKernel(CUmodule aModule, int aSize);

    float operator()(Cuda::CudaDeviceVariable& aVolOut, float radius);
};

class ApplyMaskKernel : public Cuda::CudaKernel
{
private:
    int size;

public:
    ApplyMaskKernel(CUmodule aModule, int aSize);

    float operator()(Cuda::CudaSurfaceObject3D& volume, Cuda::CudaDeviceVariable& mask, Cuda::CudaDeviceVariable& tempStore, int3 volmin, int3 volmax, int3 dimMask, int3 radiusMask, int3 centerInVol);
};

class RestoreVolumeKernel : public Cuda::CudaKernel
{
private:
    int size;

public:
    RestoreVolumeKernel(CUmodule aModule, int aSize);

    float operator()(Cuda::CudaSurfaceObject3D& volume, Cuda::CudaDeviceVariable& tempStore, int3 volmin, int3 volmax, int3 dimMask, int3 radiusMask, int3 centerInVol);
};
class DimBordersKernel : public Cuda::CudaKernel
{
public:
	DimBordersKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);
	DimBordersKernel(CUmodule aModule);

	float operator()(Cuda::CudaPitchedDeviceVariable& image, float4 crop, float4 cropDim);
};


void SetConstantValues(Cuda::CudaKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

void SetConstantValues(BPKernel& kernel, Volume<unsigned short>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

void SetConstantValues(Cuda::CudaKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

void SetConstantValues(BPKernel& kernel, Volume<float>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv);

void SetConstantValues(CTFKernel& kernel, Projection& proj, int index, float cs, float voltage);

#endif //KERNELS_H