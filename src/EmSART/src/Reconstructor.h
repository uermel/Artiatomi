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


#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H


#ifdef USE_MPI
#include <mpi.h>
#endif
#include "default.h"
#include "Projection.h"
#include "Volume.h"
#include "Kernels.h"
#include "cuda/CudaArrays.h"
#include "cuda/CudaContext.h"
#include "cuda/CudaTextures.h"
#include "cuda/CudaKernel.h"
#include "cuda/CudaDeviceProperties.h"
#include "utils/Config.h"
#include "utils/CudaConfig.h"
#include "utils/Matrix.h"
#include "io/Dm4FileStack.h"
#include "io/MRCFile.h"
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include "io/MarkerFile.h"
#include "io/writeBMP.h"
#include "io/mrcHeader.h"
#include "io/emHeader.h"
#include "io/CtfFile.h"
#include <time.h>
#include <cufft.h>
#include <npp.h>
#include <algorithm>

class KernelModuls
{
private:
	bool compilerOutput;
	bool infoOutput;

public:
	KernelModuls(Cuda::CudaContext* aCuCtx);
	CUmodule modFP;
	CUmodule modSlicer;
	CUmodule modVolTravLen;
	CUmodule modComp;
	CUmodule modWBP;
	CUmodule modBP;
	CUmodule modCTF;
	CUmodule modCTS;

};

typedef struct {
	float4 m[4];
} float4x4;

class Reconstructor
{
private:
	FPKernel fpKernel;
	SlicerKernel slicerKernel;
	VolTravLengthKernel volTravLenKernel;
	CompKernel compKernel;
	WbpWeightingKernel wbp;
	CropBorderKernel cropKernel;
	BPKernel bpKernel;
	ConvVolKernel convVolKernel;
	ConvVol3DKernel convVol3DKernel;
	CTFKernel ctf;
	CopyToSquareKernel cts;
	FourFilterKernel fourFilterKernel;
	DoseWeightingKernel doseWeightingKernel;
	ConjKernel conjKernel;
	MaxShiftKernel maxShiftKernel;
#ifdef REFINE_MODE
	MaxShiftWeightedKernel maxShiftWeightedKernel;
	FindPeakKernel findPeakKernel;
	Cuda::CudaDeviceVariable		projSquare2_d;
	RotKernel rotKernel;
	Cuda::CudaPitchedDeviceVariable projSubVols_d;
	float* ccMap;
	float* ccMapMulti;
	Cuda::CudaPitchedDeviceVariable ccMap_d;
	NppiRect roiCC1, roiCC2, roiCC3, roiCC4;
	NppiRect roiDestCC1, roiDestCC2, roiDestCC3, roiDestCC4;
#endif

	Cuda::CudaPitchedDeviceVariable realprojUS_d;
	Cuda::CudaPitchedDeviceVariable proj_d;
	Cuda::CudaPitchedDeviceVariable realproj_d;
	Cuda::CudaPitchedDeviceVariable dist_d;
	Cuda::CudaPitchedDeviceVariable filterImage_d;
	

	Cuda::CudaPitchedDeviceVariable ctf_d;
	Cuda::CudaDeviceVariable        fft_d;
	Cuda::CudaDeviceVariable		projSquare_d;
	Cuda::CudaPitchedDeviceVariable badPixelMask_d;
	Cuda::CudaPitchedDeviceVariable volTemp_d;

	cufftHandle handleR2C;
	cufftHandle handleC2R;

	NppiSize roiAll;
	NppiSize roiFFT;
	//NppiSize roiBorderSquare;
	NppiSize roiSquare;

	Cuda::CudaDeviceVariable meanbuffer;
	Cuda::CudaDeviceVariable meanval;
	Cuda::CudaDeviceVariable stdval;

	Projection& proj;
	ProjectionSource* projSource;
	CtfFile& defocus;
	MarkerFile& markers;
	Configuration::Config& config;

	int mpi_part;
	int mpi_size;
	bool skipFilter;
	int squareBorderSizeX;
	int squareBorderSizeY;
	size_t squarePointerShift;
	float* MPIBuffer;

	Matrix<float> magAnisotropy;
	Matrix<float> magAnisotropyInv;

	template<typename TVol>
	void ForwardProjectionCTF(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
	template<typename TVol>
	void ForwardProjectionNoCTF(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);

	template<typename TVol>
	void ForwardProjectionCTFROI(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
	template<typename TVol>
	void ForwardProjectionNoCTFROI(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
		
	template<typename TVol>
	void BackProjectionNoCTF(Volume<TVol>* vol, int proj_index, float SIRTCount);
	template<typename TVol>
	void BackProjectionCTF(Volume<TVol>* vol, int proj_index, float SIRTCount);

#ifdef SUBVOLREC_MODE
	template<typename TVol>
	void BackProjectionNoCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Cuda::CudaArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
	template<typename TVol>
	void BackProjectionCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Cuda::CudaArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
#endif

	template<typename TVol>
	void GetDefocusDistances(float& t_in, float& t_out, int index, Volume<TVol>* vol);

	void GetDefocusMinMax(float ray, int index, float& defocusMin, float& defocusMax);

public:
	Reconstructor(Configuration::Config& aConfig, Projection& aProj, ProjectionSource* aProjectionSource,
		 MarkerFile& aMarkers, CtfFile& aDefocus, KernelModuls& modules, int aMpi_part, int aMpi_size);
	~Reconstructor();

	Matrix<float> GetMagAnistropyMatrix(float aAmount, float angleInDeg, float dimX, float dimY);

	//If returns true, ctf_d contains the fourier Filter mask as defined by coefficients given in config file.
	bool ComputeFourFilter();
	//img_h can be of any supported type. After the call, the type is float! Make sure the array is large enough!
	void PrepareProjection(void* img_h, int proj_index, float& meanValue, float& StdValue, int& BadPixels);

	template<typename TVol>
	void PrintGeometry(Volume<TVol>* vol, int index);

	template<typename TVol>
	void ForwardProjection(Volume<TVol>* vol, Cuda::CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync = false);

	template<typename TVol>
	void ForwardProjectionROI(Volume<TVol>* vol, Cuda::CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync = false);

	template<typename TVol>
	void ForwardProjectionDistanceOnly(Volume<TVol>* vol, int index);

	template<typename TVol>
	void Compare(Volume<TVol>* vol, char* originalImage, int index);

	//Assumes image to back project stored in proj_d. SIRTCount is overridable to config-file!
	template<typename TVol>
	void BackProjection(Volume<TVol>* vol, int proj_index, float SIRTCount);
	
#ifdef SUBVOLREC_MODE
	//Assumes image to back project stored in proj_d.
	template<typename TVol>
	void BackProjection(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Cuda::CudaArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
#endif

	template<typename TVol>
	void BackProjectionWithPriorWBPFilter(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer);

	template<typename TVol>
	void RemoveProjectionFromVol(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer);

	template<typename TVol>
	void OneSARTStep(Volume<TVol>* vol, Cuda::CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, char* originalImage, float SIRTCount, float* MPIBuffer);
	

	void ResetProjectionsDevice();
	void CopyProjectionToHost(float* buffer);
	void CopyDistanceImageToHost(float* buffer);//For Debugging...
	void CopyRealProjectionToHost(float* buffer);//For Debugging...
	void CopyProjectionToDevice(float* buffer);
	void CopyDistanceImageToDevice(float* buffer);//For Debugging...
	void CopyRealProjectionToDevice(float* buffer);//For Debugging...
	void MPIBroadcast(float** buffers, int bufferCount);
#ifdef REFINE_MODE
	void CopyProjectionToSubVolumeProjection();
	float2 GetDisplacement(bool MultiPeakDetection, float* CCValue = NULL);
	void rotVol(Cuda::CudaDeviceVariable& vol, float phi, float psi, float theta);
	void setRotVolData(float* data);
	float* GetCCMap();
	float* GetCCMapMulti();
#endif

	void ConvertVolumeFP16(float* slice, int z);
	void ConvertVolume3DFP16(float* volume);
	void MatrixVector3Mul(float4x4 M, float3* v);

};

#endif // !RECONSTRUCTOR_H

