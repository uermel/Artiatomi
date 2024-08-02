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
#include "EmSartDefault.h"
#include "KernelModules.h"
#include "Projection.h"
#include "Volume.h"
#include "DeviceVolume.h"
#include "CTF.h"
#include "kernels/kernels.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "utils/Config.h"
#include "utils/CudaConfig.h"
#include <Matrix.h>
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include <MarkerFile.h>
#include "io/writeBMP.h"
#include <CtfFile.h>
#include <time.h>
#include <cufft.h>
#include <npp.h>
#include <algorithm>


class Reconstructor
{
private:
	FPKernel fpKernel;
	SlicerKernel slicerKernel;
	VolTravLengthKernel volTravLenKernel;
	CompKernel compKernel;
	SubEKernel subEKernel;
	WbpWeightingKernel wbp;
	CropBorderKernel cropKernel;
    CropBorderInvKernel cropInvKernel;
	CropSlicesKernel cropSlicesKernel;
    CropSlicesInvKernel cropSlicesInvKernel;
	BPKernel bpKernel;
	ConvVolKernel convVolKernel;
	ConvVol3DKernel convVol3DKernel;
	CTFKernel ctf;
    PostFilterSumKernel postFilterSum;
	CopyToSquareKernel cts;
	//CopyToSquareSlicesKernel ctss;
	//CopyToRectSlicesKernel ctrs;
	RectToSqrSlice r2ss;
	SqrSliceToRectSliceKernel ss2rs;
	RectSliceToSqrSliceKernel rs2ss;
	SqrSliceToRectKernel ss2r;
	FourFilterKernel fourFilterKernel;
	DoseWeightingKernel doseWeightingKernel;
	ConjKernel conjKernel;
	PCKernel pcKernel;
	MaxShiftKernel maxShiftKernel;
	DimBordersKernel dimBordersKernel;
    SplinePrefilter3DXSurf prefilter3DX;
    SplinePrefilter3DYSurf prefilter3DY;
    SplinePrefilter3DZSurf prefilter3DZ;
    SplinePrefilter3DXSurf postfilter3DX;
    SplinePrefilter3DYSurf postfilter3DY;
    SplinePrefilter3DZSurf postfilter3DZ;
    FPOrthoKernel fpOrthoKernel;
    FPOrthoSSKernel fpOrthoSSKernel;
    FPOrthoOVKernel fpOrthoOVKernel;
    FPDistOrthoKernel distOrthoKernel;
    CubicResampleKernel2D sample2D;
    CubicResampleKernel3D sample3D;
    BoxSplineDualBKernel postFilterBox;
    BoxSplineDualBReflKernel preFilterBox;
    CopyToPitchedKernel copyToPitched;
    CopyFromPitchedKernel copyFromPitched;
    AddToPitchedKernel addToPitched;
    PostFilterKernel postFilter;
    //PreFilterSpreadKernel preFilterSpread;
    SlicesToArraysKernel slicesToArrays;
    MaskedSlicesToArraysKernel maskedSlicesToArrays;
    BPOrthoSlicedKernel bpOrthoKernel;
    BPOrthoSlicedAddKernel bpOrthoAdd;
    BPOrthoSlicedAddSSKernel bpOrthoAddSS;
    Add3DKernel add3D;
    Set3DKernel set3D;
    Add3DMaskedKernel add3Dmasked;
    Mask3DKernel mask3D;
    RadialSumAbsKernel radialSum;
    PreFilterSpreadAdHocKernel preFilterSpreadAdHoc;
    PreFilterSpreadSNRKernel preFilterSpreadSNR;
    CompSpecialKernel compSpecialKernel;
    Mask3DTransformKernel mask3DTransform;
    Add3DTransformKernel add3DTransform;
    Norm3DOverlapKernel norm3Doverlap;
    Multiplicity3DKernel multiplicity3D;
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

    // START Utility vars
    // Dimensions of the projection
    uint2 projDim;
    float2 projDimF;

    // Dimensions of the FFT and correction for pixelsize
    uint2 fftDim;
    float2 fftDimF;
    float2 asymCorrFac;

    // Number of elements in projection
    size_t projSize;
    size_t projSizeF32;

    // Number of elements in FFT
    size_t fftSize;
    size_t fftSizeFC32;

    // Pitch/Stride/Step for F32/Char projection and FFT
    size_t projPitchChar;
    size_t projPitchFloat;
    size_t fftPitchComplex;
    // END Utility vars

    // START Projections
    // unsigned real proj
	Cuda::CudaPitchedDeviceVariable realprojUS_d;
	// projection/comparison result
    Cuda::CudaPitchedDeviceVariable proj_d;
    Cuda::CudaPitchedDeviceVariable proj_children_d;
    // projection without pitch
    Cuda::CudaDeviceVariable proj_dv_d;
    // float real proj
	Cuda::CudaPitchedDeviceVariable realproj_d;
    // distance image
	Cuda::CudaPitchedDeviceVariable dist_d;
    Cuda::CudaPitchedDeviceVariable dist_children_d;
    // Temp storage
	Cuda::CudaPitchedDeviceVariable filterImage_d;
    // END Projections

    // START complex projections
	Cuda::CudaPitchedDeviceVariable ctf_d;
	Cuda::CudaDeviceVariable        fft_d;
    Cuda::CudaDeviceVariable        fft_d2;
	Cuda::CudaDeviceVariable		projSquare_d;
	Cuda::CudaPitchedDeviceVariable badPixelMask_d;
	Cuda::CudaPitchedDeviceVariable volTemp_d;

    // START basis function lookup tables
    Cuda::CudaDeviceVariable d_prefilter_fft;
    // END basis function lookup tables

    // START SNR computation
    // 1D Arrays for storage
    Cuda::CudaDeviceVariable signal_power_d;
    Cuda::CudaDeviceVariable noise_power_d;
    Cuda::CudaDeviceVariable multiplicity_d;
    Cuda::CudaDeviceVariable fourier_contrib_d;
    int snrShells;

    // Arrays for interpolation
    Cuda::CudaArray1D signal_power_arr_d;
    Cuda::CudaArray1D noise_power_arr_d;

    Cuda::CudaTextureObject1D texSP;
    Cuda::CudaTextureObject1D texNP;

    float* sp_stack;
    float* np_stack;

    // Setup for halfsets
    std::vector<Cuda::CudaArray1D> signal_power_arr_d_HS = {Cuda::CudaArray1D(), Cuda::CudaArray1D()};
    std::vector<Cuda::CudaArray1D> noise_power_arr_d_HS = {Cuda::CudaArray1D(), Cuda::CudaArray1D()};

    std::vector<Cuda::CudaTextureObject1D> texSP_HS = {Cuda::CudaTextureObject1D(), Cuda::CudaTextureObject1D()};
    std::vector<Cuda::CudaTextureObject1D> texNP_HS = {Cuda::CudaTextureObject1D(), Cuda::CudaTextureObject1D()};

    std::vector<float*> sp_stack_HS = {nullptr, nullptr};
    std::vector<float*> np_stack_HS = {nullptr, nullptr};

    bool snr_loaded = false;
    // END SNR computation

    //START Exact Filter computation
    Cuda::CudaDeviceVariable det_mats_d;
    //END Exact Filter computation

    //START Distance buffer
    float* proj_distance;
    //END Distance buffer

	Cuda::CudaTextureObject2D texImage;

	cufftHandle handleR2C;
	cufftHandle handleC2R;

	NppiSize roiAll;
	NppiSize roiFFT;
	NppiSize roiSquare;

	Cuda::CudaDeviceVariable meanbuffer;
	Cuda::CudaDeviceVariable meanval;
	Cuda::CudaDeviceVariable stdval;

	Projection& proj;
	ProjectionSource* projSource;
	CtfFile& defocus;
	MarkerFile& markers;
	Configuration::Config& config;
    CTF ctfHandler;

	int mpi_part;
	int mpi_size;
	bool skipFilter;
	int squareBorderSizeX;
	int squareBorderSizeY;
	size_t squarePointerShift;
	float* MPIBuffer;

	Matrix<float> magAnisotropy;
	Matrix<float> magAnisotropyInv;

	float LUTcenter = 0;
	int maxSliceNumber = 0;
    int sliceBatchSize = 0;
	float sliceThickness = 999999999999.0f;
    vector<Cuda::CudaArray2D*> BPArrays;
    vector<Cuda::CudaSurfaceObject2D*> BPSurfaces;
    vector<Cuda::CudaTextureObject2D*> BPTextures;

    cufftHandle FFThandleR2Call;
    cufftHandle FFThandleC2Rall;

    Cuda::CudaDeviceVariable CTFbuffer_realRect;
	Cuda::CudaDeviceVariable CTFbuffer_comp1;
    Cuda::CudaDeviceVariable CTF_compute_buffer;


	template<typename TVol>
	void GetDefocusDistances(float& t_in, float& t_out, int index, Volume<TVol>* vol);

	void GetDefocusMinMax(float ray, int index, float& defocusMin, float& defocusMax);


public:
	Reconstructor(Configuration::Config& aConfig, Projection& aProj, ProjectionSource* aProjectionSource,
		 MarkerFile& aMarkers, CtfFile& aDefocus, KernelModules& modules, int aMpi_part, int aMpi_size,
         bool doHalfsets = false);
	~Reconstructor();

	Matrix<float> GetMagAnistropyMatrix(float aAmount, float angleInDeg, float dimX, float dimY);

	//img_h can be of any supported type. After the call, the type is float! Make sure the array is large enough!
	void PrepareProjection(void* img_h, int proj_index, float& meanValue, float& StdValue, int& BadPixels);
    void ExactFilter(int aIndex);

    void PlanCTFCorrection(Volume<float>* vol, int goodProjNumber, int fullProjNumber, const int* indexList);

	template<typename TVol>
	void PrintGeometry(Volume<TVol>* vol, int index);

    void ConjoinChildren(Volume<float>* parentVol,
                         std::vector<Volume<float>*>& childVols,
                         DeviceVolume& parentDevVol,
                         DeviceVolumeBuf& maskDevVol);

    // If computeSNR, expects signal power in signal_power_d
	template<typename TVol>
	void Compare(Volume<TVol>* vol, char* originalImage, uint aIndex, bool computeSNR, int iter, bool normForLength = true, float length = 1.f);

//    void CompareParticles(Volume<float>* vol,
//                          char* originalImage,
//                          uint aIndex,
//                          bool computeSNR,
//                          int iter,
//                          float numParts,
//                          bool normForLength = true,
//                          float length = 1.f);

    void CompareChildren(char* originalImage,
                          Volume<float>* vol,
                          std::vector<Volume<float>*>&  childVols,
                          DeviceVolumeFFT& childDevVol,
                          DeviceVolumeBuf& maskDevVol,
                          uint stackIdx,
                          bool computeSNR,
                          int iter);

//    void CompareSpecial(Volume<float>* parentVol, Volume<float>* childVol, char* originalImage, uint stackIdx, bool computeSNR, int iter);
    void PrepareForWBP(Volume<float>* parentVol,
                       std::vector<Volume<float>*>&  childVols,
                       char* originalImage,
                       uint stackIdx, int iter);

    void MultiplicityChildren(std::vector<Volume<float>*>&  childVols,
                              DeviceVolumeBuf& multDevVol,
                              uint stackIdx, int iter);

    void PowerChildren(std::vector<Volume<float>*>&  childVols,
                       DeviceVolumeFFT& childDevVol,
                       DeviceVolumeBuf& maskDevVol,
                       uint stackIdx,
                       int iter);

    void PowerChildrenHS(std::vector<Volume<float>*>&  childVols,
                         DeviceVolumeFFT& childDevVol,
                         std::vector<DeviceVolumeFFT*>& childDevHalfs,
                         DeviceVolumeBuf& maskDevVol,
                         uint stackIdx,
                         int iter);

    void PowerOrphans(std::vector<Volume<float>*>&  childVols,
                      DeviceVolumeFFT& childDevVol,
                      DeviceVolumeBuf& maskDevVol,
                      uint stackIdx,
                      int iter);

    void PowerOrphansHS(std::vector<Volume<float>*>&  childVols,
                        DeviceVolumeFFT& childDevVol,
                        std::vector<DeviceVolumeFFT*>& childDevHalfs,
                        DeviceVolumeBuf& maskDevVol,
                        uint stackIdx,
                        int iter);

	void SubtractError(float* error);

    void BackProjectionChildren(Volume<float>* parentVol,
                                     std::vector<Volume<float>*>& childVols,
                                     DeviceVolumeFFT& childDevVol,
                                     DeviceVolumeBuf& maskDevVol,
                                     DeviceVolumeBuf& multDevVol,
                                     int stackIdx, float SIRTCount, int iter, std::stringstream& output, bool useSNR = false);

    void BackProjectionChildrenHS(Volume<float>* parentVol,
                                  std::vector<Volume<float>*>&  childVols,
                                  DeviceVolumeFFT& childDevVol,
                                  std::vector<DeviceVolumeFFT*>& childDevHalfVols,
                                  DeviceVolumeBuf& maskDevVol,
                                  DeviceVolumeBuf& multDevVol,
                                  int stackIdx, float SIRTCount, int iter,
                                  stringstream& output, bool useSNR);

    void ForwardProjectionChildren(Volume<float>* parentVol,
                                        std::vector<Volume<float>*>& childVols,
                                        DeviceVolumeFFT& childDevVol,
                                        DeviceVolumeBuf& maskDevVol,
                                        int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                        std::stringstream& output);

    void ForwardProjectionChildrenHS(Volume<float>* parentVol,
                                     std::vector<Volume<float>*>& childVols,
                                     std::vector<DeviceVolumeFFT*>& childDevVols,
                                     DeviceVolumeBuf& maskDevVol,
                                     int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                     std::stringstream& output);

    void ForwardProjectionConjoinedChildren(Volume<float>* parentVol,
                                            std::vector<Volume<float>*>& childVols,
                                            DeviceVolume& overlapDevVol,
                                            DeviceVolumeFFT& childDevVol,
                                            DeviceVolumeBuf& maskDevVol,
                                            int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                            stringstream& output);

    void ForwardProjectionConjoinedChildrenHS(Volume<float>* parentVol,
                                              std::vector<Volume<float>*>& childVols,
                                              DeviceVolume& overlapDevVol,
                                              std::vector<DeviceVolumeFFT*>& childDevVols,
                                              DeviceVolumeBuf& maskDevVol,
                                              int stackIdx, bool volumeIsEmpty, int iter, bool noSync,
                                              stringstream& output);

    void BackProjectionSiblings(Volume<float>* parentVol,
                                std::vector<Volume<float>*>&  childVols,
                                std::vector<DeviceVolumeBuf*>& childDevVols,
                                DeviceVolumeBuf& maskDevVol,
                                DeviceVolumeBuf& multDevVol,
                                int stackIdx, float SIRTCount, int iter, std::stringstream& output, bool useSNR);

    void BackProjectionOrphans(Volume<float>* parentVol,
                               std::vector<Volume<float>*>&  childVols,
                               DeviceVolumeFFT& childDevVol,
                               DeviceVolumeBuf& maskDevVol,
                               DeviceVolumeBuf& multDevVol,
                               int stackIdx, float SIRTCount, int iter,
                               stringstream& output, bool useSNR);

    void BackProjectionOrphansHS(Volume<float>* parentVol,
                                 std::vector<Volume<float>*>&  childVols,
                                 DeviceVolumeFFT& childDevVol,
                                 std::vector<DeviceVolumeFFT*>& childDevHalfVols,
                                 DeviceVolumeBuf& maskDevVol,
                                 DeviceVolumeBuf& multDevVol,
                                 int stackIdx, float SIRTCount, int iter,
                                 stringstream& output, bool useSNR);

    void BackProjectionParent(Volume<float>* parentVol,
                                   DeviceVolume& parentDevVol,
                                   int stackIdx, float SIRTCount, int iter, std::stringstream& output, bool useSNR = false);

    void ForwardProjectionParent(Volume<float>* parentVol,
                                      DeviceVolume& parentDevVol,
                                      int stackIdx, bool volumeIsEmpty, int iter, std::stringstream& output);

    void ForwardProjectionParentMasked(Volume<float>* parentVol,
                                       std::vector<Volume<float>*>& childVols,
                                       DeviceVolume& parentDevVol,
                                       DeviceVolumeBuf& maskInvDevVol,
                                       int stackIdx, bool volumeIsEmpty, int iter, bool noSync, std::stringstream& output);


//    void BackProjectionFamily(Volume<float>* parentVol,
//                              std::vector<Volume<float>*>&  childVols,
//                              DeviceVolume& childDevVol,
//                              DeviceVolume& maskDevVol,
//                              DeviceVolume& parentDevVol,
//                              int stackIdx, float SIRTCount, int iter, bool useSNR = false);
//
//    void ForwardProjectionFamily(Volume<float>* parentVol,
//                                 std::vector<Volume<float>*>& childVols,
//                                 DeviceVolume& childDevVol,
//                                 DeviceVolume& maskDevVol,
//                                 DeviceVolume& maskInvDevVol,
//                                 DeviceVolume& parentDevVol,
//                                 int stackIdx, bool volumeIsEmpty, int iter, bool noSync);

    void DistanceChildren(Volume<float>* parentVol,
                          std::vector<Volume<float>*>& childVols,
                          DeviceVolume& maskDevVol,
                          int stackIdx, bool volumeIsEmpty, int iter, bool noSync);

    void DistanceParent(Volume<float>* parentVol,
                        int stackIdx, bool volumeIsEmpty, int iter, bool noSync);


    void computePostFilter(Volume<float>* aVol, int projIndex, int dim_x, int dim_y, float degree);
    void computePreFilter(int projIndex, int dim_x, int dim_y, float degree);

    void SaveSNR(string& aFile, string suffix = std::string());
    void LoadSNR(string& aFile, string suffix = std::string());

    void ResetProjectionsDevice();
    void ResetChildProj();
    void CopyProjectionToHost(float* buffer);
    void CopyChildProjectionToHost(float* buffer);
	void CopyDistanceImageToHost(float* buffer);//For Debugging...
    void CopyChildDistanceImageToHost(float * buffer);
	void CopyRealProjectionToHost(float* buffer);//For Debugging...
	void CopyProjectionToDevice(float* buffer);
	void CopyDistanceImageToDevice(float* buffer);//For Debugging...
	void CopyRealProjectionToDevice(float* buffer);//For Debugging...
	void MPIBroadcast(float** buffers, int bufferCount);

#ifdef REFINE_MODE
	void GetCroppedProjection(float *outImage, int2 roiMin, int2 roiMax);
    void GetCroppedProjection(float *outImage, float *inImage, int2 roiMin, int2 roiMax);
	void CopyProjectionToSubVolumeProjection();
	float2 GetDisplacement(bool MultiPeakDetection, float* CCValue = NULL);
    float2 GetDisplacementPC(bool MultiPeakDetection, float* CCValue = NULL);
	void rotVol(Cuda::CudaDeviceVariable& vol, float phi, float psi, float theta);
	void setRotVolData(float* data);
	float* GetCCMap();
	float* GetCCMapMulti();
#endif

	void ConvertVolumeFP16(float* slice, Cuda::CudaSurfaceObject3D& surf, int z);
	void ConvertVolume3DFP16(float* volume, Cuda::CudaSurfaceObject3D& surf);
	void MatrixVector3Mul(float4x4 M, float3* v);
    void MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut);
    void MatrixVector3Mul(float3x3& M, float2& val);
    void MatrixScalarMul(float4x4 &M, float v);
};

#endif // !RECONSTRUCTOR_H

