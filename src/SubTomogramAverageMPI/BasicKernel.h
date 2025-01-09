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


#ifndef BASICKERNEL_H
#define BASICKERNEL_H

#include "basics/default.h"
#include <cuda.h>
#include <CudaVariables.h>
#include <CudaKernel.h>
#include <CudaContext.h>
#include <CudaTextures.h>

using namespace Cuda;

class CudaMask
{
private:
    CudaKernel* sphericalMaskCosineCplx;
    CudaKernel* spheroidMask;

    CudaContext* ctx;
    int volSize;
    dim3 blockSize;
    dim3 gridSize;

    float oldphi = 0;
    float oldpsi = 0;
    float oldtheta = 0;

    CUstream stream;

    void runSphericalMaskCosineCplx(CudaDeviceVariable& d_odata, float radius, float taper, float3 center);
    void runSpheroidMaskKernel(CudaDeviceVariable& d_odata, float3 radius, float3 center, float rotMat[9]);

public:
    CudaMask(int aVolSize, CUstream aStream, CudaContext* context);

    void SphericalMaskCosineCplx(CudaDeviceVariable& d_odata, float radius, float taper, float3 center);
    void SpheroidMask(CudaDeviceVariable& d_odata, float3 radius, float3 center, float phi, float psi, float the);

    void SetOldAngles(float aPhi, float aPsi, float aTheta);
    void computeRotMat(float phi, float psi, float theta, float rotMat[9]);
    void multiplyRotMatrix(const float m1[9], const float m2[9], float out[9]);
};

class CudaSub
{
private:
    CudaKernel* set;
	CudaKernel* sub;
	CudaKernel* subCplx;
	CudaKernel* subCplx2;
	CudaKernel* add;
    CudaKernel* regError;

	CudaContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	CUstream stream;

    void runSetKernel(CudaDeviceVariable& d_odata, float val);
	void runAddKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void runSubKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val);
	void runSubCplxKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, float divVal);
	void runSubCplxKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, CudaDeviceVariable& divVal);
    void runRegError(CudaDeviceVariable& d_im1,
                     CudaDeviceVariable& d_im2,
                     CudaDeviceVariable& d_im1f,
                     CudaDeviceVariable& d_im2f,
                     CudaDeviceVariable& d_outVol);

public:
	CudaSub(int aVolSize, CUstream aStream, CudaContext* context);

    void Set(CudaDeviceVariable& d_odata, float val);
	void Add(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void Sub(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val);
	void SubCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, float divVal);
	void SubCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, CudaDeviceVariable& divVal);

    void RegError(CudaDeviceVariable& d_im1,
                  CudaDeviceVariable& d_im2,
                  CudaDeviceVariable& d_im1f,
                  CudaDeviceVariable& d_im2f,
                  CudaDeviceVariable& d_outVol);
};

class CudaMakeCplxWithSub
{
private:
	CudaKernel* makeReal;
	CudaKernel* makeCplxWithSub;
	CudaKernel* makeCplxWithSubSqr;

	CudaContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	CUstream stream;

	void runRealKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void runCplxWithSubKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val);
	void runCplxWithSqrSubKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val);

public:
	CudaMakeCplxWithSub(int aVolSize, CUstream aStream, CudaContext* context);

	void MakeReal(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void MakeCplxWithSub(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val);
	void MakeCplxWithSqrSub(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val);
};

class CudaBinarize
{
private:
	CudaKernel* binarize;

	CudaContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	CUstream stream;

	void runBinarizeKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float thresh);

public:
	CudaBinarize(int aVolSize, CUstream aStream, CudaContext* context);

	void Binarize(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float thresh);
};

class CudaWedgeNorm
{
private:
	CudaKernel* wedge;
    CudaKernel* wedgeWiener;

	CudaContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	CUstream stream;

	void runWedgeNormKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_partdata, CudaDeviceVariable& d_odata, int newMethod);

    void runWedgeNormWienerKernel(CudaDeviceVariable& d_wedgeSum,
                                  CudaDeviceVariable& d_partSum,
                                  CudaDeviceVariable& d_tausqr,
                                  float T);

public:
	CudaWedgeNorm(int aVolSize, CUstream aStream, CudaContext* context);

	void WedgeNorm(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_partdata, CudaDeviceVariable& d_odata, int newMethod);

    void WedgeNormWiener(CudaDeviceVariable& d_wedgeSum,
                         CudaDeviceVariable& d_partSum,
                         CudaDeviceVariable& d_tausqr,
                         float T);

};



class CudaMul
{
private:
	CudaKernel* mulVol;
	CudaKernel* mulVolCplx;
	CudaKernel* mul;
    CudaKernel* mulMaskMeanFreeCplx;

	CudaContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	CUstream stream;

	void runMulVolKernel(CudaDeviceVariable& d_idata,
                         CudaDeviceVariable& d_odata);
	void runMulVolCplxKernel(CudaDeviceVariable& d_idata,
                             CudaDeviceVariable& d_odata);
	void runMulKernel(float val,
                      CudaDeviceVariable& d_odata);
    void runMulMaskMeanFreeCplxKernel(CudaDeviceVariable& d_im1_io,
                                      CudaDeviceVariable& d_im1sqr_o,
                                      CudaDeviceVariable& d_mask1,
                                      CudaDeviceVariable& d_sumVol,
                                      CudaDeviceVariable& d_sumMask);

public:
	CudaMul(int aVolSize, CUstream aStream, CudaContext* context);

	void MulVol(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void MulVolCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void Mul(float val, CudaDeviceVariable& d_odata);

    void MulMaskMeanFreeCplx(CudaDeviceVariable& d_im1_io,
                             CudaDeviceVariable& d_im1sqr_o,
                             CudaDeviceVariable& d_mask1,
                             CudaDeviceVariable& d_sumVol,
                             CudaDeviceVariable& d_sumMask);
};



class CudaFFT
{
private:
	CudaKernel* conv;
	CudaKernel* correl;
	CudaKernel* phaseCorrel;
	CudaKernel* bandpass;
	CudaKernel* bandpassFFTShift;
	CudaKernel* fftshift;
	CudaKernel* fftshiftReal;
	CudaKernel* fftshift2;
	CudaKernel* splitDataset;
	CudaKernel* energynorm;
    CudaKernel* energynormPadfield;
    CudaKernel* energynormMaskFirst;
    CudaKernel* particleWiener;
    CudaKernel* butter;
    CudaKernel* butterfilt;

	CudaContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	CUstream stream;

	void runConvKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void runCorrelKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void runPhaseCorrelKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void runBandpassKernel(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void runBandpassFFTShiftKernel(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void runFFTShiftKernel(CudaDeviceVariable& d_vol);
	void runFFTShiftRealKernel(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut);
	void runFFTShiftKernel2(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut);
	void runSplitDataset(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOutA, CudaDeviceVariable& d_volOutB);
	void runEnergyNormKernel(CudaDeviceVariable& d_particle, CudaDeviceVariable& d_partSqr, CudaDeviceVariable& d_cccMap, CudaDeviceVariable& energyRef, CudaDeviceVariable& nVox);

    void runEnergyNormPadfieldKernel(CudaDeviceVariable& d_NCCNum,
                                     CudaDeviceVariable& d_NCCDen2_f2sqr,
                                     CudaDeviceVariable& d_NCCDen2_f2,
                                     CudaDeviceVariable& d_NCCDen1,
                                     CudaDeviceVariable& d_maskNorm);

    void runEnergyNormMaskFirstKernel(CudaDeviceVariable& d_NCCNum,
                                      CudaDeviceVariable& d_NCCDen1,
                                      CudaDeviceVariable& d_NCCDen2);

    void runParticleWienerKernel(CudaDeviceVariable& d_particle,
                                 CudaDeviceVariable& d_wedge_ctfsqr,
                                 CudaDeviceVariable& d_wedge_coverage,
                                 float wienerConst);

    void runButterFilter(CudaDeviceVariable& d_volIn,
                         CudaDeviceVariable& d_volOut,
                         int order,
                         float cutoff);

    void runButterTest(CudaDeviceVariable& d_volOut,
                    int order,
                    float cutoff);

public:
	CudaFFT(int aVolSize, CUstream aStream, CudaContext* context);

	void Conv(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void Correl(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void PhaseCorrel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata);
	void Bandpass(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void BandpassFFTShift(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void FFTShift(CudaDeviceVariable& d_vol);
	void FFTShiftReal(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut);
	void FFTShift2(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut);
	void SplitDataset(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOutA, CudaDeviceVariable& d_volOutB);
	void EnergyNorm(CudaDeviceVariable& d_particle, CudaDeviceVariable& d_partSqr, CudaDeviceVariable& d_cccMap, CudaDeviceVariable& energyRef, CudaDeviceVariable& nVox);

    void EnergyNormPadfield(CudaDeviceVariable& d_NCCNum,
                            CudaDeviceVariable& d_NCCDen2_f2sqr,
                            CudaDeviceVariable& d_NCCDen2_f2,
                            CudaDeviceVariable& d_NCCDen1,
                            CudaDeviceVariable& d_maskNorm);

    void EnergyNormMaskFirst(CudaDeviceVariable& d_NCCNum,
                             CudaDeviceVariable& d_NCCDen1,
                             CudaDeviceVariable& d_NCCDen2);

    void ParticleWiener(CudaDeviceVariable& d_particle,
                        CudaDeviceVariable& d_wedge_ctfsqr,
                        CudaDeviceVariable& d_wedge_coverage,
                        float wienerConst);

    void ButterFilter(CudaDeviceVariable& d_volIn,
                      CudaDeviceVariable& d_volOut,
                      int order,
                      float cutoff);

    void ButterTest(CudaDeviceVariable& d_volOut,
                    int order,
                    float cutoff);
};

class CudaMax
{
private:
	CudaKernel* max;
	CudaKernel* maxWithCertainty;

	CudaContext* ctx;
	dim3 blockSize;
	dim3 gridSize;

	CUstream stream;

	void runMaxKernel(CudaDeviceVariable& maxVals, CudaDeviceVariable& index, CudaDeviceVariable& val, float rphi, float rpsi, float rthe);
	void runMaxWithCertainty(CudaDeviceVariable& maxVals, CudaDeviceVariable& index, CudaDeviceVariable& val, CudaDeviceVariable& indexA, CudaDeviceVariable& indexB, float rphi, float rpsi, float rthe, int volSize, int limit);


public:
	CudaMax(CUstream aStream, CudaContext* context);

	void Max(CudaDeviceVariable& maxVals, CudaDeviceVariable& index, CudaDeviceVariable& val, float rphi, float rpsi, float rthe);
	void MaxWithCertainty(CudaDeviceVariable& maxVals, CudaDeviceVariable& index, CudaDeviceVariable& val, CudaDeviceVariable& indexA, CudaDeviceVariable& indexB, float rphi, float rpsi, float rthe, int volSize, int limit);
	//findmaxWithCertainty(float* maxVals, float* index, float* val, float* indexA, float* indexB, float rphi, float rpsi, float rthe, int volSize, int limit)
};

class CudaRadial
{
private:
    CudaKernel* sphericalSum;
    CudaKernel* line2sphere;
    CudaKernel* div;

    CudaContext* ctx;
    int volSize;
    dim3 blockSize;
    dim3 gridSize;

    CUstream stream;

    CudaArray1D lineTex;

    void runSphericalSumKernel(CudaDeviceVariable& d_wedgeSum, CudaDeviceVariable& d_ampSum, CudaDeviceVariable& d_nShell);
    void runLine2SphereKernel(CudaDeviceVariable& d_outVol);
    void runDivKernel(CudaDeviceVariable& d_ampSum, CudaDeviceVariable& d_nShell, CudaDeviceVariable& d_SNR);

public:
    CudaRadial(int aVolSize, CUstream aStream, CudaContext* context);

    void SetTexture(CudaDeviceVariable& d_idata);

    void SphericalSumKernel(CudaDeviceVariable& d_wedgeSum, CudaDeviceVariable& d_ampSum, CudaDeviceVariable& d_nShell);
    void Line2Sphere(CudaDeviceVariable& d_outVol);
    void Div(CudaDeviceVariable& d_ampSum, CudaDeviceVariable& d_nShell, CudaDeviceVariable& d_SNR);
};

class CudaCmp
{
private:
    CudaKernel* minIdx;
    CudaKernel* selIdx;

    CudaContext* ctx;
    int volSize;
    dim3 blockSize;
    dim3 gridSize;

    CUstream stream;

    void runMinIdxKernel(CudaDeviceVariable& d_im,
                         CudaDeviceVariable& d_min,
                         CudaDeviceVariable& d_minIdx1,
                         CudaDeviceVariable& d_minIdx2,
                         int idx1,
                         int idx2);

    void runSelIdxKernel(CudaDeviceVariable& d_im,
                         CudaDeviceVariable& d_minIdx,
                         CudaDeviceVariable& d_outVol,
                         int idx);

public:
    CudaCmp(int aVolSize, CUstream aStream, CudaContext* context);

    void MinIdx(CudaDeviceVariable& d_im,
                CudaDeviceVariable& d_min,
                CudaDeviceVariable& d_minIdx1,
                CudaDeviceVariable& d_minIdx2,
                int idx1,
                int idx2);

    void SelIdx(CudaDeviceVariable& d_im,
                CudaDeviceVariable& d_minIdx,
                CudaDeviceVariable& d_outVol,
                int idx);
};









#endif //BASICKERNEL_H
