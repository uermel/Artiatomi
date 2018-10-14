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


#include "BasicKernel.h"


CudaSub::CudaSub(int aVolSize, CUstream aStream, CudaContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");
	
	add = new CudaKernel("add", cuMod);
	sub = new CudaKernel("sub", cuMod);
	subCplx = new CudaKernel("subCplx", cuMod);
	subCplx2 = new CudaKernel("subCplx2", cuMod);
}

void CudaSub::runAddKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(add->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaSub::Add(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	runAddKernel(d_idata, d_odata);
}

void CudaSub::runSubKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    cudaSafeCall(cuLaunchKernel(sub->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaSub::Sub(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val)
{
	runSubKernel(d_idata, d_odata, val);
}

void CudaSub::runSubCplxKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, float divVal)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();
	CUdeviceptr val_dptr = val.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val_dptr;
    arglist[4] = &divVal;

    cudaSafeCall(cuLaunchKernel(subCplx->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaSub::SubCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, float divVal)
{
	runSubCplxKernel(d_idata, d_odata, val, divVal);
}


void CudaSub::runSubCplxKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, CudaDeviceVariable& divVal)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();
	CUdeviceptr val_dptr = val.GetDevicePtr();
	CUdeviceptr divval_dptr = divVal.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val_dptr;
    arglist[4] = &divval_dptr;

    cudaSafeCall(cuLaunchKernel(subCplx2->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaSub::SubCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, CudaDeviceVariable& val, CudaDeviceVariable& divVal)
{
	runSubCplxKernel(d_idata, d_odata, val, divVal);
}







CudaMakeCplxWithSub::CudaMakeCplxWithSub(int aVolSize, CUstream aStream, CudaContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");
	
	makeReal = new CudaKernel("makeReal", cuMod);
	makeCplxWithSub = new CudaKernel("makeCplxWithSub", cuMod);
	makeCplxWithSubSqr = new CudaKernel("makeCplxWithSquareAndSub", cuMod);
}

void CudaMakeCplxWithSub::runRealKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(makeReal->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaMakeCplxWithSub::MakeReal(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	runRealKernel(d_idata, d_odata);
}

void CudaMakeCplxWithSub::runCplxWithSubKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    cudaSafeCall(cuLaunchKernel(makeCplxWithSub->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaMakeCplxWithSub::MakeCplxWithSub(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val)
{
	runCplxWithSubKernel(d_idata, d_odata, val);
}

void CudaMakeCplxWithSub::runCplxWithSqrSubKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    cudaSafeCall(cuLaunchKernel(makeCplxWithSubSqr->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaMakeCplxWithSub::MakeCplxWithSqrSub(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata, float val)
{
	runCplxWithSqrSubKernel(d_idata, d_odata, val);
}








CudaBinarize::CudaBinarize(int aVolSize, CUstream aStream, CudaContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");
	
	binarize = new CudaKernel("binarize", cuMod);
}

void CudaBinarize::runBinarizeKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(binarize->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaBinarize::Binarize(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	runBinarizeKernel(d_idata, d_odata);
}








CudaMul::CudaMul(int aVolSize, CUstream aStream, CudaContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");
	
	mulVol = new CudaKernel("mulVol", cuMod);
	mulVolCplx = new CudaKernel("mulVolCplx", cuMod);
	mul = new CudaKernel("mul", cuMod);
}

void CudaMul::runMulVolKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(mulVol->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaMul::MulVol(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	runMulVolKernel(d_idata, d_odata);
}

void CudaMul::runMulVolCplxKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(mulVol->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaMul::MulVolCplx(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	runMulVolCplxKernel(d_idata, d_odata);
}

void CudaMul::runMulKernel(float val, CudaDeviceVariable& d_odata)
{
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &val;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(mul->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaMul::Mul(float val, CudaDeviceVariable& d_odata)
{
	runMulKernel(val, d_odata);
}





CudaFFT::CudaFFT(int aVolSize, CUstream aStream, CudaContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");
	
	conv = new CudaKernel("conv", cuMod);
	correl = new CudaKernel("correl", cuMod);
	bandpass = new CudaKernel("bandpass", cuMod);
	bandpassFFTShift = new CudaKernel("bandpassFFTShift", cuMod);
	fftshiftReal = new CudaKernel("fftshiftReal", cuMod);
	fftshift = new CudaKernel("fftshift", cuMod);
	fftshift2 = new CudaKernel("fftshift2", cuMod);
	energynorm = new CudaKernel("energynorm", cuMod);
}

void CudaFFT::runConvKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(conv->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::Conv(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	runConvKernel(d_idata, d_odata);
}

void CudaFFT::runCorrelKernel(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	CUdeviceptr in_dptr = d_idata.GetDevicePtr();
	CUdeviceptr out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    cudaSafeCall(cuLaunchKernel(correl->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::Correl(CudaDeviceVariable& d_idata, CudaDeviceVariable& d_odata)
{
	runCorrelKernel(d_idata, d_odata);
}

void CudaFFT::runBandpassFFTShiftKernel(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	CUdeviceptr vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;
    arglist[2] = &rDown;
    arglist[3] = &rUp;
    arglist[4] = &smooth;

    cudaSafeCall(cuLaunchKernel(bandpassFFTShift->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::BandpassFFTShift(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	runBandpassFFTShiftKernel(d_vol, rDown, rUp, smooth);
}

void CudaFFT::runBandpassKernel(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	CUdeviceptr vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;
    arglist[2] = &rDown;
    arglist[3] = &rUp;
    arglist[4] = &smooth;

    cudaSafeCall(cuLaunchKernel(bandpass->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::Bandpass(CudaDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	runBandpassKernel(d_vol, rDown, rUp, smooth);
}

void CudaFFT::runFFTShiftKernel(CudaDeviceVariable& d_vol)
{
	CUdeviceptr vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[2];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;

    cudaSafeCall(cuLaunchKernel(fftshift->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::FFTShift(CudaDeviceVariable& d_vol)
{
	runFFTShiftKernel(d_vol);
}

void CudaFFT::runFFTShiftKernel2(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut)
{
	CUdeviceptr voli_dptr = d_volIn.GetDevicePtr();
	CUdeviceptr volo_dptr = d_volOut.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &voli_dptr;
    arglist[2] = &volo_dptr;

    cudaSafeCall(cuLaunchKernel(fftshift2->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::FFTShift2(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut)
{
	runFFTShiftKernel2(d_volIn, d_volOut);
}

void CudaFFT::runFFTShiftRealKernel(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut)
{
	CUdeviceptr voli_dptr = d_volIn.GetDevicePtr();
	CUdeviceptr volo_dptr = d_volOut.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &voli_dptr;
    arglist[2] = &volo_dptr;

    cudaSafeCall(cuLaunchKernel(fftshiftReal->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::FFTShiftReal(CudaDeviceVariable& d_volIn, CudaDeviceVariable& d_volOut)
{
	runFFTShiftRealKernel(d_volIn, d_volOut);
}

void CudaFFT::runEnergyNormKernel(CudaDeviceVariable& d_particle, CudaDeviceVariable& d_partSqr, CudaDeviceVariable& d_cccMap, CudaDeviceVariable& energyRef, CudaDeviceVariable& nVox)
{
	CUdeviceptr particle_dptr = d_particle.GetDevicePtr();
	CUdeviceptr partSqr_dptr = d_partSqr.GetDevicePtr();
	CUdeviceptr cccMap_dptr = d_cccMap.GetDevicePtr();
	CUdeviceptr energyRef_dptr = energyRef.GetDevicePtr();
	CUdeviceptr nVox_dptr = nVox.GetDevicePtr();

    void** arglist = (void**)new void*[6];

    arglist[0] = &volSize;
    arglist[1] = &particle_dptr;
    arglist[2] = &partSqr_dptr;
    arglist[3] = &cccMap_dptr;
    arglist[4] = &energyRef_dptr;
    arglist[5] = &nVox_dptr;

    cudaSafeCall(cuLaunchKernel(energynorm->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaFFT::EnergyNorm(CudaDeviceVariable& d_particle, CudaDeviceVariable& d_partSqr, CudaDeviceVariable& d_cccMap, CudaDeviceVariable& energyRef, CudaDeviceVariable& nVox)
{
	runEnergyNormKernel(d_particle, d_partSqr, d_cccMap, energyRef, nVox);
}



CudaMax::CudaMax(CUstream aStream, CudaContext* context)
	: stream(aStream), ctx(context), blockSize(1, 1, 1), 
	  gridSize(1, 1, 1)
{
	CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");
	
	max = new CudaKernel("findmax", cuMod);
}

void CudaMax::runMaxKernel(CudaDeviceVariable& maxVals, CudaDeviceVariable& index, CudaDeviceVariable& val, float rphi, float rpsi, float rthe)
{
	CUdeviceptr maxVals_dptr = maxVals.GetDevicePtr();
	CUdeviceptr index_dptr = index.GetDevicePtr();
	CUdeviceptr val_dptr = val.GetDevicePtr();

    void** arglist = (void**)new void*[6];

    arglist[0] = &maxVals_dptr;
    arglist[1] = &index_dptr;
    arglist[2] = &val_dptr;
    arglist[3] = &rphi;
    arglist[4] = &rpsi;
    arglist[5] = &rthe;

    cudaSafeCall(cuLaunchKernel(max->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}
void CudaMax::Max(CudaDeviceVariable& maxVals, CudaDeviceVariable& index, CudaDeviceVariable& val, float rphi, float rpsi, float rthe)
{
	runMaxKernel(maxVals, index, val, rphi, rpsi, rthe);
}


CudaWedgeNorm::CudaWedgeNorm(int aVolSize, CUstream aStream, CudaContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1),
	gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	CUmodule cuMod = ctx->LoadModule("basicKernels.ptx");

	wedge = new CudaKernel("wedgeNorm", cuMod);
}

void CudaWedgeNorm::runWedgeNormKernel(CudaDeviceVariable& d_data, CudaDeviceVariable& d_partdata, CudaDeviceVariable& d_maxVal, int newMethod)
{
	CUdeviceptr in_dptr = d_data.GetDevicePtr();
	CUdeviceptr part_dptr = d_partdata.GetDevicePtr();
	CUdeviceptr out_dptr = d_maxVal.GetDevicePtr();

	void** arglist = (void**)new void*[5];

	arglist[0] = &volSize;
	arglist[1] = &in_dptr;
	arglist[2] = &d_partdata;
	arglist[3] = &out_dptr;
	arglist[4] = &newMethod;

	cudaSafeCall(cuLaunchKernel(wedge->GetCUfunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist, NULL));

	delete[] arglist;
}
void CudaWedgeNorm::WedgeNorm(CudaDeviceVariable& d_data, CudaDeviceVariable& d_partdata, CudaDeviceVariable& d_maxVal, int newMethod)
{
	runWedgeNormKernel(d_data, d_partdata, d_maxVal, newMethod);
}