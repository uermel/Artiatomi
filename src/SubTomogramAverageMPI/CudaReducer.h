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


#ifndef CUDAREDUCER_H
#define CUDAREDUCER_H

#include "basics/default.h"
#include <cuda.h>
#include <CudaVariables.h>
#include <CudaKernel.h>
#include <CudaContext.h>

using namespace Cuda;

class CudaReducer
{
private:
	CudaKernel* sum512;
	CudaKernel* sum256;
	CudaKernel* sum128;
	CudaKernel* sum64;
	CudaKernel* sum32;
	CudaKernel* sum16;
	CudaKernel* sum8;
	CudaKernel* sum4;
	CudaKernel* sum2;
	CudaKernel* sum1;

	CudaKernel* sumCplx512;
	CudaKernel* sumCplx256;
	CudaKernel* sumCplx128;
	CudaKernel* sumCplx64;
	CudaKernel* sumCplx32;
	CudaKernel* sumCplx16;
	CudaKernel* sumCplx8;
	CudaKernel* sumCplx4;
	CudaKernel* sumCplx2;
	CudaKernel* sumCplx1;

	CudaKernel* maskedSumCplx512;
	CudaKernel* maskedSumCplx256;
	CudaKernel* maskedSumCplx128;
	CudaKernel* maskedSumCplx64;
	CudaKernel* maskedSumCplx32;
	CudaKernel* maskedSumCplx16;
	CudaKernel* maskedSumCplx8;
	CudaKernel* maskedSumCplx4;
	CudaKernel* maskedSumCplx2;
	CudaKernel* maskedSumCplx1;

	CudaKernel* sumSqrCplx512;
	CudaKernel* sumSqrCplx256;
	CudaKernel* sumSqrCplx128;
	CudaKernel* sumSqrCplx64;
	CudaKernel* sumSqrCplx32;
	CudaKernel* sumSqrCplx16;
	CudaKernel* sumSqrCplx8;
	CudaKernel* sumSqrCplx4;
	CudaKernel* sumSqrCplx2;
	CudaKernel* sumSqrCplx1;

	CudaKernel* maxIndexMasked512;
	CudaKernel* maxIndexMasked256;
	CudaKernel* maxIndexMasked128;
	CudaKernel* maxIndexMasked64;
	CudaKernel* maxIndexMasked32;
	CudaKernel* maxIndexMasked16;
	CudaKernel* maxIndexMasked8;
	CudaKernel* maxIndexMasked4;
	CudaKernel* maxIndexMasked2;
	CudaKernel* maxIndexMasked1;

	CudaKernel* maxIndexMaskedCplx512;
	CudaKernel* maxIndexMaskedCplx256;
	CudaKernel* maxIndexMaskedCplx128;
	CudaKernel* maxIndexMaskedCplx64;
	CudaKernel* maxIndexMaskedCplx32;
	CudaKernel* maxIndexMaskedCplx16;
	CudaKernel* maxIndexMaskedCplx8;
	CudaKernel* maxIndexMaskedCplx4;
	CudaKernel* maxIndexMaskedCplx2;
	CudaKernel* maxIndexMaskedCplx1;

    CudaKernel* maxIndex512;
    CudaKernel* maxIndex256;
    CudaKernel* maxIndex128;
    CudaKernel* maxIndex64;
    CudaKernel* maxIndex32;
    CudaKernel* maxIndex16;
    CudaKernel* maxIndex8;
    CudaKernel* maxIndex4;
    CudaKernel* maxIndex2;
    CudaKernel* maxIndex1;

    CudaKernel* maxIndexCplx512;
    CudaKernel* maxIndexCplx256;
    CudaKernel* maxIndexCplx128;
    CudaKernel* maxIndexCplx64;
    CudaKernel* maxIndexCplx32;
    CudaKernel* maxIndexCplx16;
    CudaKernel* maxIndexCplx8;
    CudaKernel* maxIndexCplx4;
    CudaKernel* maxIndexCplx2;
    CudaKernel* maxIndexCplx1;

	CudaContext* ctx;
	int voxelCount;

	static const int maxBlocks = 64;
	static const int maxThreads = 256;

	CUstream stream;

	void getNumBlocksAndThreads(int n, int &blocks, int &threads);
	uint nextPow2(uint x);

	void runSumKernel(int size, int blocks, int threads,
                      CudaDeviceVariable& d_idata,
                      CudaDeviceVariable& d_odata);
    void runMaxIndexKernel(int size, int blocks, int threads,
                           CudaDeviceVariable& d_idata,
                           CudaDeviceVariable& d_odata,
                           CudaDeviceVariable& d_index,
                           bool readIndex);
    void runMaxIndexMaskedKernel(int size, int blocks, int threads,
                           CudaDeviceVariable& d_idata,
                           CudaDeviceVariable& d_odata,
                           CudaDeviceVariable& d_maskdata,
                           CudaDeviceVariable& d_index,
                           bool readIndex);
	void runSumCplxKernel(int size, int blocks, int threads,
                          CudaDeviceVariable& d_idata,
                          CudaDeviceVariable& d_odata);
	void runMaskedSumCplxKernel(int size, int blocks, int threads,
                                CudaDeviceVariable& d_idata,
                                CudaDeviceVariable& d_mask,
                                CudaDeviceVariable& d_odata);
	void runSumSqrCplxKernel(int size, int blocks, int threads,
                             CudaDeviceVariable& d_idata,
                             CudaDeviceVariable& d_odata);
    void runMaxIndexCplxKernel(int size, int blocks, int threads,
                               CudaDeviceVariable& d_idata,
                               CudaDeviceVariable& d_odata,
                               CudaDeviceVariable& d_index,
                               bool readIndex);
    void runMaxIndexMaskedCplxKernel(int size, int blocks, int threads,
                               CudaDeviceVariable& d_idata,
                               CudaDeviceVariable& d_odata,
                               CudaDeviceVariable& d_maskdata,
                               CudaDeviceVariable& d_index,
                               bool readIndex);

public:

	CudaReducer(int aVoxelCount, CUstream aStream, CudaContext* context);

	int GetOutBufferSize();

	void Sum(CudaDeviceVariable& d_idata,
             CudaDeviceVariable& d_odata);
    void MaxIndex(CudaDeviceVariable& d_idata,
                  CudaDeviceVariable& d_odata,
                  CudaDeviceVariable& d_index);
    void MaxIndexMasked(CudaDeviceVariable& d_idata,
                        CudaDeviceVariable& d_odata,
                        CudaDeviceVariable& d_maskdata,
                        CudaDeviceVariable& d_index);
	void SumCplx(CudaDeviceVariable& d_idata,
                 CudaDeviceVariable& d_odata);
	void MaskedSumCplx(CudaDeviceVariable& d_idata,
                       CudaDeviceVariable& d_mask,
                       CudaDeviceVariable& d_odata);
	void SumSqrCplx(CudaDeviceVariable& d_idata,
                    CudaDeviceVariable& d_odata);
    void MaxIndexCplx(CudaDeviceVariable& d_idata,
                      CudaDeviceVariable& d_odata,
                      CudaDeviceVariable& d_index);
    void MaxIndexMaskedCplx(CudaDeviceVariable& d_idata,
                            CudaDeviceVariable& d_odata,
                            CudaDeviceVariable& d_maskdata,
                             CudaDeviceVariable& d_index);
};

#endif //CUDAREDUCER_H
