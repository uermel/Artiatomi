//
// Created by uermel on 2/14/22.
//

#ifndef ARTIATOMI_ADAPTIVEREGULARIZER_H
#define ARTIATOMI_ADAPTIVEREGULARIZER_H

#include "basics/default.h"
#include <cuda.h>
#include <cufft.h>
#include "io/EMFile.h"
#include <CudaVariables.h>
#include <CudaKernel.h>
#include <CudaArrays.h>
#include <CudaTextures.h>
#include <CudaContext.h>
#include <CudaException.h>

#include "BasicKernel.h"
#include "CudaReducer.h"
#include "CudaRot.h"
#include <map>

class AdaptiveRegularizer
{
private:
    size_t sizeVol, sizeTot;

    float* sum_h;
    int*   index;
    float2* sumCplx;

    CUstream stream;
    CudaContext* ctx;

    CudaReducer reduce;
    CudaSub sub;
    CudaMakeCplxWithSub makecplx;
    CudaMul mul;
    CudaFFT fft;
    CudaMax max;
    CudaCmp cmp;
    CudaMask mask;

    // Images
    CudaDeviceVariable d_real_im1;
    CudaDeviceVariable d_real_im2;

    CudaDeviceVariable d_cplx_im1;
    CudaDeviceVariable d_cplx_im2;
    CudaDeviceVariable d_cplx_im1filt;
    CudaDeviceVariable d_cplx_im2filt;

    CudaDeviceVariable d_cplx_error_full;
    CudaDeviceVariable d_cplx_error_tmp;


    // Results
    CudaDeviceVariable d_current_min;
    CudaDeviceVariable d_min_idx;
    CudaDeviceVariable d_min_window;
    CudaDeviceVariable d_cplx_res;

    // Temp storage
    CudaDeviceVariable d_sum;

    // Filter
    CudaDeviceVariable d_cplx_window;
    CudaDeviceVariable d_real_butter;

    cufftHandle ffthandle;

public:
    AdaptiveRegularizer(size_t _sizeVol,
               CUstream _stream,
               CudaContext* _ctx);

    ~AdaptiveRegularizer();

    void findRegParams(float* aIm1,
                       float* aIm2,
                       int aOrder,
                       float aAWF,
                       int aSteps,
                       int* aRegParams);

    void regularize(float* aIm,
                    int aOrder,
                    int aSteps,
                    int* aRegParams,
                    float* aRecon);


};


#endif //ARTIATOMI_ADAPTIVEREGULARIZER_H
