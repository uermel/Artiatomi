//
// Created by uermel on 11/15/21.
//

#ifndef ARTIATOMI_CTFSLICEDKERNEL_H
#define ARTIATOMI_CTFSLICEDKERNEL_H


#include "EmSartDefault.h"
#include <CudaArrays.h>
#include <CudaContext.h>
#include <CudaTextures.h>
#include <CudaSurfaces.h>
#include <CudaKernel.h>
#include <CudaDeviceProperties.h>
#include "Projection.h"
#include "Volume.h"

class CTFSlicedKernel : public Cuda::CudaKernel
{
private:
    Cuda::CudaDeviceVariable d_offsets;
    float* h_offsets;

public:
    explicit CTFSlicedKernel(CUmodule aModule);
    CTFSlicedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);


    void AllocOffsets(int size);

    float operator()(Cuda::CudaDeviceVariable& ctf,
                     int ctf_x,
                     int ctf_y,
                     int sliceNumber,
                     float defocusMin,
                     float defocusMax,
                     vector<float>& offsets,
                     float angle,
                     bool applyForFP,
                     bool phaseFlipOnly,
                     float WienerFilterNoiseLevel,
                     size_t stride,
                     float4 betaFac);
};

void SetConstantValues(CTFSlicedKernel& kernel, Projection& proj, int index, float cs, float voltage);



////// BEGIN PostFilterSum //////
class PostFilterSumKernel : public Cuda::CudaKernel
{
private:
    Cuda::CudaDeviceVariable d_offsets;
    float* h_offsets;
    int slice_number;

public:
    explicit PostFilterSumKernel(CUmodule aModule);
    PostFilterSumKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void AllocOffsets(int size);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& filter,
                     Cuda::CudaDeviceVariable& result,
                     uint2 fftDim,
                     ctfConstants constants,
                     ctfImageConstants imageConstants,
                     vector<float>* offsets,
                     int2 minmaxSlice);
};
////// END PostFilterSum //////


////// BEGIN PostFilterSum //////
class PostFilterSumCTFfreeKernel : public Cuda::CudaKernel
{
private:
    Cuda::CudaDeviceVariable d_offsets;
    float* h_offsets;
    int slice_number;

public:
    explicit PostFilterSumCTFfreeKernel(CUmodule aModule);
    PostFilterSumCTFfreeKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void AllocOffsets(int size);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& filter,
                     Cuda::CudaDeviceVariable& result,
                     Cuda::CudaDeviceVariable& result2,
                     uint2 fftDim,
                     ctfConstants constants,
                     ctfImageConstants imageConstants,
                     vector<float>* offsets,
                     int2 minmaxSlice);
};
////// END PostFilterSum //////


////// BEGIN PreFilterSpreadAdHoc //////
class PreFilterSpreadAdHocKernel : public Cuda::CudaKernel
{
private:
    Cuda::CudaDeviceVariable d_offsets;
    float* h_offsets;

public:
    explicit PreFilterSpreadAdHocKernel(CUmodule aModule);
    PreFilterSpreadAdHocKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void AllocOffsets(int size);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& filter,
                     Cuda::CudaDeviceVariable& result,
                     uint2 fftDim,
                     ctfConstants constants,
                     ctfImageConstants imageConstants,
                     vector<float>* offsets,
                     int2 minmaxSlice);
};
////// END PreFilterSpreadAdHoc //////


////// BEGIN PreFilterSpreadSNR //////
class PreFilterSpreadSNRKernel : public Cuda::CudaKernel
{
private:
    Cuda::CudaDeviceVariable d_offsets;
    float* h_offsets;

public:
    explicit PreFilterSpreadSNRKernel(CUmodule aModule);
    PreFilterSpreadSNRKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void AllocOffsets(int size);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& filter,
                     Cuda::CudaDeviceVariable& result,
                     Cuda::CudaTextureObject1D& texSNR,
                     Cuda::CudaTextureObject1D& texNP,
                     uint2 fftDim,
                     ctfConstants constants,
                     ctfImageConstants imageConstants,
                     vector<float>* offsets,
                     int2 minmaxSlice,
                     float deconvStrength);
};
////// END PreFilterSpreadSNR //////

////// BEGIN PreFilterSpreadSNRrelion //////
class PreFilterSpreadSNRrelionKernel : public Cuda::CudaKernel
{
private:
    Cuda::CudaDeviceVariable d_offsets;
    float* h_offsets;

public:
    explicit PreFilterSpreadSNRrelionKernel(CUmodule aModule);
    PreFilterSpreadSNRrelionKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    void AllocOffsets(int size);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& filter,
                     Cuda::CudaDeviceVariable& result,
                     Cuda::CudaTextureObject1D& texSNR,
                     Cuda::CudaTextureObject1D& texNP,
                     uint2 fftDim,
                     ctfConstants constants,
                     ctfImageConstants imageConstants,
                     vector<float>* offsets,
                     int2 minmaxSlice,
                     float deconvStrength);
};
////// END PreFilterSpreadSNRrelion //////


class PostFilterKernel : public Cuda::CudaKernel
{
public:
    explicit PostFilterKernel(CUmodule aModule);
    PostFilterKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim);

    float operator()(Cuda::CudaDeviceVariable& image,
                     Cuda::CudaDeviceVariable& filter,
                     Cuda::CudaDeviceVariable& result,
                     uint2 fftDim,
                     float normFactor);
};




#endif //ARTIATOMI_CTFSLICEDKERNEL_H
