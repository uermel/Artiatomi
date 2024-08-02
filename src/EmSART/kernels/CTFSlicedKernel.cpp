//
// Created by uermel on 11/15/21.
//

#include "CTFSlicedKernel.h"

using namespace Cuda;

CTFSlicedKernel::CTFSlicedKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("ctfSliced", aModule, aGridDim, aBlockDim, 0)
{

}

CTFSlicedKernel::CTFSlicedKernel(CUmodule aModule)
        : CudaKernel("ctfSliced", aModule, make_dim3(1, 1, 1), make_dim3(4, 4, 4), 0)
{

}

void CTFSlicedKernel::AllocOffsets(int maxSliceNumber) {
    d_offsets.Alloc(maxSliceNumber * sizeof(float));
    h_offsets = new float[maxSliceNumber];
}


float CTFSlicedKernel::operator()(CudaDeviceVariable& ctf,
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
                                  float4 betaFac)
{
    CUdeviceptr ctf_dptr = ctf.GetDevicePtr();
    float _defocusMin = defocusMin * 0.000000001f;
    float _defocusMax = defocusMax * 0.000000001f;
    float _angle = angle / 180.0f * (float)M_PI;

    // Offsets in m to Device
    for(int i=0; i < sliceNumber; i++){
        printf("%f\n", offsets[i]);
        h_offsets[i] = offsets[i] * powf(10, -9);
    }

    d_offsets.CopyHostToDevice(h_offsets, sliceNumber * sizeof(float));
    CUdeviceptr offsets_dptr = d_offsets.GetDevicePtr();

    printf("\nmGridDim: %u %u %u\n", mGridDim.x, mGridDim.y, mGridDim.z);
    printf("\nmBlockDim: %u %u %u\n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
    printf("sliceNumber: %i", sliceNumber);

    void** arglist = (void**)new void*[12];

    arglist[0] = &ctf_dptr;
    arglist[1] = &ctf_x;
    arglist[2] = &ctf_y;
    arglist[3] = &sliceNumber;
    arglist[4] = &_defocusMin;
    arglist[5] = &_defocusMax;
    arglist[6] = &offsets_dptr;
    arglist[7] = &_angle;
    arglist[8] = &applyForFP;
    arglist[9] = &phaseFlipOnly;
    arglist[10] = &WienerFilterNoiseLevel;
    arglist[11] = &betaFac;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}

void SetConstantValues(CTFSlicedKernel& kernel, Projection& proj, int index, float cs, float voltage)
{
    //Set constant values
    kernel.SetConstantValue("c_cs", &cs);
    kernel.SetConstantValue("c_voltage", &voltage);
    float _openingAngle = 0.01f;
    kernel.SetConstantValue("c_openingAngle", &_openingAngle);
    float _ampContrast = 0.00f;
    kernel.SetConstantValue("c_ampContrast", &_ampContrast);
    float _phaseContrast = sqrtf(1 - _ampContrast * _ampContrast);
    kernel.SetConstantValue("c_phaseContrast", &_phaseContrast);
    float _pixelsize = proj.GetPixelSize();// * 100.0f;
    //_pixelsize = round(_pixelsize) / 100.0f;

    //TODO:revert
    _pixelsize = _pixelsize * powf(10, -9);
    //_pixelsize = 0.5f * powf(10, -9);
    kernel.SetConstantValue("c_pixelsize", &_pixelsize);
    float _pixelcount = (float)proj.GetMaxDimension();
    kernel.SetConstantValue("c_pixelcount", &_pixelcount);
    float _maxFreq = 1.0f / (_pixelsize * 2.0f);
    kernel.SetConstantValue("c_maxFreq", &_maxFreq);
    float _freqStepSize = _maxFreq / (_pixelcount / 2.0f);
    //printf("_freqStepSize: %f; _pixelsize: %f\n", _freqStepSize, _pixelsize);
    kernel.SetConstantValue("c_freqStepSize", &_freqStepSize);

    float _applyScatteringProfile = 0;
    kernel.SetConstantValue("c_applyScatteringProfile", &_applyScatteringProfile);
    float _applyEnvelopeFunction = 0;
    kernel.SetConstantValue("c_applyEnvelopeFunction", &_applyEnvelopeFunction);
}


PostFilterSumKernel::PostFilterSumKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("postFilterSum", aModule, aGridDim, aBlockDim, 0)
{

}

PostFilterSumKernel::PostFilterSumKernel(CUmodule aModule)
        : CudaKernel("postFilterSum", aModule, make_dim3(1, 1, 1), make_dim3(8, 8, 1), 0)
{

}

void PostFilterSumKernel::AllocOffsets(int maxSliceNumber) {
    d_offsets.Alloc(maxSliceNumber * sizeof(float));
    h_offsets = new float[maxSliceNumber];
    slice_number = maxSliceNumber;
}


float PostFilterSumKernel::operator()(CudaDeviceVariable& image,
                                      CudaDeviceVariable& filter,
                                      CudaDeviceVariable& result,
                                      uint2 fftDim,
                                      ctfConstants constants,
                                      ctfImageConstants imageConstants,
                                      vector<float>* offsets,
                                      int2 minmaxSlice)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    CUdeviceptr filter_dptr = filter.GetDevicePtr();
    CUdeviceptr result_dptr = result.GetDevicePtr();

    // Offsets in m to Device
    for(int i=0; i < imageConstants.sliceNumber; i++){
        h_offsets[i] = (*offsets)[i] * powf(10, -9);
    }

    d_offsets.CopyHostToDevice(h_offsets, imageConstants.sliceNumber * sizeof(float));
    CUdeviceptr offsets_dptr = d_offsets.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &image_dptr;
    arglist[1] = &filter_dptr;
    arglist[2] = &result_dptr;
    arglist[3] = &fftDim;
    arglist[4] = &constants;
    arglist[5] = &imageConstants;
    arglist[6] = &offsets_dptr;
    arglist[7] = &minmaxSlice;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}

PostFilterSumCTFfreeKernel::PostFilterSumCTFfreeKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("postFilterSumCTFfree", aModule, aGridDim, aBlockDim, 0)
{

}

PostFilterSumCTFfreeKernel::PostFilterSumCTFfreeKernel(CUmodule aModule)
        : CudaKernel("postFilterSumCTFfree", aModule, make_dim3(1, 1, 1), make_dim3(8, 8, 1), 0)
{

}

void PostFilterSumCTFfreeKernel::AllocOffsets(int maxSliceNumber) {
    d_offsets.Alloc(maxSliceNumber * sizeof(float));
    h_offsets = new float[maxSliceNumber];
    slice_number = maxSliceNumber;
}


float PostFilterSumCTFfreeKernel::operator()(CudaDeviceVariable& image,
                                      CudaDeviceVariable& filter,
                                      CudaDeviceVariable& result,
                                      CudaDeviceVariable& result2,
                                      uint2 fftDim,
                                      ctfConstants constants,
                                      ctfImageConstants imageConstants,
                                      vector<float>* offsets,
                                      int2 minmaxSlice)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    CUdeviceptr filter_dptr = filter.GetDevicePtr();
    CUdeviceptr result_dptr = result.GetDevicePtr();
    CUdeviceptr result2_dptr = result2.GetDevicePtr();

    // Offsets in m to Device
    for(int i=0; i < imageConstants.sliceNumber; i++){
        h_offsets[i] = (*offsets)[i] * powf(10, -9);
    }

    d_offsets.CopyHostToDevice(h_offsets, imageConstants.sliceNumber * sizeof(float));
    CUdeviceptr offsets_dptr = d_offsets.GetDevicePtr();

    void** arglist = (void**)new void*[9];

    arglist[0] = &image_dptr;
    arglist[1] = &filter_dptr;
    arglist[2] = &result_dptr;
    arglist[3] = &fftDim;
    arglist[4] = &constants;
    arglist[5] = &imageConstants;
    arglist[6] = &offsets_dptr;
    arglist[7] = &minmaxSlice;
    arglist[8] = &result2_dptr;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}

PostFilterKernel::PostFilterKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("postFilter", aModule, aGridDim, aBlockDim, 0)
{

}

PostFilterKernel::PostFilterKernel(CUmodule aModule)
        : CudaKernel("postFilter", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float PostFilterKernel::operator()(CudaDeviceVariable& image,
                                   CudaDeviceVariable& filter,
                                   CudaDeviceVariable& result,
                                   uint2 fftDim,
                                   float normFactor)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    CUdeviceptr filter_dptr = filter.GetDevicePtr();
    CUdeviceptr result_dptr = result.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &image_dptr;
    arglist[1] = &filter_dptr;
    arglist[2] = &result_dptr;
    arglist[3] = &fftDim;
    arglist[4] = &normFactor;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}

PreFilterSpreadAdHocKernel::PreFilterSpreadAdHocKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("preFilterSpreadAdHoc",
                     aModule, aGridDim, aBlockDim, 0),
                     h_offsets(nullptr)
{

}

PreFilterSpreadAdHocKernel::PreFilterSpreadAdHocKernel(CUmodule aModule)
        : CudaKernel("preFilterSpreadAdHoc",
                     aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(16, 16, 1), 0),
                     h_offsets(nullptr)
{

}

void PreFilterSpreadAdHocKernel::AllocOffsets(int maxSliceNumber) {
    d_offsets.Alloc(maxSliceNumber * sizeof(float));
    h_offsets = new float[maxSliceNumber];
}

float PreFilterSpreadAdHocKernel::operator()(CudaDeviceVariable& image,
                                             CudaDeviceVariable& filter,
                                             CudaDeviceVariable& result,
                                             uint2 fftDim,
                                             ctfConstants constants,
                                             ctfImageConstants imageConstants,
                                             vector<float>* offsets,
                                             int2 minmaxSlice)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    CUdeviceptr filter_dptr = filter.GetDevicePtr();
    CUdeviceptr result_dptr = result.GetDevicePtr();
    CUtexObject snr_tex = 0;

    // Offsets in m to Device
    for(int i=0; i < imageConstants.sliceNumber; i++){
        h_offsets[i] = (*offsets)[i] * powf(10, -9);
    }

    d_offsets.CopyHostToDevice(h_offsets, imageConstants.sliceNumber * sizeof(float));
    CUdeviceptr offsets_dptr = d_offsets.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &image_dptr;
    arglist[1] = &filter_dptr;
    arglist[2] = &result_dptr;
    arglist[3] = &fftDim;
    arglist[4] = &constants;
    arglist[5] = &imageConstants;
    arglist[6] = &offsets_dptr;
    arglist[7] = &minmaxSlice;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}


PreFilterSpreadSNRKernel::PreFilterSpreadSNRKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("preFilterSpreadSNR",
                     aModule, aGridDim, aBlockDim, 0),
          h_offsets(nullptr)
{

}

PreFilterSpreadSNRKernel::PreFilterSpreadSNRKernel(CUmodule aModule)
        : CudaKernel("preFilterSpreadSNR",
                     aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(16, 16, 1), 0),
          h_offsets(nullptr)
{

}

void PreFilterSpreadSNRKernel::AllocOffsets(int maxSliceNumber) {
    d_offsets.Alloc(maxSliceNumber * sizeof(float));
    h_offsets = new float[maxSliceNumber];
}

float PreFilterSpreadSNRKernel::operator()(CudaDeviceVariable& image,
                                           CudaDeviceVariable& filter,
                                           CudaDeviceVariable& result,
                                           Cuda::CudaTextureObject1D& texSNR,
                                           Cuda::CudaTextureObject1D& texNP,
                                           uint2 fftDim,
                                           ctfConstants constants,
                                           ctfImageConstants imageConstants,
                                           vector<float>* offsets,
                                           int2 minmaxSlice,
                                           float deconvStrength)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    CUdeviceptr filter_dptr = filter.GetDevicePtr();
    CUdeviceptr result_dptr = result.GetDevicePtr();
    CUtexObject snr_tex = texSNR.GetTexObject();
    CUtexObject np_tex = texNP.GetTexObject();

    // Offsets in m to Device
    for(int i=0; i < imageConstants.sliceNumber; i++){
        h_offsets[i] = (*offsets)[i] * powf(10, -9);
    }

    d_offsets.CopyHostToDevice(h_offsets, imageConstants.sliceNumber * sizeof(float));
    CUdeviceptr offsets_dptr = d_offsets.GetDevicePtr();

    void** arglist = (void**)new void*[11];

    arglist[0] = &image_dptr;
    arglist[1] = &filter_dptr;
    arglist[2] = &result_dptr;
    arglist[3] = &snr_tex;
    arglist[4] = &np_tex;
    arglist[5] = &fftDim;
    arglist[6] = &constants;
    arglist[7] = &imageConstants;
    arglist[8] = &offsets_dptr;
    arglist[9] = &minmaxSlice;
    arglist[10] = &deconvStrength;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}


PreFilterSpreadSNRrelionKernel::PreFilterSpreadSNRrelionKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("preFilterSpreadSNRrelion",
                     aModule, aGridDim, aBlockDim, 0),
          h_offsets(nullptr)
{

}

PreFilterSpreadSNRrelionKernel::PreFilterSpreadSNRrelionKernel(CUmodule aModule)
        : CudaKernel("preFilterSpreadSNRrelion",
                     aModule,
                     make_dim3(1, 1, 1),
                     make_dim3(16, 16, 1), 0),
          h_offsets(nullptr)
{

}

void PreFilterSpreadSNRrelionKernel::AllocOffsets(int maxSliceNumber) {
    d_offsets.Alloc(maxSliceNumber * sizeof(float));
    h_offsets = new float[maxSliceNumber];
}

float PreFilterSpreadSNRrelionKernel::operator()(CudaDeviceVariable& image,
                                           CudaDeviceVariable& filter,
                                           CudaDeviceVariable& result,
                                           Cuda::CudaTextureObject1D& texSNR,
                                           Cuda::CudaTextureObject1D& texNP,
                                           uint2 fftDim,
                                           ctfConstants constants,
                                           ctfImageConstants imageConstants,
                                           vector<float>* offsets,
                                           int2 minmaxSlice,
                                           float deconvStrength)
{
    CUdeviceptr image_dptr = image.GetDevicePtr();
    CUdeviceptr filter_dptr = filter.GetDevicePtr();
    CUdeviceptr result_dptr = result.GetDevicePtr();
    CUtexObject snr_tex = texSNR.GetTexObject();
    CUtexObject np_tex = texNP.GetTexObject();

    // Offsets in m to Device
    for(int i=0; i < imageConstants.sliceNumber; i++){
        h_offsets[i] = (*offsets)[i] * powf(10, -9);
    }

    d_offsets.CopyHostToDevice(h_offsets, imageConstants.sliceNumber * sizeof(float));
    CUdeviceptr offsets_dptr = d_offsets.GetDevicePtr();

    void** arglist = (void**)new void*[11];

    arglist[0] = &image_dptr;
    arglist[1] = &filter_dptr;
    arglist[2] = &result_dptr;
    arglist[3] = &snr_tex;
    arglist[4] = &np_tex;
    arglist[5] = &fftDim;
    arglist[6] = &constants;
    arglist[7] = &imageConstants;
    arglist[8] = &offsets_dptr;
    arglist[9] = &minmaxSlice;
    arglist[10] = &deconvStrength;

    float ms;

    CUevent eventStart;
    CUevent eventEnd;
    CUstream stream = 0;
    cudaSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    cudaSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    cudaSafeCall(cuEventRecord(eventStart, stream));
    cudaSafeCall(cuLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    cudaSafeCall(cuCtxSynchronize());

    cudaSafeCall(cuEventRecord(eventEnd, stream));
    cudaSafeCall(cuEventSynchronize(eventEnd));
    cudaSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    cudaSafeCall(cuEventDestroy(eventStart));
    cudaSafeCall(cuEventDestroy(eventEnd));

    delete[] arglist;
    return ms;
}