//
// Created by uermel on 10/1/21.
//

#include "CTFKernel.h"
using namespace Cuda;

CTFKernel::CTFKernel(CUmodule aModule, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("ctf", aModule, aGridDim, aBlockDim, 0)
{

}

CTFKernel::CTFKernel(CUmodule aModule)
        : CudaKernel("ctf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float CTFKernel::operator()(CudaDeviceVariable& ctf, float defocusMin, float defocusMax, float angle, bool applyForFP, bool phaseFlipOnly, float WienerFilterNoiseLevel, size_t stride, float4 betaFac)
{
    CUdeviceptr ctf_dptr = ctf.GetDevicePtr();
    //size_t stride = 2049 * sizeof(float2);
    float _defocusMin = defocusMin * 0.000000001f;
    float _defocusMax = defocusMax * 0.000000001f;
    float _angle = angle / 180.0f * (float)M_PI;

    void** arglist = (void**)new void*[9];

    arglist[0] = &ctf_dptr;
    arglist[1] = &stride;
    arglist[2] = &_defocusMin;
    arglist[3] = &_defocusMax;
    arglist[4] = &_angle;
    arglist[5] = &applyForFP;
    arglist[6] = &phaseFlipOnly;
    arglist[7] = &WienerFilterNoiseLevel;
    arglist[8] = &betaFac;

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

void SetConstantValues(CTFKernel& kernel, Projection& proj, int index, float cs, float voltage)
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

    _pixelsize = _pixelsize * powf(10, -9);
    kernel.SetConstantValue("c_pixelsize", &_pixelsize);

    float2 _pixelcount = make_float2((float)proj.GetWidth(), (float)proj.GetHeight());
    kernel.SetConstantValue("c_pixelcount", &_pixelcount);

    float _maxFreq = 1.0f / (_pixelsize * 2.0f);
    kernel.SetConstantValue("c_maxFreq", &_maxFreq);

    float2 _freqStepSize = make_float2(_maxFreq / (_pixelcount.x / 2.0f), _maxFreq / (_pixelcount.y / 2.0f));
    //printf("_freqStepSize: %f; _pixelsize: %f\n", _freqStepSize, _pixelsize);
    kernel.SetConstantValue("c_freqStepSize", &_freqStepSize);

    float _applyScatteringProfile = 0;
    kernel.SetConstantValue("c_applyScatteringProfile", &_applyScatteringProfile);
    float _applyEnvelopeFunction = 0;
    kernel.SetConstantValue("c_applyEnvelopeFunction", &_applyEnvelopeFunction);
}