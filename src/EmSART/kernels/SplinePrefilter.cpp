//
// Created by uermel on 10/4/21.
//

#include "SplinePrefilter.h"

using namespace Cuda;

void GetPoles(int degree, int &NbPoles, float4 &Poles){

    switch (degree) {
        case 2:
            NbPoles = 1;
            Poles.x = sqrt(8.0f) - 3.0f;
            break;
        case 3:
            NbPoles = 1;
            Poles.x = sqrt(3.0f) - 2.0f;
            break;
        case 4:
            NbPoles = 2;
            Poles.x = sqrt(664.0f - sqrt(438976.0f)) + sqrt(304.0f) - 19.0f;
            Poles.y = sqrt(664.0f + sqrt(438976.0f)) - sqrt(304.0f) - 19.0f;
            break;
        case 5:
            NbPoles = 2;
            Poles.x = sqrt(135.0f / 2.0f - sqrt(17745.0f / 4.0f)) + sqrt(105.0f / 4.0f)
                      - 13.0f / 2.0f;
            Poles.y = sqrt(135.0f / 2.0f + sqrt(17745.0f / 4.0f)) - sqrt(105.0f / 4.0f)
                      - 13.0f / 2.0f;
            break;
        case 6:
            NbPoles = 3;
            Poles.x = -0.48829458930304475513011803888378906211227916123938;
            Poles.y = -0.081679271076237512597937765737059080653379610398148;
            Poles.z = -0.0014141518083258177510872439765585925278641690553467;
            break;
        case 7:
            NbPoles = 3;
            Poles.x = -0.53528043079643816554240378168164607183392315234269;
            Poles.y = -0.12255461519232669051527226435935734360548654942730;
            Poles.z = -0.0091486948096082769285930216516478534156925639545994;
            break;
        case 8:
            NbPoles = 4;
            Poles.x = -0.57468690924876543053013930412874542429066157804125;
            Poles.y = -0.16303526929728093524055189686073705223476814550830;
            Poles.z = -0.023632294694844850023403919296361320612665920854629;
            Poles.w = -0.00015382131064169091173935253018402160762964054070043;
            break;
        case 9:
            NbPoles = 4;
            Poles.x = -0.60799738916862577900772082395428976943963471853991;
            Poles.y = -0.20175052019315323879606468505597043468089886575747;
            Poles.z = -0.043222608540481752133321142979429688265852380231497;
            Poles.w = -0.0021213069031808184203048965578486234220548560988624;
            break;
        default:
            exit(-1);
    }
}

void ComputeGain(int NbPoles, float4 Poles, float &Lambda) {

    // Unrolled loop to work with vector type, ugly
    Lambda = 1;

    if (NbPoles > 0)
        Lambda *= (1.0f - Poles.x) * (1.0f - 1.0f / Poles.x);

    if (NbPoles > 1)
        Lambda *= (1.0f - Poles.y) * (1.0f - 1.0f / Poles.y);

    if (NbPoles > 2)
        Lambda *= (1.0f - Poles.z) * (1.0f - 1.0f / Poles.z);

    if (NbPoles > 3)
        Lambda *= (1.0f - Poles.w) * (1.0f - 1.0f / Poles.w);
}

void ComputeHorizon(float4 Poles, int4 &Horizon){
    Horizon.x = (int)ceil(log(FLT_EPSILON) / log(fabs(Poles.x)));
    Horizon.y = (int)ceil(log(FLT_EPSILON) / log(fabs(Poles.y)));
    Horizon.z = (int)ceil(log(FLT_EPSILON) / log(fabs(Poles.z)));
    Horizon.w = (int)ceil(log(FLT_EPSILON) / log(fabs(Poles.w)));
}

uint PowTwoDivider(uint n)
{
    if (n == 0) return 0;
    uint divider = 1;
    while ((n & divider) == 0) divider <<= 1;
    return divider;
}

/// 2DX, 2DY for CUDA array to CUDA array, in place AND out of place
SplinePrefilter2DX::SplinePrefilter2DX(CUmodule aModule, int aSplineDegree,dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DX", aModule, aGridDim, aBlockDim, 0),
        NbPoles(0),
        Poles(make_float4(0, 0, 0, 0)),
        Horizon(make_int4(0, 0, 0, 0)),
        Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter2DX::SplinePrefilter2DX(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients2DX", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter2DX::operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(height), 64), 1, 1);
    SetGridDimensions(height / mBlockDim.x);

    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint p = image.GetPitch();
    uint w = (uint) width;
    uint h = (uint) height;

    void** arglist = (void**)new void*[8];

    arglist[0] = &image_dptr;
    arglist[1] = &p;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &NbPoles;
    arglist[5] = &Poles;
    arglist[6] = &Horizon;
    arglist[7] = &Lambda;

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

SplinePrefilter2DY::SplinePrefilter2DY(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DY", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter2DY::SplinePrefilter2DY(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients2DY", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter2DY::operator()(Cuda::CudaPitchedDeviceVariable& image, int width, int height)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(width), 64), 1, 1);
    SetGridDimensions(width / mBlockDim.x);

    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint p = image.GetPitch();
    uint w = (uint) width;
    uint h = (uint) height;

    void** arglist = (void**)new void*[8];

    arglist[0] = &image_dptr;
    arglist[1] = &p;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &NbPoles;
    arglist[5] = &Poles;
    arglist[6] = &Horizon;
    arglist[7] = &Lambda;

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

/// 2DX, 2DY for CUDA array to CUDA array, in place AND out of place
SplinePrefilter2DXSurf::SplinePrefilter2DXSurf(CUmodule aModule, int aSplineDegree,dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DXSurf", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter2DXSurf::SplinePrefilter2DXSurf(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients2DXSurf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter2DXSurf::operator()(Cuda::CudaSurfaceObject2D& image_in,
                                         Cuda::CudaSurfaceObject2D& image_out,
                                         int width, int height)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(height), 64), 1, 1);
    SetGridDimensions(height / mBlockDim.x);

    CUsurfObject im_in = image_in.GetSurfObject();
    CUsurfObject im_out = image_out.GetSurfObject();
    uint w = (uint) width;
    uint h = (uint) height;

    void** arglist = (void**)new void*[8];

    arglist[0] = &im_in;
    arglist[1] = &im_out;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &NbPoles;
    arglist[5] = &Poles;
    arglist[6] = &Horizon;
    arglist[7] = &Lambda;

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

SplinePrefilter2DYSurf::SplinePrefilter2DYSurf(CUmodule aModule, int aSplineDegree,dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DYSurf", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter2DYSurf::SplinePrefilter2DYSurf(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients2DYSurf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter2DYSurf::operator()(Cuda::CudaSurfaceObject2D& image_in,
                                         Cuda::CudaSurfaceObject2D& image_out,
                                         int width, int height)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(width), 64), 1, 1);
    SetGridDimensions(width / mBlockDim.x);

    CUsurfObject im_in = image_in.GetSurfObject();
    CUsurfObject im_out = image_out.GetSurfObject();
    uint w = (uint) width;
    uint h = (uint) height;

    void** arglist = (void**)new void*[8];

    arglist[0] = &im_in;
    arglist[1] = &im_out;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &NbPoles;
    arglist[5] = &Poles;
    arglist[6] = &Horizon;
    arglist[7] = &Lambda;

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

/// 2DX, 2DY for linear memory to CUDA array, out of place
SplinePrefilter2DXPtr2Surf::SplinePrefilter2DXPtr2Surf(CUmodule aModule, int aSplineDegree,dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DX_ptr2surf", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter2DXPtr2Surf::SplinePrefilter2DXPtr2Surf(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients2DX_ptr2surf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter2DXPtr2Surf::operator()(Cuda::CudaDeviceVariable& image_in,
                                             Cuda::CudaSurfaceObject2D& image_out,
                                             int width, int height, int z)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(height), 64), 1, 1);
    SetGridDimensions(height / mBlockDim.x);

    CUdeviceptr im_in = image_in.GetDevicePtr();
    CUsurfObject im_out = image_out.GetSurfObject();
    uint p = (uint) width * sizeof(float);
    uint w = (uint) width;
    uint h = (uint) height;
    // ptr offset
    size_t offset = width * height * z * sizeof(float);

    void** arglist = (void**)new void*[10];

    arglist[0] = &im_in;
    arglist[1] = &offset;
    arglist[2] = &im_out;
    arglist[3] = &p;
    arglist[4] = &w;
    arglist[5] = &h;
    arglist[6] = &NbPoles;
    arglist[7] = &Poles;
    arglist[8] = &Horizon;
    arglist[9] = &Lambda;

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

SplinePrefilter2DYPtr2Surf::SplinePrefilter2DYPtr2Surf(CUmodule aModule, int aSplineDegree,dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients2DY_ptr2surf", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter2DYPtr2Surf::SplinePrefilter2DYPtr2Surf(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients2DY_ptr2surf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter2DYPtr2Surf::operator()(Cuda::CudaDeviceVariable& image_in,
                                             Cuda::CudaSurfaceObject2D& image_out,
                                             int width, int height, int z)
{
    // Block/Grid
    SetBlockDimensions(min(PowTwoDivider(width), 64), 1, 1);
    SetGridDimensions(width / mBlockDim.x);

    CUdeviceptr im_in = image_in.GetDevicePtr();
    CUsurfObject im_out = image_out.GetSurfObject();
    uint p = (uint) width * sizeof(float);
    uint w = (uint) width;
    uint h = (uint) height;
    // ptr offset
    size_t offset = width * height * z * sizeof(float);

    void** arglist = (void**)new void*[10];

    arglist[0] = &im_in;
    arglist[1] = &offset;
    arglist[2] = &im_out;
    arglist[3] = &p;
    arglist[4] = &w;
    arglist[5] = &h;
    arglist[6] = &NbPoles;
    arglist[7] = &Poles;
    arglist[8] = &Horizon;
    arglist[9] = &Lambda;

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

SplinePrefilter3DX::SplinePrefilter3DX(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients3DX", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter3DX::SplinePrefilter3DX(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients3DX", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter3DX::operator()(Cuda::CudaDeviceVariable& image, int width, int height, int depth)
{
    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    SetBlockDimensions(dimX, dimY, 1);
    SetGridDimensions(height/mBlockDim.x, depth/mBlockDim.y, 1);

//    printf("\n");
//    printf("Block %i %i %i \n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
//    printf("Grid %i %i %i \n", mGridDim.x, mGridDim.y, mGridDim.z);

    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint p = (uint) width * sizeof(float);
    uint w = (uint) width;
    uint h = (uint) height;
    uint d = (uint) depth;

    void** arglist = (void**)new void*[9];

    arglist[0] = &image_dptr;
    arglist[1] = &p;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &d;
    arglist[5] = &NbPoles;
    arglist[6] = &Poles;
    arglist[7] = &Horizon;
    arglist[8] = &Lambda;

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

SplinePrefilter3DY::SplinePrefilter3DY(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients3DY", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter3DY::SplinePrefilter3DY(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients3DY", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter3DY::operator()(Cuda::CudaDeviceVariable& image, int width, int height, int depth)
{
    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    SetBlockDimensions(dimX, dimY, 1);
    SetGridDimensions(width/mBlockDim.x, depth/mBlockDim.y, 1);

//    printf("\n");
//    printf("Block %i %i %i \n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
//    printf("Grid %i %i %i \n", mGridDim.x, mGridDim.y, mGridDim.z);

    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint p = (uint) width * sizeof(float);
    uint w = (uint) width;
    uint h = (uint) height;
    uint d = (uint) depth;

    void** arglist = (void**)new void*[9];

    arglist[0] = &image_dptr;
    arglist[1] = &p;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &d;
    arglist[5] = &NbPoles;
    arglist[6] = &Poles;
    arglist[7] = &Horizon;
    arglist[8] = &Lambda;

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

SplinePrefilter3DZ::SplinePrefilter3DZ(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients3DZ", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter3DZ::SplinePrefilter3DZ(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients3DZ", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter3DZ::operator()(Cuda::CudaDeviceVariable& image, int width, int height, int depth)
{
    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    SetBlockDimensions(dimX, dimY, 1);
    SetGridDimensions(width/mBlockDim.x, height/mBlockDim.y, 1);

//    printf("\n");
//    printf("Block %i %i %i \n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
//    printf("Grid %i %i %i \n", mGridDim.x, mGridDim.y, mGridDim.z);

    CUdeviceptr image_dptr = image.GetDevicePtr();
    uint p = (uint) width * sizeof(float);
    uint w = (uint) width;
    uint h = (uint) height;
    uint d = (uint) depth;

    void** arglist = (void**)new void*[9];

    arglist[0] = &image_dptr;
    arglist[1] = &p;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &d;
    arglist[5] = &NbPoles;
    arglist[6] = &Poles;
    arglist[7] = &Horizon;
    arglist[8] = &Lambda;

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


SplinePrefilter3DXSurf::SplinePrefilter3DXSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients3DXSurf", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter3DXSurf::SplinePrefilter3DXSurf(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients3DXSurf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter3DXSurf::operator()(Cuda::CudaSurfaceObject3D& volume_in,
                                         Cuda::CudaSurfaceObject3D& volume_out,
                                         int width, int height, int depth)
{
    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    SetBlockDimensions(dimX, dimY, 1);
    SetGridDimensions(height/mBlockDim.x, depth/mBlockDim.y, 1);

//    printf("\n");
//    printf("Block %i %i %i \n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
//    printf("Grid %i %i %i \n", mGridDim.x, mGridDim.y, mGridDim.z);

    CUsurfObject vol_in = volume_in.GetSurfObject();
    CUsurfObject vol_out = volume_out.GetSurfObject();
    uint w = (uint) width;
    uint h = (uint) height;
    uint d = (uint) depth;

    void** arglist = (void**)new void*[9];

    arglist[0] = &vol_in;
    arglist[1] = &vol_out;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &d;
    arglist[5] = &NbPoles;
    arglist[6] = &Poles;
    arglist[7] = &Horizon;
    arglist[8] = &Lambda;

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

SplinePrefilter3DYSurf::SplinePrefilter3DYSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients3DYSurf", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter3DYSurf::SplinePrefilter3DYSurf(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients3DYSurf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter3DYSurf::operator()(Cuda::CudaSurfaceObject3D& volume_in,
                                         Cuda::CudaSurfaceObject3D& volume_out,
                                         int width, int height, int depth)
{
    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    SetBlockDimensions(dimX, dimY, 1);
    SetGridDimensions(width/mBlockDim.x, depth/mBlockDim.y, 1);

//    printf("\n");
//    printf("Block %i %i %i \n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
//    printf("Grid %i %i %i \n", mGridDim.x, mGridDim.y, mGridDim.z);

    CUsurfObject vol_in = volume_in.GetSurfObject();
    CUsurfObject vol_out = volume_out.GetSurfObject();
    uint w = (uint) width;
    uint h = (uint) height;
    uint d = (uint) depth;

    void** arglist = (void**)new void*[9];

    arglist[0] = &vol_in;
    arglist[1] = &vol_out;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &d;
    arglist[5] = &NbPoles;
    arglist[6] = &Poles;
    arglist[7] = &Horizon;
    arglist[8] = &Lambda;

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

SplinePrefilter3DZSurf::SplinePrefilter3DZSurf(CUmodule aModule, int aSplineDegree, dim3 aGridDim, dim3 aBlockDim)
        : CudaKernel("SamplesToCoefficients3DZSurf", aModule, aGridDim, aBlockDim, 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

SplinePrefilter3DZSurf::SplinePrefilter3DZSurf(CUmodule aModule, int aSplineDegree)
        : CudaKernel("SamplesToCoefficients3DZSurf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0),
          NbPoles(0),
          Poles(make_float4(0, 0, 0, 0)),
          Horizon(make_int4(0, 0, 0, 0)),
          Lambda(1)
{
    // Get spline poles
    GetPoles(aSplineDegree, NbPoles, Poles);

    // Compute gain
    ComputeGain(NbPoles, Poles, Lambda);

    // Compute Horizon
    ComputeHorizon(Poles, Horizon);
}

float SplinePrefilter3DZSurf::operator()(Cuda::CudaSurfaceObject3D& volume_in,
                                         Cuda::CudaSurfaceObject3D& volume_out,
                                         int width, int height, int depth)
{
    // Block/Grid
    uint dimX = min(min(PowTwoDivider(width), PowTwoDivider(height)), 64);
    uint dimY = min(min(PowTwoDivider(depth), PowTwoDivider(height)), 512/dimX);
    SetBlockDimensions(dimX, dimY, 1);
    SetGridDimensions(width/mBlockDim.x, height/mBlockDim.y, 1);

//    printf("\n");
//    printf("Block %i %i %i \n", mBlockDim.x, mBlockDim.y, mBlockDim.z);
//    printf("Grid %i %i %i \n", mGridDim.x, mGridDim.y, mGridDim.z);

    CUsurfObject vol_in = volume_in.GetSurfObject();
    CUsurfObject vol_out = volume_out.GetSurfObject();
    uint w = (uint) width;
    uint h = (uint) height;
    uint d = (uint) depth;

    void** arglist = (void**)new void*[9];

    arglist[0] = &vol_in;
    arglist[1] = &vol_out;
    arglist[2] = &w;
    arglist[3] = &h;
    arglist[4] = &d;
    arglist[5] = &NbPoles;
    arglist[6] = &Poles;
    arglist[7] = &Horizon;
    arglist[8] = &Lambda;

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