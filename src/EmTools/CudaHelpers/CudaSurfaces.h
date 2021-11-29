//
// Created by uermel on 2/15/20.
//

#ifndef ARTIATOMI_CUDASURFACES_H
#define ARTIATOMI_CUDASURFACES_H

#include "CudaDefault.h"
#ifdef USE_CUDA
#include "CudaException.h"
#include "CudaContext.h"
#include "CudaKernel.h"
#include "CudaVariables.h"
#include "CudaArrays.h"

namespace Cuda
{
    class CudaSurfaceObject2D
    {
    private:
        CUsurfObject mSurfObj;
        CUDA_RESOURCE_DESC mResDesc;
        CUDA_RESOURCE_VIEW_DESC mResViewDesc;

        CudaPitchedDeviceVariable* mData;
        CudaArray2D* mArray; //CUDA Array where the surface data is stored
        bool mCleanUp; //Indicates if the cuda array was created by the object itself

    public:
        explicit CudaSurfaceObject2D(CudaArray2D* aArray);
        CudaSurfaceObject2D(CudaPitchedDeviceVariable* aVariable,
                            CUarray_format aDataFormat,
                            uint aNumChannels);
        CudaSurfaceObject2D();

        void Bind(CudaArray2D* aArray);
        void Bind(CudaPitchedDeviceVariable* aVariable,
                  CUarray_format aDataFormat,
                  uint aNumChannels);

        ~CudaSurfaceObject2D();

        CudaArray2D* GetArray();

        CUsurfObject GetSurfObject();
    };

    class CudaSurfaceObject3D
    {
    private:
        CUsurfObject mSurfObj;
        CUDA_RESOURCE_DESC mResDesc;
        CUDA_RESOURCE_VIEW_DESC mResViewDesc;

        CudaArray3D* mArray; //CUDA Array where the surface data is stored
        bool mCleanUp; //Indicates if the cuda array was created by the object itself

    public:
        explicit CudaSurfaceObject3D(CudaArray3D* aArray);

        ~CudaSurfaceObject3D();

        CudaArray3D* GetArray();

        CUsurfObject GetSurfObject();
    };
}

#endif //USE_CUDA
#endif //ARTIATOMI_CUDASURFACES_H
