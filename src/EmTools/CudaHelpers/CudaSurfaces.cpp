//
// Created by uermel on 2/15/20.
//

#include "CudaSurfaces.h"

#ifdef USE_CUDA
namespace Cuda
{
    CudaSurfaceObject2D::CudaSurfaceObject2D(CudaArray2D* aArray)
            : mCleanUp(false),
              mData(nullptr),
              mSurfObj(0),
              mResDesc({}),
              mResViewDesc({})
    {
        mArray = aArray;
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));

        mResDesc.flags = 0;
        mResDesc.res.array.hArray = mArray->GetCUarray();
        mResDesc.resType = CU_RESOURCE_TYPE_ARRAY;

        cudaSafeCall(cuSurfObjectCreate(&mSurfObj, &mResDesc));

    }

    CudaSurfaceObject2D::CudaSurfaceObject2D(CudaPitchedDeviceVariable* aVariable,
                                             CUarray_format aDataFormat,
                                             uint aNumChannels)
            : mCleanUp(false),
              mArray(nullptr),
              mSurfObj(0),
              mResDesc({}),
              mResViewDesc({})
    {
        mData = aVariable;
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));

        mResDesc.flags = 0;
        mResDesc.res.pitch2D.devPtr = mData->GetDevicePtr();
        mResDesc.res.pitch2D.format = aDataFormat;
        mResDesc.res.pitch2D.height = mData->GetHeight();
        mResDesc.res.pitch2D.numChannels = aNumChannels;
        mResDesc.res.pitch2D.pitchInBytes = mData->GetPitch();
        mResDesc.res.pitch2D.width = mData->GetWidth();
        mResDesc.resType = CU_RESOURCE_TYPE_PITCH2D;

        cudaSafeCall(cuSurfObjectCreate(&mSurfObj, &mResDesc));
    }

    CudaSurfaceObject2D::CudaSurfaceObject2D()
    : mCleanUp(false),
      mData(nullptr),
      mArray(nullptr),
      mSurfObj(0),
      mResDesc({}),
      mResViewDesc({})
    {
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));
    }

    void CudaSurfaceObject2D::Bind(CudaArray2D *aArray) {
        mArray = aArray;
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));

        mResDesc.flags = 0;
        mResDesc.res.array.hArray = mArray->GetCUarray();
        mResDesc.resType = CU_RESOURCE_TYPE_ARRAY;

        cudaSafeCall(cuSurfObjectCreate(&mSurfObj, &mResDesc));
    }

    void CudaSurfaceObject2D::Bind(CudaPitchedDeviceVariable *aVariable,
                                   CUarray_format aDataFormat,
                                   uint aNumChannels){
        mData = aVariable;
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));

        mResDesc.flags = 0;
        mResDesc.res.pitch2D.devPtr = mData->GetDevicePtr();
        mResDesc.res.pitch2D.format = aDataFormat;
        mResDesc.res.pitch2D.height = mData->GetHeight();
        mResDesc.res.pitch2D.numChannels = aNumChannels;
        mResDesc.res.pitch2D.pitchInBytes = mData->GetPitch();
        mResDesc.res.pitch2D.width = mData->GetWidth();
        mResDesc.resType = CU_RESOURCE_TYPE_PITCH2D;

        cudaSafeCall(cuSurfObjectCreate(&mSurfObj, &mResDesc));
    }

    CudaSurfaceObject2D::~CudaSurfaceObject2D()
    {
        cudaSafeCall(cuSurfObjectDestroy(mSurfObj));

        if (mCleanUp && mData)
        {
            delete mData;
            mData = nullptr;
        }

        if (mCleanUp && mArray)
        {
            delete mArray;
            mArray = nullptr;
        }
    }

    CudaArray2D* CudaSurfaceObject2D::GetArray()
    {
        return mArray;
    }

    CUtexObject CudaSurfaceObject2D::GetSurfObject()
    {
        return mSurfObj;
    }

    CudaSurfaceObject3D::CudaSurfaceObject3D()
            : mCleanUp(false), mArray(nullptr), mSurfObj(0), mResDesc({}), mResViewDesc({})
    {
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));
    }

    CudaSurfaceObject3D::CudaSurfaceObject3D(CudaArray3D* aArray)
            : mCleanUp(false), mSurfObj(0), mResDesc({}), mResViewDesc({})
    {
        mArray = aArray;
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));

        mResDesc.flags = 0;
        mResDesc.res.array.hArray = mArray->GetCUarray();
        mResDesc.resType = CU_RESOURCE_TYPE_ARRAY;

        cudaSafeCall(cuSurfObjectCreate(&mSurfObj, &mResDesc));
    }

    CudaSurfaceObject3D::~CudaSurfaceObject3D()
    {
        cudaSafeCall(cuSurfObjectDestroy(mSurfObj));
        if (mCleanUp && mArray)
        {
            delete mArray;
            mArray = nullptr;
        }
    }

    void CudaSurfaceObject3D::Bind(CudaArray3D *aArray)
    {
        mArray = aArray;
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));

        mResDesc.flags = 0;
        mResDesc.res.array.hArray = mArray->GetCUarray();
        mResDesc.resType = CU_RESOURCE_TYPE_ARRAY;

        cudaSafeCall(cuSurfObjectCreate(&mSurfObj, &mResDesc));
    }

    CudaArray3D* CudaSurfaceObject3D::GetArray()
    {
        return mArray;
    }

    CUtexObject CudaSurfaceObject3D::GetSurfObject()
    {
        return mSurfObj;
    }
}
#endif //USE_CUDA