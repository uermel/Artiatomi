//
// Created by uermel on 2/15/20.
//

#include "CudaSurfaces.h"

#ifdef USE_CUDA
namespace Cuda
{
    CudaSurfaceObject3D::CudaSurfaceObject3D(CudaArray3D* aArray)
            : mCleanUp(false)
    {
        mArray = aArray;
        memset(&mResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
        //memset(&mTexDesc, 0, sizeof(CUDA_TEXTURE_DESC));
        memset(&mResViewDesc, 0, sizeof(CUDA_RESOURCE_VIEW_DESC));

        mResDesc.flags = 0;
        mResDesc.res.array.hArray = mArray->GetCUarray();
        mResDesc.resType = CU_RESOURCE_TYPE_ARRAY;

//        mTexDesc.addressMode[0] = aAddressMode0;
//        mTexDesc.addressMode[1] = aAddressMode1;
//        mTexDesc.addressMode[2] = aAddressMode2;
//        mTexDesc.filterMode = aFilterMode;
//        mTexDesc.flags = aTexRefSetFlag;

        cudaSafeCall(cuSurfObjectCreate(&mSurfObj, &mResDesc));

    }

    CudaSurfaceObject3D::~CudaSurfaceObject3D()
    {
        cudaSafeCall(cuSurfObjectDestroy(mSurfObj));
        if (mCleanUp && mArray)
        {
            delete mArray;
            mArray = NULL;
        }
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