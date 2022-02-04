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


#include "CudaVariables.h"

#ifdef USE_CUDA
namespace Cuda
{
    CudaDeviceVariable::CudaDeviceVariable(size_t aSizeInBytes)
            :mDevPtr(0), mSizeInBytes(aSizeInBytes), mIsOwner(true)
    {
        cudaSafeCall(cuMemAlloc(&mDevPtr, mSizeInBytes));
    }

    CudaDeviceVariable::CudaDeviceVariable(const CUdeviceptr & aDevPtr, bool aIsOwner)
            : mIsOwner(aIsOwner)
    {
        cudaSafeCall(cuMemGetAddressRange(NULL, &mSizeInBytes, aDevPtr));
    }

    CudaDeviceVariable::CudaDeviceVariable(CudaDeviceVariable & aDevVar, bool aIsOwner)
            : mDevPtr(aDevVar.mDevPtr), mSizeInBytes(aDevVar.mSizeInBytes), mIsOwner(aIsOwner)
    {

    }

    CudaDeviceVariable::CudaDeviceVariable()
            : mDevPtr(0), mSizeInBytes(0)
    {
    }

    void CudaDeviceVariable::Alloc(size_t aSizeInBytes)
    {
        if (mDevPtr != 0 && mIsOwner)
        {
            cudaSafeCall(cuMemFree(mDevPtr));
            mDevPtr = 0;
        }
        cudaSafeCall(cuMemAlloc(&mDevPtr, aSizeInBytes));
        mIsOwner = true;
        mSizeInBytes = aSizeInBytes;
    }

    CudaDeviceVariable::~CudaDeviceVariable()
    {
        if (mIsOwner)
            cuMemFree(mDevPtr);
    }

    void CudaDeviceVariable::CopyDeviceToDevice(CUdeviceptr aSource)
    {
        cudaSafeCall(cuMemcpyDtoD(mDevPtr, aSource, mSizeInBytes));
    }

    void CudaDeviceVariable::CopyDeviceToDevice(CudaDeviceVariable& aSource)
    {
        cudaSafeCall(cuMemcpyDtoD(mDevPtr, aSource.GetDevicePtr(), mSizeInBytes));
    }


    void CudaDeviceVariable::CopyHostToDevice(void* aSource, size_t aSizeInBytes)
    {
        size_t size = aSizeInBytes;
        if (size == 0)
            size = mSizeInBytes;
        cudaSafeCall(cuMemcpyHtoD(mDevPtr, aSource, size));
    }

    void CudaDeviceVariable::CopyDeviceToHost(void* aDest, size_t aSizeInBytes)
    {
        size_t size = aSizeInBytes;
        if (size == 0)
            size = mSizeInBytes;
        cudaSafeCall(cuMemcpyDtoH(aDest, mDevPtr, size));
    }

    void CudaDeviceVariable::CopyDeviceToDeviceAsync(CUstream stream, CudaDeviceVariable& aSource)
    {
        cudaSafeCall(cuMemcpyDtoDAsync(mDevPtr, aSource.GetDevicePtr(), mSizeInBytes, stream));
    }


    void CudaDeviceVariable::CopyHostToDeviceAsync(CUstream stream, void* aSource, size_t aSizeInBytes)
    {
        size_t size = aSizeInBytes;
        if (size == 0)
            size = mSizeInBytes;
        cudaSafeCall(cuMemcpyHtoDAsync(mDevPtr, aSource, size, stream));
    }

    void CudaDeviceVariable::CopyDeviceToHostAsync(CUstream stream, void* aDest, size_t aSizeInBytes)
    {
        size_t size = aSizeInBytes;
        if (size == 0)
            size = mSizeInBytes;
        cudaSafeCall(cuMemcpyDtoHAsync(aDest, mDevPtr, size, stream));
    }

    size_t CudaDeviceVariable::GetSize()
    {
        return mSizeInBytes;
    }

    CUdeviceptr CudaDeviceVariable::GetDevicePtr()
    {
        return mDevPtr;
    }

    void CudaDeviceVariable::Memset(uchar aValue)
    {
        cudaSafeCall(cuMemsetD8(mDevPtr, aValue, mSizeInBytes));
    }



    bool CudaPitchedDeviceVariable::IsReallyPitched()
    {
        return IsReallyPitched(mElementSize);
    }

    bool CudaPitchedDeviceVariable::IsReallyPitched(uint aElementSize)
    {
        if (aElementSize == 4 || aElementSize == 8 || aElementSize == 16)
            return true;

        return false;
    }

    CudaPitchedDeviceVariable::CudaPitchedDeviceVariable(size_t aWidthInBytes, size_t aHeight, uint aElementSize)
            : mPitch(0), mHeight(aHeight), mWidthInBytes(aWidthInBytes), mElementSize(aElementSize), mIsOwner(true)
    {
        if (IsReallyPitched())
        {
            cudaSafeCall(cuMemAllocPitch(&mDevPtr, &mPitch, mWidthInBytes, mHeight, mElementSize));
            mSizeInBytes = aHeight * mPitch;
        }
        else
        {
            mSizeInBytes = aHeight * mWidthInBytes;
            mPitch = mWidthInBytes;
            cudaSafeCall(cuMemAlloc(&mDevPtr, mSizeInBytes));
        }
    }

    CudaPitchedDeviceVariable::CudaPitchedDeviceVariable(CudaPitchedDeviceVariable& aDevVar, bool aIsOwner)
            : mPitch(aDevVar.mPitch), mHeight(aDevVar.mHeight), mWidthInBytes(aDevVar.mWidthInBytes), mElementSize(aDevVar.mElementSize), mIsOwner(aIsOwner), mSizeInBytes(aDevVar.mSizeInBytes)
    {

    }

    CudaPitchedDeviceVariable::CudaPitchedDeviceVariable(CUdeviceptr aPtr, size_t aWidthInBytes, size_t aHeight, size_t aPitch, uint aElementSize, bool aIsOwner)
            : mPitch(aPitch), mHeight(aHeight), mWidthInBytes(aWidthInBytes), mElementSize(aElementSize), mIsOwner(aIsOwner), mSizeInBytes(aPitch * aHeight)
    {

    }

    CudaPitchedDeviceVariable::CudaPitchedDeviceVariable()
            : mPitch(0), mHeight(0), mWidthInBytes(0), mElementSize(0), mDevPtr(0)
    {
    }

    void CudaPitchedDeviceVariable::Alloc(size_t aWidthInBytes, size_t aHeight, uint aElementSize)
    {
        if (mDevPtr != 0 && mIsOwner)
        {
            cudaSafeCall(cuMemFree(mDevPtr));
            mDevPtr = 0;
        }
        mPitch = 0;
        mHeight = aHeight;
        mWidthInBytes = aWidthInBytes;
        mElementSize = aElementSize;

        cudaSafeCall(cuMemAllocPitch(&mDevPtr, &mPitch, mWidthInBytes, mHeight, mElementSize));
        mSizeInBytes = aHeight * mPitch;
        mIsOwner = true;
    }

    CudaPitchedDeviceVariable::~CudaPitchedDeviceVariable()
    {
        if (mIsOwner)
            cuMemFree(mDevPtr);
    }

    void CudaPitchedDeviceVariable::CopyDeviceToDevice(CudaPitchedDeviceVariable& aSource)
    {
        if (IsReallyPitched())
        {
            CUDA_MEMCPY2D params;
            memset(&params, 0, sizeof(params));
            params.srcDevice = aSource.GetDevicePtr();
            params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            params.srcPitch = aSource.GetPitch();
            params.dstDevice = mDevPtr;
            params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            params.dstPitch = mPitch;
            params.Height = mHeight;
            params.WidthInBytes = mWidthInBytes;

            cudaSafeCall(cuMemcpy2D(&params));
        }
        else
        {
            cudaSafeCall(cuMemcpyDtoD(mDevPtr, aSource.mDevPtr, mSizeInBytes));
        }
    }

    void CudaPitchedDeviceVariable::CopyHostToDevice(void* aSource)
    {
        if (IsReallyPitched())
        {
            CUDA_MEMCPY2D params;
            memset(&params, 0, sizeof(params));
            params.srcHost = aSource;
            //params.srcPitch = mWidthInBytes;
            params.srcMemoryType = CU_MEMORYTYPE_HOST;
            params.dstDevice = mDevPtr;
            params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            params.dstPitch = mPitch;
            params.Height = mHeight;
            params.WidthInBytes = mWidthInBytes;

            cudaSafeCall(cuMemcpy2D(&params));
        }
        else
        {
            cudaSafeCall(cuMemcpyHtoD(mDevPtr, aSource, mSizeInBytes));
        }
    }
    void CudaPitchedDeviceVariable::CopyDeviceToHost(void* aDest)
    {
        if (IsReallyPitched())
        {
            CUDA_MEMCPY2D params;
            memset(&params, 0, sizeof(params));
            params.srcDevice = mDevPtr;
            params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            params.srcPitch = mPitch;
            params.dstHost = aDest;
            params.dstMemoryType = CU_MEMORYTYPE_HOST;
            //params.dstPitch = mWidthInBytes;
            params.Height = mHeight;
            params.WidthInBytes = mWidthInBytes;


            cudaSafeCall(cuMemcpy2D(&params));
        }
        else
        {
            cudaSafeCall(cuMemcpyDtoH(aDest, mDevPtr, mSizeInBytes));
        }
    }

    size_t CudaPitchedDeviceVariable::GetSize()
    {
        return mSizeInBytes;
    }

    CUdeviceptr CudaPitchedDeviceVariable::GetDevicePtr()
    {
        return mDevPtr;
    }

    size_t CudaPitchedDeviceVariable::GetPitch()
    {
        return mPitch;
    }

    uint CudaPitchedDeviceVariable::GetElementSize()
    {
        return mElementSize;
    }

    size_t CudaPitchedDeviceVariable::GetWidth()
    {
        return mWidthInBytes / mElementSize;
    }

    size_t CudaPitchedDeviceVariable::GetWidthInBytes()
    {
        return mWidthInBytes;
    }

    size_t CudaPitchedDeviceVariable::GetHeight()
    {
        return mHeight;
    }

    void CudaPitchedDeviceVariable::Memset(uchar aValue)
    {
        cudaSafeCall(cuMemsetD8(mDevPtr, aValue, mSizeInBytes));
    }


    //CudaPageLockedHostVariable
    CudaPageLockedHostVariable::CudaPageLockedHostVariable(size_t aSizeInBytes, uint aFlags)
            : mSizeInBytes(aSizeInBytes), mHostPtr(0)
    {
        cudaSafeCall(cuMemHostAlloc(&mHostPtr, mSizeInBytes, aFlags));
    }

    CudaPageLockedHostVariable::~CudaPageLockedHostVariable()
    {
        //cudaSafeCall(cuMemFreeHost(mHostPtr));
        cuMemFreeHost(mHostPtr);
    }

    size_t CudaPageLockedHostVariable::GetSize()
    {
        return mSizeInBytes;
    }

    void* CudaPageLockedHostVariable::GetHostPtr()
    {
        return mHostPtr;
    }
}
#endif //USE_CUDA
