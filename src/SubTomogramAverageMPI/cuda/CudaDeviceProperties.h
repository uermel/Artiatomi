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


#ifndef CUDADEVICEPROPERTIES_H
#define CUDADEVICEPROPERTIES_H

#include "CudaDefault.h"
#include "CudaException.h"
#include "CudaArrays.h"

namespace Cuda
{
    //!  Retrieves the device properties of a CUDA device. 
    /*!
      \author Michael Kunz
      \date   January 2010
      \version 1.0
    */
    //Retrieves the device properties of a CUDA device. 
    class CudaDeviceProperties
    {
    private:
        int mDeviceID;
        int mClockRate;
        int mMaxBlockDim[3];
        int mMaxGridDim[3];
        int mMaxThreadsPerBlock;
        int mMemPitch;
        int mRegsPerBlock;
        int mSharedMemPerBlock;
        int mTextureAlign;
        int mTotalConstantMemory;
        string mDeviceName;
        float mComputeCapability;
        float mDriverVersion;
        size_t mTotalGlobalMemory;
        int mMultiProcessorCount;
        int mWarpSize;
        int mGpuOverlap;
        int mKernelExecTimeoutEnabled;
        int mIntegrated;
        int mCanMapHostMemory;
        CUcomputemode mComputeMode;
        int mMaximumTexture1DWidth;
        int mMaximumTexture2DWidth;
        int mMaximumTexture2DHeight;
        int mMaximumTexture3DWidth;
        int mMaximumTexture3DHeight;
        int mMaximumTexture3DDepth;
        int mMaximumTexture2DArrayWidth;
        int mMaximumTexture2DArrayHeight;
        int mMaximumTexture2DArrayNumSlices;
        int mSurfaceAllignment;
        int mConcurrentKernels;
        int mECCEnabled;
        int mPCIBusID;
        int mPCIDeviceID;
        int mTCCDriver;
        int mMemoryClockRate;
        int mGlobalMemoryBusWidth;
        int mL2CacheSize;
        int mMaxThreadsPerMultiProcessor;
        int mAsyncEngineCount;
        int mUnifiedAddressing;
        int mMaximumTexture1DLayeredWidth;
        int mMaximumTexture1DLayeredLayers;
        int mPCIDomainID;

    public:	
        //! CudaDeviceProperties constructor
        /*!
            While instantiation, CudaDeviceProperties retrieves the device properties using the CUDA Driver API 
            \param aDevice The CUDA Device 
            \param aDeviceID The ID of the CUDA Device to use
        */
        //CudaDeviceProperties constructor
        CudaDeviceProperties(CUdevice aDevice, int aDeviceID);
        
        //! CudaDeviceProperties destructor
        //CudaDeviceProperties destructor
        ~CudaDeviceProperties();

        //! Peak clock frequency in kilohertz
        int GetClockRate();

        //! Maximum block dimensions
        int* GetMaxBlockDim();

        //! Maximum grid dimensions
        int* GetMaxGridDim();

        //! Maximum number of threads per block
        int GetMaxThreadsPerBlock();

        //! Maximum pitch in bytes allowed by memory copies
        int GetMemPitch();

        //! Maximum number of 32-bit registers available per block
        int GetRegsPerBlock();

        //! Maximum shared memory available per block in bytes 
        int GetSharedMemPerBlock();

        //! Alignment requirement for textures
        int GetTextureAlign();

        //! Constant memory available on device in bytes
        int GetTotalConstantMemory();

        //! Name of the device
        string GetDeviceName();

        //! Compute capability version 
        float GetComputeCapability();

        //! CUDA Driver API version
        float GetDriverVersion();

        //! Size of the device memory in bytes
        size_t GetTotalGlobalMemory();

        //! Number of multiprocessors on device
        int GetMultiProcessorCount();

        //! Warp size in threads
        int GetWarpSize();

        //! Device can possibly copy memory and execute a kernel concurrently
        bool GetGpuOverlap();

        //! Specifies whether there is a run time limit on kernels
        bool GetGernelExecTimeoutEnabled();

        //! Device is integrated with host memory
        bool GetIntegrated();

        //! Device can map host memory into CUDA address space
        bool GetCanMapHostMemory();

        //! Compute mode
        CUcomputemode GetComputeMode();

        //! Maximum 1D texture width
        int GetMaximumTexture1DWidth();

        //! Maximum 2D texture width
        int GetMaximumTexture2DWidth();

        //! Maximum 2D texture height
        int GetMaximumTexture2DHeight();

        //! Maximum 3D texture width 
        int GetMaximumTexture3DWidth();

        //! Maximum 3D texture height
        int GetMaximumTexture3DHeight();

        //! Maximum 3D texture depth
        int GetMaximumTexture3DDepth();

        //! Maximum texture array width
        int GetMaximumTexture2DArrayWidth();

        //! Maximum texture array height
        int GetMaximumTexture2DArrayHeight();

        //! Maximum slices in a texture array 
        int GetMaximumTexture2DArrayNumSlices();

        //! Alignment requirement for surfaces 
        int GetSurfaceAllignment();

        //! Device can possibly execute multiple kernels concurrently
        bool GetConcurrentKernels();

        //! Device has ECC support enabled 
        bool GetECCEnabled();

        //! PCI bus ID of the device 
        int GetPCIBusID();

        //! PCI device ID of the device
        int GetPCIDeviceID();

        //! Device is using TCC driver model 
        bool GetTCCDriver();
                
        //! Peak memory clock frequency in kilohertz
        int GetMemoryClockRate();
                
        //! Global memory bus width in bits
        int GetGlobalMemoryBusWidth();
                
        //! Size of L2 cache in bytes 
        int GetL2CacheSize();
                
        //! Maximum resident threads per multiprocessor
        int GetMaxThreadsPerMultiProcessor();
                
        //! Number of asynchronous engines
        int GetAsyncEngineCount();
                
        //! Device shares a unified address space with the host 
        bool GetUnifiedAddressing();
                
        //! Maximum 1D layered texture width
        int GetMaximumTexture1DLayeredWidth();
                
        //! Maximum layers in a 1D layered texture 
        int GetMaximumTexture1DLayeredLayers();
                
        //! PCI domain ID of the device
        int GetPCIDomainID();


        //! Prints detailed information on the CUdevice to std::cout
        void PrintProperties();
    };
}

#endif //CUDADEVICEPROPERTIES_H