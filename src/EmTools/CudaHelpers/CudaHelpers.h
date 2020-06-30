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


#ifndef CUDAHELPERS_H
#define CUDAHELPERS_H

#include "../Basics/Default.h"

#ifdef USE_CUDA
#include "CudaDefault.h"
#include "../Threading/ThreadPool.h"
#include "../Threading/SpecificBackgroundThread.h"
#include "CudaContext.h"
#include "CudaArrays.h"
#include "CudaDeviceProperties.h"
#include "CudaException.h"
#include "CudaKernel.h"
#include "CudaTextures.h"
#include "CudaVariables.h"
#include <mutex>

namespace Cuda
{
	class CudaThreadBoundContext
	{
	public:
		static CudaContext* Get(int deviceID = 0);
		static void Cleanup();

		template<class F1, class... Args1> static inline auto Run(F1&& f, Args1&&... args)
		{
			RunInCudaThread(SetCurrent);
			return SingletonThread::Get(CUDA_THREAD_ID)->enqueue(f, args...);
		}

	private:
		static std::recursive_mutex _mutex;
		static CudaThreadBoundContext* _instance;
		static void SetCurrent();
		CudaContext* _ctx;

		CudaThreadBoundContext() = default;
		CudaThreadBoundContext(const CudaThreadBoundContext&) = delete;
		CudaThreadBoundContext(CudaThreadBoundContext&&) = delete;
	};
}


#endif //USE_CUDA
#endif // !CUDAHELPERS_H
