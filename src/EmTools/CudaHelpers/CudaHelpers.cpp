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


#include "CudaHelpers.h"
#include "../MKLog/MKLog.h"

#ifdef USE_CUDA
namespace Cuda
{
	CudaContext* CudaThreadBoundContext::Get(int deviceID)
	{
		std::lock_guard<std::recursive_mutex> lock(_mutex);

		if (!_instance)
		{
			_instance = new CudaThreadBoundContext();
			MKLOG("CudaThreadBoundContext created.");
			auto ret = RunInCudaThread(CudaContext::CreateInstance, deviceID, CU_CTX_SCHED_AUTO);
			_instance->_ctx = ret.get();
			MKLOG("Cuda context created on device %d.", deviceID);
		}

		return _instance->_ctx;
	}

	void CudaThreadBoundContext::Cleanup()
	{
		std::lock_guard<std::recursive_mutex> lock(_mutex);
		RunInCudaThread(CudaContext::DestroyContext, Get());
		MKLOG("Cuda context destoyed.");
		delete _instance;
		_instance = NULL;
		MKLOG("CudaThreadBoundContext deleted.");
	}

	void CudaThreadBoundContext::SetCurrent()
	{
		Get()->SetCurrent();
	}

	CudaThreadBoundContext* CudaThreadBoundContext::_instance = NULL;
	std::recursive_mutex CudaThreadBoundContext::_mutex;
}
#endif //USE_CUDA