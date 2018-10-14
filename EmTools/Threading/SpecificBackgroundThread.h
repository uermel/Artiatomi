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


#ifndef SPECIFICBACKGROUNDTHREAD_H
#define SPECIFICBACKGROUNDTHREAD_H

#include "ThreadPool.h"
#include <mutex>

#define RunInThread(a, ...) (__runInThread(10, a, __VA_ARGS__))
#define RunInCudaThread(a, ...) (__runInThread(CUDA_THREAD_ID, a, __VA_ARGS__))
#define RunInOpenCLThread(a, ...) (__runInThread(OPENCL_THREAD_ID, a, __VA_ARGS__))

class SingletonThread
{
public:
	static ThreadPool* Get(size_t id);

private:
	static std::recursive_mutex _mutex;
	static std::vector<std::pair<size_t, ThreadPool*> > threadPools;
	static SingletonThread* _instance;

	SingletonThread(const SingletonThread&) = delete;
	SingletonThread(SingletonThread&&) = delete;

	SingletonThread();
};


template<class F1, class... Args1> inline auto __runInThread(size_t id, F1&& f, Args1&&... args)
{
	/*using return_type = typename std::result_of<F1(Args1...)>::type;

	auto task = std::make_shared< std::packaged_task<return_type()> >(
		std::bind(std::forward<F1>(f), std::forward<Args1>(args)...)
		);

	std::future<return_type> res = task->get_future();
	(*task)();
	return res;*/
	return SingletonThread::Get(id)->enqueue(f, args...);
}
#endif
