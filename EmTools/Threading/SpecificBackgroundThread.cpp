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


#include "SpecificBackgroundThread.h"


SingletonThread* SingletonThread::_instance = NULL;
std::recursive_mutex SingletonThread::_mutex;
std::vector<std::pair<size_t, ThreadPool*> > SingletonThread::threadPools;

ThreadPool* SingletonThread::Get(size_t id)
{
	std::lock_guard<std::recursive_mutex> lock(_mutex);

	if (!_instance)
	{
		_instance = new SingletonThread();
	}

	for (size_t i = 0; i < _instance->threadPools.size(); i++)
	{
		if (_instance->threadPools[i].first == id)
			return _instance->threadPools[i].second;
	}

	_instance->threadPools.push_back(std::pair<size_t, ThreadPool*>(id, new ThreadPool(1)));
	return _instance->threadPools[_instance->threadPools.size() - 1].second;
}

SingletonThread::SingletonThread()
{

}