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


#include "BufferRequestSet.h"

BufferRequestSet::BufferRequestSet()
{
}

BufferRequestSet::BufferRequestSet(const BufferRequestSet& copy)
	: mBufferRequests(copy.mBufferRequests)
{
	
}

void BufferRequestSet::AddBufferRequest(std::shared_ptr<BufferRequest> request)
{
	mBufferRequests.push_back(request);
}

std::vector<std::shared_ptr<BufferRequest> > BufferRequestSet::GetRequests()
{
	return mBufferRequests;
}

void BufferRequestSet::Clear()
{
	mBufferRequests.clear();
}

void BufferRequestSet::PopBack()
{
	mBufferRequests.pop_back();
}
