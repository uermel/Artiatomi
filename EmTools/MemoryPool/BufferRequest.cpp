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


#include "BufferRequest.h"


BufferRequest::BufferRequest(BufferType_enum aBufferType, DataType_enum aDataType, size_t aWidth, size_t aHeight, size_t aDepth) :
	mBufferType(aBufferType),
	mRequestedSizeInBytes(GetDataTypeSize(aDataType) * aWidth * aHeight * aDepth),
	mTypeSize(GetDataTypeSize(aDataType)),
	mDataType(aDataType),
	mPtr(NULL),
	mAllocatedPitch(0),
	mAllocatedSizeInBytes(0),
	mWidth(aWidth),
	mHeight(aHeight),
	mDepth(aDepth)
{

}

BufferRequest::BufferRequest(const BufferRequest& copy)
{
	mBufferType = copy.mBufferType;
	mRequestedSizeInBytes = copy.mRequestedSizeInBytes;
	mTypeSize = copy.mTypeSize;
	mDataType = copy.mDataType;
	mPtr = copy.mPtr;
	mAllocatedPitch = copy.mAllocatedPitch;
	mAllocatedSizeInBytes = copy.mAllocatedSizeInBytes;
	mWidth = copy.mWidth;
	mHeight = copy.mHeight;
	mDepth = copy.mDepth;
}

BufferRequest::BufferRequest(BufferRequest&& move)
{
	mBufferType = move.mBufferType;
	mRequestedSizeInBytes = move.mRequestedSizeInBytes;
	mTypeSize = move.mTypeSize;
	mDataType = move.mDataType;
	mPtr = move.mPtr;
	mAllocatedPitch = move.mAllocatedPitch;
	mAllocatedSizeInBytes = move.mAllocatedSizeInBytes;
	mWidth = move.mWidth;
	mHeight = move.mHeight;
	mDepth = move.mDepth;
}