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


#include "CudaException.h"
#include <sstream>

#ifdef USE_CUDA
Cuda::CudaException::CudaException()
	: mFileName(), mMessage(), mLine(0)
{

}

Cuda::CudaException::~CudaException() throw()
{

}

Cuda::CudaException::CudaException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine()
{
	stringstream ss;
	ss << "CUDA error: " << mMessage << ".";
	mMessage = ss.str();
}

Cuda::CudaException::CudaException(string aFileName, int aLine, string aMessage, CUresult aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{
	stringstream ss;
	ss << "CUDA Driver API error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	mMessage = ss.str();
}

const char* Cuda::CudaException::what() const throw()
{
	return mMessage.c_str();
}

string Cuda::CudaException::GetMessage() const
{
	return mMessage;
}






Cuda::CufftException::CufftException()
	: mFileName(), mMessage(), mLine(0)
{

}

Cuda::CufftException::~CufftException() throw()
{

}

Cuda::CufftException::CufftException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine()
{
	stringstream ss;
	ss << "CUDA CUFFT error: " << mMessage << ".";
	mMessage = ss.str();
}

Cuda::CufftException::CufftException(string aFileName, int aLine, string aMessage, cufftResult aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{
	stringstream ss;
	ss << "CUDA CUFFT error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	mMessage = ss.str();
}

const char* Cuda::CufftException::what() const throw()
{
	return mMessage.c_str();
}

string Cuda::CufftException::GetMessage() const
{
	return mMessage;
}






Cuda::NppException::NppException()
	: mFileName(), mMessage(), mLine(0)
{

}

Cuda::NppException::~NppException() throw()
{

}

Cuda::NppException::NppException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine()
{
	stringstream ss;
	ss << "CUDA NPP error: " << mMessage << ".";
	mMessage = ss.str();
}

Cuda::NppException::NppException(string aFileName, int aLine, string aMessage, NppStatus aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{
	stringstream ss;
	ss << "CUDA NPP error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	mMessage = ss.str();
}

const char* Cuda::NppException::what() const throw()
{
	return mMessage.c_str();
}

string Cuda::NppException::GetMessage() const
{
	return mMessage;
}





Cuda::CusolverException::CusolverException()
	: mFileName(), mMessage(), mLine(0)
{

}

Cuda::CusolverException::~CusolverException() throw()
{

}

Cuda::CusolverException::CusolverException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine()
{
	stringstream ss;
	ss << "CUDA CUSOLVER error: " << mMessage << ".";
	mMessage = ss.str();
}

Cuda::CusolverException::CusolverException(string aFileName, int aLine, string aMessage, cusolverStatus_t aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{
	stringstream ss;
	ss << "CUDA SOLVER error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	mMessage = ss.str();
}

const char* Cuda::CusolverException::what() const throw()
{
	return mMessage.c_str();
}

string Cuda::CusolverException::GetMessage() const
{
	return mMessage;
}
#endif