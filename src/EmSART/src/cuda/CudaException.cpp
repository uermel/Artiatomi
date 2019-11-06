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

}

Cuda::CudaException::CudaException(string aFileName, int aLine, string aMessage, CUresult aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{

}

const char* Cuda::CudaException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string Cuda::CudaException::GetMessage() const
{
	if (mFileName.length() == 0)
		return mMessage;

	stringstream ss;
	ss << "CUDA Driver API error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	return ss.str();
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

}

Cuda::CufftException::CufftException(string aFileName, int aLine, string aMessage, cufftResult aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{

}

const char* Cuda::CufftException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string Cuda::CufftException::GetMessage() const
{
	if (mFileName.length() == 0)
		return mMessage;

	stringstream ss;
	ss << "CUDA CUFFT error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	return ss.str();
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

}

Cuda::NppException::NppException(string aFileName, int aLine, string aMessage, NppStatus aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{

}

const char* Cuda::NppException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string Cuda::NppException::GetMessage() const
{
	if (mFileName.length() == 0)
		return mMessage;

	stringstream ss;
	ss << "CUDA NPP error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	return ss.str();
}