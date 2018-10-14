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


#include "OpenCLException.h"
#include <sstream>

OpenCL::OpenCLException::OpenCLException()
	: mFileName(), mMessage(), mLine(0), mErr(-1)
{

}

OpenCL::OpenCLException::~OpenCLException() throw()
{

}

OpenCL::OpenCLException::OpenCLException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine(), mErr(-1)
{
	stringstream ss;
	ss << "OpenCL error: ";
	ss << aMessage << ".";
	mMessage = ss.str();
}

OpenCL::OpenCLException::OpenCLException(string aFileName, int aLine, string aMessage, cl_int aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{
	stringstream ss;
	ss << "OpenCL API error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";

	mMessage = ss.str();
}

const char* OpenCL::OpenCLException::what() const throw()
{
	return mMessage.c_str();
}

//string OpenCL::OpenCLException::GetMessage() const
//{
//	return mMessage;
//}

//string OpenCL::OpenCLException::GetMessageW() const
//{
//	return GetMessage();
//}


