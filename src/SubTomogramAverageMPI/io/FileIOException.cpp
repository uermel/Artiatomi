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


#include "FileIOException.h"

FileIOException::FileIOException()
	: mFileName(), mMessage()
{

}

FileIOException::~FileIOException() throw()
{

}

FileIOException::FileIOException(string aMessage)
	: mFileName(), mMessage(aMessage)
{

}

FileIOException::FileIOException(string aFileName, string aMessage)
	: mFileName(aFileName), mMessage(aMessage)
{

}

const char* FileIOException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string FileIOException::GetMessage() const throw()
{
	if (mFileName.length() == 0 && mMessage.length() == 0)
		return "FileIOException";
	if (mFileName.length() == 0 && mMessage.length() > 0)
		return mMessage;

	stringstream ss;
	ss << "Could not access file '";
	ss << mFileName << "'. " << mMessage << endl;
	return ss.str();
}