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


#ifndef MEMORYEXCEPTION_H
#define MEMORYEXCEPTION_H

#include "../Basics/Default.h"
#include <sstream>

using namespace std;

//!  An exception thrown while memory allocation. 
/*!
MemoryException is thrown, if a memory allcoation fails.
\author Michael Kunz
\date   October 2016
\version 1.0
*/
class MemoryException : public exception
{
protected:
	size_t mAllocSize;

public:
	MemoryException();

	~MemoryException() throw();


	//! FileIOException constructor
	/*!
	\param aMessage Ecxeption message
	*/
	//FileIOException constructor
	MemoryException(size_t aAllocSize);

	//! Returns "FileIOException"
	//Returns "FileIOException"
	virtual const char* what() const throw();

	//! Returns an error message
	//Returns an error message
	virtual string GetMessage() const throw();
};

#endif