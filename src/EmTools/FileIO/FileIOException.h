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


#ifndef FILEIOEXCEPETION_H
#define FILEIOEXCEPETION_H

#include "../Basics/Default.h"
#include <sstream>

using namespace std;

//!  An exception thrown while accessing Files. 
/*!
FileIOException is thrown, if a file cannot be accessed as intended.
\author Michael Kunz
\date   September 2011
\version 1.0
*/
class FileIOException : public exception
{
protected:
	string mFileName;
	string mMessage;

public:
	FileIOException();

	~FileIOException() throw();


	//! FileIOException constructor
	/*!
	\param aMessage Ecxeption message
	*/
	//FileIOException constructor
	FileIOException(string aMessage);

	//! FileIOException constructor
	/*!
	\param aFileName Name of the file provoking the exception
	\param aMessage Ecxeption message
	*/
	//FileIOException constructor
	FileIOException(string aFileName, string aMessage);

	//! Returns "FileIOException"
	//Returns "FileIOException"
	virtual const char* what() const throw();

	//! Returns an error message
	//Returns an error message
	virtual string GetMessage() const throw();
};

#endif