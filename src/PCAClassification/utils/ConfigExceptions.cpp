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


#include "ConfigExceptions.h"

namespace Configuration
{
	ConfigException::ConfigException()
	{

	}

	ConfigException::~ConfigException() throw()
	{

	}

	const char* ConfigException::what() const throw()
	{
		return "ConfigException";
	}

	string ConfigException::GetMessage()
	{
		return "ConfigException";
	}

	ConfigValueException::ConfigValueException()
		:mConfigFile(), mConfigEntry(), mType()
	{
	
	}
	
	ConfigValueException::~ConfigValueException() throw()
	{

	}

	ConfigValueException::ConfigValueException(string aConfigFile, string aConfigEntry, string aType)
		:mConfigFile(aConfigFile), mConfigEntry(aConfigEntry), mType(aType)
	{
	
	}

	const char* ConfigValueException::what() const throw()
	{
		string str = GetMessage();

		char* cstr = new char [str.size()+1];
		strcpy (cstr, str.c_str());

		return cstr;
	}

	string ConfigValueException::GetMessage() const throw()
	{
		string retVal = "The value for property '";
		retVal += mConfigEntry + "' in file '" + mConfigFile + "' doesn't match it's type. It should be of type '";
		retVal += mType + "'.";
		return retVal;
	}

	void ConfigValueException::setValue(string aConfigFile, string aConfigEntry, string aType)
	{
		mConfigFile = aConfigFile;
		mConfigEntry = aConfigEntry;
		mType = aType;
	}

	ConfigPropertyException::ConfigPropertyException()
		:mConfigFile(), mConfigEntry()
	{
	
	}

	ConfigPropertyException::~ConfigPropertyException() throw()
	{
	
	}

	ConfigPropertyException::ConfigPropertyException(string aConfigFile, string aConfigEntry)
		:mConfigFile(aConfigFile), mConfigEntry(aConfigEntry)
	{
	
	}

	const char* ConfigPropertyException::what() const throw()
	{
		string str = GetMessage();

		char* cstr = new char [str.size()+1];
		strcpy (cstr, str.c_str());

		return cstr;
	}

	string ConfigPropertyException::GetMessage() const throw()
	{
		string retVal = "The property '";
		retVal += mConfigEntry + "' is missing in file '" + mConfigFile + "'.";
		return retVal;
	}

	void ConfigPropertyException::setValue(string aConfigFile, string aConfigEntry)
	{
		mConfigFile = aConfigFile;
		mConfigEntry = aConfigEntry;
	}


	ConfigFileException::ConfigFileException()
		:mConfigFile()
	{
	
	}

	ConfigFileException::~ConfigFileException() throw()
	{
	
	}

	ConfigFileException::ConfigFileException(string aConfigFile)
		:mConfigFile(aConfigFile)
	{
	
	}

	const char* ConfigFileException::what() const throw()
	{
		string str = GetMessage();

		char* cstr = new char [str.size()+1];
		strcpy (cstr, str.c_str());

		return cstr;
	}

	string ConfigFileException::GetMessage() const throw()
	{
		string retVal = "Cannot read the configuration file '";
		retVal += mConfigFile + "'.";
		return retVal;
	}

	void ConfigFileException::setValue(string aConfigFile)
	{
		mConfigFile = aConfigFile;
	}
}
