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


#ifndef CONFIGEXCEPTIONS_H
#define CONFIGEXCEPTIONS_H

#include "UtilsDefault.h"

using namespace std;

namespace Configuration
{
	//! Baseclass for exceptions occuring while processing a configuration file
	//Baseclass for exceptions occuring while processing a configuration file
	class ConfigException: public exception
	{
	protected:

	public:
		ConfigException();

		~ConfigException() throw();

		virtual const char* what() const throw();

		virtual string GetMessage();
	};
	
	//! Thrown when a value in the configuration file has the wrong value / type
	//Thrown when a value in the configuration file has the wrong value / type
	class ConfigValueException: public ConfigException
	{
		private:
			string mConfigFile, mConfigEntry, mType;

		public:
			ConfigValueException();

			~ConfigValueException() throw();

			ConfigValueException(string aConfigFile, string aConfigEntry, string aType);

			virtual const char* what() const throw();

			void setValue(string aConfigFile, string aConfigEntry, string aType);

			virtual string GetMessage() const throw();
	};
	
	//! Thrown when a missing property is fetched
	//Thrown when a missing property is fetched
	class ConfigPropertyException: public ConfigException
	{
		private:
			string mConfigFile, mConfigEntry;

		public:
			ConfigPropertyException();

			~ConfigPropertyException() throw();

			ConfigPropertyException(string aConfigFile, string aConfigEntry);

			virtual const char* what() const throw();

			void setValue(string aConfigFile, string aConfigEntry);

			virtual string GetMessage() const throw();
	};
	
	//! Thrown when the configuration file can't be read
	//Thrown when the configuration file can't be read
	class ConfigFileException: public ConfigException
	{
		private:
			string mConfigFile;

		public:
			ConfigFileException();

			~ConfigFileException() throw();

			ConfigFileException(string aConfigFile);

			virtual const char* what() const throw();

			void setValue(string aConfigFile);

			virtual string GetMessage() const throw();
	};
} //end namespace Configuration
#endif //CONFIGEXCEPTIONS_H