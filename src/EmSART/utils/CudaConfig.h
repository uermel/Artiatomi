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


#ifndef CUDACONFIG_H
#define CUDACONFIG_H

#include "UtilsDefault.h"
#include <map>
#include <list>
#include "ConfigExceptions.h"

using namespace std;

namespace Configuration
{
	//! Parse structured config files
	/*!
		Config files contains lines with name-value assignements in the form "<name> = <value>".
	   Trailing and leading whitespace is stripped. Parsed config entries are stored in
	   a symbol map.

	   Lines beginning with '#' are a comment and ignored.

	   Config files may be structured (to arbitrary depth). To start a new config sub group
	   (or sub section) use a line in the form of "<name> = (".
	   Subsequent entries are stured in the sub group, until a line containing ")" is found.

	   Values may reuse already defined names as a variable which gets expanded during
	   the parsing process. Names for expansion are searched from the current sub group
	   upwards. Finally the process environment is searched, so also environment
	   variables may be used as expansion symbols in the config file.
	*/
	//Parse structured config files
	class CudaConfig {
		public:
	    private:
			//! Parse config file aConfigFile
			/*!
				If the process environment
				is provided, environment variables can be used as expansion symbols.
			*/
			//Parse config file 'aConfigFile'
			CudaConfig(string aConfigFile, int argc, char** argv, char** appEnvp = 0);
			static CudaConfig* config;

		public:
			uint	FPKernelMaxReg;
			string	FPKernelName;
			dim3	FPBlockSize;
			uint	SlicerKernelMaxReg;
			string	SlicerKernelName;
			dim3	SlicerBlockSize;
			uint	VolTravLenKernelMaxReg;
			string	VolTravLenKernelName;
			dim3	VolTravLenBlockSize;
			uint	CompKernelMaxReg;
			string	CompKernelName;
			dim3	CompBlockSize;
			uint	BPKernelMaxReg;
			string	BPKernelName;
			dim3	BPBlockSize;
			bool	CompilerOutput;
			bool	InfoOutput;


            static CudaConfig& GetConfig();
            static CudaConfig& GetConfig(string aConfigFile, int argc, char** argv, char** appEnvp = 0);

			~CudaConfig();

			//! Get string config entry
			//Get string config entry
			string GetString(string aName);

			//! get boolean config entry
			/*!
				A value of Yes/yes/YES/true/True/TRUE leads to true,
				all other values leads to false.
			*/
			//get boolean config entry
			bool GetBool(string aName);

			//! get double config entry; value is parsed using stringstream
			// get double config entry; value is parsed using stringstream
			double GetDouble(string name);

			//! get float config entry; value is parsed using stringstream
			// get float config entry; value is parsed using stringstream
			float GetFloat(string aName);

			//! get int config entry; value is parsed using stringstream
			// get int config entry; value is parsed using stringstream
			int GetInt(string aName);

			//! get dim3 config entry; value is parsed using stringstream
			// get dim3 config entry; value is parsed using stringstream
			dim3 GetDim3(string aName);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			float3 GetFloat3(string aName);

			//! get the symbol map (e.g. for iterating over all symbols)
			// get the symbol map (e.g. for iterating over all symbols)
			inline map<string, string>& GetSymbols() {
				return symbols;
			}

			//! get config sub group
			// get config sub group
			inline CudaConfig* GetGroup(string aName) {
				return groups[aName];
			}

			//! get config sub group map (e.g. for iterating over all groups)
			// get config sub group map (e.g. for iterating over all groups)
			inline map<string, CudaConfig*>& GetGroups() {
				return groups;
			}

		private:
			// private constructor for sub groups
			CudaConfig(string name, string parentDebugInfo);

			// helper functions for parsing
			void add(string name, string value);
			void split(string in, string& left, string& right, char c);
			void trim(string& s);
			void symbolExpand(string& s);
			void symbolExpand(map<string, string>& symbols, string& s);
			void envSymbolExpand(string& s);
			void replaceChar(string& str, char replace, char by);

			// config group symbol map
			map<string, string> symbols;

			// environment symbol map
			map<string, string> envSymbols;

			// config sub group map
			map<string, CudaConfig*> groups;

			// stack of config groups for parsing (only used in top config element)
			list<CudaConfig*> groupStack;

			// debug info used for logging messages
			string mDebugInfo;

			string mConfigFileName;
	};
} //end namespace Configuration

#endif //CONFIG_H
