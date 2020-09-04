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


#ifndef CONFIG_H
#define CONFIG_H

#include "../basics/default.h"
#include <map>
#include <list>
#include "ConfigExceptions.h"

using namespace std;

namespace Configuration
{
	enum Symmetry
	{
		Symmetry_None,
		Symmetry_Rotate180,
		Symmetry_Shift,
		Symmetry_Helical,
		Symmetry_Rotational,
        Symmetry_Transform
	};

	enum NamingConvention
	{
		NC_ParticleOnly, //Particle nr from line 4
		NC_TomogramParticle //Tomogram nr from line 5 and particle nr from line 6
	};

	enum CorrelationMethod
	{
		CM_CrossCorrelation,
		CM_PhaseCorrelation
	};

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
	class Config {
		public:
		
	    private:
			//! Parse config file aConfigFile
			/*!
				If the process environment
				is provided, environment variables can be used as expansion symbols.
			*/
			//Parse config file 'aConfigFile'
			Config(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp = 0);
			static Config* config;
			bool logAllowed;

		public:
			vector<int> CudaDeviceIDs;
			string	MotiveList;
			string	WedgeFile;
			bool SingleWedge;
			//vector<int> WedgeIndices;
			string	Particles;
			string	Path;
			vector<string>	Reference;
			string	Mask;
			string	MaskCC;
			int		StartIteration;
			int		EndIteration;
			float	AngIter;
			float	AngIncr;
			float	PhiAngIter;
			float	PhiAngIncr;
			int		LowPass;
			int		HighPass;
			int		Sigma;
			bool	ClearAngles;
			int		ClearAnglesIteration;
			float	BestParticleRatio;
			Symmetry ApplySymmetry;
			string SymmetryFile;
			float3	ShiftSymmetryVector[3];
			bool    MultiReference;
			NamingConvention NamingConv;
			vector<int> Classes;
			bool BinarizeMask;
			bool RotateMaskCC;
			string FilterFileName;
			bool UseFilterVolume;
			bool LinearInterpolation;
			bool CouplePhiToPsi;
			float HelicalRise;
			float HelicalTwist;
			int HelicalRepeatStart;
			int HelicalRepeatEnd; 
			float RotationalAngleStep;
			int RotationalCount;
			float BFactor;
			float PixelSize;
			bool ComputeCCValOnly;
			CorrelationMethod Correlation;
			int CertaintyMaxDistance;


            static Config& GetConfig();
            static Config& GetConfig(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp = 0);

			~Config();

			//! Get string config entry
			//Get string config entry
			string GetString(string aName);

			//! Get string config entry (for optional entries; does not throw exception if not found)
			//Get string config entry (for optional entries; does not throw exception if not found)
			string GetStringOptional(string aName);

			//! get boolean config entry
			/*!
				A value of Yes/yes/YES/true/True/TRUE leads to true,
				all other values leads to false.
			*/
			//get boolean config entry
			bool GetBool(string aName);

			//! get boolean config entry
			/*!
				A value of Yes/yes/YES/true/True/TRUE leads to true,
				all other values leads to false.
			*/
			//get boolean config entry
			bool GetBool(string aName, bool defaultVal);

			//! get double config entry; value is parsed using stringstream
			// get double config entry; value is parsed using stringstream
			double GetDouble(string name);

			//! get float config entry; value is parsed using stringstream
			// get float config entry; value is parsed using stringstream
			float GetFloat(string aName);

			//! get float config entry; value is parsed using stringstream
			// get float config entry; value is parsed using stringstream
			float GetFloat(string aName, float defaultVal);

			//! get int config entry; value is parsed using stringstream
			// get int config entry; value is parsed using stringstream
			int GetInt(string aName);

			//! get int config entry; value is parsed using stringstream
			// get int config entry; value is parsed using stringstream
			int GetInt(string aName, int defaultVal);

			//! get dim3 config entry; value is parsed using stringstream
			// get dim3 config entry; value is parsed using stringstream
			dim3 GetDim3(string aName);

			//! get float2 config entry; value is parsed using stringstream
			// get float2 config entry; value is parsed using stringstream
			float2 GetFloat2(string aName);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			float3 GetFloat3(string aName);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			float3 GetFloatOrFloat3(string aName);

			//! get float4 config entry; value is parsed using stringstream. If value not existent, default value is returned.
			// get float4 config entry; value is parsed using stringstream If value not existent, default value is returned.
			float4 GetFloat4(string aName, float4 defaultVal);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			vector<int> GetVectorInt(string aName, bool optional = false);

			//! get the symbol map (e.g. for iterating over all symbols)
			// get the symbol map (e.g. for iterating over all symbols)
			inline map<string, string>& GetSymbols() {
				return symbols;
			}

			string GetConfigFileName();

			//! get config sub group
			// get config sub group
			inline Config* GetGroup(string aName) {
				return groups[aName];
			}

			//! get config sub group map (e.g. for iterating over all groups)
			// get config sub group map (e.g. for iterating over all groups)
			inline map<string, Config*>& GetGroups() {
				return groups;
			}


		private:
			// private constructor for sub groups
			Config(string name, string parentDebugInfo);

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
			map<string, Config*> groups;

			// stack of config groups for parsing (only used in top config element)
			list<Config*> groupStack;

			// debug info used for logging messages
			string mDebugInfo;

			string mConfigFileName;
	};
} //end namespace Configuration

#endif //CONFIG_H
