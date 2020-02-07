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

#include "UtilsDefault.h"
#include <map>
#include <list>
#include "ConfigExceptions.h"
#include "../Kernels.h"
#include <MotiveListe.h>

using namespace std;

//#define REFINE_MODE 1

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
	class Config {
		public:
		enum FILE_SAVE_MODE
		{
			FSM_RAW,
			FSM_EM,
			FSM_MRC
		};
		enum CTF_MODE
		{
			CTFM_YES,
			CTFM_NO
		};
		enum FILE_READ_MODE
		{
			FRM_DM4,
			FRM_MRC
		};
		enum PROJ_NORMALIZATION_MODE
		{
			PNM_MEAN,
			PNM_STANDARD_DEV,
			PNM_NONE
		};

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
			vector<int>	CudaDeviceIDs;
			string	ProjectionFile;
			string	OutVolumeFile;
			string	MarkerFile;
			float	Lambda;
			float	AddTiltAngle;
			float	AddTiltXAngle;
			int		Iterations;
			dim3	RecDimensions;
			bool    UseFixPsiAngle;
			float   PsiAngle;
			float3  VolumeShift;
			int		ReferenceMarker;
			int		OverSampling;
			float	PhiAngle;
			float3  VoxelSize;
			float2  DimLength;
			float2  CutLength;
			float4  Crop;
			float4  CropDim;
			bool	SkipFilter;
			int     fourFilterLP;
			int     fourFilterLPS;
			int     fourFilterHP;
			int     fourFilterHPS;
			int     SIRTCount;
			//float   ProjNormVal;
			//bool    Filtered;
			CTF_MODE CtfMode;
			string  CtfFile;
			float	BadPixelValue;
			bool	CorrectBadPixels;
			float4	CTFBetaFac;
			bool	FP16Volume;
			bool	WriteVolumeAsFP16;
			float	ProjectionScaleFactor;
			PROJ_NORMALIZATION_MODE ProjectionNormalization;
			bool	WBP_NoSART;
			FilterMethod WBPFilter;
			float	Cs;
			float	Voltage;
			float	MagAnisotropyAmount;
			float	MagAnisotropyAngleInDeg;
			bool	IgnoreZShiftForCTF;
			float	CTFSliceThickness;
			bool	DebugImages;
			bool	DownWeightTiltsForWBP;
			bool	DoseWeighting;
			vector<float> AccumulatedDose;
			bool	SwitchCTFDirectionForIMOD;
			bool	PhaseFlipOnly;
			float	WienerFilterNoiseLevel;
			
#ifdef REFINE_MODE
			int SizeSubVol;
			float VoxelSizeSubVol;
			string MotiveList;
			float ScaleMotivelistShift;
			float ScaleMotivelistPosition;
			string Reference;
			int MaxShift;
			string ShiftOutputFile;
			MotiveList::GroupMode_enum GroupMode;
			int GroupSize;
			float MaxDistance;
			string CCMapFileName;
			float SpeedUpDistance;
			vector<string> SupportingReferences;
			vector<string> SupportingMotiveLists;
			float MaxDistanceSupport;
			bool MultiPeakDetection;
			int TomogramIndex;
#endif

#ifdef SUBVOLREC_MODE
			int SizeSubVol;
			float VoxelSizeSubVol;
			string MotiveList;
			float ScaleMotivelistShift;
			float ScaleMotivelistPosition;
			string SubVolPath;
			string ShiftInputFile;
			int BatchSize;
			int MaxShift;
			MotiveList::NamingConvention_enum NamingConv;
#endif

            static Config& GetConfig();
            static Config& GetConfig(string aConfigFile, int argc, char** argv, int mpiPart, char** appEnvp = 0);

			~Config();

			FILE_SAVE_MODE GetFileSaveMode();

			FILE_READ_MODE GetFileReadMode();

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

			//! get dim3 config entry; value is parsed using stringstream
			// get dim3 config entry; value is parsed using stringstream
			dim3 GetDim3(string aName);

			//! get float2 config entry; value is parsed using stringstream
			// get float2 config entry; value is parsed using stringstream
			float2 GetFloat2(string aName);

			//! get float2 config entry; value is parsed using stringstream
			// get float2 config entry; value is parsed using stringstream
			float2 GetFloat2(string aName, float2 defaultVal);

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
			vector<int> GetVectorInt(string aName);

			//! get float3 config entry; value is parsed using stringstream
			// get float3 config entry; value is parsed using stringstream
			vector<float> GetVectorFloat(string aName);

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
