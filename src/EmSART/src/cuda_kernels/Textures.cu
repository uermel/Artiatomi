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


#ifndef TEXTURES_CU
#define TEXTURES_CU

#include <cuda.h>
#include "Constants.h"

texture< VOXEL, 3, cudaReadModeNormalizedFloat > t_dataset;
texture< float4, 1, cudaReadModeElementType > t_transfer_function;
texture< float4, 1, cudaReadModeElementType > t_pre_integration_1D;
texture< float4, 2, cudaReadModeElementType > t_pre_integration_2D;
//texture<float4, 1, cudaReadModeElementType> transferTex;

//__device__ cudaArray *t_array_dataset;
cudaArray* t_array_dataset;
cudaArray* t_array_transfer_function;
cudaArray* t_array_pre_integration_1D;
cudaArray* t_array_pre_integration_2D;

#endif
