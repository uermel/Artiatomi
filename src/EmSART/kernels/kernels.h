//
// Created by uermel on 9/30/21.
//

#ifndef ARTIATOMI_KERNELS_H
#define ARTIATOMI_KERNELS_H

// Projection system
#include "FPKernel.h"
#include "SlicerKernel.h"
#include "VolTravLengthKernel.h"
#include "FPLUTKernel.h"
#include "FPLUTSlicedKernel.h"
#include "FPDistSlicedKernel.h"
#include "FPDistOrthoKernel.h"
#include "FPOrthoKernel.h"
#include "BPKernel.h"
#include "BPLUTKernel.h"
#include "BPLUTBWKernel.h"
#include "BPLUTBWCGKernel.h"
#include "BPLUTBlockKernel.h"
#include "BPLUTBlockNoDivKernel.h"
#include "BPLUTVBlockSlicedKernel.h"
#include "BPLUTSlicedKernel.h"
#include "BPOrthoSlicedKernel.h"

// 2D Image processing
#include "CompKernel.h"
#include "ConjKernel.h"
#include "CopyToSquareKernel.h"
#include "RectToSqrSlice.h"
#include "CopyToRectSlicesKernel.h"
#include "CropBorderKernel.h"
#include "CTFKernel.h"
#include "CTFSlicedKernel.h"
#include "DimBordersKernel.h"
#include "DoseWeightingKernel.h"
#include "FindPeakKernel.h"
#include "FourFilterKernel.h"
#include "MaxShiftKernel.h"
#include "MaxShiftWeightedKernel.h"
#include "PCKernel.h"
#include "SubEKernel.h"
#include "WbpWeightingKernel.h"
#include "SplinePrefilter.h"
#include "OversampleKernel.h"
#include "ComputeLUTKernel.h"
#include "FFTShiftKernel.h"
#include "RadialSumKernel.h"
#include "FreqSampleKernel.h"

#include "RectToSqrSlice.h"
#include "SqrSliceToRectSliceKernel.h"
#include "RectSliceToSqrSliceKernel.h"
#include "SqrSliceToRectKernel.h"
#include "CropSlicesKernel.h"

// 3D Image processing
#include "ApplyMaskKernel.h"
#include "ConvVolKernel.h"
#include "ConvVol3DKernel.h"
#include "RestoreVolumeKernel.h"
#include "RotKernel.h"
#include "SphericalMaskKernel.h"
#include "CubicResampleKernel.h"

#endif //ARTIATOMI_KERNELS_H
