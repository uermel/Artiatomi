//
// Created by uermel on 10/4/21.
//

//__device__ __constant__ int NbPoles;
//__device__ __constant__ float Poles[4];
#include <cuda.h>
#include <texture_fetch_functions.h>
#include "cutil_math.h"
#include <cfloat>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef signed char schar;

inline __device__ __host__ uint UMIN(uint a, uint b)
{
    return a < b ? a : b;
}

inline __device__ __host__ uint PowTwoDivider(uint n)
{
    if (n == 0) return 0;
    uint divider = 1;
    while ((n & divider) == 0) divider <<= 1;
    return divider;
}

//#define Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline
//#define Pole7_0 -0.53528043079643816554240378168164607183392315234269
//#define Pole7_1 -0.12255461519232669051527226435935734360548654942730
//#define Pole7_2 -0.0091486948096082769285930216516478534156925639545994

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define M_PI_F 3.14159265358979323846f

#ifndef iu
#define iu make_float2(0, 1)
#endif

//--------------------------------------------------------------------------
// Local GPU device procedures
//--------------------------------------------------------------------------

/// Recursive filters for linear memory (in-place)
__host__ __device__ float InitialCausalCoefficient(
        float* c,			// coefficients
        uint DataLength,	// number of coefficients
        float z,            // Pole
        uint Horizon,
        int step)			// element interleave in bytes
{
    // this initialization corresponds to clamping boundaries
    // accelerated loop
    float zn = z;
    float Sum = *c;
    for (uint n = 0; n < Horizon; n++) {
        Sum += zn * *c;
        zn *= z;
        c = (float*)((uchar*)c + step);
    }
    return(Sum);
}

__host__ __device__ float InitialAntiCausalCoefficient(
        float* c,			// last coefficient
        uint DataLength,	// number of samples or coefficients
        float z,
        int step)			// element interleave in bytes
{
    // this initialization corresponds to clamping boundaries
    return((z / (z - 1.0f)) * *c);
}

__host__ __device__ void ConvertToInterpolationCoefficients(
        float* coeffs,		// input samples --> output coefficients
        uint DataLength,	// number of samples or coefficients
        int step,           // element interleave in bytes
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)		// Gain
{

    // Vector to array
    const float z[4] = {Poles.x, Poles.y, Poles.z, Poles.w};
    const int h[4] = {Horizon.x, Horizon.y, Horizon.z, Horizon.w};

    // Loop over all poles
    for (uint k = 0; k < NbPoles; k++){

        if (k > 0) Lambda = 1;
        // causal initialization
        float* c = coeffs;
        float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        *c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, z[k], UMIN(h[k], DataLength), step);

        // causal recursion
        for (uint n = 1; n < DataLength; n++) {
            c = (float*)((uchar*)c + step);
            *c = previous_c = Lambda * *c + z[k] * previous_c;
        }

        // anticausal initialization
        *c = previous_c = InitialAntiCausalCoefficient(c, DataLength, z[k], step);

        // anticausal recursion
        for (int n = DataLength - 2; 0 <= n; n--) {
            c = (float*)((uchar*)c - step);
            *c = previous_c = z[k] * (previous_c - *c);
        }
    }
}

/// Recursive filters for linear memory (in-place)
__host__ __device__ float InitialCausalCoefficientSurf2D(
        CUsurfObject c,	    // coefficients
        int3 coord,
        uint DataLength,	// number of coefficients
        float z,            // Pole
        uint Horizon,
        int3 step)			// step in voxel coords
{
    // this initialization corresponds to clamping boundaries
    // accelerated loop
    float zn = z;
    float Sum;
    float val;
    surf2Dread(&Sum, c, coord.x * 4, coord.y);
    for (uint n = 0; n < Horizon; n++) {
        surf2Dread(&val, c, coord.x * 4, coord.y);
        Sum += zn * val;
        zn *= z;
        coord.x += step.x;
        coord.y += step.y;
        coord.z += step.z;
        //c = (float*)((uchar*)c + step);
    }
    return(Sum);
}

__host__ __device__ float InitialCausalCoefficientSurf3D(
        CUsurfObject c,	    // coefficients
        int3 coord,
        uint DataLength,	// number of coefficients
        float z,            // Pole
        uint Horizon,
        int3 step)			// step in voxel coords
{
    // this initialization corresponds to clamping boundaries
    // accelerated loop
    float zn = z;
    float Sum;
    float val;
    surf3Dread(&Sum, c, coord.x * 4, coord.y, coord.z);
    for (uint n = 0; n < Horizon; n++) {
        surf3Dread(&val, c, coord.x * 4, coord.y, coord.z);
        Sum += zn * val;
        zn *= z;
        coord.x += step.x;
        coord.y += step.y;
        coord.z += step.z;
        //c = (float*)((uchar*)c + step);
    }
    return(Sum);
}

__host__ __device__ float InitialAntiCausalCoefficientSurf(
        float c,			// last coefficient
        uint DataLength,	// number of samples or coefficients
        float z)			// Pole
{
    // this initialization corresponds to clamping boundaries
    return((z / (z - 1.0f)) * c);
}

/// C2IC for linear memory input, cuda array output (in-place AND out-of-place)
__host__ __device__ void ConvertToInterpolationCoefficientsPtr2Surf2D(
        float* coeffs_in,// input samples --> intput coefficients
        CUsurfObject coeffs_out,// input samples --> output coefficients
        int3 coord,
        uint DataLength,	// number of samples or coefficients
        uint stepbytes,     // element interleave in bytes
        int3 step,          // step in coords
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)		// Gain
{

    // Vector to array
    const float z[4] = {Poles.x, Poles.y, Poles.z, Poles.w};
    const int h[4] = {Horizon.x, Horizon.y, Horizon.z, Horizon.w};

    //CUsurfObject coeffs_read = coeffs_in;

    // Loop over all poles
    for (uint k = 0; k < NbPoles; k++){

        // All coeffs need to be multiplied with gain once. This happens below. After first iter, Lambda should thus be
        // 1.
        if (k > 0) Lambda = 1;

        // causal initialization
        float c;
        float* cptr = coeffs_in;
        //surf3Dread(&c, coeffs, coord.x * 4, coord.y, coord.z);
        float previous_c;  //cache the previously calculated c rather than look it up again (faster!)

        // Initially, values should be read from the input array. After the first causal recursion, the output array is
        // filled, and thus values should now be read from there for future recursions.
        if (k == 0) {
            previous_c = Lambda * InitialCausalCoefficient(cptr, DataLength, z[k], UMIN(h[k], DataLength), stepbytes);
            surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
        } else {
            previous_c = Lambda * InitialCausalCoefficientSurf2D(coeffs_out, coord, DataLength, z[k], UMIN(h[k], DataLength), step);
            surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
        }



        // causal recursion
        for (uint n = 1; n < DataLength; n++) {
            // Next coefficient
            // Surf
            coord.x += step.x;
            coord.y += step.y;
            coord.z += step.z;

            // ptr
            cptr = (float*)((uchar*)cptr + stepbytes);

            if (k == 0) {
                c = *cptr;
            } else {
                surf2Dread(&c, coeffs_out, coord.x * 4, coord.y);
            }
            //c = (float*)((uchar*)c + step);
            previous_c = Lambda * c + z[k] * previous_c;
            surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
            //*c = previous_c = Lambda * *c + z[k] * previous_c;
        }

        // anticausal initialization
        previous_c = InitialAntiCausalCoefficientSurf(previous_c, DataLength, z[k]);
        surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
        //*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, z[k], step);

        // Now we start reading from output array instead of input array.
        //coeffs_read = coeffs_out;

        // anticausal recursion
        for (int n = DataLength - 2; 0 <= n; n--) {
            coord.x -= step.x;
            coord.y -= step.y;
            coord.z -= step.z;
            surf2Dread(&c, coeffs_out, coord.x * 4, coord.y);
            //c = (float*)((uchar*)c - step);
            previous_c = z[k] * (previous_c - c);
            surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
            //*c = previous_c = z[k] * (previous_c - *c);
        }
    }
}

/// C2IC for cuda array input, cuda array output (in-place AND out-of-place)
__host__ __device__ void ConvertToInterpolationCoefficientsSurf2D(
        CUsurfObject coeffs_in,// input samples --> intput coefficients
        CUsurfObject coeffs_out,// input samples --> output coefficients (can be same as input)
        int3 coord,
        uint DataLength,	// number of samples or coefficients
        int3 step,          // element interleave in bytes
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)		// Gain
{

    // Vector to array
    const float z[4] = {Poles.x, Poles.y, Poles.z, Poles.w};
    const int h[4] = {Horizon.x, Horizon.y, Horizon.z, Horizon.w};

    // Initially, values should be read from the input array. After the first causal recursion, the output array is
    // filled, and thus values should now be read from there for future recursions.
    CUsurfObject coeffs_read = coeffs_in;

    // Loop over all poles
    for (uint k = 0; k < NbPoles; k++){

        // All coeffs need to be multiplied with gain once. This happens below. After first iter, Lambda should thus be
        // 1.
        if (k > 0) Lambda = 1;

        // causal initialization
        float c;
        //surf3Dread(&c, coeffs, coord.x * 4, coord.y, coord.z);
        float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        previous_c = Lambda * InitialCausalCoefficientSurf2D(coeffs_read, coord, DataLength, z[k], UMIN(h[k], DataLength), step);
        surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);

        // causal recursion
        for (uint n = 1; n < DataLength; n++) {
            coord.x += step.x;
            coord.y += step.y;
            coord.z += step.z;
            surf2Dread(&c, coeffs_read, coord.x * 4, coord.y);
            //c = (float*)((uchar*)c + step);
            previous_c = Lambda * c + z[k] * previous_c;
            surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
            //*c = previous_c = Lambda * *c + z[k] * previous_c;
        }

        // anticausal initialization
        previous_c = InitialAntiCausalCoefficientSurf(previous_c, DataLength, z[k]);
        surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
        //*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, z[k], step);

        // Now we start reading from output array instead of input array.
        coeffs_read = coeffs_out;

        // anticausal recursion
        for (int n = DataLength - 2; 0 <= n; n--) {
            coord.x -= step.x;
            coord.y -= step.y;
            coord.z -= step.z;
            surf2Dread(&c, coeffs_read, coord.x * 4, coord.y);
            //c = (float*)((uchar*)c - step);
            previous_c = z[k] * (previous_c - c);
            surf2Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y);
            //*c = previous_c = z[k] * (previous_c - *c);
        }
    }
}

/// C2IC for cuda array input, cuda array output (in-place AND out-of-place)
__host__ __device__ void ConvertToInterpolationCoefficientsSurf3D(
        CUsurfObject coeffs_in,// input samples --> intput coefficients
        CUsurfObject coeffs_out,// input samples --> output coefficients (can be same as input)
        int3 coord,
        uint DataLength,	// number of samples or coefficients
        int3 step,          // element interleave in bytes
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)		// Gain
{

    // Vector to array
    const float z[4] = {Poles.x, Poles.y, Poles.z, Poles.w};
    const int h[4] = {Horizon.x, Horizon.y, Horizon.z, Horizon.w};

    // Initially, values should be read from the input array. After the first causal recursion, the output array is
    // filled, and thus values should now be read from there for future recursions.
    CUsurfObject coeffs_read = coeffs_in;

    // Loop over all poles
    for (uint k = 0; k < NbPoles; k++){

        // All coeffs need to be multiplied with gain once. This happens below. After first iter, Lambda should thus be
        // 1.
        if (k > 0) Lambda = 1;

        // causal initialization
        float c;
        //surf3Dread(&c, coeffs, coord.x * 4, coord.y, coord.z);
        float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
        previous_c = Lambda * InitialCausalCoefficientSurf3D(coeffs_read, coord, DataLength, z[k], UMIN(h[k], DataLength), step);
        surf3Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y, coord.z);

        // causal recursion
        for (uint n = 1; n < DataLength; n++) {
            coord.x += step.x;
            coord.y += step.y;
            coord.z += step.z;
            surf3Dread(&c, coeffs_read, coord.x * 4, coord.y, coord.z);
            //c = (float*)((uchar*)c + step);
            previous_c = Lambda * c + z[k] * previous_c;
            surf3Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y, coord.z);
            //*c = previous_c = Lambda * *c + z[k] * previous_c;
        }

        // anticausal initialization
        previous_c = InitialAntiCausalCoefficientSurf(previous_c, DataLength, z[k]);
        surf3Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y, coord.z);
        //*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, z[k], step);

        // Now we start reading from output array instead of input array.
        coeffs_read = coeffs_out;

        // anticausal recursion
        for (int n = DataLength - 2; 0 <= n; n--) {
            coord.x -= step.x;
            coord.y -= step.y;
            coord.z -= step.z;
            surf3Dread(&c, coeffs_read, coord.x * 4, coord.y, coord.z);
            //c = (float*)((uchar*)c - step);
            previous_c = z[k] * (previous_c - c);
            surf3Dwrite(previous_c, coeffs_out, coord.x * 4, coord.y, coord.z);
            //*c = previous_c = z[k] * (previous_c - *c);
        }
    }
}

////// 2DX, 2DY for linear memory in-place filter
extern "C"
__global__
void SamplesToCoefficients2DX(
        float* image,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the image
        uint height,		// height of the image
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    float* line = (float*)((uchar*)image + y * pitch);  //direct access

    ConvertToInterpolationCoefficients(line, width, sizeof(float), NbPoles, Poles, Horizon, Lambda);
}

extern "C"
__global__
void SamplesToCoefficients2DY(
        float* image,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the image
        uint height,		// height of the image
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    float* line = image + x;  //direct access

    ConvertToInterpolationCoefficients(line, height, pitch, NbPoles, Poles, Horizon, Lambda);
}

////// 2DX, 2DY for linear memory input, cuda array output (out-of-place)
extern "C"
__global__
void SamplesToCoefficients2DX_ptr2surf(
        float* image_in,		// input coefficients
        size_t offset,          // input pointer offset in bytes
        CUsurfObject image_out, // output coefficients
        uint pitch,			// width in bytes
        uint width,			// width of the image
        uint height,		// height of the image
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    float* line = (float*)((uchar*)image_in + offset + y * pitch);  //direct access
    int3 coord = make_int3(0, y, 0);
    const int3 step = make_int3(1, 0, 0);

    ConvertToInterpolationCoefficientsPtr2Surf2D(line,
                                                 image_out,
                                                 coord,
                                                 width,
                                                 sizeof(float),
                                                 step,
                                                 NbPoles,
                                                 Poles,
                                                 Horizon,
                                                 Lambda);
}

extern "C"
__global__
void SamplesToCoefficients2DY_ptr2surf(
        float* image_in,		// input coefficients
        size_t offset,          // input pointer offset in bytes
        CUsurfObject image_out, // output coefficients
        uint pitch,			// width in bytes
        uint width,			// width of the image
        uint height,		// height of the image
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    float* line = (float*)((uchar*) image_in + offset) + x;  //direct access
    int3 coord = make_int3(x, 0, 0);
    const int3 step = make_int3(0, 1, 0);

    ConvertToInterpolationCoefficientsPtr2Surf2D(line,
                                                 image_out,
                                                 coord,
                                                 width,
                                                 pitch,
                                                 step,
                                                 NbPoles,
                                                 Poles,
                                                 Horizon,
                                                 Lambda);
}

////// 2DX, 2DY for cuda array input, cuda array output (in-place AND out-of-place)
extern "C"
__global__
void SamplesToCoefficients2DXSurf(
        CUsurfObject image_in, // input coeffs
        CUsurfObject image_out, // output coeffs
        uint width,			// width of the image
        uint height,		// height of the image
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    //float* line = (float*)((uchar*)image + y * pitch);  //direct access
    int3 coord = make_int3(0, y, 0);
    const int3 step = make_int3(1, 0, 0);

    ConvertToInterpolationCoefficientsSurf2D(image_in,
                                             image_out,
                                             coord,
                                             width,
                                             step,
                                             NbPoles,
                                             Poles,
                                             Horizon,
                                             Lambda);
}

extern "C"
__global__
void SamplesToCoefficients2DYSurf(
        CUsurfObject image_in, // input coeffs
        CUsurfObject image_out, // output coeffs
        uint width,			// width of the image
        uint height,		// height of the image
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    //float* line = image + x;  //direct access
    int3 coord = make_int3(x, 0, 0);
    const int3 step = make_int3(0, 1, 0);

    ConvertToInterpolationCoefficientsSurf2D(image_in,
                                             image_out,
                                             coord,
                                             height,
                                             step,
                                             NbPoles,
                                             Poles,
                                             Horizon,
                                             Lambda);
}

////// 3DX, 3DY, 3DZ for linear memory (in-place)
extern "C"
__global__
void SamplesToCoefficients3DX(
        float* volume,		// in-place processing
        uint pitch,         // width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth,			// depth of the volume
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    const uint z = blockIdx.y * blockDim.y + threadIdx.y;
    const uint startIdx = (z * height + y) * pitch;

    float* ptr = (float*)((uchar*)volume + startIdx);
    ConvertToInterpolationCoefficients(ptr, width, sizeof(float), NbPoles, Poles, Horizon, Lambda);
}

extern "C"
__global__
void SamplesToCoefficients3DY(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth,			// depth of the volume
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in y-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint z = blockIdx.y * blockDim.y + threadIdx.y;
    const uint startIdx = z * height * pitch;

    float* ptr = (float*)((uchar*)volume + startIdx);
    ConvertToInterpolationCoefficients(ptr + x, height, pitch, NbPoles, Poles, Horizon, Lambda);
}

extern "C"
__global__
void SamplesToCoefficients3DZ(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth,			// depth of the volume
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in z-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint startIdx = y * pitch;
    const uint slice = height * pitch;

    float* ptr = (float*)((uchar*)volume + startIdx);
    ConvertToInterpolationCoefficients(ptr + x, depth, slice, NbPoles, Poles, Horizon, Lambda);
}

////// 3DX, 3DY, 3DZ for cuda array input, cuda array output (in-place AND out-of-place)
extern "C"
__global__
void SamplesToCoefficients3DXSurf(
        CUsurfObject volume_in, // input coefficients
        CUsurfObject volume_out, // output coefficients (can be same as input)
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth,			// depth of the volume
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in x-direction
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    const uint z = blockIdx.y * blockDim.y + threadIdx.y;
    int3 coord = make_int3(0, y, z);
    const int3 step = make_int3(1, 0, 0);

    ConvertToInterpolationCoefficientsSurf3D(volume_in, volume_out, coord, width, step, NbPoles, Poles, Horizon, Lambda);
}

extern "C"
__global__
void SamplesToCoefficients3DYSurf(
        CUsurfObject volume_in, // input coefficients
        CUsurfObject volume_out, // output coefficients (can be same as input)
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth,			// depth of the volume
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in y-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint z = blockIdx.y * blockDim.y + threadIdx.y;
    int3 coord = make_int3(x, 0, z);
    const int3 step = make_int3(0, 1, 0);

    ConvertToInterpolationCoefficientsSurf3D(volume_in, volume_out, coord, height, step, NbPoles, Poles, Horizon, Lambda);
}

extern "C"
__global__
void SamplesToCoefficients3DZSurf(
        CUsurfObject volume_in, // input coefficients
        CUsurfObject volume_out, // output coefficients (can be same as input)
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth,			// depth of the volume
        int NbPoles,        // Num Poles
        float4 Poles,       // Poles
        int4 Horizon,       // Horizon
        float Lambda)       // Gain
{
    // process lines in z-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    int3 coord = make_int3(x, y, 0);
    const int3 step = make_int3(0, 0, 1);

    ConvertToInterpolationCoefficientsSurf3D(volume_in, volume_out, coord, depth, step, NbPoles, Poles, Horizon, Lambda);
}


__inline__ __device__ float sinc(float val)
{
    //return ((val == 0) ? 1 : (sin(val*M_PI)/(val*M_PI)));
    //return (fabs(val) < 1e-4) ? 1 : (sinpif(val)/(val*M_PI_F));

    // Seems safe if using sinpif. Not safe using __sinf(val*M_PI_F).
    return (val == 0) ? 1 : (sinpif(val)/(val*M_PI_F));
}

extern "C"
__global__
void computeLUT(int pixelcount,
                float freqStepSize,
                float2 Xi_x, // Spline basis vector X on proj
                float2 Xi_y, // Spline basis vector Y on proj
                float2 Xi_z, // Spline basis vector Z on proj
                float3 nu,   // Multiplicity
                float2* outIm,
                size_t stride)
{
    //compute x,y indices
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid (C2C transform)
//    if (x >= pixelcount) return;
//    if (y >= pixelcount) return;
//
//    float xx = (float)x;
//    float yy = (float)y;
//
//    if (xx > 0.5f * (float)pixelcount)
//        xx = -1.f * ((float)pixelcount - xx);
//    if (yy > 0.5f * (float)pixelcount)
//        yy = -1.f * ((float)pixelcount - yy);

    // Grid (C2R transform)
    if (x >= pixelcount * 0.5f + 1) return;
    if (y >= pixelcount) return;

    float xx = (float)x;
    float yy = (float)y;

    if (yy > 0.5f * (float)pixelcount)
        yy = -1.f * ((float)pixelcount - yy);

    float sp = 1;
    xx = (float) xx * freqStepSize;
    yy = (float) yy * freqStepSize;

    // Compute FFT of the spline
    float om_x = xx * Xi_x.x + yy * Xi_x.y;
    float om_y = xx * Xi_y.x + yy * Xi_y.y;
    float om_z = xx * Xi_z.x + yy * Xi_z.y;
    // Box spline directions
    float sinc_x = pow(fabs(sinc(om_x)), nu.x);
    float sinc_y = pow(fabs(sinc(om_y)), nu.y);
    float sinc_z = pow(fabs(sinc(om_z)), nu.z);
    // Projection directions
    float sinc_px = pow(fabs(sinc(xx)), 4);
    float sinc_py = pow(fabs(sinc(yy)), 4);

    sp = sp * sinc_x * sinc_y * sinc_z * sinc_px * sinc_py;

    // Out
    *(((float2 *) ((char *) outIm + stride * y)) + x) = make_float2(sp, 0);
}

extern "C"
__global__
void computeBoxSplineDualB(int2 pixelcount,
                           float2 freqStepSize,
                           float2 Xi_x, // Spline basis vector X on proj
                           float2 Xi_y, // Spline basis vector Y on proj
                           float2 Xi_z, // Spline basis vector Z on proj
                           float3 nu,   // Multiplicity
                           float thickness,
                           float2* outIm)
{
    //compute x,y indices
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid (C2R transform)
    if (x >= pixelcount.x/2 + 1) return;
    if (y >= pixelcount.y) return;

    float xx = (float)x;
    float yy = (float)y;

    if (yy > 0.5f * (float)pixelcount.y)
        yy = -1.f * ((float)pixelcount.y - yy);

    // Frequency grid
    xx = (float) xx * freqStepSize.x;
    yy = (float) yy * freqStepSize.y;

    // Step sizes of the projected 3D tensor B-Splines in 2D
    float om_x = xx * Xi_x.x + yy * Xi_x.y;
    float om_y = xx * Xi_y.x + yy * Xi_y.y;
    float om_z = xx * Xi_z.x + yy * Xi_z.y;

    // Oversampled splines
    float spl = 0.f;      // conv2(Proj(BoxSpline), BSpline)
    float dual_spl = 0.f; // 1/BSpline^2
    for (int k = -2; k<=2; k++){
        for (int l = -2; l<=2; l++){
            // Direction plus period
            float om_xkl = om_x + ((float)k) * Xi_x.x + ((float)l) * Xi_x.y;
            float om_ykl = om_y + ((float)k) * Xi_y.x + ((float)l) * Xi_y.y;
            float om_zkl = om_z + ((float)k) * Xi_z.x + ((float)l) * Xi_z.y;

            // Volume box spline values
            float boxsp_vx = pow(fabs(sinc(om_xkl)), nu.x);
            float boxsp_vy = pow(fabs(sinc(om_ykl)), nu.y);
            float boxsp_vz = pow(fabs(sinc(om_zkl)), nu.z);

            // Projection tensor B-Spline values
            float bsp_px_py = pow(fabs(sinc(xx + ((float)k))), 4.f) * pow(fabs(sinc(yy + ((float)l))), 4.f);
            //float bsp_py = pow(fabs(sinc(yy + ((float)l))), 4);

            // Add it up
            spl += boxsp_vx * boxsp_vy * boxsp_vz * bsp_px_py;
            dual_spl += bsp_px_py*bsp_px_py;//pow(bsp_px_py, 2.f);
        }
    }

    // Orthogonal projection post-filter
    float sp = spl / dual_spl;

    // Out
    outIm[(pixelcount.x/2+1) * y + x] = make_float2(sp * thickness, 0);
}

__device__ __forceinline__ float2 cmulf(float2 a, float2 b)
{
    float2 res;
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res;
}

__device__ __forceinline__ float2 cexpf (float2 z)
{
    float2 res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

extern "C"
__global__
void computeBoxSplineDualBRefl(int2 pixelcount,
                               float2 freqStepSize,
                               float2 Xi_x, // Spline basis vector X on proj
                               float2 Xi_y, // Spline basis vector Y on proj
                               float2 Xi_z, // Spline basis vector Z on proj
                               float3 nu,   // Multiplicity
                               float thickness,
                               float2* outIm)
{
    //compute x,y indices
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid (C2R transform)
    if (x >= pixelcount.x/2 + 1) return;
    if (y >= pixelcount.y) return;

    float xx = (float)x;
    float yy = (float)y;

    if (yy > 0.5f * (float)pixelcount.y)
        yy = -1.f * ((float)pixelcount.y - yy);

    // Frequency grid
    xx = (float) xx * freqStepSize.x;
    yy = (float) yy * freqStepSize.y;

    // Step sizes of the projected 3D tensor B-Splines in 2D
    float om_x = xx * Xi_x.x + yy * Xi_x.y;
    float om_y = xx * Xi_y.x + yy * Xi_y.y;
    float om_z = xx * Xi_z.x + yy * Xi_z.y;

    // Oversampled splines
    float spl = 0.f;      // conv2(Proj(BoxSpline), BSpline)
    float dual_spl = 0.f; // 1/BSpline^2
    for (int k = -2; k<=2; k++){
        for (int l = -2; l<=2; l++){
            // Direction plus period
            float om_xkl = om_x + ((float)k) * Xi_x.x + ((float)l) * Xi_x.y;
            float om_ykl = om_y + ((float)k) * Xi_y.x + ((float)l) * Xi_y.y;
            float om_zkl = om_z + ((float)k) * Xi_z.x + ((float)l) * Xi_z.y;

            // Volume box spline values
            float boxsp_vx = pow(fabs(sinc(om_xkl)), nu.x);
            float boxsp_vy = pow(fabs(sinc(om_ykl)), nu.y);
            float boxsp_vz = pow(fabs(sinc(om_zkl)), nu.z);

            // Projection tensor B-Spline values
            float bsp_px_py = pow(fabs(sinc(xx + ((float)k))), 4.f) * pow(fabs(sinc(yy + ((float)l))), 4.f);
            //float bsp_py = pow(fabs(sinc(yy + ((float)l))), 4);

            // Add it up
            spl += boxsp_vx * boxsp_vy * boxsp_vz * bsp_px_py;
            dual_spl += bsp_px_py*bsp_px_py;//pow(bsp_px_py, 2.f);
        }
    }

    // Orthogonal projection pre-filter
    float sp = spl / dual_spl;

    // Reflection
    //float2 shift = cexpf(-1.f * iu * 2.f * M_PI * (-1 * xx + -1 * yy));
    float2 shift = make_float2(1, 0);
    float2 val = make_float2(sp * shift.x, sp * shift.y);

    // Out
    outIm[(pixelcount.x/2+1) * y + x] = val;
}