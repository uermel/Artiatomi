//
// Created by uermel on 10/4/21.
//

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

#define Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline

//--------------------------------------------------------------------------
// Local GPU device procedures
//--------------------------------------------------------------------------

__host__ __device__ float InitialCausalCoefficient(
        float* c,			// coefficients
        uint DataLength,	// number of coefficients
        int step)			// element interleave in bytes
{
    const uint Horizon = UMIN(12, DataLength);

    // this initialization corresponds to clamping boundaries
    // accelerated loop
    float zn = Pole;
    float Sum = *c;
    for (uint n = 0; n < Horizon; n++) {
        Sum += zn * *c;
        zn *= Pole;
        c = (float*)((uchar*)c + step);
    }
    return(Sum);
}

__host__ __device__ float InitialAntiCausalCoefficient(
        float* c,			// last coefficient
        uint DataLength,	// number of samples or coefficients
        int step)			// element interleave in bytes
{
    // this initialization corresponds to clamping boundaries
    return((Pole / (Pole - 1.0f)) * *c);
}

__host__ __device__ void ConvertToInterpolationCoefficients(
        float* coeffs,		// input samples --> output coefficients
        uint DataLength,	// number of samples or coefficients
        int step)			// element interleave in bytes
{
    // compute the overall gain
    const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);

    // causal initialization
    float* c = coeffs;
    float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
    *c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
    // causal recursion
    for (uint n = 1; n < DataLength; n++) {
        c = (float*)((uchar*)c + step);
        *c = previous_c = Lambda * *c + Pole * previous_c;
    }
    // anticausal initialization
    *c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
    // anticausal recursion
    for (int n = DataLength - 2; 0 <= n; n--) {
        c = (float*)((uchar*)c - step);
        *c = previous_c = Pole * (previous_c - *c);
    }
}

extern "C"
__global__
void SamplesToCoefficients2DX(
        float* image,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the image
        uint height)		// height of the image
{
    // process lines in x-direction
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    float* line = (float*)((uchar*)image + y * pitch);  //direct access

    ConvertToInterpolationCoefficients(line, width, sizeof(float));
}

extern "C"
__global__
void SamplesToCoefficients2DY(
        float* image,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the image
        uint height)		// height of the image
{
    // process lines in x-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    float* line = image + x;  //direct access

    ConvertToInterpolationCoefficients(line, height, pitch);
}

extern "C"
__global__ void SamplesToCoefficients3DX(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth)			// depth of the volume
{
    // process lines in x-direction
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    const uint z = blockIdx.y * blockDim.y + threadIdx.y;
    const uint startIdx = (z * height + y) * pitch;

    float* ptr = (float*)((uchar*)volume + startIdx);
    ConvertToInterpolationCoefficients(ptr, width, sizeof(float));
}

extern "C"
__global__ void SamplesToCoefficients3DY(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth)			// depth of the volume
{
    // process lines in y-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint z = blockIdx.y * blockDim.y + threadIdx.y;
    const uint startIdx = z * height * pitch;

    float* ptr = (float*)((uchar*)volume + startIdx);
    ConvertToInterpolationCoefficients(ptr + x, height, pitch);
}

extern "C"
__global__ void SamplesToCoefficients3DZ(
        float* volume,		// in-place processing
        uint pitch,			// width in bytes
        uint width,			// width of the volume
        uint height,		// height of the volume
        uint depth)			// depth of the volume
{
    // process lines in z-direction
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint startIdx = y * pitch;
    const uint slice = height * pitch;

    float* ptr = (float*)((uchar*)volume + startIdx);
    ConvertToInterpolationCoefficients(ptr + x, depth, slice);
}