#ifndef ARTIATOMI_COMMON_TYPES_H
#define ARTIATOMI_COMMON_TYPES_H
#include <vector_types.h>
#include <vector_functions.h>
#include "cutil_math.h"

// Matrix types
typedef struct {
    float4 m[4];
} float4x4;

typedef struct {
    float3 m[3];
} float3x3;

// CTF constants
typedef struct {
    // CTF
    float cs;
    float voltage;
    float openingAngle;
    float ampContrast;
    float phaseContrast;

    // CTF correction
    float WienerFilterNoiseLevel;

    // CTF helper
    float cs_m;
    float lambda;
} ctfConstants;

// CTF image constants
typedef struct {
    // FFT
    float pixelsize;
    float2 pixelcount;
    float maxFreq;
    float2 freqStepSize;
    float2 asymCorrFac;

    // Slices
    int sliceNumber;
    float ctfCenter;
    float entryPoint;
    float sliceThickness;
    int sliceBatchCount;

    // Defocus, Astigmatism, B
    float phaseShift;
    float defocusMin;
    float defocusMax;
    float astigAngle;

    float B;
    float Bsqr;
    float Bcub;
} ctfImageConstants;

// FP constants
typedef struct {
    float cs;
    float voltage;
    float openingAngle;
    float ampContrast;
    float phaseContrast;
    float pixelsize;
    float2 pixelcount;
    float maxFreq;
    float2 freqStepSize;
} fpConstants;

// BP constants
// CTF constants
typedef struct {
    float cs;
    float voltage;
    float openingAngle;
    float ampContrast;
    float phaseContrast;
    float pixelsize;
    float2 pixelcount;
    float maxFreq;
    float2 freqStepSize;
} bpConstants;

// Modes for frequency sampling
enum SF_EXTRAP_MODE {
    SF_EXTRAP_TEX,
    SF_EXTRAP_VAL
};

// transform vector by matrix
__host__ __device__
inline void MatrixVector3Mul(const float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
{
    xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
    yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
    //erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
}

__host__ __device__
inline void MatrixVector3Mul(const float3x3& M, float3* v)
{
    float3 erg;
    erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z;
    erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z;
    erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z;
    *v = erg;
}

__host__ __device__
inline void MatrixVector3Mul(const float4x4& M, float3& v, float3& erg)
{
    erg.x = M.m[0].x * v.x + M.m[0].y * v.y + M.m[0].z * v.z + 1.f * M.m[0].w;
    erg.y = M.m[1].x * v.x + M.m[1].y * v.y + M.m[1].z * v.z + 1.f * M.m[1].w;
    erg.z = M.m[2].x * v.x + M.m[2].y * v.y + M.m[2].z * v.z + 1.f * M.m[2].w;
}

__host__ __device__
inline void MatrixVector3Mul(const float4x4& M, float3* v)
{
    float3 erg;
    erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
    erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
    erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
    *v = erg;
}

#endif //ARTIATOMI_COMMON_TYPES_H
