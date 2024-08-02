//
// Created by uermel on 9/14/23.
//

#include "CTF.h"

#define h ((float)6.63E-34f) //Planck's quantum
#define c ((float)3.00E+08f) //Light speed
#define E0 (511.f) //keV

CTF::CTF(Configuration::Config &aConfig,
         Projection &aProj,
         CtfFile &aCtf,
         int aMpi_rank) :
    mpi_rank(aMpi_rank),
    config(aConfig),
    proj(aProj),
    defocus(aCtf),
    c_global(),
    c_image(new ctfImageConstants[aProj.GetProjCount()]),
    maxSliceNum(1)
{
    // Set constants
    c_global.cs = config.Cs;
    c_global.voltage = config.Voltage;
    c_global.openingAngle = 0.01f;
    c_global.ampContrast = config.AmplitudeContrast;
    c_global.phaseContrast = sqrtf(1 - c_global.ampContrast * c_global.ampContrast);
    c_global.WienerFilterNoiseLevel = config.WienerFilterNoiseLevel;

    // Helper
    c_global.cs_m = config.Cs * 0.001f;
    c_global.lambda = ((h * c) / sqrtf(((2.f * E0 * c_global.voltage * 1000.0f * 1000.0f) + (c_global.voltage * c_global.voltage * 1000.0f * 1000.0f)) * 1.602E-19f * 1.602E-19f));

    // Parameters for all projections (slices later because they may differ depending on the volume)
    for (int i=0; i<proj.GetProjCount(); i++){

        // To avoid planning for shitty projections
        if (!proj.IsGood(i))
            continue;

        // FFT constants
        c_image[i].pixelsize = proj.GetPixelSize() * powf(10, -9);
        c_image[i].pixelcount = make_float2((float)proj.GetWidth(), (float)proj.GetHeight());
        c_image[i].maxFreq = 1.0f / (c_image[i].pixelsize * 2.0f);
        c_image[i].freqStepSize = make_float2(c_image[i].maxFreq / ((float)proj.GetMaxDimension() / 2.0f),
                                              c_image[i].maxFreq / ((float)proj.GetMaxDimension() / 2.0f));

        c_image[i].asymCorrFac = proj.GetAsymCorrFactor();

        // Defocus, Astigmatism, B (read only when doing CTF correction)
        if (config.CtfMode == Configuration::Config::CTFM_YES) {
            c_image[i].defocusMin = defocus.GetMinDefocus((uint) i) * 0.000000001f;
            c_image[i].defocusMax = defocus.GetMaxDefocus((uint) i) * 0.000000001f;
            c_image[i].astigAngle = defocus.GetAstigmatismAngle((uint) i) / 180 * (float) M_PI;
        }

        c_image[i].B = config.CTFBetaFac.y;
        c_image[i].Bsqr = config.CTFBetaFac.z;
        c_image[i].Bcub = config.CTFBetaFac.w;

        c_image[i].phaseShift = 0.f;
    }


}

CTF::~CTF(){
    offsets.clear();
}

ctfConstants CTF::GetGlobalConstants()
{
    return c_global;
}

ctfImageConstants CTF::GetImageConstants(int aIndex)
{
    return c_image[aIndex];
}

vector<float>* CTF::GetDefocusOffsets(int aIndex)
{
    return &offsets[aIndex];
}

vector<float> CTF::GetDefocusOffsetsBatched(int aIndex, int aBatch)
{
    int2 minmax = GetMinMaxSlice(aIndex, aBatch);
    vector<float> ret(&offsets[aIndex][minmax.x], &offsets[aIndex][minmax.y]);

    return ret;
}

int CTF::GetMaxSliceNumber() const
{
    return maxSliceNum;
}

int CTF::GetBatchCount(int aIndex) const
{
    return c_image[aIndex].sliceBatchCount;
}

int2 CTF::GetMinMaxSlice(int aIndex, int aBatch)
{
    // Start and End of batch
    int2 minmax = {};
    minmax.x = aBatch * config.CTFSliceBatch;
    minmax.y = min(c_image[aIndex].sliceNumber - 1, minmax.x + config.CTFSliceBatch - 1);

    return minmax;
}

int CTF::GetBatchSize(int aIndex, int aBatch)
{
    int2 minmax = GetMinMaxSlice(aIndex, aBatch);
    return minmax.y - minmax.x + 1;
}

void CTF::CTFSlices(Volume<float> *aVol,
                    int index,
                    int &sliceNumber,
                    float &ctfCenter,
                    float &thickness,
                    float &entryPoint,
                    std::vector<float> &defocusOffsets)
{
    if (config.CtfMode == Configuration::Config::CTFM_NO){
        sliceNumber = 1;
        ctfCenter = 0;
        thickness = -DIST;
        defocusOffsets.push_back(0);
        return;
    } else {
        // Detector Matrix and Volume Matrix
        Matrix<double> M_det = proj.DetectorMatrix<double>(index);
        Matrix<double> M_vol = aVol->VolumeMatrix();

        // Volume center
        Matrix<double> center(4, 1);

        if (config.IgnoreZShiftForCTF) {
            // CTF is correct at global center
            center(0, 0) = 0;
            center(1, 0) = 0;
            center(2, 0) = 0;
            center(3, 0) = 1;
        } else {
            // CTF is correct at volume center
            center(0, 0) = 0;
            center(1, 0) = 0;
            center(2, 0) = -config.VolumeShift.z;
            center(3, 0) = 1;
        }

        // Projected center
        center = M_det * center;

//        printf("\nStackIdx: %i\n", index);
//        printf("center: %f %f %f %f\n", center(0,0), center(1, 0), center(2, 0), center(3, 0));

        // Distance from center, distance at entry and exit point
        double d;
        double d_in = 2 * - DIST;
        double d_out = 2 * DIST;

        Matrix<double> point(4, 1);
        float3 volDim = aVol->GetSubVolumeDimension(mpi_rank);

        for (int x = 0; x <= 1; x++) {
            for (int y = 0; y <= 1; y++) {
                for (int z = 0; z <= 1; z++) {
                    // Corners of the volume
                    point(0, 0) = (double) x * volDim.x;
                    point(1, 0) = (double) y * volDim.y;
                    point(2, 0) = (double) z * volDim.z;
                    point(3, 0) = 1;

//                    printf("corner %i %i %i point before: %f %f %f %f\n", x, y, z, point(0,0), point(1, 0), point(2, 0), point(3, 0));

                    // To global frame, then project
                    point = M_vol * point;

//                    printf("corner %i %i %i point global: %f %f %f %f\n", x, y, z, point(0,0), point(1, 0), point(2, 0), point(3, 0));


                    point = M_det * point;

//                    printf("corner %i %i %i point after: %f %f %f %f\n", x, y, z, point(0,0), point(1, 0), point(2, 0), point(3, 0));

                    // Distance to the projected center
                    d = center(2, 0) - point(2, 0);

//                    printf("corner %i %i %i d: %f\n", x, y, z, d);


                    if (d < d_in) d_in = d;
                    if (d > d_out) d_out = d;
                }
            }
        }

//        printf("d_in %f d_out %f\n", d_in, d_out);

        // Now compute the number of slices
        double apix = proj.GetPixelSize();

        // Slice thickness in Global Coords
        double thickness_glb = config.CTFSliceThickness / apix;
        double thickness_nm = config.CTFSliceThickness;

        // Number of total necessary slices for either side
        double negNum = ceil(abs(d_in) / thickness_glb);
        double posNum = ceil(abs(d_out) / thickness_glb);

        // The first slice will start at center - 1/2 * thickness_glb - negNum * thickness_glb
        // and will have ctf offset of -negNum * thickness_nm
        double volcenter = center(2, 0);
        double entryGlob = volcenter - 0.5 * thickness_glb - negNum * thickness_glb;
        double entryNM = -negNum * thickness_nm;

        // The slice number of any point can then be determined by:
        // floor((center - t - entryGlob) / thickness) ; where t is the projected point's z-coord.

        // There will be negNum + posNum + 1 slices
        int c_sliceNumber = (int) negNum + (int) posNum + 1;


        // Entry point/thickness in nm
        //double entryNM = d_in * apix;
        //double thick = config.CTFSliceThickness;

        for (int sliceIdx = 0; sliceIdx < c_sliceNumber; sliceIdx++) {
            double offset = entryNM + ((double) sliceIdx) * config.CTFSliceThickness;// - thick / 2.f;

//            printf("offset: %f %i %f\n", offset, sliceIdx, sliceIdx * thickness_glb);
            defocusOffsets.push_back((float) offset);
        }

        // Output
        sliceNumber = c_sliceNumber;
        ctfCenter = (float)center(2, 0);
        thickness = config.CTFSliceThickness / proj.GetPixelSize();
        entryPoint = (float)entryGlob;
    }
}

void CTF::PlanSlices(Volume<float>* aVol)
{
    // Slices for all projections
    maxSliceNum = 0;

    // Plan all slices
    for (int i=0; i<proj.GetProjCount(); i++){

        int sliceNumber;
        float ctfCenter;
        float thickness;
        float entry;
        auto off = new vector<float>;

        // Push empty vector to keep indexes consistent
        if (!proj.IsGood(i)){
            offsets.push_back(*off);
            continue;
        }

        // Compute the slices needed for this volume/projection combination
        CTFSlices(aVol, i, sliceNumber,ctfCenter, thickness, entry, *off);
        maxSliceNum = max(maxSliceNum, sliceNumber);

        // Slices
        c_image[i].sliceNumber = sliceNumber;
        c_image[i].ctfCenter = ctfCenter;
        c_image[i].entryPoint = entry;
        c_image[i].sliceThickness = thickness;
        offsets.push_back(*off);

        // Slice batches
        int batchCount = 1;
        if (sliceNumber > config.CTFSliceBatch) {
            batchCount = (sliceNumber + config.CTFSliceBatch - 1) / config.CTFSliceBatch;
        }

        c_image[i].sliceBatchCount = batchCount;
    }
}
