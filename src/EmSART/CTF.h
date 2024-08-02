//
// Created by uermel on 9/14/23.
//

#ifndef ARTIATOMI_CTF_H
#define ARTIATOMI_CTF_H

#include "EmSartDefault.h"
#include "Projection.h"
#include "Volume.h"
#include "utils/Config.h"
#include "CtfFile.h"
#include <cstdlib>

class CTF {
private:
    Configuration::Config& config;
    Projection& proj;
    CtfFile& defocus;

    ctfConstants c_global;
    ctfImageConstants* c_image;

    std::vector<std::vector<float>> offsets;

    int mpi_rank;
    int maxSliceNum;

public:
    CTF(Configuration::Config& aConfig,
        Projection &aProj,
        CtfFile &aCtf,
        int aMpi_rank = 0);
    ~CTF();

    ctfConstants GetGlobalConstants();
    ctfImageConstants GetImageConstants(int aIndex);
    vector<float>* GetDefocusOffsets(int aIndex);
    vector<float> GetDefocusOffsetsBatched(int aIndex, int batch);
    int GetMaxSliceNumber() const;
    int GetBatchCount(int aIndex) const;
    int GetBatchSize(int aIndex, int aBatch);
    int2 GetMinMaxSlice(int aIndex, int aBatch);

    void CTFSlices(Volume<float>* aVol,
                   int index,
                   int &sliceNum,
                   float &ctfCenter,
                   float &thickness,
                   float &entryPoint,
                   std::vector<float> &off);

    void PlanSlices(Volume<float>* aVol);
};


#endif //ARTIATOMI_CTF_H
