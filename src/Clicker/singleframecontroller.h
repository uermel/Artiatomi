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


#ifndef SINGLEFRAMECONTROLLER_H
#define SINGLEFRAMECONTROLLER_H

#include <QObject>
#include "baseframecontroller.h"
#include "SingleFrame.h"

class SingleFrameController : public BaseFrameController
{
    Q_OBJECT

//    struct SingleFrameControllerData
//    {
//      cl_mem devPtr;
//      OpenCL::OpenCLKernel* kernel;
//      float minVal;
//      float maxVal;
//    };

public:
    explicit SingleFrameController(QObject *parent = 0);
    ~SingleFrameController();
    int GetWidth();
    int GetHeight();
    bool GetIsMultiChannel();
    float GetPixelSize();
    void CloseFile();
    virtual QMatrix4x4 GetAlignmentMatrix();

public slots:
    int openFile(QString filename);

private:

    static void FileLoadStatusUpdate(FileReader::FileReaderStatus status);
    static SingleFrameController* myController;
//    float mMinValues[3];
//    float mMaxValues[3];
//    float mMeanValues[3];
//    float mStdValues[3];
//    bool mIsRGB;
//    float mPixelSize;
//    int mHistogramBinCount;
//    QVector<int> mHistogramRorGray;
//    QVector<int> mHistogramG;
//    QVector<int> mHistogramB;

//    int mDimX;
//    int mDimY;
//    DataType_enum mDatatype;

//    cl_context mCtx;
//    cl_program mProgram;
//    OpenCL::OpenCLDeviceVariable* mDevVar;
//    OpenCL::OpenCLKernel* mKernel8uC1;
//    OpenCL::OpenCLKernel* mKernel8uC2;
//    OpenCL::OpenCLKernel* mKernel8uC3;
//    OpenCL::OpenCLKernel* mKernel8uC4;
//    OpenCL::OpenCLKernel* mKernel16uC1;
//    OpenCL::OpenCLKernel* mKernel16sC1;
//    OpenCL::OpenCLKernel* mKernel32sC1;
//    OpenCL::OpenCLKernel* mKernel32fC1;

//    OpenCL::OpenCLKernel* mKernelSumDiff8uC1;
//    OpenCL::OpenCLKernel* mKernelSumDiff8uC2;
//    OpenCL::OpenCLKernel* mKernelSumDiff8uC3;
//    OpenCL::OpenCLKernel* mKernelSumDiff8uC4;
//    OpenCL::OpenCLKernel* mKernelSumDiff16uC1;
//    OpenCL::OpenCLKernel* mKernelSumDiff16sC1;
//    OpenCL::OpenCLKernel* mKernelSumDiff32sC1;
//    OpenCL::OpenCLKernel* mKernelSumDiff32fC1;
//    OpenCL::OpenCLKernel* mKernelSumDiff32fC1Final;
//    OpenCL::OpenCLKernel* mKernelSumDiff32fC2Final;
//    OpenCL::OpenCLKernel* mKernelSumDiff32fC3Final;

//    OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC1;
//    OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC2;
//    OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC3;
//    OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC4;
//    OpenCL::OpenCLKernel* mKernelSumDiffSqr16uC1;
//    OpenCL::OpenCLKernel* mKernelSumDiffSqr16sC1;
//    OpenCL::OpenCLKernel* mKernelSumDiffSqr32sC1;
//    OpenCL::OpenCLKernel* mKernelSumDiffSqr32fC1;

//    OpenCL::OpenCLKernel* mKernelMinMax8uC1;
//    OpenCL::OpenCLKernel* mKernelMinMax8uC2;
//    OpenCL::OpenCLKernel* mKernelMinMax8uC3;
//    OpenCL::OpenCLKernel* mKernelMinMax8uC4;
//    OpenCL::OpenCLKernel* mKernelMinMax16uC1;
//    OpenCL::OpenCLKernel* mKernelMinMax16sC1;
//    OpenCL::OpenCLKernel* mKernelMinMax32sC1;
//    OpenCL::OpenCLKernel* mKernelMinMax32fC1;
//    OpenCL::OpenCLKernel* mKernelMinMax32fC1Final;
//    OpenCL::OpenCLKernel* mKernelMinMax32fC2Final;
//    OpenCL::OpenCLKernel* mKernelMinMax32fC3Final;

//    OpenCL::OpenCLKernel* mKernelHistogram8uC1;
//    OpenCL::OpenCLKernel* mKernelHistogram8uC2;
//    OpenCL::OpenCLKernel* mKernelHistogram8uC3;
//    OpenCL::OpenCLKernel* mKernelHistogram8uC4;
//    OpenCL::OpenCLKernel* mKernelHistogram16uC1;
//    OpenCL::OpenCLKernel* mKernelHistogram16sC1;
//    OpenCL::OpenCLKernel* mKernelHistogram32sC1;
//    OpenCL::OpenCLKernel* mKernelHistogram32fC1;
//    SingleFrameControllerData data;

    SingleFrame* mImage;

//    void computeImageStatistics();
//    void computeHistogram();
//    void computeImageStatisticsOpenCL();
//    void computeHistogramOpenCL();

public:
//    static void processWithOpenCL(int width, int height, cl_mem output, float minVal, float maxVal, void* userData);
//    SingleFrameControllerData* GetUserData();
};

#endif // SINGLEFRAMECONTROLLER_H
