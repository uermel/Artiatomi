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


#ifndef BASEFRAMECONTROLLER_H
#define BASEFRAMECONTROLLER_H

#include <QObject>
#include <CL/cl.h>
#include <OpenCLHelpers.h>
#include <OpenCLException.h>
#include <OpenCLKernel.h>
#include <OpenCLDeviceVariable.h>
#include <QVector>
#include <QEvent>
#include <QMatrix4x4>

#define ProgressUpdateEventID 1001


class QProgressUpdateEvent : public QEvent
{
public:
    explicit QProgressUpdateEvent(int aValue);
    int GetValue();

private:
    int value;
};

class BaseFrameController : public QObject
{
    Q_OBJECT

    struct FrameControllerData
    {
      cl_mem devPtr;
      OpenCL::OpenCLKernel* kernel;
      OpenCL::OpenCLKernel* kernelFilter;
      float minVal;
      float maxVal;
      float scale;
      bool useFilter;
    };

public:
    explicit BaseFrameController(QObject *parent = 0);
    virtual ~BaseFrameController();
    virtual int GetWidth() = 0;
    virtual int GetHeight() = 0;
    virtual bool GetIsMultiChannel() = 0;
    virtual float GetPixelSize() = 0;
    virtual void CloseFile() = 0;
    //Overridden Slot that is called when a Custom Event is caught
    void customEvent(QEvent* e);
    virtual QMatrix4x4 GetAlignmentMatrix();

signals:
    void MinValueChanged(float minValue);
    void MaxValueChanged(float minValue);
    void MeanValueChanged(float minValue);
    void StdValueChanged(float minValue);
    void Std3ValueChanged(float minValue);
    void ValueRangeChanged(float value);
    void NegValueRangeChanged(float value);

    void MinValuesChanged(float minValues[3]);
    void MaxValuesChanged(float maxValues[3]);
    void MeanValuesChanged(float meanValues[3]);
    void StdValuesChanged(float stdValues[3]);
    void IsRGBChanged(bool isRGB);
    void PixelSizeChanged(float pixelsize);
    void DimensionsChanged(int dimX, int dimY);
    void DimensionXChanged(int dimX);
    void DimensionYChanged(int dimY);
    void HistogramChanged(QVector<int>* red, QVector<int>* green, QVector<int>* blue);
    void MinMaxDatatypeChanged(float minVal, float maxVal);

    void StartPreparingData();
    void ProgressStatusChanged(int);
    void AlignmentMatrixChanged(QMatrix4x4 matrix);

protected:
    static const char* openCLSource;
    float mMinValues[3];
    float mMaxValues[3];
    float mMeanValues[3];
    float mStdValues[3];
    bool mIsRGB;
    float mPixelSize;
    int mHistogramBinCount;
    QVector<int> mHistogramRorGray;
    QVector<int> mHistogramG;
    QVector<int> mHistogramB;

    int mDimX;
    int mDimY;
    DataType_enum mDatatype;

    static cl_context mCtx;
    static cl_program mProgram;

    OpenCL::OpenCLDeviceVariable* mDevVar;
    FrameControllerData data;

    static OpenCL::OpenCLKernel* mKernel8uC1;
    static OpenCL::OpenCLKernel* mKernel8uC1Filter;
    static OpenCL::OpenCLKernel* mKernel8uC2;
    static OpenCL::OpenCLKernel* mKernel8uC3;
    static OpenCL::OpenCLKernel* mKernel8uC4;
    static OpenCL::OpenCLKernel* mKernel16uC1;
    static OpenCL::OpenCLKernel* mKernel16uC1Filter;
    static OpenCL::OpenCLKernel* mKernel16sC1;
    static OpenCL::OpenCLKernel* mKernel16sC1Filter;
    static OpenCL::OpenCLKernel* mKernel32sC1;
    static OpenCL::OpenCLKernel* mKernel32sC1Filter;
    static OpenCL::OpenCLKernel* mKernel32fC1;
    static OpenCL::OpenCLKernel* mKernel32fC1Filter;

    static OpenCL::OpenCLKernel* mKernelSumDiff8uC1;
    static OpenCL::OpenCLKernel* mKernelSumDiff8uC2;
    static OpenCL::OpenCLKernel* mKernelSumDiff8uC3;
    static OpenCL::OpenCLKernel* mKernelSumDiff8uC4;
    static OpenCL::OpenCLKernel* mKernelSumDiff16uC1;
    static OpenCL::OpenCLKernel* mKernelSumDiff16sC1;
    static OpenCL::OpenCLKernel* mKernelSumDiff32sC1;
    static OpenCL::OpenCLKernel* mKernelSumDiff32fC1;
    static OpenCL::OpenCLKernel* mKernelSumDiff32fC1Final;
    static OpenCL::OpenCLKernel* mKernelSumDiff32fC2Final;
    static OpenCL::OpenCLKernel* mKernelSumDiff32fC3Final;

    static OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC1;
    static OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC2;
    static OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC3;
    static OpenCL::OpenCLKernel* mKernelSumDiffSqr8uC4;
    static OpenCL::OpenCLKernel* mKernelSumDiffSqr16uC1;
    static OpenCL::OpenCLKernel* mKernelSumDiffSqr16sC1;
    static OpenCL::OpenCLKernel* mKernelSumDiffSqr32sC1;
    static OpenCL::OpenCLKernel* mKernelSumDiffSqr32fC1;

    static OpenCL::OpenCLKernel* mKernelMinMax8uC1;
    static OpenCL::OpenCLKernel* mKernelMinMax8uC2;
    static OpenCL::OpenCLKernel* mKernelMinMax8uC3;
    static OpenCL::OpenCLKernel* mKernelMinMax8uC4;
    static OpenCL::OpenCLKernel* mKernelMinMax16uC1;
    static OpenCL::OpenCLKernel* mKernelMinMax16sC1;
    static OpenCL::OpenCLKernel* mKernelMinMax32sC1;
    static OpenCL::OpenCLKernel* mKernelMinMax32fC1;
    static OpenCL::OpenCLKernel* mKernelMinMax32fC1Final;
    static OpenCL::OpenCLKernel* mKernelMinMax32fC2Final;
    static OpenCL::OpenCLKernel* mKernelMinMax32fC3Final;

    static OpenCL::OpenCLKernel* mKernelHistogram8uC1;
    static OpenCL::OpenCLKernel* mKernelHistogram8uC2;
    static OpenCL::OpenCLKernel* mKernelHistogram8uC3;
    static OpenCL::OpenCLKernel* mKernelHistogram8uC4;
    static OpenCL::OpenCLKernel* mKernelHistogram16uC1;
    static OpenCL::OpenCLKernel* mKernelHistogram16sC1;
    static OpenCL::OpenCLKernel* mKernelHistogram32sC1;
    static OpenCL::OpenCLKernel* mKernelHistogram32fC1;

protected:
    void computeImageStatisticsOpenCL();
    void computeHistogramOpenCL(OpenCL::OpenCLDeviceVariable* a, OpenCL::OpenCLDeviceVariable* b, OpenCL::OpenCLDeviceVariable* c);

    void LoadKernel();
    bool mIsFileLoaded;
    bool mIsLockedForLoading;
    QString mFilename;

public:
    static void processWithOpenCL(int width, int height, cl_mem output, float minVal, float maxVal, void* userData);
    FrameControllerData* GetUserData();
    bool GetIsLoaded();
    QString GetFilename();
};

#endif // BASEFRAMECONTROLLER_H
