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


#ifndef MYQOPENGLWIDGET_H
#define MYQOPENGLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <OpenCLHelpers.h>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <OpenCLKernel.h>
#include <CL/cl.h>


class MyQOpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit MyQOpenGLWidget(QWidget *parent = 0);
    ~MyQOpenGLWidget();

    void MakeOpenCLCurrent();
    void MakeOpenGLCurrent();
    void SetOpenCLProcessor(void(*processWithOpenCL)(int width, int height, cl_mem output, float minContrast, float maxContrast, void* userData));
    void SetOpenCLProcessorUserData(void* userData);
    void StopOpenGL();
    void StartOpenGL();

protected:
    void resizeGL(int, int);
    void initializeGL();
    void paintGL();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void mouseDoubleClickEvent( QMouseEvent * e );

public slots:
    // slots for xyz-rotation slider
    void setXShift(float value);
    void setYShift(float value);
    void setZoom(float value);
    void setNewImageDimensions(int sizeX, int sizeY, bool isMultiChannel);
    void setImageCenterValue(float center);
    void setImageContrastWidthValue(float contrastWidth);
    void setLinearInterpolation(bool value);
    void setPixelSize(float value);
    void setShowScaleBar(bool value);
    //void refreshImage(float minVal, float maxVal, bool gamma);
    void updateImage();
    void setViewMatrix(QMatrix4x4 aMatrix);
    void setMarkerPositions(QVector<float>& aX, QVector<float>& aY, QVector<float>& xAlig, QVector<float>& yAlig);
    void setActiveMarker(int aMarkerIdx);
    void setMarkerSize(int size);
    void SetShowAlignedMarkers(bool aShow);
    void SetShowAlignedMarkersAsArrows(bool aShowAsArrow);

signals:
    // signaling rotation from mouse movement
    void xShiftChanged(float value);
    void yShiftChanged(float value);
    void zoomChanged(float value);
    void imageCenterValueChanged(float center);
    void imageContrastWidthValueChanged(float contrastWidth);
    void linearInterpolationChanged(bool value);
    void pixelSizeChanged(float value);
    void showScaleBarChanged(bool value);
    void mousePositionChanged(int x, int y, bool isHit);
    void mouseClicked(int x, int y, bool isHit, Qt::MouseButtons buttons);
    void markerSizeChanged(int size);

private:

    void setRatioXY(int width, int height);
    float xShift, yShift;
    float ratioX, ratioY;
    float zoom;
    int imageWidth, imageHeight;
    float imageMinValue;
    float imageMaxValue;
    float imageCenterValue;
    float imageContrastWidthValue;
    void draw();
    void getOpenCLMemoryBuffer();
    void modifyTextureWithOpenCL();
    void computeScaleBarVertices(float addBorder, int scaleBarWidth);
    void computeScaleBarWidth(int& widthInPixelsToShow, QString& label);
    void drawScaleBar();
    void drawMarker();
    void* mUserData;
    typedef void (*processWithOpenCL)(int width, int height, cl_mem output, float minVal, float maxVal, void* userData);
    processWithOpenCL mOpenCLProcessor;
    bool mIsMultiChannel;


    static int createContext(MyQOpenGLWidget* widget);
    QPoint lastPos;
    QOpenGLContext* mSecondContext;
    cl_context cl_ctx;
    cl_mem mCLBuffer;
    OpenCL::OpenCLKernel* mCLKernel;
    OpenCL::OpenCLKernel* mCLFilterKernel;

    QOffscreenSurface* mOffscreenSurf;
    QSurface* mMySurface;

    QOpenGLShaderProgram* mShaderProg;
    QMatrix4x4 mBaseMatrix;
    QMatrix4x4 mViewMatrix;
    QMatrix4x4 mMatrix;
    uint mSamplerLocation;
    QOpenGLBuffer* mBufferVertices;
    QOpenGLBuffer* mBufferUV;
    QOpenGLBuffer* mBufferMarker;
    QOpenGLBuffer* mBufferReference;
    QOpenGLBuffer* mBufferScaleBarWhite;
    QOpenGLBuffer* mBufferScaleBarBlack;
    uint mTexID;
    bool linearInterpolation;

    float mVertices[3*4];
    const float mUV[2*4];
    float mVerticesScaleBar[3*4];
    float mVerticesCircle[3*72];
    float mMarkerSize;

    bool mStopOpenGL;
    float mPixelSizeInNM;
    bool mShowScalebar;
    QVector<float> mMarkerX;
    QVector<float> mMarkerY;
    QVector<float> mMarkerXAlig;
    QVector<float> mMarkerYAlig;
    int mActiveMarker;
    bool mShowAlignedMarkers;
    bool mShowAlignedMarkersAsArrows;
};

#endif // MYQOPENGLWIDGET_H
