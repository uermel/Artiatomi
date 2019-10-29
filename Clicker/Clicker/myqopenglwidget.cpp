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


#include "myqopenglwidget.h"
#include <iostream>
#include <QThread>
#include <CL/cl_gl.h>
#include "singletonoffscreensurface.h"
#include <OpenCLException.h>
#include <QtDebug>
#include <QPainter>
#define GL_GLEXT_PROTOTYPES
#include <qopenglext.h>
#include <math.h>

//#define NoSecondContext


MyQOpenGLWidget::MyQOpenGLWidget(QWidget *parent):
    QOpenGLWidget(parent), xShift(0), yShift(0), zoom(1), imageWidth(2048), imageHeight(2048),
    ratioX(1), ratioY(1), mOpenCLProcessor(NULL), mUserData(NULL), imageMinValue(0), imageMaxValue(1),
    imageContrastWidthValue(1), imageCenterValue(0.5f), linearInterpolation(true), mStopOpenGL(false), mIsMultiChannel(false),
    mBufferVertices(NULL), mBufferUV(NULL), mBufferScaleBarWhite(NULL), mBufferScaleBarBlack(NULL), mBufferReference(NULL), mBufferMarker(NULL),
    mUV{0, 0, 1, 0, 1, 1, 0, 1}, mShaderProg(NULL), mCLBuffer(0), mCLKernel(NULL), mPixelSizeInNM(0), mShowScalebar(false),
    mMatrix(), mViewMatrix(), mBaseMatrix(), mActiveMarker(-1), mMarkerSize(10.0f), mShowAlignedMarkers(false), mShowAlignedMarkersAsArrows(false)
{
    setMouseTracking(true);
}

MyQOpenGLWidget::~MyQOpenGLWidget()
{
    if (mBufferUV)
    {
        mBufferUV->destroy();
        delete mBufferUV;
    }
    if (mBufferVertices)
    {
        mBufferVertices->destroy();
        delete mBufferVertices;
    }
    if (mBufferScaleBarWhite)
    {
        mBufferScaleBarWhite->destroy();
        delete mBufferScaleBarWhite;
    }
    if (mBufferScaleBarBlack)
    {
        mBufferScaleBarBlack->destroy();
        delete mBufferScaleBarBlack;
    }
    if (mBufferReference)
    {
        mBufferReference->destroy();
        delete mBufferReference;
    }
    if (mBufferMarker)
    {
        mBufferMarker->destroy();
        delete mBufferMarker;
    }
    if (mShaderProg)
    {
        delete mShaderProg;
    }
    if (mCLKernel)
    {
        delete mCLKernel;
    }
#ifndef NoSecondContext
    if (mSecondContext)
    {
        delete mSecondContext;
    }
#endif
}

int MyQOpenGLWidget::createContext(MyQOpenGLWidget* widget)
{
#ifndef NoSecondContext
    widget->mSecondContext = new QOpenGLContext(0);
    widget->mSecondContext->setShareContext(widget->context());
    widget->mSecondContext->create();
    widget->mSecondContext->makeCurrent(widget->mOffscreenSurf);
#endif
    return 0;
}

int makeContextCurrent(QOpenGLContext* ctx, QSurface* surf)
{
#ifndef NoSecondContext
    ctx->makeCurrent(surf);
#endif
    return 0;
}

int makeContextDone(QOpenGLContext* ctx)
{
#ifndef NoSecondContext
    ctx->doneCurrent();
#endif
    return 0;
}

void MyQOpenGLWidget::MakeOpenCLCurrent()
{
#ifndef NoSecondContext
    context()->doneCurrent();
    mMySurface = context()->surface();
    auto ret = RunInOpenCLThread(makeContextCurrent, mSecondContext, mOffscreenSurf);
    ret.get();
    //qDebug() << "OpenCL Current";
#endif
}

void MyQOpenGLWidget::MakeOpenGLCurrent()
{
#ifndef NoSecondContext
    //qDebug() << "OpenGL Current";
    auto ret = RunInOpenCLThread(makeContextDone, mSecondContext);
    ret.get();
    context()->makeCurrent(mMySurface);
    makeCurrent();
#endif
}

void MyQOpenGLWidget::SetOpenCLProcessor(void (*processWithOpenCL)(int, int, cl_mem, float, float, void *))
{
    mOpenCLProcessor = processWithOpenCL;
}

void MyQOpenGLWidget::SetOpenCLProcessorUserData(void *userData)
{
    mUserData = userData;
}

void MyQOpenGLWidget::StopOpenGL()
{
    //qDebug() << "OpenGL OFF";
    mStopOpenGL = true;
}

void MyQOpenGLWidget::StartOpenGL()
{
    //qDebug() << "OpenGL ON";
    mStopOpenGL = false;
}

void MyQOpenGLWidget::initializeGL()
{
    //qDebug() << "initializeGL";

#ifndef NoSecondContext
    mOffscreenSurf = SingletonOffscreenSurface::Get();
    mMySurface = context()->surface();
    context()->doneCurrent();
    auto ret12 = RunInOpenCLThread(createContext, this);
    ret12.get();
    //MakeOpenCLCurrent();
#endif
    cl_ctx = OpenCL::OpenCLThreadBoundContext::GetCtxOpenGL();
//    //cl_ctx = OpenCL::OpenCLThreadBoundContext::GetCtx(0, 0);

    MakeOpenGLCurrent();
//    // Set up the rendering context, load shaders and other resources, etc.:


    initializeOpenGLFunctions();

    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(2);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable(GL_CONVOLUTION_2D);




    const char* fragmentShader = "#version 330\n"
            "\nuniform sampler2D tex; //this is the texture\n"
            "uniform float imageMinValue; \n"
            "uniform float imageMaxValue; \n"
            "uniform vec4 fixedColor; \n"
            "uniform int useFixedColor; \n"
            //"uniform int useFilter; \n"
            //"uniform int width; \n"
            //"uniform int height; \n"
            //"const float step_w = 1.0/width; \n"
            //"const float step_h = 1.0/height; \n"
            //""
            //"uniform vec2 offset[8];// = { vec2(-step_w, -step_h), vec2(0.0, -step_h), vec2(step_w, -step_h), \n"
            //"                vec2(-step_w, 0.0), vec2(step_w, 0.0), \n"
            //"                vec2(-step_w, step_h), vec2(0.0, step_h), vec2(step_w, step_h) }; \n"
            //""
            "in vec2 UV; //this is the texture coord\n"
            "out vec4 finalColor; //this is the output color of the pixel\n"
            "void main() {\n"
            "      if (useFixedColor == 1) \n"
            "      {\n"
            "          finalColor = fixedColor;\n"
            "      }\n"
            "      else \n"
            "      { \n"
            "          finalColor = texture(tex, UV);//texture(tex, fragTexCoord);\n"
            //"          if (useFilter == 1){ \n"
            //"            for(i=0; i<8; i++ ) \n"
            //"            { \n"
            //"              vec4 tmp = texture(tex, UV + offset[i]); \n"
           // "              finalColor += tmp / 9.0f; \n"
            //"            } \n"
            //"          } \n"
            "          finalColor.xyz = finalColor.xyz - imageMinValue;\n"
            "          finalColor.xyz = finalColor.xyz / (imageMaxValue - imageMinValue);\n"
            "          finalColor.w = 1;\n"
            "      } \n"
            "}";

    const char* vertexShader = "#version 330\n"
                               "layout(location = 0) in vec3 vertexPosition_modelspace;\n"
                               "layout(location = 1) in vec2 vertexUV;\n"
                               "out vec2 UV;"
                               "uniform mat4 MVP;"
                               "void main() {\n"
                               "gl_Position = MVP * vec4(vertexPosition_modelspace,1);\n"
                               "UV = vertexUV;\n"
                               "}";

    mShaderProg = new QOpenGLShaderProgram();
    mShaderProg->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShader);
    mShaderProg->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShader);
    if (!mShaderProg->link())
    {
        QString log = mShaderProg->log();
        qDebug() << log;
    }
    mShaderProg->bind();

    //QOpenGLTexture doesn't work: go basic...
    mTexID = -1;
    glGenTextures(1, &mTexID);
    glBindTexture(GL_TEXTURE_2D, mTexID);

    //unsigned char* data = new unsigned char[imageWidth * imageHeight * 4];
    //memset(data, 0, imageWidth * imageHeight * 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (mIsMultiChannel)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, imageWidth, imageHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        GLint swizzleMask[] = {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }
    else
    {

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, imageWidth, imageHeight, 0, GL_RED, GL_FLOAT, NULL);
        GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }
    glGenerateMipmap(GL_TEXTURE_2D);
    //delete[] data;
    mBaseMatrix.setToIdentity();
    mBaseMatrix.ortho(-2, +2, -2, +2, 1.0f, 15.0f);
    mBaseMatrix.translate(0, 0, -10);

    mMatrix = mBaseMatrix * mViewMatrix;

    mShaderProg->setUniformValue("MVP", mMatrix);
    mShaderProg->setUniformValue("imageMinValue", imageMinValue);
    mShaderProg->setUniformValue("imageMaxValue", imageMaxValue);
    QVector4D vecColor = {1, 1, 1, 1};
    mShaderProg->setUniformValue("fixedColor", vecColor);
    int useFixedColor = 0;
    mShaderProg->setUniformValue("useFixedColor", useFixedColor);


    mSamplerLocation = mShaderProg->uniformLocation("tex");


    mVertices[0] = -imageWidth / 2; mVertices[1] = imageHeight / 2; mVertices[2] = 0;
    mVertices[3] = imageWidth / 2; mVertices[4] = imageHeight / 2; mVertices[5] = 0;
    mVertices[6] = imageWidth / 2; mVertices[7] = -imageHeight / 2; mVertices[8] = 0;
    mVertices[9] = -imageWidth / 2; mVertices[10] = -imageHeight / 2; mVertices[11] = 0;


    mBufferVertices = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    mBufferVertices->setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
    mBufferVertices->create();
    mBufferVertices->bind();
    mBufferVertices->allocate(mVertices, sizeof(mVertices));

    mBufferUV = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    mBufferUV->setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
    mBufferUV->create();
    mBufferUV->bind();
    mBufferUV->allocate(mUV, sizeof(mUV));

    mBufferMarker = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    mBufferMarker->setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
    mBufferMarker->create();
    mBufferMarker->bind();
    mBufferMarker->allocate(mVerticesCircle, sizeof(mVerticesCircle));

    mBufferReference = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    mBufferReference->setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
    mBufferReference->create();

    mBufferScaleBarWhite = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    mBufferScaleBarWhite->setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
    mBufferScaleBarWhite->create();
    mBufferScaleBarWhite->bind();
    mBufferScaleBarWhite->allocate(mVerticesScaleBar, sizeof(mVerticesScaleBar));

    mBufferScaleBarBlack = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
    mBufferScaleBarBlack->setUsagePattern(QOpenGLBuffer::UsagePattern::StaticDraw);
    mBufferScaleBarBlack->create();
    mBufferScaleBarWhite->bind();
    mBufferScaleBarWhite->allocate(mVerticesScaleBar, sizeof(mVerticesScaleBar));


    getOpenCLMemoryBuffer();

    const char *source =
    "__kernel void Main(__write_only image2d_t image, int dimX, int dimY)\n"
    "{\n"
    " int x = get_global_id(0);\n"
    " int y = get_global_id(1);\n"
    " if (x>=dimX || y >=dimY) return;\n"
    " float4 color = (float4)(0.0, 0.0, 0.0, 1.0);\n"
    " if ((y / 128) % 2 > 0 && (x / 128) % 2 > 0)\n"
    "   color = (float4)(1.0, 1.0, 1.0, 1.0);\n"
    " if ((y / 128) % 2 == 0 && (x / 128) % 2 == 0)\n"
    "   color = (float4)(1.0, 1.0, 1.0, 1.0);\n"
    " write_imagef(image, (int2)(x, y), color); \n"
    "}\n";


    MakeOpenCLCurrent();
    mCLKernel = new OpenCL::OpenCLKernel("Main", source);
    mCLKernel->SetProblemSize(16, 16, imageWidth, imageHeight);


    MakeOpenGLCurrent();
    modifyTextureWithOpenCL();
    //qDebug() << "End initializeGL";
}

void MyQOpenGLWidget::resizeGL(int width, int height)
{
    //qDebug() << "resizeGL";
    MakeOpenGLCurrent();
    // Update projection matrix and other size-related settings:

    setRatioXY(width, height);


    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);

    //qDebug() << "End resizeGL";
}

void MyQOpenGLWidget::paintGL()
{
    //qDebug() << "paintGL";
    if (mStopOpenGL)
    {
        //qDebug() << "End paintGL";
        return;
    }



    QPainter painter(this);
    makeCurrent();

    QString caption;
    int scaleWidth;
    computeScaleBarWidth(scaleWidth, caption);
    computeScaleBarVertices(0, scaleWidth);
    mBufferScaleBarBlack->bind();
    mBufferScaleBarBlack->allocate(mVerticesScaleBar, sizeof(mVerticesScaleBar));
    computeScaleBarVertices(2, scaleWidth);
    mBufferScaleBarWhite->bind();
    mBufferScaleBarWhite->allocate(mVerticesScaleBar, sizeof(mVerticesScaleBar));

    // Draw the scene:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mShaderProg->bind();

    mBaseMatrix.setToIdentity();
    mBaseMatrix.ortho(-imageWidth / 2, imageWidth / 2, -imageHeight / 2, imageHeight / 2, -15.0f, 15.0f);
    mBaseMatrix.scale(zoom, zoom, 1);
    mBaseMatrix.translate(imageWidth * xShift / this->width(), -imageHeight * yShift / this->height(), 0);
    mBaseMatrix.scale(ratioX, ratioY, 1);

    mMatrix = mBaseMatrix * mViewMatrix;

    mShaderProg->setUniformValue("MVP", mMatrix);
    QVector4D vecColor = {0, 1, 0, 1};
    mShaderProg->setUniformValue("fixedColor", vecColor);
    int useFixedColor = 0;
    mShaderProg->setUniformValue("useFixedColor", useFixedColor);
    glBindTexture(GL_TEXTURE_2D, mTexID);

    draw();
    drawMarker();

    if (mPixelSizeInNM > 0 && mShowScalebar)
    {
        drawScaleBar();
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glUseProgram(0);
    painter.endNativePainting();

    if (mPixelSizeInNM > 0 && mShowScalebar)
    {
        painter.setPen(Qt::white);
        painter.setFont(QFont("Arial", 10, QFont::Bold));
        painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
        painter.drawText(22, height() - 23, caption); // z = pointT4.z + distOverOp / 4
    }
    painter.end();
}

cl_mem CreateBuffer(cl_context ctx, uint texID, cl_mem oldBuffer)
{
    cl_int err = 0;

    //try to release the old buffer first:
    if (oldBuffer != 0)
    {
        err = clReleaseMemObject(oldBuffer);
    }
//    else
//        qDebug() << "Old CL Buffer is NULL";
    if (err!= CL_SUCCESS)
    {
        qDebug() << "Could not release old buffer... error code: " << err;
    }
//    cl_image_format format2;
//    format2.image_channel_data_type = CL_UNORM_INT8;
//    format2.image_channel_order = CL_RGBA;
//    cl_image_desc desc;
//    memset(&desc, 0, (sizeof(desc)));
//    desc.image_height = height;
//    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
//    desc.image_width = width;


//    cl_mem ret = clCreateImage(ctx, CL_MEM_READ_WRITE, &format2, &desc, 0, &err);
    cl_mem ret = clCreateFromGLTexture2D(ctx, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texID, &err);
    if (err != CL_SUCCESS)
    {
        throw OpenCL::OpenCLException("Failed to map OpenGL texture to OpenCL buffer.");
    }
//    size_t w;
//    size_t h;
//    size_t elSize;
//    cl_image_format format;

//    err = clGetImageInfo(ret, CL_IMAGE_WIDTH, sizeof(size_t), &w, 0);
//    err = clGetImageInfo(ret, CL_IMAGE_HEIGHT, sizeof(size_t), &h, 0);
//    err = clGetImageInfo(ret, CL_IMAGE_ELEMENT_SIZE , sizeof(size_t), &elSize, 0);
//    err = clGetImageInfo(ret, CL_IMAGE_FORMAT , sizeof(cl_image_format), &format, 0);
//    qDebug() << "Width: " << w << " Height: " << h;

    return ret;
}

void MyQOpenGLWidget::getOpenCLMemoryBuffer()
{
    MakeOpenCLCurrent();

    auto ret = RunInOpenCLThread(CreateBuffer, cl_ctx, mTexID, mCLBuffer);
    mCLBuffer = ret.get();

    MakeOpenGLCurrent();
    makeCurrent();
    glGenerateMipmap(GL_TEXTURE_2D);
}

void MyQOpenGLWidget::modifyTextureWithOpenCL()
{
    MakeOpenCLCurrent();

    OpenCL::OpenCLThreadBoundContext::Sync();
    OpenCL::OpenCLThreadBoundContext::AcquireOpenGL(mCLBuffer);
    if (mOpenCLProcessor)
    {
        //(*mOpenCLProcessor)(imageWidth, imageHeight, mCLBuffer, imageMinValue, imageMaxValue, mUserData);
        (*mOpenCLProcessor)(imageWidth, imageHeight, mCLBuffer, 0.0f, 1.0f, mUserData);
    }
    else
    {
        //use default kernel creating checkerboard pattern...
        mCLKernel->SetProblemSize(16, 16, imageWidth, imageHeight);
        mCLKernel->Run(mCLBuffer, imageWidth, imageHeight);
    }
    //try
    //{
    OpenCL::OpenCLThreadBoundContext::Sync();
    OpenCL::OpenCLThreadBoundContext::ReleaseOpenGL(mCLBuffer);
    //}
    //catch (std::exception& ex)
    //{

    //}
    MakeOpenGLCurrent();
    makeCurrent();
    glBindTexture(GL_TEXTURE_2D, mTexID);
    glGenerateMipmap(GL_TEXTURE_2D);
}

void MyQOpenGLWidget::computeScaleBarVertices(float addBorder, int scaleBarWidth)
{

    float width = scaleBarWidth + 2 * addBorder;
    float height = 15 + 2 * addBorder;
    float X = 20 - addBorder;
    float Y = 20 - addBorder;
    mVerticesScaleBar[0] = X; mVerticesScaleBar[1] = Y + height; mVerticesScaleBar[2] = 0;
    mVerticesScaleBar[3] = X + width; mVerticesScaleBar[4] = Y + height; mVerticesScaleBar[5] = 0;
    mVerticesScaleBar[6] = X + width; mVerticesScaleBar[7] = Y; mVerticesScaleBar[8] = 0;
    mVerticesScaleBar[9] = X; mVerticesScaleBar[10] = Y; mVerticesScaleBar[11] = 0;
}

void MyQOpenGLWidget::computeScaleBarWidth(int &widthInPixelsToShow, QString &label)
{
    float pixelRatio;
    if (ratioX == 1)
    {
        pixelRatio = width() / (float)imageWidth;
    }
    else
    {
        pixelRatio = height() / (float)imageHeight;
    }
    pixelRatio *= zoom;

    bool switchToMicrons = false;
    float micronFactor = 1.0f;
    if (100.0f / pixelRatio * mPixelSizeInNM > 500.0f)
    {
        switchToMicrons = true;
        micronFactor = 0.001f;
    }

    float desiredCaption;
    float pixelWidth;

    desiredCaption = 0.5f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);

    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("0.5 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 0.5 - Width: " << pixelWidth;
    desiredCaption = 0.8f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("0.8 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 0.8 - Width: " << pixelWidth;
    desiredCaption = 1.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("1 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 1 - Width: " << pixelWidth;
    desiredCaption = 3.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("3 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 3 - Width: " << pixelWidth;
    desiredCaption = 5.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("5 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 5 - Width: " << pixelWidth;
    desiredCaption = 8.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("8 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 8 - Width: " << pixelWidth;
    desiredCaption = 10.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("10 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 10 - Width: " << pixelWidth;
    desiredCaption = 30.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("30 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 30 - Width: " << pixelWidth;
    desiredCaption = 50.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("50 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 50 - Width: " << pixelWidth;
    desiredCaption = 80.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("80 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 80 - Width: " << pixelWidth;
    desiredCaption = 100.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("100 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 100 - Width: " << pixelWidth;
    desiredCaption = 300.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("300 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 300 - Width: " << pixelWidth;
    desiredCaption = 500.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("500 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 500 - Width: " << pixelWidth;
    desiredCaption = 800.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("800 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    //qDebug() << "Desired: 800 - Width: " << pixelWidth;
    desiredCaption = 1000.f;
    pixelWidth = desiredCaption * pixelRatio / (mPixelSizeInNM * micronFactor);
    if (pixelWidth < 300 && pixelWidth >= 100)
    {
        label = QString::asprintf("1000 %s", switchToMicrons ? "µm" : "nm");
        widthInPixelsToShow = round(pixelWidth);
        return;
    }
    else
    {
        if (pixelWidth >= 300)
        {
            label = QString::asprintf("%0.2f %s", 300.0f / pixelRatio * (mPixelSizeInNM * micronFactor), switchToMicrons ? "µm" : "nm");
            widthInPixelsToShow = 300;
            return;
        }
        else
        {
            label = QString::asprintf("%0.2f %s", 100.0f / pixelRatio * (mPixelSizeInNM * micronFactor), switchToMicrons ? "µm" : "nm");
            widthInPixelsToShow = 100;
            return;
        }
    }

    //width of scalebar in pixel / pixelRatio * pixelSize = beschriftung

}

void MyQOpenGLWidget::drawScaleBar()
{
    QMatrix4x4 mat;
    mat.setToIdentity();
    mat.ortho(0, width(), 0, height(), -15.0f, 15.0f);
    mat.translate(0, 0, 10);

    mShaderProg->setUniformValue("MVP", mat);
    QVector4D vecColor = {1, 1, 1, 1};
    mShaderProg->setUniformValue("fixedColor", vecColor);
    int useFixedColor = 1;
    mShaderProg->setUniformValue("useFixedColor", useFixedColor);

    mBufferUV->bind();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);
    mBufferScaleBarWhite->bind(),
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_QUADS, 0, 4*3);

    mat.translate(0, 0, 0.1f);
    mShaderProg->setUniformValue("MVP", mat);

    vecColor = {0, 0, 0, 0};
    mShaderProg->setUniformValue("fixedColor", vecColor);

    mBufferUV->bind();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);
    mBufferScaleBarBlack->bind(),
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_QUADS, 0, 4*3);
    //    mBufferVertices->bind();
    //    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    //    glEnableVertexAttribArray(0);
}

void MyQOpenGLWidget::drawMarker()
{

    mShaderProg->setUniformValue("MVP", mMatrix);
    int useFixedColor = 1;
    mShaderProg->setUniformValue("useFixedColor", useFixedColor);

    if (mShowAlignedMarkers && mShowAlignedMarkersAsArrows)
    {
        memset(mVerticesCircle, 0, sizeof(mVerticesCircle));

        for (int i = 0; i < mMarkerXAlig.size(); i++)
        {
            if (mMarkerXAlig[i] < 0)
            {
                continue;
            }
            /*QVector4D vecColor = {0, 0, 1, 1};

            mShaderProg->setUniformValue("fixedColor", vecColor);
            int counter = 0;
            for (int ang = 0; ang < 360; ang += 5)
            {
                float degInRad = ang * 3.1415f / 180.0f;
                mVerticesCircle[3*counter + 0] = mMarkerXAlig[i] + cosf(degInRad) * (mMarkerSize+2) - imageWidth / 2.0f;
                mVerticesCircle[3*counter + 1] = mMarkerYAlig[i] + sinf(degInRad) * (mMarkerSize+2) - imageHeight / 2.0f;
                mVerticesCircle[3*counter + 2] = 0;
                counter++;
            }


            mBufferUV->bind();
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(1);
            mBufferMarker->bind();
            mBufferMarker->allocate(mVerticesCircle, sizeof(mVerticesCircle));
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(0);

            glLineWidth(2);
            glDrawArrays(GL_LINE_LOOP, 0, 72);*/

            QVector4D vecColor = {1, 0, 0, 1};

            mShaderProg->setUniformValue("fixedColor", vecColor);


            QTransform trans1;
            trans1.rotate(10);
            QTransform trans2;
            trans2.rotate(-10);
            QVector2D vecDir(mMarkerX[i] - mMarkerXAlig[i], mMarkerY[i] - mMarkerYAlig[i]);
            float length = vecDir.length();

            if (length < 0.5f)
            {
                mVerticesCircle[0] = mMarkerX[i] - mMarkerSize * 0.5f * sqrtf(2.f) - imageWidth / 2.0f;
                mVerticesCircle[1] = mMarkerY[i] - mMarkerSize * 0.5f * sqrtf(2.f) - imageHeight / 2.0f;
                mVerticesCircle[2] = 0;
                mVerticesCircle[3] = mMarkerX[i] + mMarkerSize * 0.5f * sqrtf(2.f) - imageWidth / 2.0f;
                mVerticesCircle[4] = mMarkerY[i] + mMarkerSize * 0.5f * sqrtf(2.f) - imageHeight / 2.0f;
                mVerticesCircle[5] = 0;

                mVerticesCircle[6] = mMarkerX[i] - mMarkerSize * 0.5f * sqrtf(2.f) - imageWidth / 2.0f;
                mVerticesCircle[7] = mMarkerY[i] + mMarkerSize * 0.5f * sqrtf(2.f) - imageHeight / 2.0f;
                mVerticesCircle[8] = 0;
                mVerticesCircle[9] = mMarkerX[i] + mMarkerSize * 0.5f * sqrtf(2.f) - imageWidth / 2.0f;
                mVerticesCircle[10] = mMarkerY[i] - mMarkerSize * 0.5f * sqrtf(2.f) - imageHeight / 2.0f;
                mVerticesCircle[11] = 0;

                mVerticesCircle[12] = 0;
                mVerticesCircle[13] = 0;
                mVerticesCircle[14] = 0;
                mVerticesCircle[15] = 0;
                mVerticesCircle[16] = 0;
                mVerticesCircle[17] = 0;


                mBufferUV->bind();
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
                glEnableVertexAttribArray(1);
                mBufferMarker->bind();
                mBufferMarker->allocate(mVerticesCircle, sizeof(mVerticesCircle));
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
                glEnableVertexAttribArray(0);

                float minmaxWidth[2];
                glGetFloatv(GL_SMOOTH_LINE_WIDTH_RANGE, minmaxWidth);

                if (mMarkerSize < 3)
                    glLineWidth(1);
                else
                    glLineWidth(2.0f);
                glDrawArrays(GL_LINES, 0, 12);
            }
            else
            {
                QVector2D pos(mMarkerXAlig[i], mMarkerYAlig[i]);
                vecDir.normalize();

                if (length > mMarkerSize)
                    vecDir *= mMarkerSize;

                QVector3D vecDir1 = trans1 * vecDir.toVector3D() + pos;
                QVector3D vecDir2 = trans2 * vecDir.toVector3D() + pos;

                mVerticesCircle[0] = mMarkerX[i] - imageWidth / 2.0f;
                mVerticesCircle[1] = mMarkerY[i] - imageHeight / 2.0f;
                mVerticesCircle[2] = 0;
                mVerticesCircle[3] = mMarkerXAlig[i] - imageWidth / 2.0f;
                mVerticesCircle[4] = mMarkerYAlig[i] - imageHeight / 2.0f;
                mVerticesCircle[5] = 0;

                mVerticesCircle[6] = mMarkerXAlig[i] - imageWidth / 2.0f;
                mVerticesCircle[7] = mMarkerYAlig[i] - imageHeight / 2.0f;
                mVerticesCircle[8] = 0;
                mVerticesCircle[9] = vecDir1.x() - imageWidth / 2.0f;
                mVerticesCircle[10] = vecDir1.y() - imageHeight / 2.0f;
                mVerticesCircle[11] = 0;

                mVerticesCircle[12] = mMarkerXAlig[i] - imageWidth / 2.0f;
                mVerticesCircle[13] = mMarkerYAlig[i] - imageHeight / 2.0f;
                mVerticesCircle[14] = 0;
                mVerticesCircle[15] = vecDir2.x() - imageWidth / 2.0f;
                mVerticesCircle[16] = vecDir2.y() - imageHeight / 2.0f;
                mVerticesCircle[17] = 0;


                mBufferUV->bind();
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
                glEnableVertexAttribArray(1);
                mBufferMarker->bind();
                mBufferMarker->allocate(mVerticesCircle, sizeof(mVerticesCircle));
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
                glEnableVertexAttribArray(0);

                glLineWidth(2);
                glDrawArrays(GL_LINES, 0, 18);
            }
        }
    }
    else
    {
        if (mShowAlignedMarkers && !mShowAlignedMarkersAsArrows)
        {
            for (int i = 0; i < mMarkerXAlig.size(); i++)
            {
                if (mMarkerXAlig[i] < 0)
                {
                    continue;
                }
                QVector4D vecColor = {0, 0, 1, 1};

                mShaderProg->setUniformValue("fixedColor", vecColor);
                int counter = 0;
                for (int ang = 0; ang < 360; ang += 5)
                {
                    float degInRad = ang * 3.1415f / 180.0f;
                    mVerticesCircle[3*counter + 0] = mMarkerXAlig[i] + cosf(degInRad) * (mMarkerSize+2) - imageWidth / 2.0f;
                    mVerticesCircle[3*counter + 1] = mMarkerYAlig[i] + sinf(degInRad) * (mMarkerSize+2) - imageHeight / 2.0f;
                    mVerticesCircle[3*counter + 2] = 0;
                    counter++;
                }


                mBufferUV->bind();
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
                glEnableVertexAttribArray(1);
                mBufferMarker->bind();
                mBufferMarker->allocate(mVerticesCircle, sizeof(mVerticesCircle));
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
                glEnableVertexAttribArray(0);

                glLineWidth(2);
                glDrawArrays(GL_LINE_LOOP, 0, 72);
            }
        }
    }

    if (!(mShowAlignedMarkers && mShowAlignedMarkersAsArrows))
    {
        for (int i = 0; i < mMarkerX.size(); i++)
        {
            if (mMarkerX[i] < 0)
            {
                continue;
            }
            QVector4D vecColor = {1, 0, 0, 1};
            if (i == mActiveMarker)
                vecColor = {0, 1, 0, 1};
            mShaderProg->setUniformValue("fixedColor", vecColor);
            int counter = 0;
            for (int ang = 0; ang < 360; ang += 5)
            {
                float degInRad = ang * 3.1415f / 180.0f;
                mVerticesCircle[3*counter + 0] = mMarkerX[i] + cosf(degInRad) * mMarkerSize - imageWidth / 2.0f;
                mVerticesCircle[3*counter + 1] = mMarkerY[i] + sinf(degInRad) * mMarkerSize - imageHeight / 2.0f;
                //mVerticesCircle[3*counter + 0] = 2000 + cosf(degInRad) * 10.0f - imageWidth / 2.0f;
                //mVerticesCircle[3*counter + 1] = 2000 + sinf(degInRad) * 10.0f - imageHeight / 2.0f;
                mVerticesCircle[3*counter + 2] = 0;
                counter++;
            }


            mBufferUV->bind();
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(1);
            mBufferMarker->bind();
            mBufferMarker->allocate(mVerticesCircle, sizeof(mVerticesCircle));
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glEnableVertexAttribArray(0);

            glLineWidth(2);
            glDrawArrays(GL_LINE_LOOP, 0, 72);
        }
    }
}

void MyQOpenGLWidget::draw()
{
    //modifyTextureWithOpenCL();
    mShaderProg->setUniformValue("imageMinValue", imageMinValue);
    mShaderProg->setUniformValue("imageMaxValue", imageMaxValue);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mTexID);
    glUniform1i(mSamplerLocation, 0);

    mBufferUV->bind();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    mBufferVertices->bind(),
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_QUADS, 0, 4*3);

}

QSize MyQOpenGLWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize MyQOpenGLWidget::sizeHint() const
{
    return QSize(400, 400);
}

void MyQOpenGLWidget::setXShift(float value)
{
    if (xShift != value) {
        xShift = value;
        emit xShiftChanged(xShift);
        update();
    }
}

void MyQOpenGLWidget::setYShift(float value)
{
    if (yShift != value) {
        yShift = value;
        emit yShiftChanged(yShift);
        update();
    }
}

void MyQOpenGLWidget::setZoom(float value)
{
    if (zoom != value) {
        zoom = value;
        emit zoomChanged(zoom);
        update();
    }
}

void MyQOpenGLWidget::mouseDoubleClickEvent( QMouseEvent * e )
{
    if ( e->button() == Qt::LeftButton )
    {
        StopOpenGL();
        setZoom(1);
        setXShift(0);
        StartOpenGL();
        setYShift(0);
    }
}

void MyQOpenGLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();

    QMatrix4x4 invMat = mMatrix.inverted();

    QVector4D vector(event->x() / (float)width() * 2.0f - 1.0f, -event->y() / (float)height() * 2.0f + 1.0f, 0, 1);
    QVector4D position = invMat * vector;

    position += QVector4D(imageWidth / 2, imageHeight / 2, 0, 0);
    int px = (int)position.x();
    int py = (int)position.y();
    bool isHit = px >= 0 && px < imageWidth && py >= 0 && py < imageHeight;

    mouseClicked(px, py, isHit, event->buttons());
}

void MyQOpenGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    QMatrix4x4 invMat = mMatrix.inverted();

    QVector4D vector(event->x() / (float)width() * 2.0f - 1.0f, -event->y() / (float)height() * 2.0f + 1.0f, 0, 1);
    QVector4D position = invMat * vector;

    position += QVector4D(imageWidth / 2, imageHeight / 2, 0, 0);
    int px = (int)position.x();
    int py = (int)position.y();
    bool isHit = px >= 0 && px < imageWidth && py >= 0 && py < imageHeight;

    mousePositionChanged(px, py, isHit);

    //qDebug() << (int)position.x() << (int)position.y();

    if (event->buttons() & Qt::LeftButton)
    {
        StopOpenGL();
        setXShift(xShift + dx / zoom);
        StartOpenGL();
        setYShift(yShift + dy / zoom);
    } else if (event->buttons() & Qt::RightButton) {
//        setXShift(xShift + dx);
//        setYShift(yShift + dx);
    }

    lastPos = event->pos();
}

void MyQOpenGLWidget::wheelEvent(QWheelEvent *event)
{
    float delta = event->delta();
    float valX = (event->pos().x() - width() / 2) / zoom;
    float valY = (event->pos().y() - height() / 2) / zoom;
    if (delta < 0)
    {
        StopOpenGL();
        setZoom(zoom + zoom * delta / 1200.0f);

        float shiftDiffX = (event->pos().x() - width() / 2) / zoom - valX;
        float shiftDiffY = (event->pos().y() - height() / 2) / zoom - valY;

        setXShift(xShift + shiftDiffX);
        StartOpenGL();
        setYShift(yShift + shiftDiffY);
    }
    else
    {
        StopOpenGL();
        float diff = (zoom + delta / 1200.0f) * delta / 1200.0f;
        setZoom(zoom + diff);

        float shiftDiffX = (event->pos().x() - width() / 2) / zoom - valX;
        float shiftDiffY = (event->pos().y() - height() / 2) / zoom - valY;

        setXShift(xShift + shiftDiffX);
        StartOpenGL();
        setYShift(yShift + shiftDiffY);

    }

}

void MyQOpenGLWidget::setNewImageDimensions(int sizeX, int sizeY, bool isMultiChannel)
{
    //qDebug() << "setNewImageDimensions";
    mMarkerX.clear();
    mMarkerY.clear();
    mMarkerXAlig.clear();
    mMarkerYAlig.clear();
    mActiveMarker = -1;
    if (imageWidth == sizeX && imageHeight == sizeY && mIsMultiChannel == isMultiChannel)
    {
        modifyTextureWithOpenCL();
        update();
        return;
    }

    MakeOpenGLCurrent();
    imageWidth = sizeX;
    imageHeight = sizeY;

    setRatioXY(width(), height());

    glDeleteTextures(1, &mTexID);
    glGenTextures(1, &mTexID);
    glBindTexture(GL_TEXTURE_2D, mTexID);

    if(linearInterpolation)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, imageWidth, imageHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    mIsMultiChannel = isMultiChannel;
    if (mIsMultiChannel)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, imageWidth, imageHeight, 0, GL_RGBA, GL_FLOAT, NULL);
        GLint swizzleMask[] = {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, imageWidth, imageHeight, 0, GL_RED, GL_FLOAT, NULL);
        GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }

    glGenerateMipmap(GL_TEXTURE_2D);

    mVertices[0] = -imageWidth / 2; mVertices[1] = imageHeight / 2; mVertices[2] = 0;
    mVertices[3] = imageWidth / 2; mVertices[4] = imageHeight / 2; mVertices[5] = 0;
    mVertices[6] = imageWidth / 2; mVertices[7] = -imageHeight / 2; mVertices[8] = 0;
    mVertices[9] = -imageWidth / 2; mVertices[10] = -imageHeight / 2; mVertices[11] = 0;

    //write new vertice locations
    mBufferVertices->bind();
    mBufferVertices->allocate(mVertices, sizeof(mVertices));
    //request the new openCL pointer to the texture
    getOpenCLMemoryBuffer();
    //adopt the kernel dimensions to the new size
    mCLKernel->SetProblemSize(16, 16, imageWidth, imageHeight);

    modifyTextureWithOpenCL();
    //repaint
    update();
    //qDebug() << "End setNewImageDimensions";
}

void MyQOpenGLWidget::updateImage()
{
    mMarkerX.clear();
    mMarkerY.clear();
    mMarkerXAlig.clear();
    mMarkerYAlig.clear();
    mActiveMarker = -1;
    modifyTextureWithOpenCL();
    update();
}

void MyQOpenGLWidget::setViewMatrix(QMatrix4x4 aMatrix)
{
    mViewMatrix = aMatrix;
    update();
}

void MyQOpenGLWidget::setMarkerPositions(QVector<float>& aX, QVector<float>& aY, QVector<float>& xAlig, QVector<float>& yAlig)
{
    mMarkerX = aX;
    mMarkerY = aY;
    mMarkerXAlig = xAlig;
    mMarkerYAlig = yAlig;
    update();
}

void MyQOpenGLWidget::setActiveMarker(int aMarkerIdx)
{
    mActiveMarker = aMarkerIdx;
    update();
}

void MyQOpenGLWidget::setMarkerSize(int size)
{
    if (mMarkerSize != size)
    {
        mMarkerSize = size;
        emit markerSizeChanged(mMarkerSize);
        update();
    }
}

void MyQOpenGLWidget::SetShowAlignedMarkers(bool aShow)
{
    mShowAlignedMarkers = aShow;
    update();
}

void MyQOpenGLWidget::SetShowAlignedMarkersAsArrows(bool aShowAsArrow)
{
    mShowAlignedMarkersAsArrows = aShowAsArrow;
    update();
}

void MyQOpenGLWidget::setImageCenterValue(float center)
{
    if (imageCenterValue != center)
    {
        imageCenterValue = center;
        imageMinValue = imageCenterValue - imageContrastWidthValue;
        imageMaxValue = imageCenterValue + imageContrastWidthValue;
        emit imageCenterValueChanged(imageCenterValue);
        update();
    }
}

void MyQOpenGLWidget::setImageContrastWidthValue(float contrastWidth)
{
    if (imageContrastWidthValue != contrastWidth - 0.5f)
    {
        imageContrastWidthValue = contrastWidth - 0.5f;
        imageMinValue = imageCenterValue - imageContrastWidthValue;
        imageMaxValue = imageCenterValue + imageContrastWidthValue;
        emit imageCenterValueChanged(imageContrastWidthValue);
        update();
    }
}

void MyQOpenGLWidget::setLinearInterpolation(bool value)
{
    if (linearInterpolation != value)
    {
        linearInterpolation = value;
        if (linearInterpolation)
        {
            glBindTexture(GL_TEXTURE_2D, mTexID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D, mTexID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        //paintGL();
        emit linearInterpolationChanged(linearInterpolation);
        modifyTextureWithOpenCL();
        update();
    }
}

void MyQOpenGLWidget::setPixelSize(float value)
{
    if (mPixelSizeInNM <= 0)
    {
        setShowScaleBar(false);
    }
    if (mPixelSizeInNM != value)
    {
        mPixelSizeInNM = value;

        emit pixelSizeChanged(mPixelSizeInNM);
        update();
    }
}

void MyQOpenGLWidget::setShowScaleBar(bool value)
{
    value = value && mPixelSizeInNM > 0;
    if (mShowScalebar != value)
    {
        mShowScalebar = value;

        emit showScaleBarChanged(mShowScalebar);
        update();
    }
}

void MyQOpenGLWidget::setRatioXY(int width, int height)
{
    if (width < height)
    {
        ratioY = (float)width / (float)height * ((float)imageHeight / (float)imageWidth);
        ratioX = 1;
        if (ratioY > 1.0f)
        {
            ratioX = 1.0f / ratioY;
            ratioY = 1.0f;
        }
        //qDebug() << "Ratio Y: " << ratioY << "\n";
    }
    else
    {
        ratioX = (float)height / (float)width * ((float)imageWidth / (float)imageHeight);
        ratioY = 1;
        if (ratioX > 1.0f)
        {
            ratioY = 1.0f / ratioX;
            ratioX = 1.0f;
        }
        //qDebug() << "Ratio X: " << ratioX << "\n";
    }
}
