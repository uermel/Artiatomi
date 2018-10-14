#-------------------------------------------------
#
# Project created by QtCreator 2016-11-16T11:08:56
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ThirdQt
TEMPLATE = app

LIBS += -lopengl32

SOURCES += main.cpp\
        mymainwindow.cpp \
    myqlineedit.cpp \
    myqopenglwidget.cpp \
    singletonoffscreensurface.cpp \
    floatlineedit.cpp \
    floatslider.cpp \
    histogramwidget.cpp \
    singleframecontroller.cpp \
    intlineedit.cpp \
    baseframecontroller.cpp \
    tiltseriescontroller.cpp \
    markerlist.cpp \
    cropsizeselector.cpp \
    alignmentreport.cpp \
    intslider.cpp \
    reconstructionassistant.cpp \
    squarelabel.cpp

HEADERS  += mymainwindow.h \
    myqlineedit.h \
    myqopenglwidget.h \
    singletonoffscreensurface.h \
    floatlineedit.h \
    floatslider.h \
    histogramwidget.h \
    singleframecontroller.h \
    intlineedit.h \
    baseframecontroller.h \
    tiltseriescontroller.h \
    markerlist.h \
    cropsizeselector.h \
    alignmentreport.h \
    intslider.h \
    reconstructionassistant.h \
    squarelabel.h

FORMS    += mymainwindow.ui \
    alignmentreport.ui \
    reconstructionassistant.ui

#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lCudaHelpers
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lCudaHelpers
#else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lCudaHelpers

#INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/CudaHelpers
#DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/CudaHelpers

#win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libCudaHelpers.a
#else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libCudaHelpers.a
#else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/CudaHelpers.lib
#else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/CudaHelpers.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libCudaHelpers.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lFileIO
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lFileIO
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lFileIO

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/FileIO
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/FileIO

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libFileIO.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libFileIO.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/FileIO.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/FileIO.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libFileIO.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lFilterGraph
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lFilterGraph
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lFilterGraph

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/FilterGraph
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/FilterGraph

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libFilterGraph.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libFilterGraph.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/FilterGraph.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/FilterGraph.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libFilterGraph.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lMemoryPool
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lMemoryPool
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lMemoryPool

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/MemoryPool
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/MemoryPool

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libMemoryPool.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libMemoryPool.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/MemoryPool.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/MemoryPool.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libMemoryPool.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lMKLog
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lMKLog
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lMKLog

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/MKLog
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/MKLog

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libMKLog.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libMKLog.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/MKLog.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/MKLog.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libMKLog.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lOpenCLHelpers
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lOpenCLHelpers
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lOpenCLHelpers

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/OpenCLHelpers
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/OpenCLHelpers

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libOpenCLHelpers.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libOpenCLHelpers.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/OpenCLHelpers.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/OpenCLHelpers.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libOpenCLHelpers.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lThreading
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lThreading
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lThreading

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/Threading
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/Threading

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libThreading.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libThreading.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/Threading.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/Threading.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libThreading.a

unix|win32: LIBS += -L$$PWD/'../../../../../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64/' -lOpenCL
#unix|win32: LIBS += -L$$PWD/'../../../../../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64/' -lcuda

INCLUDEPATH += $$PWD/'../../../../../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include'
DEPENDPATH += $$PWD/'../../../../../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/include'

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lMinimization
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lMinimization
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lMinimization

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/Minimization
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/Minimization

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libMinimization.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libMinimization.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/Minimization.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/Minimization.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libMinimization.a

unix|win32: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/libs/FFTW/ -llibfftw3f-3

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/libs/FFTW
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/libs/FFTW

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/release/ -lCrossCorrelation
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/debug/ -lCrossCorrelation
else:unix: LIBS += -L$$PWD/../../../../../Source/Repos/EmTools/x64/ -lCrossCorrelation

INCLUDEPATH += $$PWD/../../../../../Source/Repos/EmTools/CrossCorrelation
DEPENDPATH += $$PWD/../../../../../Source/Repos/EmTools/CrossCorrelation

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/libCrossCorrelation.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/libCrossCorrelation.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/release/CrossCorrelation.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/debug/CrossCorrelation.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../../Source/Repos/EmTools/x64/libCrossCorrelation.a
