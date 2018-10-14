#-------------------------------------------------
#
# Project created by QtCreator 2016-11-16T11:08:56
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Clicker
TEMPLATE = app

CONFIG += static

#LIBS += -lopengl32

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

QMAKE_CXXFLAGS += -std=c++11 -fpermissive
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS +=  -fopenmp

#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lCudaHelpers
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lCudaHelpers
#else:unix: LIBS += -L$$PWD/../EmTools/x64/ -lCudaHelpers

#INCLUDEPATH += $$PWD/../EmTools/CudaHelpers
#DEPENDPATH += $$PWD/../EmTools/CudaHelpers

#win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libCudaHelpers.a
#else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libCudaHelpers.a
#else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/CudaHelpers.lib
#else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/CudaHelpers.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libCudaHelpers.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lFileIO
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lFileIO
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lFileIO
else:unix: SOURCES += $$PWD/../EmTools/FileIO/*.cpp

INCLUDEPATH += $$PWD/../EmTools/FileIO
#DEPENDPATH += $$PWD/../EmTools/FileIO

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libFileIO.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libFileIO.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/FileIO.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/FileIO.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libFileIO.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lFilterGraph
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lFilterGraph
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lFilterGraph
else:unix: SOURCES += $$PWD/../EmTools/FilterGraph/*.cpp

INCLUDEPATH += $$PWD/../EmTools/FilterGraph
#DEPENDPATH += $$PWD/../EmTools/FilterGraph

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libFilterGraph.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libFilterGraph.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/FilterGraph.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/FilterGraph.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libFilterGraph.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lMemoryPool
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lMemoryPool
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lMemoryPool
else:unix: SOURCES += $$PWD/../EmTools/MemoryPool/*.cpp

INCLUDEPATH += $$PWD/../EmTools/MemoryPool
#DEPENDPATH += $$PWD/../EmTools/MemoryPool

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libMemoryPool.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libMemoryPool.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/MemoryPool.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/MemoryPool.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libMemoryPool.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lMKLog
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lMKLog
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lMKLog
else:unix: SOURCES += $$PWD/../EmTools/MKLog/*.cpp

INCLUDEPATH += $$PWD/../EmTools/MKLog
#DEPENDPATH += $$PWD/../EmTools/MKLog

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libMKLog.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libMKLog.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/MKLog.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/MKLog.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libMKLog.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lOpenCLHelpers
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lOpenCLHelpers
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lOpenCLHelpers
else:unix: SOURCES += $$PWD/../EmTools/OpenCLHelpers/*.cpp

INCLUDEPATH += $$PWD/../EmTools/OpenCLHelpers
#DEPENDPATH += $$PWD/../EmTools/OpenCLHelpers

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libOpenCLHelpers.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libOpenCLHelpers.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/OpenCLHelpers.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/OpenCLHelpers.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libOpenCLHelpers.a

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lThreading
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lThreading
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lThreading
else:unix: SOURCES += $$PWD/../EmTools/Threading/*.cpp

INCLUDEPATH += $$PWD/../EmTools/Threading
#DEPENDPATH += $$PWD/../EmTools/Threading

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libThreading.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libThreading.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/Threading.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/Threading.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libThreading.a

unix|win32: LIBS += -L/usr/local/cuda-8.0/lib/x64/ -lOpenCL
#unix|win32: LIBS += -L$$PWD/'../../../../../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/lib/x64/' -lcuda

INCLUDEPATH += /usr/local/cuda-8.0/include
#DEPENDPATH += /usr/local/cuda-8.0/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lMinimization
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lMinimization
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lMinimization
else:unix: SOURCES += $$PWD/../EmTools/Minimization/*.cpp

INCLUDEPATH += $$PWD/../EmTools/Minimization
#DEPENDPATH += $$PWD/../EmTools/Minimization

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libMinimization.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libMinimization.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/Minimization.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/Minimization.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libMinimization.a

unix|win32: LIBS += -lfftw3f

#INCLUDEPATH += $$PWD/../EmTools/libs/FFTW
#DEPENDPATH += $$PWD/../EmTools/libs/FFTW

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../EmTools/x64/Release/ -lCrossCorrelation
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../EmTools/x64/debug/ -lCrossCorrelation
#else:unix: LIBS += -L$$PWD/../EmTools/x64/Release -lCrossCorrelation
else:unix: SOURCES += $$PWD/../EmTools/CrossCorrelation/*.cpp

INCLUDEPATH += $$PWD/../EmTools/CrossCorrelation
#DEPENDPATH += $$PWD/../EmTools/CrossCorrelation

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/libCrossCorrelation.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/libCrossCorrelation.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/Release/CrossCorrelation.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../EmTools/x64/debug/CrossCorrelation.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../EmTools/x64/libCrossCorrelation.a
