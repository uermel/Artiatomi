cmake_minimum_required(VERSION 3.14)
project(Artiatomi LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(EMTOOLS_PATH "EmTools" CACHE PATH "Path to EmTools source.")
set(CLICKER_PATH "Clicker/Clicker" CACHE PATH "Path to Clicker source.")

######################### CUDA stuff #############################
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_library(CUDA_LIBRARY cuda ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUFFT_LIBRARY cufft ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY libnppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPI_LIBRARY libnppi ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set(CUDA_LIBS ${CUDA_LIBRARY} ${CUFFT_LIBRARY} ${NPPC_LIBRARY} ${NPPI_LIBRARY})
######################### CUDA stuff #############################

######################### OpenCL stuff #############################

find_package(OpenCL REQUIRED)

######################### OpenCL stuff #############################

######################### Qt stuff #############################
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(Qt5 COMPONENTS Core Widgets DBus OpenGL REQUIRED)
find_package(OpenGL REQUIRED)

######################### Qt stuff #############################

######################### System stuff ################################

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED )

######################### System stuff ################################

######################### MKLog #####################################
set(MK_LOG_SRC
        ${EMTOOLS_PATH}/MKLog/MKLog.cpp)

set(MK_LOG_INC
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/MKLog)

add_library(MKLog
        ${MK_LOG_SRC})

target_include_directories(MKLog PUBLIC ${MK_LOG_INC})
######################### MKLog #####################################

######################### Threading ################################
set(THREADING_SRC
        ${EMTOOLS_PATH}/Threading/SpecificBackgroundThread.cpp)

set(THREADING_INC
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/Threading)

set(THREADING_LIB
        Threads::Threads)

add_library(Threading
        ${THREADING_SRC})

target_include_directories(Threading PUBLIC ${THREADING_INC})
target_link_libraries(Threading ${THREADING_LIB})
######################### Threading ################################

######################### CudaHelpers ################################
set(CUDA_HELPERS_SRC
        ${EMTOOLS_PATH}/CudaHelpers/CudaArrays.cpp
        ${EMTOOLS_PATH}/CudaHelpers/CudaContext.cpp
        ${EMTOOLS_PATH}/CudaHelpers/CudaDeviceProperties.cpp
        ${EMTOOLS_PATH}/CudaHelpers/CudaException.cpp
        ${EMTOOLS_PATH}/CudaHelpers/CudaHelpers.cpp
        ${EMTOOLS_PATH}/CudaHelpers/CudaKernel.cpp
        ${EMTOOLS_PATH}/CudaHelpers/CudaTextures.cpp
        ${EMTOOLS_PATH}/CudaHelpers/CudaVariables.cpp
        ${EMTOOLS_PATH}/CudaHelpers/NPPImageBase.cpp
        ${EMTOOLS_PATH}/CudaHelpers/NPPImages.cpp)

set(CUDA_HELPERS_INC
        ${EMTOOLS_PATH}/CudaHelpers
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/MemoryPool
        ${EMTOOLS_PATH}/Threading
        ${EMTOOLS_PATH}/MKLog
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CUDA_HELPERS_LIB
        MemoryPool
        Threading
        MKLog
        ${CUDA_LIBS})

add_library(CudaHelpers
        ${CUDA_HELPERS_SRC})

target_include_directories(CudaHelpers PUBLIC ${CUDA_HELPERS_INC})
target_link_libraries(CudaHelpers ${CUDA_HELPERS_LIB})
######################### CudaHelpers ################################


######################### FilterGraph #####################################
set(FILTER_GRAPH_SRC
        ${EMTOOLS_PATH}/FilterGraph/FilterPoint2D.cpp
        ${EMTOOLS_PATH}/FilterGraph/FilterPoint3D.cpp
        ${EMTOOLS_PATH}/FilterGraph/FilterROI3D.cpp
        ${EMTOOLS_PATH}/FilterGraph/FilterROI.cpp
        ${EMTOOLS_PATH}/FilterGraph/FilterSize3D.cpp
        ${EMTOOLS_PATH}/FilterGraph/FilterSize.cpp
        ${EMTOOLS_PATH}/FilterGraph/HostSinkElement2D.cpp
        ${EMTOOLS_PATH}/FilterGraph/HostSourceElement2D.cpp
        ${EMTOOLS_PATH}/FilterGraph/HostSplitterElement2D.cpp
        ${EMTOOLS_PATH}/FilterGraph/IComputeParameters.cpp
        ${EMTOOLS_PATH}/FilterGraph/IFilterElement.cpp
        ${EMTOOLS_PATH}/FilterGraph/Matrix.cpp
        ${EMTOOLS_PATH}/FilterGraph/PointF.cpp)

set(FILTER_GRAPH_INC
        ${EMTOOLS_PATH}/FilterGraph
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/MemoryPool)

set(FILTER_GRAPH_LIB
        MemoryPool)

add_library(FilterGraph
        ${FILTER_GRAPH_SRC})

target_include_directories(FilterGraph PUBLIC ${FILTER_GRAPH_INC})
target_link_libraries(FilterGraph ${FILTER_GRAPH_LIB})
######################### FilterGraph #####################################

######################### Minimization ################################
set(MINIMIZATION_SRC
        ${EMTOOLS_PATH}/Minimization/levmar.cpp)

set(MINIMIZATION_INC
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/Minimization)

add_library(Minimization
        ${MINIMIZATION_SRC})

target_include_directories(Minimization PUBLIC ${MINIMIZATION_INC})
######################### Minimization ################################

######################### FileIO #####################################
set(FILEIO_SRC
        ${EMTOOLS_PATH}/FileIO/Dm3File.cpp
        #${EMTOOLS_PATH}/FileIO/Dm3FileStack.cpp
        ${EMTOOLS_PATH}/FileIO/Dm3FileTag.cpp
        ${EMTOOLS_PATH}/FileIO/Dm3FileTagDirectory.cpp
        ${EMTOOLS_PATH}/FileIO/Dm4File.cpp
        #${EMTOOLS_PATH}/FileIO/Dm4FileStack.cpp
        ${EMTOOLS_PATH}/FileIO/Dm4FileTag.cpp
        ${EMTOOLS_PATH}/FileIO/Dm4FileTagDirectory.cpp
        ${EMTOOLS_PATH}/FileIO/EmFile.cpp
        ${EMTOOLS_PATH}/FileIO/File.cpp
        ${EMTOOLS_PATH}/FileIO/FileIOException.cpp
        ${EMTOOLS_PATH}/FileIO/FileReader.cpp
        ${EMTOOLS_PATH}/FileIO/FileWriter.cpp
        ${EMTOOLS_PATH}/FileIO/ImageBase.cpp
        ${EMTOOLS_PATH}/FileIO/ImodFiducialFile.cpp
        ${EMTOOLS_PATH}/FileIO/MarkerFile.cpp
        ${EMTOOLS_PATH}/FileIO/MDocFile.cpp
        ${EMTOOLS_PATH}/FileIO/MovieStack.cpp
        ${EMTOOLS_PATH}/FileIO/MRCFile.cpp
        ${EMTOOLS_PATH}/FileIO/SERFile.cpp
        ${EMTOOLS_PATH}/FileIO/SimpleFileList.cpp
        ${EMTOOLS_PATH}/FileIO/SingleFrame.cpp
        ${EMTOOLS_PATH}/FileIO/TIFFFile.cpp
        ${EMTOOLS_PATH}/FileIO/TiltSeries.cpp
        ${EMTOOLS_PATH}/FileIO/Volume.cpp)

set(FILEIO_INC
        ${EMTOOLS_PATH}/FileIO
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/MKLog
        ${EMTOOLS_PATH}/FilterGraph
        ${EMTOOLS_PATH}/Minimization)

set(FILEIO_LIB
        MKLog
        FilterGraph
        Minimization)

add_library(FileIO
        ${FILEIO_SRC})

target_include_directories(FileIO PUBLIC ${FILEIO_INC})
target_link_libraries(FileIO ${FILEIO_LIB})
######################### FileIO #####################################


######################### OpenCLHelpers #####################################
set(OPENCL_HELPERS_SRC
        ${EMTOOLS_PATH}/OpenCLHelpers/OpenCLDeviceVariable.cpp
        ${EMTOOLS_PATH}/OpenCLHelpers/OpenCLException.cpp
        ${EMTOOLS_PATH}/OpenCLHelpers/OpenCLHelpers.cpp
        ${EMTOOLS_PATH}/OpenCLHelpers/OpenCLKernel.cpp
        )

set(OPENCL_HELPERS_INC
        ${EMTOOLS_PATH}/OpenCLHelpers
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/MKLog
        ${EMTOOLS_PATH}/Threading
        ${OpenCL_INCLUDE_DIRS})

set(OPENCL_HELPERS_LIB
        MKLog
        Threading
        ${OpenCL_INCLUDE_DIRS}
        Qt5::Widgets
        Qt5::Core
        Qt5::OpenGL
        Qt5::DBus
        ${OPENGL_LIBRARIES}
        ${OpenCL_LIBRARIES})

add_library(OpenCLHelpers
        ${OPENCL_HELPERS_SRC})

target_include_directories(OpenCLHelpers PUBLIC ${OPENCL_HELPERS_INC})
target_link_libraries(OpenCLHelpers ${OPENCL_HELPERS_LIB})
######################### OpenCLHelpers #####################################

######################### MemoryPool ################################
set(MEMORY_POOL_SRC
        ${EMTOOLS_PATH}/MemoryPool/BufferRequest.cpp
        ${EMTOOLS_PATH}/MemoryPool/BufferRequestSet.cpp
        ${EMTOOLS_PATH}/MemoryPool/HostBufferRequest.cpp
        ${EMTOOLS_PATH}/MemoryPool/HostPitchedBufferRequest.cpp
        ${EMTOOLS_PATH}/MemoryPool/MemoryException.cpp
        ${EMTOOLS_PATH}/MemoryPool/MemoryPool.cpp)

set(MEMORY_POOL_INC
        ${EMTOOLS_PATH}/MemoryPool
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/MKLog
        ${EMTOOLS_PATH}/CudaHelpers
        ${EMTOOLS_PATH}/OpenCLHelpers)

set(MEMORY_POOL_LIB
        MKLog
        CudaHelpers
        OpenCLHelpers)

add_library(MemoryPool
        ${MEMORY_POOL_SRC})

target_include_directories(MemoryPool PUBLIC ${MEMORY_POOL_INC})
target_link_libraries(MemoryPool ${MEMORY_POOL_LIB})
######################### MemoryPool ################################

######################### CC #####################################
set(CC_SRC
        ${EMTOOLS_PATH}/CrossCorrelation/CrossCorrelator.cpp)

set(CC_INC
        ${EMTOOLS_PATH}/CrossCorrelation
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/FileIO
        ${EMTOOLS_PATH}/FilterGraph)

set(CC_LIB
        FilterGraph
        FileIO
        fftw3f)

set(CC_LIB_EXT
        fftw3f)

add_library(CrossCorrelation
        ${CC_SRC})

target_include_directories(CrossCorrelation PUBLIC ${CC_INC})
target_link_libraries(CrossCorrelation ${CC_LIB})
######################### CC #####################################

####################################################################################################
####################################################################################################


######################### Clicker #####################################
set(CLICKER_SRC
        ${CLICKER_PATH}/mymainwindow.ui
        ${CLICKER_PATH}/alignmentreport.cpp
        ${CLICKER_PATH}/baseframecontroller.cpp
        ${CLICKER_PATH}/cropsizeselector.cpp
        ${CLICKER_PATH}/floatlineedit.cpp
        ${CLICKER_PATH}/floatslider.cpp
        ${CLICKER_PATH}/histogramwidget.cpp
        ${CLICKER_PATH}/intlineedit.cpp
        ${CLICKER_PATH}/intslider.cpp
        ${CLICKER_PATH}/main.cpp
        ${CLICKER_PATH}/markerlist.cpp
        ${CLICKER_PATH}/mymainwindow.cpp
        ${CLICKER_PATH}/myqlineedit.cpp
        ${CLICKER_PATH}/myqopenglwidget.cpp
        ${CLICKER_PATH}/reconstructionassistant.cpp
        ${CLICKER_PATH}/singleframecontroller.cpp
        ${CLICKER_PATH}/singletonoffscreensurface.cpp
        ${CLICKER_PATH}/squarelabel.cpp
        ${CLICKER_PATH}/tiltseriescontroller.cpp

        ${MKLOG_SRC}
        ${FILEIO_SRC}
        ${CC_SRC}
        ${OPENCL_HELPERS_SRC})

set(CLICKER_INC
        ${CLICKER_PATH}
        ${EMTOOLS_PATH}/MKLog
        ${EMTOOLS_PATH}/FileIO
        ${EMTOOLS_PATH}/CrossCorrelation
        ${EMTOOLS_PATH}/OpenCLHelpers)

set(CLICKER_LIB
        #MKLog
        #FileIO
        #CrossCorrelation
        #OpenCLHelpers
        ${CC_LIB_EXT}
        Qt5::Widgets
        Qt5::Core
        Qt5::OpenGL
        Qt5::DBus
        m)

add_executable(Clicker
        ${CLICKER_SRC})

target_include_directories(Clicker PUBLIC ${CLICKER_INC})
target_link_libraries(Clicker ${CLICKER_LIB})
target_compile_definitions(Clicker PUBLIC USE_OPENCL)
#target_compile_definitions(Clicker PUBLIC USE_OPENCL)
######################### Clicker #####################################