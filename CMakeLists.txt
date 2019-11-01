cmake_minimum_required(VERSION 3.14)
project(Artiatomi LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)


set(EMTOOLS_PATH "EmTools" CACHE PATH "Path to EmTools source.")
set(CLICKER_PATH "Clicker/Clicker" CACHE PATH "Path to Clicker source.")
set(ISA_PATH "ImageStackAlignator" CACHE PATH "Path to ImageStackAlignator source.")
set(CUDA_PATH "/usr/local/cuda-8.0" CACHE PATH "Path to cuda 8 root")

######################### CUDA #############################
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(CUDA 8.0 REQUIRED)

find_library(CUDA_LIBRARY cuda ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUFFT_LIBRARY cufft ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPC_LIBRARY nppc  ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NPPI_LIBRARY nppi ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set(CUDA_LIBS ${CUDA_LIBRARIES})
message("${CUDA_LIBS}")
######################### CUDA #############################

######################### OpenCL/GL #############################
find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
######################### OpenCL #############################

######################### Qt #############################
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(Qt5 COMPONENTS Core Widgets DBus OpenGL REQUIRED)
######################### Qt #############################

######################### System ################################
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
######################### System ################################

######################### MPI ################################
find_package(MPI REQUIRED)
######################### MPI ################################

####################################################################################################
####################################################################################################

######################### MKLog #####################################
set(MK_LOG_SRC
        ${EMTOOLS_PATH}/MKLog/MKLog.cpp)

set(MK_LOG_INC
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/MKLog)

set(MK_LOG_LIB
        )
######################### MKLog #####################################

######################### Threading ################################
set(THREADING_SRC
        ${EMTOOLS_PATH}/Threading/SpecificBackgroundThread.cpp)

set(THREADING_INC
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/Threading)

set(THREADING_LIB
        Threads::Threads)
######################### Threading ################################

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
        ${MEMORY_POOL_INC})

set(FILTER_GRAPH_LIB
        ${MEMORY_POOL_LIB})

#add_library(FilterGraph
#        ${FILTER_GRAPH_SRC})

#target_include_directories(FilterGraph PUBLIC ${FILTER_GRAPH_INC})
#target_link_libraries(FilterGraph ${FILTER_GRAPH_LIB})
######################### FilterGraph #####################################

######################### Minimization ################################
set(MINIMIZATION_SRC
        ${EMTOOLS_PATH}/Minimization/levmar.cpp)

set(MINIMIZATION_INC
        ${EMTOOLS_PATH}/Basics
        ${EMTOOLS_PATH}/Minimization)

set(MINIMIZATION_LIB
        )

#target_include_directories(Minimization PUBLIC ${MINIMIZATION_INC})
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
        ${MK_LOG_INC}
        ${FILTER_GRAPH_INC}
        ${MINIMIZATION_INC})

set(FILEIO_LIB
        )

#add_library(FileIO
#        ${FILEIO_SRC})

#target_include_directories(FileIO PUBLIC ${FILEIO_INC})
#target_link_libraries(FileIO ${FILEIO_LIB})
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
        ${MK_LOG_INC}
        ${THREADING_INC}
        ${OpenCL_INCLUDE_DIRS})

set(OPENCL_HELPERS_LIB
        ${MK_LOG_LIB}
        ${THREADING_LIB}
        Qt5::Widgets
        Qt5::Core
        Qt5::OpenGL
        Qt5::DBus
        ${OPENGL_LIBRARIES}
        ${OpenCL_LIBRARIES})
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
        ${MK_LOG_INC}
        ${CUDA_HELPERS_INC}
        ${OPENCL_HELPERS_INC})

set(MEMORY_POOL_LIB
        ${MK_LOG_LIB}
        ${OPENCL_HELPERS_LIB}
        ${CUDA_HELPERS_LIB})

#add_library(MemoryPool
#        ${MEMORY_POOL_SRC})

#target_include_directories(MemoryPool PUBLIC ${MEMORY_POOL_INC})
#target_link_libraries(MemoryPool ${MEMORY_POOL_LIB})
######################### MemoryPool ################################

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
        ${MEMORY_POOL_INC}
        ${THREADING_INC}
        ${MK_LOG_INC}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CUDA_HELPERS_LIB
        ${CUDA_LIBS}
        ${THREADING_LIB})
######################### CudaHelpers ################################

######################### CC #####################################
set(CC_SRC
        ${EMTOOLS_PATH}/CrossCorrelation/CrossCorrelator.cpp)

set(CC_INC
        ${EMTOOLS_PATH}/CrossCorrelation
        ${EMTOOLS_PATH}/Basics
        ${FILEIO_INC}
        ${FILTER_GRAPH_INC}
        ${EMTOOLS_PATH}/FileIO
        ${EMTOOLS_PATH}/FilterGraph)

set(CC_LIB
        ${FILTER_GRAPH_LIB}
        ${FILEIO_LIB}
        fftw3f)
######################### CC #####################################

####################################################################################################
####################################################################################################


####################################################################################################
#############################   Applications   #####################################################
####################################################################################################

######################### Clicker #####################################
set(CLICKER_SRC
        ${MKLOG_SRC}
        ${FILEIO_SRC}
        ${CC_SRC}
        ${OPENCL_HELPERS_SRC}
        ${MINIMIZATION_SRC}
        ${THREADING_SRC}
        ${FILTER_GRAPH_SRC}
        ${MEMORY_POOL_SRC}

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
        ${CLICKER_PATH}/tiltseriescontroller.cpp)

set(CLICKER_INC
        ${CLICKER_PATH}
        ${MK_LOG_INC}
        ${FILEIO_INC}
        ${CC_INC}
        ${OPENCL_HELPERS_INC}
        ${MINIMIZATION_INC}
        ${THREADING_INC}
        ${FILTER_GRAPH_INC}
        ${MEMORY_POOL_INC}
        )

set(CLICKER_LIB
        ${OPENCL_HELPERS_LIB}
        ${CC_LIB}
        ${FILEIO_LIB}
        ${MK_LOG_LIB}
        ${MINIMIZATION_LIB}
        ${THREADING_LIB}
        ${FILTER_GRAPH_LIB}
        ${MEMORY_POOL_LIB}
        Qt5::Widgets
        Qt5::Core
        Qt5::OpenGL
        Qt5::DBus
        m)

add_executable(Clicker
        ${CLICKER_SRC})

target_include_directories(Clicker PUBLIC ${CLICKER_INC})
target_link_libraries(Clicker ${CLICKER_LIB})
target_compile_definitions(Clicker PUBLIC NO_CUDA)
######################### Clicker #####################################



##############################################################
set(ISA_SRC
        ${FILEIO_SRC}
        ${CUDA_HELPERS_SRC}
        ${MKLOG_SRC}
        ${THREADING_SRC}
        ${FILTER_GRAPH_SRC}
        ${MEMORY_POOL_SRC}
        ${MINIMIZATION_SRC}

        ${ISA_PATH}/AlignmentOptions.cpp
        ${ISA_PATH}/BaseAligner.cpp
        ${ISA_PATH}/CudaCrossCorrelator.cpp
        ${ISA_PATH}/ImageStackAlignator.cpp
        ${ISA_PATH}/Kernels.cpp)

set(ISA_INC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${MPI_INCLUDE_PATH}
        ${EMTOOLS_PATH})

set(ISA_LIB
        ${CUDA_LIBS}
        ${MPI_LIBRARIES}
        ${FILEIO_LIB}
        ${MK_LOG_LIB}
        ${MINIMIZATION_LIB}
        ${THREADING_LIB}
        ${FILTER_GRAPH_LIB}
        ${MEMORY_POOL_LIB}
        m
        cuda
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_nppi_LIBRARY}
        ${CUDA_nppc_LIBRARY}
        ${MPI_LIBRARIES})

message(${ISA_LIB})
set(CMAKE_VERBOSE_MAKEFILE ON)

# PTX-Kernel file
cuda_compile_ptx(ISA_PTX_FILES ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/kernel.ptx COMMAND ${CMAKE_COMMAND} -E rename ${ISA_PTX_FILES} ${CMAKE_BINARY_DIR}/kernel.ptx DEPENDS ${ISA_PTX_FILES})
add_custom_target(isa_ptx ALL
        DEPENDS ${CMAKE_BINARY_DIR}/kernel.ptx ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu
        SOURCES ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu)

# Executable
add_executable(ImageStackAlignator
        ${ISA_SRC})

# Executable depends on ptx files
add_dependencies(ImageStackAlignator isa_ptx)

target_include_directories(ImageStackAlignator PUBLIC ${ISA_INC})
target_link_libraries(ImageStackAlignator ${ISA_LIB})
target_compile_definitions(ImageStackAlignator PUBLIC NO_OPENCL)
target_compile_options(ImageStackAlignator PUBLIC -fpermissive)
######################### ImageStackAlignator #####################################


######################### EmSART #####################################
set(EMSART_SRC
        ${FILEIO_SRC}
        ${CUDA_HELPERS_SRC}
        ${MKLOG_SRC}
        ${THREADING_SRC}
        ${FILTER_GRAPH_SRC}
        ${MEMORY_POOL_SRC}
        ${MINIMIZATION_SRC}

        ${ISA_PATH}/AlignmentOptions.cpp
        ${ISA_PATH}/BaseAligner.cpp
        ${ISA_PATH}/CudaCrossCorrelator.cpp
        ${ISA_PATH}/ImageStackAlignator.cpp
        ${ISA_PATH}/Kernels.cpp)

set(EmSART_INC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${MPI_INCLUDE_PATH}
        ${EMTOOLS_PATH})

set(EmSART_LIB
        ${CUDA_LIBS}
        ${MPI_LIBRARIES}
        ${FILEIO_LIB}
        ${MK_LOG_LIB}
        ${MINIMIZATION_LIB}
        ${THREADING_LIB}
        ${FILTER_GRAPH_LIB}
        ${MEMORY_POOL_LIB}
        m
        cuda
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_nppi_LIBRARY}
        ${CUDA_nppc_LIBRARY}
        ${MPI_LIBRARIES})

message(${ISA_LIB})
set(CMAKE_VERBOSE_MAKEFILE ON)

# PTX-Kernel file
#cuda_compile_ptx(ISA_PTX_FILES ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu)
#add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/kernel.ptx COMMAND ${CMAKE_COMMAND} -E rename ${ISA_PTX_FILES} ${CMAKE_BINARY_DIR}/kernel.ptx DEPENDS ${ISA_PTX_FILES})
#add_custom_target(isa_ptx ALL
#        DEPENDS ${CMAKE_BINARY_DIR}/kernel.ptx ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu
#        SOURCES ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu)

# Executable
add_executable(EmSART
        ${EMSART_SRC})

# Executable depends on ptx files
#add_dependencies(ImageStackAlignator isa_ptx)


target_include_directories(EmSART PUBLIC ${ISA_INC})
target_link_libraries(EmSART ${ISA_LIB})
target_compile_definitions(EmSART PUBLIC NO_OPENCL)
target_compile_options(EmSART PUBLIC -fpermissive)
######################### EmSART #####################################



######################### Tutorial #####################################
find_package(LATEX COMPONENTS PDFLATEX)
include(docs/UseLATEX.cmake)

file(GLOB_RECURSE tutorial_files RELATIVE ${CMAKE_SOURCE_DIR} docs/Tutorial/*.tex)

ADD_LATEX_DOCUMENT(
        docs/Tutorial/Tutorial.tex
        INPUTS ${tutorial_files}
        FORCE_PDF
)
######################### Tutorial #####################################