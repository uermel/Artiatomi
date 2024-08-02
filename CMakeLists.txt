cmake_minimum_required(VERSION 3.24)
project(Artiatomi LANGUAGES CUDA CXX)
include(CheckLanguage)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

check_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

set(EMTOOLS_PATH "src/EmTools" CACHE PATH "Path to EmTools source.")
set(CLICKER_PATH "src/Clicker" CACHE PATH "Path to Clicker source.")
set(EMSART_PATH "src/EmSART" CACHE PATH "Path to EmSART source.")
set(ISA_PATH "src/ImageStackAlignator" CACHE PATH "Path to ImageStackAlignator source.")
set(STA_PATH "src/SubTomogramAverageMPI" CACHE PATH "Path to SubTomogramAverageMPI source.")

#if(UNIX)
#	set(CUDA_PATH "/usr/local/cuda-10.2" CACHE PATH "Path to CUDA root")
#endif()

#if(WIN32)
#	set(MIKTEX_BINARY_PATH "C:/Program Files/MiKTeX 2.9/miktex/bin/x64/")
#endif()

######################### CUDA #############################
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CUDA_LIBS
		CUDA::cuda_driver
		CUDA::cufft
		CUDA::nppc
		CUDA::nppial
		CUDA::nppicc
		CUDA::nppidei
		CUDA::nppif
		CUDA::nppig
		CUDA::nppim
		CUDA::nppist
		CUDA::nppisu
		CUDA::nppitc
		CUDA::npps
		CUDA::cudart)

find_program(bin_to_c
		NAMES bin2c
		PATHS ${CUDAToolkit_BIN_DIR}
		)
######################### CUDA #############################

######################### OpenCL #############################
#if (UNIX)
#	set(OpenCL_INCLUDE_DIR "/usr/local/cuda/include")
#	set(OpenCL_LIBRARY "/usr/local/cuda/lib64/libOpenCL.so")
#endif()

find_package(OpenCL REQUIRED)
find_package(OpenGL)
######################### OpenCL #############################

set(CURSES_USE_NCURSES TRUE)
include(FindCurses)
find_package(Curses)


######################### FFTW ###########################
if(WIN32)
	set(FFTW_ROOT "${CMAKE_SOURCE_DIR}/${EMTOOLS_PATH}/libs/FFTW")
	set(FFTW_INC ${FFTW_ROOT})
	set(FFTW_LIB ${FFTW_ROOT}/libfftw3f-3.lib)
else()
	set(FFTW_LIB "fftw3f")
endif()
##########################################################


######################### Qt #############################
#set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTORCC ON)
#set(CMAKE_AUTOUIC ON)

#if(CMAKE_VERSION VERSION_LESS "3.7.0")
#    set(CMAKE_INCLUDE_CURRENT_DIR ON)
#endif()

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
		${EMTOOLS_PATH}/FilterGraph/Matrix.cpp
		${EMTOOLS_PATH}/FilterGraph/PointF.cpp
		${EMTOOLS_PATH}/FilterGraph/FilterSize.cpp
		${EMTOOLS_PATH}/FilterGraph/FilterROI.cpp
		${EMTOOLS_PATH}/FilterGraph/FilterPoint2D.cpp)

set(FILTER_GRAPH_INC
		${EMTOOLS_PATH}/FilterGraph
		${EMTOOLS_PATH}/Basics)

set(FILTER_GRAPH_LIB
		)

######################### FilterGraph #####################################

######################### Minimization ################################
set(MINIMIZATION_SRC
		${EMTOOLS_PATH}/Minimization/levmar.cpp)

set(MINIMIZATION_INC
		${EMTOOLS_PATH}/Basics
		${EMTOOLS_PATH}/Minimization)

set(MINIMIZATION_LIB
		)

######################### Minimization ################################

######################### FileIO #####################################
set(FILEIO_SRC
		${EMTOOLS_PATH}/FileIO/CtfFile.cpp
		${EMTOOLS_PATH}/FileIO/Dm3File.cpp
		${EMTOOLS_PATH}/FileIO/Dm3FileTag.cpp
		${EMTOOLS_PATH}/FileIO/Dm3FileTagDirectory.cpp
		${EMTOOLS_PATH}/FileIO/Dm4File.cpp
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
		${EMTOOLS_PATH}/FileIO/MotiveList.cpp
		${EMTOOLS_PATH}/FileIO/MovieStack.cpp
		${EMTOOLS_PATH}/FileIO/MRCFile.cpp
		${EMTOOLS_PATH}/FileIO/SERFile.cpp
		${EMTOOLS_PATH}/FileIO/ShiftFile.cpp
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

######################### CudaHelpers ################################
set(CUDA_HELPERS_SRC
		${EMTOOLS_PATH}/CudaHelpers/CudaArrays.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaContext.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaDeviceProperties.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaException.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaHelpers.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaKernel.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaTextures.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaSurfaces.cpp
		${EMTOOLS_PATH}/CudaHelpers/CudaVariables.cpp
		${EMTOOLS_PATH}/CudaHelpers/NPPImageBase.cpp
		${EMTOOLS_PATH}/CudaHelpers/NPPImages.cpp)

set(CUDA_HELPERS_INC
		${EMTOOLS_PATH}/CudaHelpers
		${EMTOOLS_PATH}/Basics
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
		${EMTOOLS_PATH}/FilterGraph
		${FFTW_INC})

set(CC_LIB
		${FILTER_GRAPH_LIB}
		${FILEIO_LIB}
		${FFTW_LIB})
######################### CC #####################################

######################### STA FileIO #####################################
set(STA_FILEIO_SRC
		${STA_PATH}/io/EMFile.cpp
		${STA_PATH}/io/File.cpp
		${STA_PATH}/io/FileIOException.cpp
		${STA_PATH}/io/FileReader.cpp
		${STA_PATH}/io/FileWriter.cpp
		${STA_PATH}/io/Image.cpp
		${STA_PATH}/io/ImageConverterMethods.cpp
		${STA_PATH}/io/MotiveList.cpp)

set(STA_FILEIO_INC
		${STA_PATH}/io
		${STA_PATH}/basics
		${MK_LOG_INC}
		${FILTER_GRAPH_INC}
		${MINIMIZATION_INC})

set(STA_FILEIO_LIB
		)
######################### STA FileIO #####################################

######################### STA CudaHelpers ################################
set(STA_CUDA_HELPERS_SRC
		${STA_PATH}/cuda/CudaArrays.cpp
		${STA_PATH}/cuda/CudaContext.cpp
		${STA_PATH}/cuda/CudaDeviceProperties.cpp
		${STA_PATH}/cuda/CudaException.cpp
		${STA_PATH}/cuda/CudaGral.cpp
		${STA_PATH}/cuda/CudaKernel.cpp
		${STA_PATH}/cuda/CudaTextures.cpp
		${STA_PATH}/cuda/CudaVariables.cpp)

set(STA_CUDA_HELPERS_INC
		${STA_PATH}/cuda
		${STA_PATH}/basics
		${THREADING_INC}
		${MK_LOG_INC}
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(STA_CUDA_HELPERS_LIB
		${CUDA_LIBS}
		${THREADING_LIB})
######################### STA CudaHelpers ################################

######################### STA Config ################################
set(STA_CONFIG_SRC
		${STA_PATH}/config/Config.cpp
		${STA_PATH}/config/ConfigExceptions.cpp)

set(STA_CONFIG_INC
		${STA_PATH}/config
		${STA_PATH}/basics)

set(STA_CONFIG_LIB)
######################### STA Config ################################
####################################################################################################
####################################################################################################


####################################################################################################
#############################   Applications   #####################################################
####################################################################################################

######################### Clicker #####################################
set(CLICKER_SRC
		${MK_LOG_SRC}
		${FILEIO_SRC}
		${CC_SRC}
		${OPENCL_HELPERS_SRC}
		${MINIMIZATION_SRC}
		${THREADING_SRC}
		${FILTER_GRAPH_SRC}

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
		)

set(CLICKER_LIB
		${OPENCL_HELPERS_LIB}
		${CC_LIB}
		${FILEIO_LIB}
		${MK_LOG_LIB}
		${MINIMIZATION_LIB}
		${THREADING_LIB}
		${FILTER_GRAPH_LIB}
		Qt5::Widgets
		Qt5::Core
		Qt5::OpenGL
		Qt5::DBus
		)

if(UNIX)
	set(CLICKER_LIB
			${CLICKER_LIB}
			m)
endif()

if(WIN32)
	add_executable(Clicker WIN32
			${CLICKER_SRC})
else()
	add_executable(Clicker
			${CLICKER_SRC})
endif()

target_include_directories(Clicker PUBLIC ${CLICKER_INC})
target_link_libraries(Clicker ${CLICKER_LIB})
target_compile_definitions(Clicker PUBLIC NO_CUDA)
######################### Clicker #####################################

##############################################################
######################### ImageStackAlignator ################
set(ISA_SRC
		${FILEIO_SRC}
		${CUDA_HELPERS_SRC}
		${MK_LOG_SRC}
		${THREADING_SRC}
		${FILTER_GRAPH_SRC}
		${MINIMIZATION_SRC}

		${ISA_PATH}/AlignmentOptions.cpp
		${ISA_PATH}/BaseAligner.cpp
		${ISA_PATH}/CudaCrossCorrelator.cpp
		${ISA_PATH}/ImageStackAlignator.cpp
		${ISA_PATH}/Kernels.cpp
		)

set(ISA_INC
		${CUDAToolkit_INCLUDE_DIRS}
		${EMTOOLS_PATH})

set(ISA_LIB
		${CUDA_LIBS}
		${FILEIO_LIB}
		${MK_LOG_LIB}
		${MINIMIZATION_LIB}
		${THREADING_LIB}
		${FILTER_GRAPH_LIB}
		)

if(UNIX)
	set(ISA_LIB
			${ISA_LIB}
			m)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

# PTX-Kernel file
add_library(isa_ptx OBJECT ${ISA_PATH}/ImageStackAlignatorKernels/KernelCC.cu)
target_compile_definitions(isa_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET isa_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(isa_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/KernelCC.cu.h)
add_custom_command(
		OUTPUT ${isa_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:isa_ptx>"
		"-DOUTPUT:FILEPATH=${isa_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:isa_ptx>
		COMMENT "Converting PTX to header."
)

# Executable
add_executable(ImageStackAlignator
		${ISA_SRC} ${isa_ptx_header})

# Executable depends on ptx files
add_dependencies(ImageStackAlignator isa_ptx)

target_include_directories(
		ImageStackAlignator
		PUBLIC ${ISA_INC}
		PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(
		ImageStackAlignator
		PUBLIC ${ISA_LIB}
)
target_compile_definitions(ImageStackAlignator PUBLIC NO_OPENCL)
target_compile_options(ImageStackAlignator PUBLIC -fpermissive)
######################################################################

######################### EmSART Common ##############################
# Common definitions for EmSART, EmSARTRefine and EmSARTSubVolumes
set(EMSARTCOMMON_SRC
		${MK_LOG_SRC}
		${FILEIO_SRC}
		${FILTER_GRAPH_SRC}
		${MINIMIZATION_SRC}
		${THREADING_SRC}
		${CUDA_HELPERS_SRC}

		#${EMSART_PATH}/Kernels.cpp
		${EMSART_PATH}/Projection.cpp
		${EMSART_PATH}/Volume.cpp
		${EMSART_PATH}/Reconstructor.cpp
		${EMSART_PATH}/CTF.cpp
		${EMSART_PATH}/DeviceVolume.cpp
		${EMSART_PATH}/KernelModules.cpp

		${EMSART_PATH}/utils/ConfigExceptions.cpp
		${EMSART_PATH}/utils/CudaConfig.cpp
		${EMSART_PATH}/utils/log.cpp
		${EMSART_PATH}/utils/SimpleLogger.cpp
		${EMSART_PATH}/utils/Config.cpp
		${EMSART_PATH}/utils/ConvexPolygon.cpp

		${EMSART_PATH}/io/FileSource.cpp
		${EMSART_PATH}/io/MPISource.cpp)


set(EMSARTCOMMON_INC
		${EMSART_PATH}
		${FILEIO_INC}
		${MK_LOG_INC}
		${FILTER_GRAPH_INC}
		${MINIMIZATION_INC}
		${THREADING_INC}
		${CUDA_HELPERS_INC}
		${MPI_CXX_INCLUDE_DIRS}
		${CURSES_INCLUDE_DIRS})

set(EMSARTCOMMON_LIB
		${FILEIO_LIB}
		${MK_LOG_LIB}
		${CUDA_HELPERS_LIB}
		${MPI_CXX_LIBRARIES}
		${FILTER_GRAPH_LIB}
		${MINIMIZATION_LIB}
		${THREADING_LIB}
		${CURSES_LIBRARIES})

# PTX-Kernel file
# BackProjectionSquareOS
add_library(bp_ptx OBJECT ${EMSART_PATH}/cuda_kernels/BackProjectionSquareOS.cu)
target_compile_definitions(bp_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET bp_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(bp_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/BackProjectionSquareOS.cu.h)
add_custom_command(
		OUTPUT ${bp_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:bp_ptx>"
		"-DOUTPUT:FILEPATH=${bp_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:bp_ptx>
		COMMENT "Converting PTX to header."
)

#ForwardProjectionRayMarcher_TL
add_library(fp_ptx OBJECT ${EMSART_PATH}/cuda_kernels/ForwardProjectionRayMarcher_TL.cu)
target_compile_definitions(fp_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET fp_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(fp_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/ForwardProjectionRayMarcher_TL.cu.h)
add_custom_command(
		OUTPUT ${fp_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:fp_ptx>"
		"-DOUTPUT:FILEPATH=${fp_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:fp_ptx>
		COMMENT "Converting PTX to header."
)

# ForwardProjectionSlicer
add_library(fpd_ptx OBJECT ${EMSART_PATH}/cuda_kernels/ForwardProjectionSlicer.cu)
target_compile_definitions(fpd_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET fpd_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(fpd_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/ForwardProjectionSlicer.cu.h)
add_custom_command(
		OUTPUT ${fpd_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:fpd_ptx>"
		"-DOUTPUT:FILEPATH=${fpd_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:fpd_ptx>
		COMMENT "Converting PTX to header."
)

# ctf
add_library(ctf_ptx OBJECT ${EMSART_PATH}/cuda_kernels/ctf.cu)
target_compile_definitions(ctf_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET ctf_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(ctf_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/ctf.cu.h)
add_custom_command(
		OUTPUT ${ctf_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:ctf_ptx>"
		"-DOUTPUT:FILEPATH=${ctf_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:ctf_ptx>
		COMMENT "Converting PTX to header."
)

# CTS
add_library(cts_ptx OBJECT ${EMSART_PATH}/cuda_kernels/CopyToSquare.cu)
target_compile_definitions(cts_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET cts_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(cts_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/CopyToSquare.cu.h)
add_custom_command(
		OUTPUT ${cts_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:cts_ptx>"
		"-DOUTPUT:FILEPATH=${cts_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:cts_ptx>
		COMMENT "Converting PTX to header."
)

# CTS
add_library(compare_ptx OBJECT ${EMSART_PATH}/cuda_kernels/Compare.cu)
target_compile_definitions(compare_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET compare_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(compare_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/Compare.cu.h)
add_custom_command(
		OUTPUT ${compare_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:compare_ptx>"
		"-DOUTPUT:FILEPATH=${compare_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:compare_ptx>
		COMMENT "Converting PTX to header."
)

# WBP weighting
add_library(wbpw_ptx OBJECT ${EMSART_PATH}/cuda_kernels/wbpWeighting.cu)
target_compile_definitions(wbpw_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET wbpw_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(wbpw_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/wbpWeighting.cu.h)
add_custom_command(
		OUTPUT ${wbpw_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:wbpw_ptx>"
		"-DOUTPUT:FILEPATH=${wbpw_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:wbpw_ptx>
		COMMENT "Converting PTX to header."
)

# FP LUT
add_library(fplut_ptx OBJECT ${EMSART_PATH}/cuda_kernels/ForwardProjectionLUT.cu)
target_compile_definitions(fplut_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET fplut_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(fplut_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/ForwardProjectionLUT.cu.h)
add_custom_command(
		OUTPUT ${fplut_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:fplut_ptx>"
		"-DOUTPUT:FILEPATH=${fplut_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:fplut_ptx>
		COMMENT "Converting PTX to header."
)

# FP LUT
add_library(bplut_ptx OBJECT ${EMSART_PATH}/cuda_kernels/BackProjectionLUT.cu)
target_compile_definitions(bplut_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET bplut_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(bplut_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/BackProjectionLUT.cu.h)
add_custom_command(
		OUTPUT ${bplut_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:bplut_ptx>"
		"-DOUTPUT:FILEPATH=${bplut_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:bplut_ptx>
		COMMENT "Converting PTX to header."
)

# Splines
add_library(splines_ptx OBJECT ${EMSART_PATH}/cuda_kernels/splines.cu)
target_compile_definitions(splines_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET splines_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(splines_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/splines.cu.h)
add_custom_command(
		OUTPUT ${splines_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:splines_ptx>"
		"-DOUTPUT:FILEPATH=${splines_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:splines_ptx>
		COMMENT "Converting PTX to header."
)
######################################################################

######################### EmSART Kernels ################################
set(EMSART_KERNELS_SRC
		${EMSART_PATH}/kernels/BPKernel.cpp
		${EMSART_PATH}/kernels/FPKernel.cpp
		${EMSART_PATH}/kernels/PCKernel.cpp
		${EMSART_PATH}/kernels/CTFKernel.cpp
		${EMSART_PATH}/kernels/CTFSlicedKernel.cpp
		#${EMSART_PATH}/kernels/RotKernel.cpp
		${EMSART_PATH}/kernels/CompKernel.cpp
		${EMSART_PATH}/kernels/ConjKernel.cpp
		${EMSART_PATH}/kernels/SubEKernel.cpp
		${EMSART_PATH}/kernels/BPLUTKernel.cpp
		${EMSART_PATH}/kernels/BPLUTBWKernel.cpp
		${EMSART_PATH}/kernels/BPLUTBWCGKernel.cpp
		${EMSART_PATH}/kernels/BPLUTBlockKernel.cpp
		${EMSART_PATH}/kernels/BPLUTBlockNoDivKernel.cpp
		${EMSART_PATH}/kernels/BPLUTVBlockSlicedKernel.cpp
		${EMSART_PATH}/kernels/BPLUTSlicedKernel.cpp
		${EMSART_PATH}/kernels/FPLUTKernel.cpp
		${EMSART_PATH}/kernels/FPLUTSlicedKernel.cpp
		${EMSART_PATH}/kernels/FPDistSlicedKernel.cpp
		${EMSART_PATH}/kernels/SlicerKernel.cpp
		${EMSART_PATH}/kernels/ConvVolKernel.cpp
		${EMSART_PATH}/kernels/FindPeakKernel.cpp
		${EMSART_PATH}/kernels/MaxShiftKernel.cpp
		${EMSART_PATH}/kernels/ApplyMaskKernel.cpp
		${EMSART_PATH}/kernels/ConvVol3DKernel.cpp
		${EMSART_PATH}/kernels/CropBorderKernel.cpp
		${EMSART_PATH}/kernels/DimBordersKernel.cpp
		${EMSART_PATH}/kernels/FourFilterKernel.cpp
		${EMSART_PATH}/kernels/CopyToSquareKernel.cpp
		#${EMSART_PATH}/kernels/CopyToSquareSlicesKernel.cpp
		#${EMSART_PATH}/kernels/CopyToRectSlicesKernel.cpp
		${EMSART_PATH}/kernels/WbpWeightingKernel.cpp
		${EMSART_PATH}/kernels/DoseWeightingKernel.cpp
		${EMSART_PATH}/kernels/RestoreVolumeKernel.cpp
		${EMSART_PATH}/kernels/SphericalMaskKernel.cpp
		${EMSART_PATH}/kernels/SplinePrefilter.cpp
		${EMSART_PATH}/kernels/VolTravLengthKernel.cpp
		${EMSART_PATH}/kernels/MaxShiftWeightedKernel.cpp
		${EMSART_PATH}/kernels/CubicResampleKernel.cpp
		${EMSART_PATH}/kernels/OversampleKernel.cpp
		${EMSART_PATH}/kernels/RectToSqrSlice.cpp
		${EMSART_PATH}/kernels/RectSliceToSqrSliceKernel.cpp
		${EMSART_PATH}/kernels/SqrSliceToRectSliceKernel.cpp
		${EMSART_PATH}/kernels/SqrSliceToRectKernel.cpp
		${EMSART_PATH}/kernels/CropSlicesKernel.cpp
		${EMSART_PATH}/kernels/ComputeLUTKernel.cpp
		${EMSART_PATH}/kernels/FFTShiftKernel.cpp
		${EMSART_PATH}/kernels/FPOrthoKernel.cpp
		${EMSART_PATH}/kernels/FPDistOrthoKernel.cpp
		${EMSART_PATH}/kernels/BPOrthoSlicedKernel.cpp
		${EMSART_PATH}/kernels/RadialSumKernel.cpp
		${EMSART_PATH}/kernels/FreqSampleKernel.cpp)

set(EMSART_KERNELS_INC
		${EMSART_PATH}/kernels/
		)
######################### EmSART Kernels ################################

######################### EmSART #####################################
set(EMSART_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSart.cpp
		)

set(EMSART_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSART_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSART
		${EMSART_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSART bp_ptx)
add_dependencies(EmSART fp_ptx)
add_dependencies(EmSART fpd_ptx)
add_dependencies(EmSART ctf_ptx)
add_dependencies(EmSART cts_ptx)
add_dependencies(EmSART compare_ptx)
add_dependencies(EmSART wbpw_ptx)
add_dependencies(EmSART fplut_ptx)
add_dependencies(EmSART bplut_ptx)
add_dependencies(EmSART splines_ptx)


target_include_directories(EmSART PUBLIC ${EMSART_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSART ${EMSART_LIB})
target_compile_definitions(EmSART PUBLIC NO_OPENCL)
target_compile_options(EmSART PUBLIC -fpermissive)
######################### EmSART #####################################

######################### EmSARTSubtract #####################################
set(EMSARTSUBTRACT_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSartSubtract.cpp
		)

set(EMSARTSUBTRACT_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSARTSUBTRACT_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTSubtract
		${EMSARTSUBTRACT_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTSubtract bp_ptx)
add_dependencies(EmSARTSubtract fp_ptx)
add_dependencies(EmSARTSubtract fpd_ptx)
add_dependencies(EmSARTSubtract ctf_ptx)
add_dependencies(EmSARTSubtract cts_ptx)
add_dependencies(EmSARTSubtract compare_ptx)
add_dependencies(EmSARTSubtract wbpw_ptx)
add_dependencies(EmSARTSubtract fplut_ptx)
add_dependencies(EmSARTSubtract bplut_ptx)
add_dependencies(EmSARTSubtract splines_ptx)


target_include_directories(EmSARTSubtract PUBLIC ${EMSARTSUBTRACT_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTSubtract ${EMSARTSUBTRACT_LIB})
target_compile_definitions(EmSARTSubtract PUBLIC NO_OPENCL SUBVOLREC_MODE)
target_compile_options(EmSARTSubtract PUBLIC -fpermissive)
######################### EmSARTSubtract #####################################

######################### EmSARTParticle #####################################
set(EMSARTPARTICLE_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSartParticle.cpp
		)

set(EMSARTPARTICLE_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSARTPARTICLE_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTParticle
		${EMSARTPARTICLE_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTParticle bp_ptx)
add_dependencies(EmSARTParticle fp_ptx)
add_dependencies(EmSARTParticle fpd_ptx)
add_dependencies(EmSARTParticle ctf_ptx)
add_dependencies(EmSARTParticle cts_ptx)
add_dependencies(EmSARTParticle compare_ptx)
add_dependencies(EmSARTParticle wbpw_ptx)
add_dependencies(EmSARTParticle fplut_ptx)
add_dependencies(EmSARTParticle bplut_ptx)
add_dependencies(EmSARTParticle splines_ptx)


target_include_directories(EmSARTParticle PUBLIC ${EMSARTPARTICLE_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTParticle ${EMSARTPARTICLE_LIB})
target_compile_definitions(EmSARTParticle PUBLIC NO_OPENCL SUBVOLREC_MODE)
target_compile_options(EmSARTParticle PUBLIC -fpermissive)
######################### EmSARTParticle #####################################

######################### EmSARTParticleHS #####################################
set(EMSARTPARTICLEHS_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSartParticleHS.cpp
		)

set(EMSARTPARTICLEHS_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSARTPARTICLEHS_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTParticleHS
		${EMSARTPARTICLEHS_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTParticleHS bp_ptx)
add_dependencies(EmSARTParticleHS fp_ptx)
add_dependencies(EmSARTParticleHS fpd_ptx)
add_dependencies(EmSARTParticleHS ctf_ptx)
add_dependencies(EmSARTParticleHS cts_ptx)
add_dependencies(EmSARTParticleHS compare_ptx)
add_dependencies(EmSARTParticleHS wbpw_ptx)
add_dependencies(EmSARTParticleHS fplut_ptx)
add_dependencies(EmSARTParticleHS bplut_ptx)
add_dependencies(EmSARTParticleHS splines_ptx)


target_include_directories(EmSARTParticleHS PUBLIC ${EMSARTPARTICLEHS_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTParticleHS ${EMSARTPARTICLEHS_LIB})
target_compile_definitions(EmSARTParticleHS PUBLIC NO_OPENCL SUBVOLREC_MODE)
target_compile_options(EmSARTParticleHS PUBLIC -fpermissive)
######################### EmSARTSubtract #####################################


######################### EmSARTAlign #####################################
set(EMSARTALIGN_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSartAlign.cpp
		)

set(EMSARTALIGN_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSARTALIGN_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTAlign
		${EMSARTALIGN_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTAlign bp_ptx)
add_dependencies(EmSARTAlign fp_ptx)
add_dependencies(EmSARTAlign fpd_ptx)
add_dependencies(EmSARTAlign ctf_ptx)
add_dependencies(EmSARTAlign cts_ptx)
add_dependencies(EmSARTAlign compare_ptx)
add_dependencies(EmSARTAlign wbpw_ptx)
add_dependencies(EmSARTAlign fplut_ptx)
add_dependencies(EmSARTAlign bplut_ptx)
add_dependencies(EmSARTAlign splines_ptx)


target_include_directories(EmSARTAlign PUBLIC ${EMSARTALIGN_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTAlign ${EMSARTALIGN_LIB})
target_compile_definitions(EmSARTAlign PUBLIC NO_OPENCL SUBVOLREC_MODE)
target_compile_options(EmSARTAlign PUBLIC -fpermissive)
######################### EmSARTAlign #####################################

######################### EmSARTROI #####################################
set(EMSARTROI_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSartROI.cpp
		)

set(EMSART_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSART_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTROI
		${EMSARTROI_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTROI bp_ptx)
add_dependencies(EmSARTROI fp_ptx)
add_dependencies(EmSARTROI fpd_ptx)
add_dependencies(EmSARTROI ctf_ptx)
add_dependencies(EmSARTROI cts_ptx)
add_dependencies(EmSARTROI compare_ptx)
add_dependencies(EmSARTROI wbpw_ptx)
add_dependencies(EmSARTROI fplut_ptx)
add_dependencies(EmSARTROI bplut_ptx)
add_dependencies(EmSARTROI splines_ptx)


target_include_directories(EmSARTROI PUBLIC ${EMSART_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTROI ${EMSART_LIB})
target_compile_definitions(EmSARTROI PUBLIC NO_OPENCL)
target_compile_options(EmSARTROI PUBLIC -fpermissive)
######################### EmSARTROI #####################################

######################### EmSARTProject #####################################
set(EMSARTPROJECT_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSartProject.cpp
		)

set(EMSARTPROJECT_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSARTPROJECT_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTProject
		${EMSARTPROJECT_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTProject bp_ptx)
add_dependencies(EmSARTProject fp_ptx)
add_dependencies(EmSARTProject fpd_ptx)
add_dependencies(EmSARTProject ctf_ptx)
add_dependencies(EmSARTProject cts_ptx)
add_dependencies(EmSARTProject compare_ptx)
add_dependencies(EmSARTProject wbpw_ptx)
add_dependencies(EmSARTProject fplut_ptx)
add_dependencies(EmSARTProject bplut_ptx)
add_dependencies(EmSARTProject splines_ptx)


target_include_directories(EmSARTProject PUBLIC ${EMSARTPROJECT_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTProject ${EMSARTPROJECT_LIB})
target_compile_definitions(EmSARTProject PUBLIC NO_OPENCL SUBVOLREC_MODE)
target_compile_options(EmSARTProject PUBLIC -fpermissive)
######################### EmSARTProject #####################################

######################### EmSARTRefine ###############################
set(EMSARTREFINE_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_PATH}/EmSartRefine.cpp
		)

set(EMSARTREFINE_INC
		${EMSARTCOMMON_INC}
		)

set(EMSARTREFINE_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTRefine
		${EMSARTREFINE_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTRefine bp_ptx)
add_dependencies(EmSARTRefine fp_ptx)
add_dependencies(EmSARTRefine fpd_ptx)
add_dependencies(EmSARTRefine ctf_ptx)
add_dependencies(EmSARTRefine cts_ptx)
add_dependencies(EmSARTRefine compare_ptx)
add_dependencies(EmSARTRefine wbpw_ptx)
add_dependencies(EmSARTRefine fplut_ptx)
add_dependencies(EmSARTRefine bplut_ptx)
add_dependencies(EmSARTRefine splines_ptx)


target_include_directories(EmSARTRefine PUBLIC ${EMSARTREFINE_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTRefine ${EMSARTREFINE_LIB})
target_compile_definitions(EmSARTRefine PUBLIC NO_OPENCL REFINE_MODE)
target_compile_options(EmSARTRefine PUBLIC -fpermissive)
######################### EmSARTRefine ###############################

######################### EmSARTSubVolumes ###########################
set(EMSARTSUBVOLS_SRC
		${EMSARTCOMMON_SRC}
		${EMSART_KERNELS_SRC}
		${EMSART_PATH}/EmSartSubVolumes.cpp
		)

set(EMSARTSUBVOLS_INC
		${EMSARTCOMMON_INC}
		${EMSART_KERNELS_INC}
		)

set(EMSARTSUBVOLS_LIB
		${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(
		EmSARTSubVols
		${EMSARTSUBVOLS_SRC}
		${bp_ptx_header}
		${fp_ptx_header}
		${fpd_ptx_header}
		${ctf_ptx_header}
		${cts_ptx_header}
		${compare_ptx_header}
		${wbpw_ptx_header}
		${fplut_ptx_header}
		${bplut_ptx_header}
		${splines_ptx_header}
)

# Executable depends on ptx files
add_dependencies(EmSARTSubVols bp_ptx)
add_dependencies(EmSARTSubVols fp_ptx)
add_dependencies(EmSARTSubVols fpd_ptx)
add_dependencies(EmSARTSubVols ctf_ptx)
add_dependencies(EmSARTSubVols cts_ptx)
add_dependencies(EmSARTSubVols compare_ptx)
add_dependencies(EmSARTSubVols wbpw_ptx)
add_dependencies(EmSARTSubVols fplut_ptx)
add_dependencies(EmSARTSubVols bplut_ptx)
add_dependencies(EmSARTSubVols splines_ptx)


target_include_directories(EmSARTSubVols PUBLIC ${EMSARTSUBVOLS_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EmSARTSubVols ${EMSARTSUBVOLS_LIB})
target_compile_definitions(EmSARTSubVols PUBLIC NO_OPENCL SUBVOLREC_MODE)
target_compile_options(EmSARTSubVols PUBLIC -fpermissive)
######################### EmSARTSubVolumes ###############################


######################### Tutorial #####################################
find_package(LATEX COMPONENTS PDFLATEX)

if(LATEX_FOUND)
	include(docs/UseLATEX.cmake)

	file(GLOB_RECURSE tutorial_files RELATIVE ${CMAKE_SOURCE_DIR} docs/Tutorial/*.tex)

	ADD_LATEX_DOCUMENT(
			docs/Tutorial/Tutorial.tex
			INPUTS ${tutorial_files}
			FORCE_PDF
	)
endif()
######################### Tutorial #####################################


######################### STA Common ########################
set(STA_COMMON_SRC
		${STA_FILEIO_SRC}
		${STA_CUDA_HELPERS_SRC}
		${STA_CONFIG_SRC}
		${MK_LOG_SRC}
		${THREADING_SRC}
		${FILTER_GRAPH_SRC}
		${MINIMIZATION_SRC}

		${STA_PATH}/AvgProcess.cpp
		${STA_PATH}/BasicKernel.cpp
		${STA_PATH}/CudaReducer.cpp
		${STA_PATH}/CudaRot.cpp
		${STA_PATH}/stdafx.cpp
		)

set(STA_COMMON_INC
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
		${MPI_CXX_INCLUDE_DIRS}
		${STA_FILEIO_INC}
		${STA_CUDA_HELPERS_INC}
		${STA_CONFIG_INC}
		${STA_PATH}/SubTomogramAverageMPI)

set(STA_COMMON_LIB
		${CUDA_LIBS}
		${MPI_CXX_LIBRARIES}
		${STA_FILEIO_LIB}
		${STA_CUDA_HELPERS_LIB}
		${STA_CONFIG_LIB}
		${MK_LOG_LIB}
		${MINIMIZATION_LIB}
		${THREADING_LIB}
		${FILTER_GRAPH_LIB}
		)

set(CMAKE_VERBOSE_MAKEFILE ON)

# PTX-Kernel files
# ReducerKernels
add_library(red_ptx OBJECT ${STA_PATH}/cuda_kernels/ReducerKernels.cu)
target_compile_definitions(red_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET red_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(red_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/ReducerKernels.cu.h)
add_custom_command(
		OUTPUT ${red_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:red_ptx>"
		"-DOUTPUT:FILEPATH=${red_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:red_ptx>
		COMMENT "Converting PTX to header."
)

# ReducerKernels
add_library(basic_ptx OBJECT ${STA_PATH}/cuda_kernels/basicKernels.cu)
target_compile_definitions(basic_ptx PRIVATE "CUDA_PTX_COMPILATION")
set_property(
		TARGET basic_ptx
		PROPERTY CUDA_PTX_COMPILATION ON
)
set(basic_ptx_header ${CMAKE_CURRENT_BINARY_DIR}/basicKernels.cu.h)
add_custom_command(
		OUTPUT ${basic_ptx_header}
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS:STRING=$<TARGET_OBJECTS:basic_ptx>"
		"-DOUTPUT:FILEPATH=${basic_ptx_header}"
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS $<TARGET_OBJECTS:basic_ptx>
		COMMENT "Converting PTX to header."
)
######################### STA Common ########################

######################### SubTomogramAverageMPI ########################

set(STA_SRC
		${STA_COMMON_SRC}
		${STA_PATH}/SubTomogramAverageMPI.cpp
		)

set(STA_INC
		${STA_COMMON_INC}
		)

set(STA_LIB
		${STA_COMMON_LIB}
		)

# Executable
add_executable(SubTomogramAverageMPI
		${STA_SRC}
		${red_ptx_header}
		${basic_ptx_header})

# Executable depends on ptx files
add_dependencies(SubTomogramAverageMPI red_ptx)
add_dependencies(SubTomogramAverageMPI basic_ptx)

target_include_directories(SubTomogramAverageMPI PUBLIC ${STA_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(SubTomogramAverageMPI ${STA_LIB})
target_compile_definitions(SubTomogramAverageMPI PUBLIC NO_OPENCL USE_MPI)
target_compile_options(SubTomogramAverageMPI PUBLIC -fpermissive)
######################### SubTomogramAverageMPI ########################

######################### AddParticles ########################

set(ADDPARTICLES_SRC
		${STA_COMMON_SRC}
		${STA_PATH}/AddParticles.cpp
		)

set(ADDPARTICLES_INC
		${STA_COMMON_INC}
		)

set(ADDPARTICLES_LIB
		${STA_COMMON_LIB}
		)

# Executable
add_executable(AddParticles
		${ADDPARTICLES_SRC}
		${red_ptx_header}
		${basic_ptx_header})

# Executable depends on ptx files
add_dependencies(AddParticles red_ptx)
add_dependencies(AddParticles basic_ptx)

target_include_directories(AddParticles PUBLIC ${ADDPARTICLES_INC} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AddParticles ${ADDPARTICLES_LIB})
target_compile_definitions(AddParticles PUBLIC NO_OPENCL USE_MPI)
target_compile_options(AddParticles PUBLIC -fpermissive)
######################### AddParticles ########################


