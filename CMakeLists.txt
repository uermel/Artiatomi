cmake_minimum_required(VERSION 3.13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.1/bin/nvcc)
project(Artiatomi LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Debug)

set(CUDA_COMPILE_OPTIONS --use_fast_math)
#set(CUDA_COMPILE_OPTIONS -g -G -lineinfo) #to debug kernels



set(CMAKE_CXX_STANDARD 14)


set(EMTOOLS_PATH "src/EmTools" CACHE PATH "Path to EmTools source.")
set(CLICKER_PATH "src/Clicker" CACHE PATH "Path to Clicker source.")
set(EMSART_PATH "src/EmSART" CACHE PATH "Path to EmSART source.")
set(ISA_PATH "src/ImageStackAlignator" CACHE PATH "Path to ImageStackAlignator source.")
set(STA_PATH "src/SubTomogramAverageMPI" CACHE PATH "Path to SubTomogramAverageMPI source.")

if(UNIX)
	set(CUDA_PATH "/usr/local/cuda-10.2" CACHE PATH "Path to CUDA root")
endif()

if(WIN32)
	set(MIKTEX_BINARY_PATH "C:/Program Files/MiKTeX 2.9/miktex/bin/x64/")
endif()

######################### CUDA #############################
#set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_PATH})
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(CUDA 9.2 REQUIRED)

set(CUDA_LIBS ${CUDA_CUDA_LIBRARY} 
				${CUDA_CUFFT_LIBRARIES}
				${CUDA_nppc_LIBRARY}
				${CUDA_nppial_LIBRARY}
				${CUDA_nppicc_LIBRARY}
				${CUDA_nppicom_LIBRARY}
				${CUDA_nppidei_LIBRARY}
				${CUDA_nppif_LIBRARY} 
				${CUDA_nppig_LIBRARY}
				${CUDA_nppim_LIBRARY}
				${CUDA_nppist_LIBRARY}
				${CUDA_nppisu_LIBRARY}
				${CUDA_nppitc_LIBRARY}
				${CUDA_npps_LIBRARY}
				${CUDA_LIBRARIES})

#set(CUDA_LIBS ${CUDA_LIBRARIES})
set(BIN2C_EXEC /usr/local/cuda-10.1/bin/bin2c) #in cuda toolkit binary path
######################### CUDA #############################
if (UNIX)
	set(OpenCL_INCLUDE_DIR "/usr/local/cuda/include")
	set(OpenCL_LIBRARY "/usr/local/cuda/lib64/libOpenCL.so")
endif()
	
find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)

message( ${OpenCL_INCLUDE_DIR} )
message( ${OpenCL_LIBRARIES} )
#message( ${OPENGL_INCLUDE_DIR} )
#message( ${OPENGL_LIBRARIES} )
######################### OpenCL #############################

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
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
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
cuda_compile_ptx(ISA_PTX_FILES ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu.h COMMAND ${BIN2C_EXEC} -p 0 --const --name KernelCC ${ISA_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu.h DEPENDS ${ISA_PTX_FILES})
add_custom_target(isa_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu.h ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu
        SOURCES ${ISA_PATH}/ImageStackAlignatorKernels/kernel.cu)
		
# Executable
add_executable(ImageStackAlignator
        ${ISA_SRC})

# Executable depends on ptx files
add_dependencies(ImageStackAlignator isa_ptx_header)

target_include_directories(ImageStackAlignator PUBLIC ${ISA_INC})
target_link_libraries(ImageStackAlignator ${ISA_LIB})
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
		
        ${EMSART_PATH}/Kernels.cpp
        ${EMSART_PATH}/Projection.cpp
        ${EMSART_PATH}/Volume.cpp
        ${EMSART_PATH}/Reconstructor.cpp
		
        ${EMSART_PATH}/utils/ConfigExceptions.cpp
        ${EMSART_PATH}/utils/CudaConfig.cpp
        ${EMSART_PATH}/utils/log.cpp
        ${EMSART_PATH}/utils/SimpleLogger.cpp
        ${EMSART_PATH}/utils/Config.cpp
		
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
        )
		
set(EMSARTCOMMON_LIB
        ${FILEIO_LIB}
		${MK_LOG_LIB}
        ${CUDA_HELPERS_LIB}
        ${MPI_CXX_LIBRARIES}
        ${FILTER_GRAPH_LIB}
        ${MINIMIZATION_LIB}
        ${THREADING_LIB}
		)
	
# PTX-Kernel file
cuda_compile_ptx(BP_PTX_FILES ${EMSART_PATH}/cuda_kernels/BackProjectionSquareOS.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/BackProjectionSquareOS.cu.h 
				   COMMAND ${BIN2C_EXEC} -p 0 --const --name KernelBackProjectionOS ${BP_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/BackProjectionSquareOS.cu.h DEPENDS ${BP_PTX_FILES})
add_custom_target(bp_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/BackProjectionSquareOS.cu.h ${EMSART_PATH}/cuda_kernels/BackProjectionSquareOS.cu
        SOURCES ${EMSART_PATH}/cuda_kernels/BackProjectionSquareOS.cu)
		
cuda_compile_ptx(FPRM_PTX_FILES ${EMSART_PATH}/cuda_kernels/ForwardProjectionRayMarcher_TL.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ForwardProjectionRayMarcher_TL.cu.h 
				   COMMAND ${BIN2C_EXEC} -p 0 --const --name KernelForwardProjectionRayMarcher_TL ${FPRM_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ForwardProjectionRayMarcher_TL.cu.h DEPENDS ${FPRM_PTX_FILES})
add_custom_target(fprm_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ForwardProjectionRayMarcher_TL.cu.h ${EMSART_PATH}/cuda_kernels/ForwardProjectionRayMarcher_TL.cu
        SOURCES ${EMSART_PATH}/cuda_kernels/ForwardProjectionRayMarcher_TL.cu)
		
cuda_compile_ptx(FPD_PTX_FILES ${EMSART_PATH}/cuda_kernels/ForwardProjectionSlicer.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ForwardProjectionSlicer.cu.h 
				   COMMAND ${BIN2C_EXEC} -p 0 --const --name KernelForwardProjectionSlicer ${FPD_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ForwardProjectionSlicer.cu.h DEPENDS ${FPD_PTX_FILES})
add_custom_target(fpd_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ForwardProjectionSlicer.cu.h ${EMSART_PATH}/cuda_kernels/ForwardProjectionSlicer.cu
        SOURCES ${EMSART_PATH}/cuda_kernels/ForwardProjectionSlicer.cu)
		
cuda_compile_ptx(CTF_PTX_FILES ${EMSART_PATH}/cuda_kernels/ctf.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ctf.cu.h 
				   COMMAND ${BIN2C_EXEC} -p 0 --const --name Kernelctf ${CTF_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ctf.cu.h DEPENDS ${CTF_PTX_FILES})
add_custom_target(ctf_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/ctf.cu.h ${EMSART_PATH}/cuda_kernels/ctf.cu
        SOURCES ${EMSART_PATH}/cuda_kernels/ctf.cu)
		
cuda_compile_ptx(CTS_PTX_FILES ${EMSART_PATH}/cuda_kernels/CopyToSquare.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/CopyToSquare.cu.h 
				   COMMAND ${BIN2C_EXEC} -p 0 --const --name KernelCopyToSquare ${CTS_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/CopyToSquare.cu.h DEPENDS ${CTS_PTX_FILES})
add_custom_target(cts_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/CopyToSquare.cu.h ${EMSART_PATH}/cuda_kernels/CopyToSquare.cu
        SOURCES ${EMSART_PATH}/cuda_kernels/CopyToSquare.cu)
		
cuda_compile_ptx(COMPARE_PTX_FILES ${EMSART_PATH}/cuda_kernels/Compare.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/Compare.cu.h 
				   COMMAND ${BIN2C_EXEC} -p 0 --const --name KernelCompare ${COMPARE_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/Compare.cu.h DEPENDS ${COMPARE_PTX_FILES})
add_custom_target(compare_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/Compare.cu.h ${EMSART_PATH}/cuda_kernels/Compare.cu
        SOURCES ${EMSART_PATH}/cuda_kernels/Compare.cu)
		
cuda_compile_ptx(WBP_PTX_FILES ${EMSART_PATH}/cuda_kernels/wbpWeighting.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/wbpWeighting.cu.h 
				   COMMAND ${BIN2C_EXEC} -p 0 --const --name KernelWbpWeighting ${WBP_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/wbpWeighting.cu.h DEPENDS ${WBP_PTX_FILES})
add_custom_target(wbp_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${EMSART_PATH}/cuda_kernels/wbpWeighting.cu.h ${EMSART_PATH}/cuda_kernels/wbpWeighting.cu
        SOURCES ${EMSART_PATH}/cuda_kernels/wbpWeighting.cu)
		
######################################################################

######################### EmSART #####################################
set(EMSART_SRC
        ${EMSARTCOMMON_SRC}
        ${EMSART_PATH}/EmSart.cpp
        )
		
set(EMSART_INC
        ${EMSARTCOMMON_INC}
        )
		
set(EMSART_LIB
        ${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(EmSART
        ${EMSART_SRC})

# Executable depends on ptx files
add_dependencies(EmSART bp_ptx_header)	
add_dependencies(EmSART fprm_ptx_header)	
add_dependencies(EmSART fpd_ptx_header)		
add_dependencies(EmSART ctf_ptx_header)		
add_dependencies(EmSART cts_ptx_header)		
add_dependencies(EmSART compare_ptx_header)	
add_dependencies(EmSART wbp_ptx_header)	


target_include_directories(EmSART PUBLIC ${EMSART_INC})
target_link_libraries(EmSART ${EMSART_LIB})
target_compile_definitions(EmSART PUBLIC NO_OPENCL USE_MPI)
target_compile_options(EmSART PUBLIC -fpermissive)
######################### EmSART #####################################

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
add_executable(EmSARTRefine
        ${EMSARTREFINE_SRC})

# Executable depends on ptx files
add_dependencies(EmSARTRefine bp_ptx_header)	
add_dependencies(EmSARTRefine fprm_ptx_header)	
add_dependencies(EmSARTRefine fpd_ptx_header)		
add_dependencies(EmSARTRefine ctf_ptx_header)		
add_dependencies(EmSARTRefine cts_ptx_header)		
add_dependencies(EmSARTRefine compare_ptx_header)	
add_dependencies(EmSARTRefine wbp_ptx_header)	


target_include_directories(EmSARTRefine PUBLIC ${EMSARTREFINE_INC})
target_link_libraries(EmSARTRefine ${EMSARTREFINE_LIB})
target_compile_definitions(EmSARTRefine PUBLIC NO_OPENCL REFINE_MODE)
target_compile_options(EmSARTRefine PUBLIC -fpermissive)
######################### EmSARTRefine ###############################

######################### EmSARTSubVolumes ###########################
set(EMSARTSUBVOLS_SRC
        ${EMSARTCOMMON_SRC}
		${EMSART_PATH}/EmSartSubVolumes.cpp
        )
		
set(EMSARTSUBVOLS_INC
        ${EMSARTCOMMON_INC}
        )
		
set(EMSARTSUBVOLS_LIB
        ${EMSARTCOMMON_LIB}
		)

# Executable
add_executable(EmSARTSubVols
        ${EMSARTSUBVOLS_SRC})

# Executable depends on ptx files
add_dependencies(EmSARTSubVols bp_ptx_header)	
add_dependencies(EmSARTSubVols fprm_ptx_header)	
add_dependencies(EmSARTSubVols fpd_ptx_header)		
add_dependencies(EmSARTSubVols ctf_ptx_header)		
add_dependencies(EmSARTSubVols cts_ptx_header)		
add_dependencies(EmSARTSubVols compare_ptx_header)	
add_dependencies(EmSARTSubVols wbp_ptx_header)	


target_include_directories(EmSARTSubVols PUBLIC ${EMSARTSUBVOLS_INC})
target_link_libraries(EmSARTSubVols ${EMSARTSUBVOLS_LIB})
target_compile_definitions(EmSARTSubVols PUBLIC NO_OPENCL USE_MPI SUBVOLREC_MODE)
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
cuda_compile_ptx(STA_PTX_FILES ${STA_PATH}/cuda_kernels/kernel.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${STA_PATH}/cuda_kernels/kernel.cu.h COMMAND ${BIN2C_EXEC} -p 0 --const --name SubTomogramAverageKernel ${STA_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${STA_PATH}/cuda_kernels/kernel.cu.h DEPENDS ${STA_PTX_FILES})
add_custom_target(sta_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${STA_PATH}/cuda_kernels/kernel.cu.h ${STA_PATH}/cuda_kernels/kernel.cu
        SOURCES ${STA_PATH}/cuda_kernels/kernel.cu)

cuda_compile_ptx(STA_BASIC_PTX_FILES ${STA_PATH}/cuda_kernels/basicKernels.cu OPTIONS ${CUDA_COMPILE_OPTIONS})
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/${STA_PATH}/cuda_kernels/basicKernels.cu.h COMMAND ${BIN2C_EXEC} -p 0 --const --name SubTomogramAverageBasicKernel ${STA_BASIC_PTX_FILES} > ${CMAKE_SOURCE_DIR}/${STA_PATH}/cuda_kernels/basicKernels.cu.h DEPENDS ${STA_BASIC_PTX_FILES})
add_custom_target(sta_basic_ptx_header ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/${STA_PATH}/cuda_kernels/basicKernels.cu.h ${STA_PATH}/cuda_kernels/basicKernels.cu
        SOURCES ${STA_PATH}/cuda_kernels/basicKernels.cu)
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
        ${STA_SRC})

# Executable depends on ptx files
add_dependencies(SubTomogramAverageMPI sta_ptx_header)  
add_dependencies(SubTomogramAverageMPI sta_basic_ptx_header)    

target_include_directories(SubTomogramAverageMPI PUBLIC ${STA_INC})
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
        ${ADDPARTICLES_SRC})

# Executable depends on ptx files
add_dependencies(AddParticles sta_ptx_header)  
add_dependencies(AddParticles sta_basic_ptx_header)    

target_include_directories(AddParticles PUBLIC ${ADDPARTICLES_INC})
target_link_libraries(AddParticles ${ADDPARTICLES_LIB})
target_compile_definitions(AddParticles PUBLIC NO_OPENCL USE_MPI)
target_compile_options(AddParticles PUBLIC -fpermissive)
######################### AddParticles ########################


######################### MotlTest #####################################
set(TEST_SRC
		${MK_LOG_SRC}
		${FILEIO_SRC}
		${CC_SRC}
		${MINIMIZATION_SRC}
		${THREADING_SRC}
		${FILTER_GRAPH_SRC}

		${EMTOOLS_PATH}/FileIO/motl_test.cpp)

set(TEST_INC
		${CLICKER_PATH}
		${MK_LOG_INC}
		${FILEIO_INC}
		${CC_INC}
		${MINIMIZATION_INC}
		${THREADING_INC}
		${FILTER_GRAPH_INC}
		)

set(TEST_LIB
		${CC_LIB}
		${FILEIO_LIB}
		${MK_LOG_LIB}
		${MINIMIZATION_LIB}
		${THREADING_LIB}
		${FILTER_GRAPH_LIB}
		)

add_executable(test
			${TEST_SRC})


target_include_directories(test PUBLIC ${TEST_INC})
target_link_libraries(test ${TEST_LIB})
target_compile_definitions(test PUBLIC NO_CUDA)
######################### MotlTest #####################################



######################### VolTest #####################################
set(VOLTEST_SRC
		${EMSARTCOMMON_SRC}

		${EMSART_PATH}/volume_test.cpp)

set(VOLTEST_INC
		${EMSARTCOMMON_INC}
		)

set(VOLTEST_LIB
		${EMSARTCOMMON_LIB}
		)

add_executable(volTest
		${VOLTEST_SRC})

# Executable depends on ptx files
add_dependencies(volTest bp_ptx_header)
add_dependencies(volTest fprm_ptx_header)
add_dependencies(volTest fpd_ptx_header)
add_dependencies(volTest ctf_ptx_header)
add_dependencies(volTest cts_ptx_header)
add_dependencies(volTest compare_ptx_header)
add_dependencies(volTest wbp_ptx_header)

target_include_directories(volTest PUBLIC ${VOLTEST_INC})
target_link_libraries(volTest ${VOLTEST_LIB})
target_compile_definitions(volTest PUBLIC USE_MPI NO_OPENCL REFINE_MODE)
target_compile_options(volTest PUBLIC -fpermissive)
######################### VolTest #####################################