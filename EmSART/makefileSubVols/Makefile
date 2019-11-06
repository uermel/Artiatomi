CC=mpic++
CFLAGS=-c -Wall -std=c++11 -DUSE_MPI -DSUBVOLREC_MODE
INCLUDES=-I/usr/local/cuda-8.0/include
LIBDIRS=-L/usr/lib -L/usr/local/cuda-8.0/lib64/
LIBS=/usr/local/cuda-8.0/lib64/libcufft.so /usr/local/cuda-8.0/lib64/libnppc.so /usr/local/cuda-8.0/lib64/libnppi.so -lcuda
LDFLAGS=""
SOURCES=../src/cuda/CudaArrays.cpp ../src/cuda/CudaContext.cpp ../src/cuda/CudaDeviceProperties.cpp ../src/cuda/CudaException.cpp ../src/cuda/CudaKernel.cpp ../src/cuda/CudaTextures.cpp ../src/cuda/CudaVariables.cpp ../src/io/CtfFile.cpp ../src/io/Dm4File.cpp ../src/io/File.cpp ../src/io/ImageConverterMethods.cpp ../src/io/Dm4FileStack.cpp ../src/io/FileIOException.cpp ../src/io/MPISource.cpp ../src/io/Dm4FileTag.cpp ../src/io/FileReader.cpp ../src/io/MRCFile.cpp ../src/io/Dm4FileTagDirectory.cpp ../src/io/FileWriter.cpp ../src/io/MarkerFile.cpp ../src/io/EMFile.cpp ../src/io/Image.cpp ../src/io/writeBMP.cpp ../src/io/ShiftFile.cpp ../src/utils/Config.cpp ../src/utils/ConfigExceptions.cpp ../src/utils/CudaConfig.cpp ../src/utils/Matrix.cpp ../src/utils/log.cpp ../src/EmSartSubVolumes.cpp ../src/Kernels.cpp ../src/Projection.cpp ../src/utils/SimpleLogger.cpp ../src/Volume.cpp ../src/Reconstructor.cpp ../src/io/MotiveList.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=EmSartSubVolumes

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm -f EmSartSubVolumes $(OBJECTS)
