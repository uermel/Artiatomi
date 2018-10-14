CC=mpic++
CFLAGS=-c -Wall -std=c++11
INCLUDES=-I/usr/local/cuda-8.0/include
LIBDIRS=-L/usr/lib -L/usr/local/cuda-8.0/lib64/
LIBS=/usr/local/cuda-8.0/lib64/libcufft.so /usr/local/cuda-8.0/lib64/libnppc.so /usr/local/cuda-8.0/lib64/libnppi.so -lcuda
LDFLAGS=""
SOURCES=../AvgProcess.cpp ../BasicKernel.cpp ../CudaReducer.cpp ../CudaRot.cpp ../stdafx.cpp ../AddParticles.cpp ../io/EMFile.cpp ../io/File.cpp ../io/FileIOException.cpp ../io/FileReader.cpp ../io/FileWriter.cpp ../io/Image.cpp ../io/ImageConverterMethods.cpp ../io/MotiveList.cpp ../cuda/CudaArrays.cpp ../cuda/CudaContext.cpp ../cuda/CudaDeviceProperties.cpp ../cuda/CudaException.cpp ../cuda/CudaGral.cpp ../cuda/CudaKernel.cpp ../cuda/CudaTextures.cpp ../cuda/CudaVariables.cpp ../config/Config.cpp ../config/ConfigExceptions.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=AddParticles

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@
