CC=mpic++
CFLAGS=-c -fpermissive -std=c++14 -DUSE_CUDA 
INCLUDES=-I/usr/local/cuda-8.0/include
LIBDIRS=-L/usr/lib -L/usr/local/cuda-8.0/lib64/
LIBS=/usr/local/cuda-8.0/lib64/libcufft.so /usr/local/cuda-8.0/lib64/libnppc.so /usr/local/cuda-8.0/lib64/libnppi.so -lcuda
LDFLAGS=""
SOURCES=AlignmentOptions.cpp CudaCrossCorrelator.cpp ImageStackAlignator.cpp Kernels.cpp ../Threading/SpecificBackgroundThread.cpp ../FilterGraph/Matrix.cpp ../FilterGraph/PointF.cpp ../MKLog/MKLog.cpp ../CudaHelpers/CudaArrays.cpp ../CudaHelpers/CudaContext.cpp ../CudaHelpers/CudaDeviceProperties.cpp ../CudaHelpers/CudaException.cpp ../CudaHelpers/CudaHelpers.cpp ../CudaHelpers/CudaKernel.cpp ../CudaHelpers/CudaTextures.cpp ../CudaHelpers/CudaVariables.cpp ../CudaHelpers/NPPImageBase.cpp ../CudaHelpers/NPPImages.cpp ../Minimization/levmar.cpp ../FileIO/Dm3File.cpp ../FileIO/Dm3FileTag.cpp ../FileIO/Dm3FileTagDirectory.cpp ../FileIO/Dm4File.cpp ../FileIO/Dm4FileTag.cpp ../FileIO/Dm4FileTagDirectory.cpp ../FileIO/EmFile.cpp ../FileIO/File.cpp ../FileIO/FileIOException.cpp ../FileIO/FileReader.cpp ../FileIO/FileWriter.cpp ../FileIO/ImageBase.cpp ../FileIO/ImodFiducialFile.cpp ../FileIO/MRCFile.cpp ../FileIO/MarkerFile.cpp ../FileIO/SimpleFileList.cpp ../FileIO/MDocFile.cpp ../FileIO/MovieStack.cpp ../FileIO/SERFile.cpp ../FileIO/SingleFrame.cpp ../FileIO/TIFFFile.cpp ../FileIO/TiltSeries.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=ImageStackAlignator

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm ImageStackAlignator $(OBJECTS)