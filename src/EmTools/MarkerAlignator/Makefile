CC=g++
CFLAGS=-c -fpermissive -std=c++14 -fopenmp  
INCLUDES=
LIBDIRS=-L/usr/lib 
LIBS=
LDFLAGS=-fopenmp
SOURCES=MarkerAlignator.cpp ../Threading/SpecificBackgroundThread.cpp ../FilterGraph/Matrix.cpp ../FilterGraph/PointF.cpp ../MKLog/MKLog.cpp  ../Minimization/levmar.cpp ../FileIO/Dm3File.cpp ../FileIO/Dm3FileTag.cpp ../FileIO/Dm3FileTagDirectory.cpp ../FileIO/Dm4File.cpp ../FileIO/Dm4FileTag.cpp ../FileIO/Dm4FileTagDirectory.cpp ../FileIO/EmFile.cpp ../FileIO/File.cpp ../FileIO/FileIOException.cpp ../FileIO/FileReader.cpp ../FileIO/FileWriter.cpp ../FileIO/ImageBase.cpp ../FileIO/ImodFiducialFile.cpp ../FileIO/MRCFile.cpp ../FileIO/MarkerFile.cpp ../FileIO/SimpleFileList.cpp ../FileIO/MDocFile.cpp ../FileIO/MovieStack.cpp ../FileIO/SERFile.cpp ../FileIO/SingleFrame.cpp ../FileIO/TIFFFile.cpp ../FileIO/TiltSeries.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=MarkerAlignator

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

clean:
	rm MarkerAlignator $(OBJECTS)