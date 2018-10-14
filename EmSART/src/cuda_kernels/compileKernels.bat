@echo off
echo Current directory:
echo %CD%
if "%VCSETUPDONE%"=="TRUE" (
	ECHO Skip VC setup...
) else (
	ECHO Setup VC environment...
	call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
	set VCSETUPDONE=TRUE
)

set NVCCEXE="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvcc.exe"
set INCLUDE="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include"
set OUTPUT="%CD%"
set INPUT="%CD%"
set OPTIONS=-arch compute_35 --machine 64 -ptx --use_fast_math
if "%1"=="-D" (
	echo "DEBUG Modus!"
	set OPTIONS=%OPTIONS% -g -G -lineinfo
)
rem %NVCCEXE% %OPTIONS% -o %OUTPUT%\BackProjectionSquareOS.ptx %INPUT%\BackProjectionSquareOS.cu
rem %NVCCEXE% %OPTIONS% -o %OUTPUT%\ForwardProjectionRayMarcher_TL.ptx %INPUT%\ForwardProjectionRayMarcher_TL.cu
rem %NVCCEXE% %OPTIONS% -o %OUTPUT%\ForwardProjectionSlicer.ptx %INPUT%\ForwardProjectionSlicer.cu
%NVCCEXE% %OPTIONS% -o %OUTPUT%\ctf.ptx %INPUT%\ctf.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\cl.exe"
rem %NVCCEXE% %OPTIONS% -o %OUTPUT%\CopyToSquare.ptx %INPUT%\CopyToSquare.cu
rem %NVCCEXE% %OPTIONS% -o %OUTPUT%\Compare.ptx %INPUT%\Compare.cu
%NVCCEXE% %OPTIONS% -o %OUTPUT%\wbpWeighting.ptx %INPUT%\wbpWeighting.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\cl.exe"


bin2c -p 0 --name KernelBackProjectionOS %OUTPUT%\BackProjectionSquareOS.ptx > %OUTPUT%\BackProjectionSquareOS.h
bin2c -p 0 --name KernelForwardProjectionRayMarcher_TL %OUTPUT%\ForwardProjectionRayMarcher_TL.ptx > %OUTPUT%\ForwardProjectionRayMarcher_TL.h
bin2c -p 0 --name KernelForwardProjectionSlicer %OUTPUT%\ForwardProjectionSlicer.ptx > %OUTPUT%\ForwardProjectionSlicer.h
bin2c -p 0 --name Kernelctf %OUTPUT%\ctf.ptx > %OUTPUT%\ctf.h
bin2c -p 0 --name KernelCopyToSquare %OUTPUT%\CopyToSquare.ptx > %OUTPUT%\CopyToSquare.h
bin2c -p 0 --name KernelCompare %OUTPUT%\Compare.ptx > %OUTPUT%\Compare.h
bin2c -p 0 --name KernelWbpWeighting %OUTPUT%\wbpWeighting.ptx > %OUTPUT%\wbpWeighting.h