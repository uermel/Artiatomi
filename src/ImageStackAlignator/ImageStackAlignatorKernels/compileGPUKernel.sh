#!/bin/bash

# AS
#
# This script creates header files from cu files
#
# kernel.h is from the original repository
# kernel.cu.h is generated by this script

# ToDo name (KernelCC) should be generated by the Filename
# to allow for multiple kernelfiles in the same folder


for filename in $(ls *.cu); do
	nvcc --ptx -c -o $filename.code $filename
	bin2c -p 0 --name KernelCC $filename.code > $filename.h
done