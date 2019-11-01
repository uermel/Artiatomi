# Artiatomi
This is the official Artiatomi cryo electron tomography package

it consists of the following tools or applications:
- **EmSART**: The "main" tool, performing tomographic reconstruction
  of cryo-EM tilt series. It accepts DM3/4 file series or MRC/ST
  files as input and writes the result in one EM file. MRC output
  is also implemented but not well tested. Core of the
  implemented method is super-sampling SART and three-
  dimensional CTF correction.
  References: 
  1) Kunz, M. & Frangakis, A.S. (2014). _[Super-sampling SART with ordered subsets.](https://doi.org/10.1016/j.jsb.2014.09.010)_
     J Struct Biol. 2014 Nov;188(2):107-15.
  2) Kunz, M. & Frangakis, A.S. (2017). _[Three-dimensional CTF correction improves the resolution of electron tomograms.](https://doi.org/10.1016/j.jsb.2016.06.016)_
	 J Struct Biol. 2017 Feb;197(2):114-122.

  If you use **EmSART** for reconstruction of your EM data, please
  make sure to properly cite the methods used.
  If you develop software based on this implemantation, please
  also cite one of these papers.
  --> sub folder: EmSART
- **EmSARTRefine**: Finds the individual shift of each sub-volume
  on each tilt of the tilt-series.
  ref: Paper to be published
  sub folder: EmSART
- **EmSARTSubVolumes**: Reconstructs sub-volumes with weighted back-
  projection with three-dimensional CTF-correction without the 
  need of reconstructing a full volume. Includes the individual
  shift information.
  sub folder: EmSART
- **MarkerAlignator**: A command line tool to align marker files,
  provided either as EM-file (old TOM-toolbox format) or a set
  of IMOD marker files (*.fid, *.tlt, *.prexg). In theory, one 
  could also convert the final IMOD alignment files, but this 
  turned out to be ambiguous due to post-processing in IMOD, why 
  we prefer to align the tilt-series independently from the 
  clicked positions in the micrographs.
  sub folder: EmTools/MarkerAlignator
- **ImageStackAligntor**: A command line tool to align dose-
  fractionation stacks.
  The first step consists of minimizing the consistency error of
  a shift-matrix (ref: Electron counting and beam-induced motion 
  correction enable near atomic resolution single particle cryoEM
  Xueming Li et al., Nat Methods. 2013 Jun; 10(6): 584–590.) 
  followed by iterative re-aligning one frame to the sum of the 
  other aligned frames until convergence. To avoid interpolation
  artifacts, we also search only for full-pixel shifts.
  sub folder: EmTools/ImageStackAlignator
- **Clicker**: A GUI tool to visualize tilt series and click gold beads.
  sub folder: Clicker
- **SubTomogramAveraging**: This application performs sub-tomogram
  averaging as done in the old TOM-toolbox, with full GPU acceleration
  and the capabilility to run on a compute cluster using MPI.
  sub folder: SubTomogramAverageMPI

# Compilation:
All development was done on Windows and Visual Studio in varying 
versions. For Linux, each tool has a hand written make file for now.
Some applications can be built using cmake.



# Dependcies:
- Cuda 8.0 
- openMPI on Linux, version of your choice
- MPICH on Windows, http://www.mpich.org/static/downloads/1.4.1p1/mpich2-1.4.1p1-win-x86-64.msi
- For Clicker only: Qt 5.5, we still have troubles to get that
  compiled on different machines...

# Cuda kernels:
the kernels are mostly only compiled on Windows either through 
visualStudio or a batch file. The resulting ptx-file is then 
converted to a byte-array in a header file which is then used 
during compilation of the host-code, both for Windows and Linux. 
The main reason for this manual compilation step is the use of 
the CUDA driver API due to historic reasons.

# Missing symbols while linking:
The same code is re-used for different tools (especially EmSART 
and its falvours) with different code segments enabled or 
disabled by pre-processor macros. If you encounter linking errors,
run a "make clean" before the "make" step to correct for this.