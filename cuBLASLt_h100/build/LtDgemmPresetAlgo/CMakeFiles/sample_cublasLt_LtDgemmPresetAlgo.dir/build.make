# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build

# Include any dependencies generated for this target.
include LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/depend.make

# Include the progress variables for this target.
include LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/progress.make

# Include the compile flags for this target's objects.
include LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/flags.make

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.o: LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/flags.make
LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.o: ../LtDgemmPresetAlgo/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.o"
	cd /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.o -c /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/LtDgemmPresetAlgo/main.cpp

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.i"
	cd /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/LtDgemmPresetAlgo/main.cpp > CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.i

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.s"
	cd /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/LtDgemmPresetAlgo/main.cpp -o CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.s

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.o: LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/flags.make
LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.o: ../LtDgemmPresetAlgo/sample_cublasLt_LtDgemmPresetAlgo.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.o"
	cd /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo && /usr/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/LtDgemmPresetAlgo/sample_cublasLt_LtDgemmPresetAlgo.cu -o CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.o

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sample_cublasLt_LtDgemmPresetAlgo
sample_cublasLt_LtDgemmPresetAlgo_OBJECTS = \
"CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.o" \
"CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.o"

# External object files for target sample_cublasLt_LtDgemmPresetAlgo
sample_cublasLt_LtDgemmPresetAlgo_EXTERNAL_OBJECTS =

LtDgemmPresetAlgo/sample_cublasLt_LtDgemmPresetAlgo: LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/main.cpp.o
LtDgemmPresetAlgo/sample_cublasLt_LtDgemmPresetAlgo: LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/sample_cublasLt_LtDgemmPresetAlgo.cu.o
LtDgemmPresetAlgo/sample_cublasLt_LtDgemmPresetAlgo: LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/build.make
LtDgemmPresetAlgo/sample_cublasLt_LtDgemmPresetAlgo: LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable sample_cublasLt_LtDgemmPresetAlgo"
	cd /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/build: LtDgemmPresetAlgo/sample_cublasLt_LtDgemmPresetAlgo

.PHONY : LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/build

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/clean:
	cd /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo && $(CMAKE_COMMAND) -P CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/cmake_clean.cmake
.PHONY : LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/clean

LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/depend:
	cd /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/LtDgemmPresetAlgo /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo /home/ubuntu/chiang/cuda_test/CUDALibrarySamples/cuBLASLt/build/LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : LtDgemmPresetAlgo/CMakeFiles/sample_cublasLt_LtDgemmPresetAlgo.dir/depend

