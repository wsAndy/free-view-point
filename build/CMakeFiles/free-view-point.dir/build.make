# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.7.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.7.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sheng/Desktop/free-view-point

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sheng/Desktop/free-view-point/build

# Include any dependencies generated for this target.
include CMakeFiles/free-view-point.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/free-view-point.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/free-view-point.dir/flags.make

CMakeFiles/free-view-point.dir/src/test_lib.cc.o: CMakeFiles/free-view-point.dir/flags.make
CMakeFiles/free-view-point.dir/src/test_lib.cc.o: ../src/test_lib.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/sheng/Desktop/free-view-point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/free-view-point.dir/src/test_lib.cc.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/free-view-point.dir/src/test_lib.cc.o -c /Users/sheng/Desktop/free-view-point/src/test_lib.cc

CMakeFiles/free-view-point.dir/src/test_lib.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/free-view-point.dir/src/test_lib.cc.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sheng/Desktop/free-view-point/src/test_lib.cc > CMakeFiles/free-view-point.dir/src/test_lib.cc.i

CMakeFiles/free-view-point.dir/src/test_lib.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/free-view-point.dir/src/test_lib.cc.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sheng/Desktop/free-view-point/src/test_lib.cc -o CMakeFiles/free-view-point.dir/src/test_lib.cc.s

CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires:

.PHONY : CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires

CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides: CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires
	$(MAKE) -f CMakeFiles/free-view-point.dir/build.make CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides.build
.PHONY : CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides

CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides.build: CMakeFiles/free-view-point.dir/src/test_lib.cc.o


CMakeFiles/free-view-point.dir/src/tool.cpp.o: CMakeFiles/free-view-point.dir/flags.make
CMakeFiles/free-view-point.dir/src/tool.cpp.o: ../src/tool.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/sheng/Desktop/free-view-point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/free-view-point.dir/src/tool.cpp.o"
	/usr/bin/clang++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/free-view-point.dir/src/tool.cpp.o -c /Users/sheng/Desktop/free-view-point/src/tool.cpp

CMakeFiles/free-view-point.dir/src/tool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/free-view-point.dir/src/tool.cpp.i"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sheng/Desktop/free-view-point/src/tool.cpp > CMakeFiles/free-view-point.dir/src/tool.cpp.i

CMakeFiles/free-view-point.dir/src/tool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/free-view-point.dir/src/tool.cpp.s"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sheng/Desktop/free-view-point/src/tool.cpp -o CMakeFiles/free-view-point.dir/src/tool.cpp.s

CMakeFiles/free-view-point.dir/src/tool.cpp.o.requires:

.PHONY : CMakeFiles/free-view-point.dir/src/tool.cpp.o.requires

CMakeFiles/free-view-point.dir/src/tool.cpp.o.provides: CMakeFiles/free-view-point.dir/src/tool.cpp.o.requires
	$(MAKE) -f CMakeFiles/free-view-point.dir/build.make CMakeFiles/free-view-point.dir/src/tool.cpp.o.provides.build
.PHONY : CMakeFiles/free-view-point.dir/src/tool.cpp.o.provides

CMakeFiles/free-view-point.dir/src/tool.cpp.o.provides.build: CMakeFiles/free-view-point.dir/src/tool.cpp.o


# Object files for target free-view-point
free__view__point_OBJECTS = \
"CMakeFiles/free-view-point.dir/src/test_lib.cc.o" \
"CMakeFiles/free-view-point.dir/src/tool.cpp.o"

# External object files for target free-view-point
free__view__point_EXTERNAL_OBJECTS =

../lib/libfree-view-point.dylib: CMakeFiles/free-view-point.dir/src/test_lib.cc.o
../lib/libfree-view-point.dylib: CMakeFiles/free-view-point.dir/src/tool.cpp.o
../lib/libfree-view-point.dylib: CMakeFiles/free-view-point.dir/build.make
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_dnn.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_ml.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_objdetect.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_shape.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_stitching.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_superres.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_videostab.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_calib3d.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_features2d.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_flann.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_highgui.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_photo.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_video.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_videoio.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_imgcodecs.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_imgproc.3.3.0.dylib
../lib/libfree-view-point.dylib: /usr/local/lib/libopencv_core.3.3.0.dylib
../lib/libfree-view-point.dylib: CMakeFiles/free-view-point.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/sheng/Desktop/free-view-point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../lib/libfree-view-point.dylib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/free-view-point.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/free-view-point.dir/build: ../lib/libfree-view-point.dylib

.PHONY : CMakeFiles/free-view-point.dir/build

CMakeFiles/free-view-point.dir/requires: CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires
CMakeFiles/free-view-point.dir/requires: CMakeFiles/free-view-point.dir/src/tool.cpp.o.requires

.PHONY : CMakeFiles/free-view-point.dir/requires

CMakeFiles/free-view-point.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/free-view-point.dir/cmake_clean.cmake
.PHONY : CMakeFiles/free-view-point.dir/clean

CMakeFiles/free-view-point.dir/depend:
	cd /Users/sheng/Desktop/free-view-point/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sheng/Desktop/free-view-point /Users/sheng/Desktop/free-view-point /Users/sheng/Desktop/free-view-point/build /Users/sheng/Desktop/free-view-point/build /Users/sheng/Desktop/free-view-point/build/CMakeFiles/free-view-point.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/free-view-point.dir/depend

