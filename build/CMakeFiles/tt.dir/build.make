# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.9.6/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.9.6/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sheng/Desktop/free-view-point

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sheng/Desktop/free-view-point/build

# Include any dependencies generated for this target.
include CMakeFiles/tt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tt.dir/flags.make

CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o: CMakeFiles/tt.dir/flags.make
CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o: ../Examples/vgftv/tt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/sheng/Desktop/free-view-point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o -c /Users/sheng/Desktop/free-view-point/Examples/vgftv/tt.cpp

CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sheng/Desktop/free-view-point/Examples/vgftv/tt.cpp > CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.i

CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sheng/Desktop/free-view-point/Examples/vgftv/tt.cpp -o CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.s

CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.requires:

.PHONY : CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.requires

CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.provides: CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.requires
	$(MAKE) -f CMakeFiles/tt.dir/build.make CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.provides.build
.PHONY : CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.provides

CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.provides.build: CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o


# Object files for target tt
tt_OBJECTS = \
"CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o"

# External object files for target tt
tt_EXTERNAL_OBJECTS =

tt: CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o
tt: CMakeFiles/tt.dir/build.make
tt: ../lib/libfree-view-point.dylib
tt: /usr/local/lib/libopencv_dnn.3.3.0.dylib
tt: /usr/local/lib/libopencv_ml.3.3.0.dylib
tt: /usr/local/lib/libopencv_objdetect.3.3.0.dylib
tt: /usr/local/lib/libopencv_shape.3.3.0.dylib
tt: /usr/local/lib/libopencv_stitching.3.3.0.dylib
tt: /usr/local/lib/libopencv_superres.3.3.0.dylib
tt: /usr/local/lib/libopencv_videostab.3.3.0.dylib
tt: /usr/local/lib/libopencv_calib3d.3.3.0.dylib
tt: /usr/local/lib/libopencv_features2d.3.3.0.dylib
tt: /usr/local/lib/libopencv_flann.3.3.0.dylib
tt: /usr/local/lib/libopencv_highgui.3.3.0.dylib
tt: /usr/local/lib/libopencv_photo.3.3.0.dylib
tt: /usr/local/lib/libopencv_video.3.3.0.dylib
tt: /usr/local/lib/libopencv_videoio.3.3.0.dylib
tt: /usr/local/lib/libopencv_imgcodecs.3.3.0.dylib
tt: /usr/local/lib/libopencv_imgproc.3.3.0.dylib
tt: /usr/local/lib/libopencv_core.3.3.0.dylib
tt: CMakeFiles/tt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/sheng/Desktop/free-view-point/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tt.dir/build: tt

.PHONY : CMakeFiles/tt.dir/build

CMakeFiles/tt.dir/requires: CMakeFiles/tt.dir/Examples/vgftv/tt.cpp.o.requires

.PHONY : CMakeFiles/tt.dir/requires

CMakeFiles/tt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tt.dir/clean

CMakeFiles/tt.dir/depend:
	cd /Users/sheng/Desktop/free-view-point/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sheng/Desktop/free-view-point /Users/sheng/Desktop/free-view-point /Users/sheng/Desktop/free-view-point/build /Users/sheng/Desktop/free-view-point/build /Users/sheng/Desktop/free-view-point/build/CMakeFiles/tt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tt.dir/depend

