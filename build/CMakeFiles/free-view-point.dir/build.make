# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sheng/free-view-point

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sheng/free-view-point/build

# Include any dependencies generated for this target.
include CMakeFiles/free-view-point.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/free-view-point.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/free-view-point.dir/flags.make

CMakeFiles/free-view-point.dir/src/test_lib.cc.o: CMakeFiles/free-view-point.dir/flags.make
CMakeFiles/free-view-point.dir/src/test_lib.cc.o: ../src/test_lib.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sheng/free-view-point/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/free-view-point.dir/src/test_lib.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/free-view-point.dir/src/test_lib.cc.o -c /home/sheng/free-view-point/src/test_lib.cc

CMakeFiles/free-view-point.dir/src/test_lib.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/free-view-point.dir/src/test_lib.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sheng/free-view-point/src/test_lib.cc > CMakeFiles/free-view-point.dir/src/test_lib.cc.i

CMakeFiles/free-view-point.dir/src/test_lib.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/free-view-point.dir/src/test_lib.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sheng/free-view-point/src/test_lib.cc -o CMakeFiles/free-view-point.dir/src/test_lib.cc.s

CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires:
.PHONY : CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires

CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides: CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires
	$(MAKE) -f CMakeFiles/free-view-point.dir/build.make CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides.build
.PHONY : CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides

CMakeFiles/free-view-point.dir/src/test_lib.cc.o.provides.build: CMakeFiles/free-view-point.dir/src/test_lib.cc.o

CMakeFiles/free-view-point.dir/src/tool.cpp.o: CMakeFiles/free-view-point.dir/flags.make
CMakeFiles/free-view-point.dir/src/tool.cpp.o: ../src/tool.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sheng/free-view-point/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/free-view-point.dir/src/tool.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/free-view-point.dir/src/tool.cpp.o -c /home/sheng/free-view-point/src/tool.cpp

CMakeFiles/free-view-point.dir/src/tool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/free-view-point.dir/src/tool.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sheng/free-view-point/src/tool.cpp > CMakeFiles/free-view-point.dir/src/tool.cpp.i

CMakeFiles/free-view-point.dir/src/tool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/free-view-point.dir/src/tool.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sheng/free-view-point/src/tool.cpp -o CMakeFiles/free-view-point.dir/src/tool.cpp.s

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

../lib/libfree-view-point.so: CMakeFiles/free-view-point.dir/src/test_lib.cc.o
../lib/libfree-view-point.so: CMakeFiles/free-view-point.dir/src/tool.cpp.o
../lib/libfree-view-point.so: CMakeFiles/free-view-point.dir/build.make
../lib/libfree-view-point.so: /usr/local/lib/libopencv_videostab.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_video.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_ts.a
../lib/libfree-view-point.so: /usr/local/lib/libopencv_superres.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_stitching.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_photo.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_ocl.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_objdetect.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_nonfree.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_ml.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_legacy.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_imgproc.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_highgui.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_gpu.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_flann.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_features2d.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_core.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_contrib.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_calib3d.so.2.4.13
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libfree-view-point.so: /usr/lib/libpcl_common.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libfree-view-point.so: /usr/lib/libpcl_kdtree.so
../lib/libfree-view-point.so: /usr/lib/libpcl_octree.so
../lib/libfree-view-point.so: /usr/lib/libpcl_search.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libqhull.so
../lib/libfree-view-point.so: /usr/lib/libpcl_surface.so
../lib/libfree-view-point.so: /usr/lib/libpcl_sample_consensus.so
../lib/libfree-view-point.so: /usr/lib/libOpenNI.so
../lib/libfree-view-point.so: /usr/lib/libOpenNI2.so
../lib/libfree-view-point.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkGenericFiltering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkGeovis.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkCharts.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libpcl_io.so
../lib/libfree-view-point.so: /usr/lib/libpcl_filters.so
../lib/libfree-view-point.so: /usr/lib/libpcl_features.so
../lib/libfree-view-point.so: /usr/lib/libpcl_keypoints.so
../lib/libfree-view-point.so: /usr/lib/libpcl_registration.so
../lib/libfree-view-point.so: /usr/lib/libpcl_segmentation.so
../lib/libfree-view-point.so: /usr/lib/libpcl_recognition.so
../lib/libfree-view-point.so: /usr/lib/libpcl_visualization.so
../lib/libfree-view-point.so: /usr/lib/libpcl_people.so
../lib/libfree-view-point.so: /usr/lib/libpcl_outofcore.so
../lib/libfree-view-point.so: /usr/lib/libpcl_tracking.so
../lib/libfree-view-point.so: /usr/lib/libpcl_apps.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libpthread.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libqhull.so
../lib/libfree-view-point.so: /usr/lib/libOpenNI.so
../lib/libfree-view-point.so: /usr/lib/libOpenNI2.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../lib/libfree-view-point.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkGenericFiltering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkGeovis.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkCharts.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libGLU.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libGL.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libSM.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libICE.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libX11.so
../lib/libfree-view-point.so: /usr/lib/x86_64-linux-gnu/libXext.so
../lib/libfree-view-point.so: /usr/local/lib/libopencv_nonfree.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_ocl.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_gpu.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_photo.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_objdetect.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_legacy.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_video.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_ml.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_calib3d.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_features2d.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_highgui.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_imgproc.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_flann.so.2.4.13
../lib/libfree-view-point.so: /usr/local/lib/libopencv_core.so.2.4.13
../lib/libfree-view-point.so: /usr/lib/libpcl_common.so
../lib/libfree-view-point.so: /usr/lib/libpcl_kdtree.so
../lib/libfree-view-point.so: /usr/lib/libpcl_octree.so
../lib/libfree-view-point.so: /usr/lib/libpcl_search.so
../lib/libfree-view-point.so: /usr/lib/libpcl_surface.so
../lib/libfree-view-point.so: /usr/lib/libpcl_sample_consensus.so
../lib/libfree-view-point.so: /usr/lib/libpcl_io.so
../lib/libfree-view-point.so: /usr/lib/libpcl_filters.so
../lib/libfree-view-point.so: /usr/lib/libpcl_features.so
../lib/libfree-view-point.so: /usr/lib/libpcl_keypoints.so
../lib/libfree-view-point.so: /usr/lib/libpcl_registration.so
../lib/libfree-view-point.so: /usr/lib/libpcl_segmentation.so
../lib/libfree-view-point.so: /usr/lib/libpcl_recognition.so
../lib/libfree-view-point.so: /usr/lib/libpcl_visualization.so
../lib/libfree-view-point.so: /usr/lib/libpcl_people.so
../lib/libfree-view-point.so: /usr/lib/libpcl_outofcore.so
../lib/libfree-view-point.so: /usr/lib/libpcl_tracking.so
../lib/libfree-view-point.so: /usr/lib/libpcl_apps.so
../lib/libfree-view-point.so: /usr/lib/libvtkViews.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkInfovis.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkWidgets.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkVolumeRendering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkHybrid.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkParallel.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkRendering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkImaging.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkGraphics.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkIO.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkFiltering.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtkCommon.so.5.8.0
../lib/libfree-view-point.so: /usr/lib/libvtksys.so.5.8.0
../lib/libfree-view-point.so: CMakeFiles/free-view-point.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library ../lib/libfree-view-point.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/free-view-point.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/free-view-point.dir/build: ../lib/libfree-view-point.so
.PHONY : CMakeFiles/free-view-point.dir/build

CMakeFiles/free-view-point.dir/requires: CMakeFiles/free-view-point.dir/src/test_lib.cc.o.requires
CMakeFiles/free-view-point.dir/requires: CMakeFiles/free-view-point.dir/src/tool.cpp.o.requires
.PHONY : CMakeFiles/free-view-point.dir/requires

CMakeFiles/free-view-point.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/free-view-point.dir/cmake_clean.cmake
.PHONY : CMakeFiles/free-view-point.dir/clean

CMakeFiles/free-view-point.dir/depend:
	cd /home/sheng/free-view-point/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sheng/free-view-point /home/sheng/free-view-point /home/sheng/free-view-point/build /home/sheng/free-view-point/build /home/sheng/free-view-point/build/CMakeFiles/free-view-point.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/free-view-point.dir/depend

