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
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sheng/free-view-point

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sheng/free-view-point/build

# Include any dependencies generated for this target.
include CMakeFiles/ftv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ftv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ftv.dir/flags.make

CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o: CMakeFiles/ftv.dir/flags.make
CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o: ../Examples/vgftv/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sheng/free-view-point/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o -c /home/sheng/free-view-point/Examples/vgftv/main.cpp

CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/sheng/free-view-point/Examples/vgftv/main.cpp > CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.i

CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/sheng/free-view-point/Examples/vgftv/main.cpp -o CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.s

CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.requires:
.PHONY : CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.requires

CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.provides: CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ftv.dir/build.make CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.provides.build
.PHONY : CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.provides

CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.provides.build: CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o

# Object files for target ftv
ftv_OBJECTS = \
"CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o"

# External object files for target ftv
ftv_EXTERNAL_OBJECTS =

../Examples/vgftv/ftv: CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o
../Examples/vgftv/ftv: CMakeFiles/ftv.dir/build.make
../Examples/vgftv/ftv: ../lib/libfree-view-point.so
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_xphoto.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_xobjdetect.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_tracking.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_surface_matching.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_structured_light.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_stereo.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_saliency.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_rgbd.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_reg.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_plot.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_optflow.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_ximgproc.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_line_descriptor.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_hdf.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_fuzzy.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_dpm.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_dnn.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_datasets.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_text.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_face.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_ccalib.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_bioinspired.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_bgsegm.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_aruco.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_videostab.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_superres.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_stitching.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_xfeatures2d.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_shape.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_video.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_photo.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_objdetect.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_calib3d.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_features2d.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_ml.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_highgui.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_videoio.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_imgcodecs.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_imgproc.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_flann.so.3.1.0
../Examples/vgftv/ftv: /home/sheng/anaconda2/lib/libopencv_core.so.3.1.0
../Examples/vgftv/ftv: /usr/lib/libvtkGenericFiltering.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkGeovis.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkCharts.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkViews.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkInfovis.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkWidgets.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkVolumeRendering.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkHybrid.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkParallel.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkRendering.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkImaging.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkGraphics.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkIO.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkFiltering.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtkCommon.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/libvtksys.so.5.8.0
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_system.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libpthread.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_common.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../Examples/vgftv/ftv: /usr/local/lib/libpcl_kdtree.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_octree.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_search.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_sample_consensus.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_filters.so
../Examples/vgftv/ftv: /usr/lib/libOpenNI.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_io.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_features.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_ml.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_segmentation.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_keypoints.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libqhull.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_surface.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_registration.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_recognition.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_visualization.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_people.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_outofcore.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_stereo.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_tracking.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_system.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libpthread.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_common.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../Examples/vgftv/ftv: /usr/local/lib/libpcl_kdtree.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_octree.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_search.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_sample_consensus.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_filters.so
../Examples/vgftv/ftv: /usr/lib/libOpenNI.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_io.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_features.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_ml.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_segmentation.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_keypoints.so
../Examples/vgftv/ftv: /usr/lib/x86_64-linux-gnu/libqhull.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_surface.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_registration.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_recognition.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_visualization.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_people.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_outofcore.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_stereo.so
../Examples/vgftv/ftv: /usr/local/lib/libpcl_tracking.so
../Examples/vgftv/ftv: CMakeFiles/ftv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../Examples/vgftv/ftv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ftv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ftv.dir/build: ../Examples/vgftv/ftv
.PHONY : CMakeFiles/ftv.dir/build

CMakeFiles/ftv.dir/requires: CMakeFiles/ftv.dir/Examples/vgftv/main.cpp.o.requires
.PHONY : CMakeFiles/ftv.dir/requires

CMakeFiles/ftv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ftv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ftv.dir/clean

CMakeFiles/ftv.dir/depend:
	cd /home/sheng/free-view-point/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sheng/free-view-point /home/sheng/free-view-point /home/sheng/free-view-point/build /home/sheng/free-view-point/build /home/sheng/free-view-point/build/CMakeFiles/ftv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ftv.dir/depend
