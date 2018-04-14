cmake_minimum_required(VERSION 2.8)

project(free-view-point)

find_package(OpenCV REQUIRED)
find_Package(Eigen3 REQUIRED)
#find_package(PCL REQUIRED)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
#${PCL_INCLUDE_DIRS}
)

#add_definitions(${PCL_DEFINITIONS})
#link_directories(${PCL_LIBRARY_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library( ${PROJECT_NAME} SHARED

src/test_lib.cc
src/tool.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
#${PCL_LIBRARIES}
#${PROJECT_SOURCE_DIR}/Thirdparty/libDBoW2.so
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/test)

add_executable(test
Examples/test/test.cc)
target_link_libraries(test ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/vgftv)
add_executable(ftv
Examples/vgftv/main.cpp)
target_link_libraries(ftv ${PROJECT_NAME})

add_executable( tt
Examples/vgftv/tt.cpp)
target_link_libraries(tt ${PROJECT_NAME})
