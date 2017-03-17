cmake_minimum_required(VERSION 2.8)

project(free-view-point)

find_package(OpenCV REQUIRED)
find_Package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library( ${PROJECT_NAME} SHARED

src/test_lib.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PCL_LIBRARIES}
#${PROJECT_SOURCE_DIR}/Thirdparty/libDBoW2.so
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/test)

add_executable(test
Examples/test/test.cc)
target_link_libraries(test ${PROJECT_NAME})
