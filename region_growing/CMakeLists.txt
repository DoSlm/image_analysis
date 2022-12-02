cmake_minimum_required(VERSION 3.21)

# Set project name
project (region_growing)

#
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O2")
set(SOURCE_FILES main.cpp)
add_executable(region_growing ${SOURCE_FILES})

# Include all necessary directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(Images)

# Make Images directory
file(MAKE_DIRECTORY build/Images)

# Define the target
target_link_libraries(region_growing ${OpenCV_LIBS})