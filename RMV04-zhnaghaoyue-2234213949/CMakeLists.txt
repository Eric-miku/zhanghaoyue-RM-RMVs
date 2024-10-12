cmake_minimum_required(VERSION 3.0.0)
project(task)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 找到 OpenCV 库
find_package(OpenCV REQUIRED)


aux_source_directory(./src ALL_SRCS)
include_directories(include

    ${OpenCV_INCLUDE_DIRS}
)


add_executable(task main.cpp ${ALL_SRCS})
target_link_libraries(task ${OpenCV_LIBS})
