cmake_minimum_required (VERSION 2.8)

project (GenerateNoise)

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g -ggdb" )
set ( SRC_DIR "src" )
set ( BIN_DIR "bin" )

find_package( OpenCV REQUIRED )

add_executable(${BIN_DIR}/GenerateNoise
  ${SRC_DIR}/main.cc
)

target_link_libraries (${BIN_DIR}/GenerateNoise
  ${OpenCV_LIBS}
)
