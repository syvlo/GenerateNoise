cmake_minimum_required (VERSION 2.8)

project (GenerateNoise)

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=gnu++11 -Wall -Werror -g -ggdb" )
set ( SRC_DIR "src" )
set ( BIN_DIR "bin" )

find_package( OpenCV REQUIRED )

add_executable(${BIN_DIR}/GenerateNoise
  ${SRC_DIR}/main.cc
  ${SRC_DIR}/ImwHelper.cc
)

target_link_libraries (${BIN_DIR}/GenerateNoise
  ${OpenCV_LIBS}
)
