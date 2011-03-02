CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

IF(COMMAND cmake_policy)
  # Library paths like /path/to/mylib.so aren't translated into
  # -L/path/to -lmylib
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND cmake_policy)

PROJECT(syrah)

INCLUDE(CTest)

SET(CMAKE_VERBOSE_MAKEFILE OFF)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

INCLUDE (${CMAKE_ROOT}/Modules/FindThreads.cmake)

IF (CMAKE_COMPILER_IS_GNUCXX)
 SET(CMAKE_CXX_FLAGS "-msse4.1 ${CMAKE_CXX_FLAGS}")
ENDIF ()

IF (CMAKE_USE_PTHREADS_INIT)
 SET(SYRAH_THREAD_LIB ${CMAKE_THREAD_LIBS_INIT})
ELSE(CMAKE_USE_PTHREADS_INIT)
  IF(WIN32)
   SET(SYRAH_THREAD_LIB Winmm.lib)
  ENDIF(WIN32)
ENDIF(CMAKE_USE_PTHREADS_INIT)

ADD_EXECUTABLE(bringup unitTests/bringup.cc)
ADD_EXECUTABLE(taskBringup unitTests/taskBringup.cc)
ADD_EXECUTABLE(saxpy benchmarks/saxpy/saxpy.cc)
ADD_EXECUTABLE(trig benchmarks/trig/trig.cc)
ADD_EXECUTABLE(memcpy benchmarks/memcpy/memcpy.cc)

TARGET_LINK_LIBRARIES(taskBringup ${SYRAH_THREAD_LIB})
