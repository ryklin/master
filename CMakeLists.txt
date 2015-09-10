# COMEDI for CMake (you're welcome)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
        if (GCC_VERSION VERSION_LESS 4.7)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++0x")
        else ()
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++11")
        endif ()
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native-std=c++11")
    endif ()

project(beast-tracker)


include_directories(
    SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}    
)
SET(COMEDI_INSTALL "/usr/local/")
FIND_FILE(COMEDILIB_FOUND "comedilib.h" ${COMEDI_INSTALL}/include)
SET(COMEDI_INCLUDE_DIRS "${COMEDI_INSTALL}/include")
SET(COMEDI_LINK_DIRS "${COMEDI_INSTALL}/lib")
SET(COMEDI_LIBS "comedi")
FIND_LIBRARY(COMEDI_LIBRARY NAMES comedilib comedi HINTS "${COMEDI_LINK_DIRS}")


find_package( OpenCV REQUIRED )
find_package (TBB REQUIRED)
find_package (Boost REQUIRED)
add_executable( beast-tracker beast-tracker.cpp )
target_link_libraries( beast-tracker ${OpenCV_LIBS} )

find_library( FLYCAPTURE2 flycapture)
target_link_libraries( beast-tracker ${FLYCAPTURE2})

target_link_libraries( beast-tracker ${COMEDI_LIBRARY} )

