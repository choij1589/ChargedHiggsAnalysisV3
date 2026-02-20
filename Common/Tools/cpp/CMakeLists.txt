cmake_minimum_required(VERSION 3.20)
project(CommonTools)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROOT
find_package(ROOT REQUIRED COMPONENTS RIO Tree RooFit RooFitCore)

# Include ROOT headers
include_directories(${ROOT_INCLUDE_DIRS})

# Generate ROOT dictionary
file(GLOB HEADERS include/*.h)
list(FILTER HEADERS EXCLUDE REGEX "LinkDef.h")
ROOT_GENERATE_DICTIONARY(G__CommonTools
    ${HEADERS}
    LINKDEF include/LinkDef.h
)

# Define the library with source files
file(GLOB SOURCES src/*.cc)
add_library(CommonTools SHARED
    ${SOURCES}
    G__CommonTools.cxx
)

target_include_directories(CommonTools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(CommonTools PUBLIC ${ROOT_LIBRARIES})

install(TARGETS CommonTools DESTINATION lib)

# Install .rootmap and .pcm files to the lib
set(ROOTMAP "${CMAKE_CURRENT_BINARY_DIR}/libCommonTools.rootmap")
set(PCM "${CMAKE_CURRENT_BINARY_DIR}/libCommonTools_rdict.pcm")
install(FILES ${ROOTMAP} ${PCM} DESTINATION lib)
