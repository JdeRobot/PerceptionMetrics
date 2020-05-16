cmake_minimum_required(VERSION 2.8)
project(config)

### Project config
include_directories(
    include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

set(HEADERS
    include/config/config.h
    include/config/loader.hpp
    include/config/properties.hpp
    include/config/stdutils.hpp
)

set(SOURCES
    src/loader.cpp
    src/properties.cpp
)


add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}  ${YAML_CPP_LIBRARIES} ${GLOG_LIBRARIES})

## Export library variables (like find_package)
set(${PROJECT_NAME}_FOUND 1 CACHE BOOL "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_LIBRARY_DIRS "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Find(${PROJECT_NAME})")
set(${PROJECT_NAME}_LIBRARIES "${PROJECT_NAME}" CACHE STRINGS "Find(${PROJECT_NAME})")
