
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${config_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${CMAKE_CURRENT_LIST_DIR}
        ${QT_INCLUDE_DIRS}
        ${DeepLearningSuite_INCLUDE_DIR}

)

add_executable (converter  converter.cpp)


TARGET_LINK_LIBRARIES(converter
        DeepLearningSuite
        ${EXTRA_LIBS}
        ${QT_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${Ice_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ssl
        crypto
        )
