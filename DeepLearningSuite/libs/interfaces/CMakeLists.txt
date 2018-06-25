if (ZeroCIce_FOUND)
    project(JderobotInterfaces)

    set(SLICE_NEW_STYLE "ON")
    if (NOT DEFINED SLICE_NEW_STYLE)

    include (${CMAKE_CURRENT_LIST_DIR}/slice/CMakeLists.txt)

    include_directories(
            ${CMAKE_CURRENT_LIST_DIR}/cpp/jderobot
            ${CMAKE_CURRENT_LIST_DIR}/cpp/
    )
    add_library (JderobotInterfaces SHARED ${SOURCE_FILES} ${ICE_FILES})
    TARGET_LINK_LIBRARIES(JderobotInterfaces ${ice2_LIBRARIES})

    else()

    ### CMakeCache build control

    set(build_interfaces_cpp "ON" CACHE BOOL "Build Ice interfaces for CPP")


    message(STATUS "Building ICE Interfaces for:")
    if (build_interfaces_cpp)
        message(STATUS "\tCPP")
    endif()

    set(SLICE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/slice)
    set(SLICE_CPP_DIR  ${CMAKE_CURRENT_BINARY_DIR}/cpp)

    FILE(GLOB_RECURSE SLICE_FILES
        RELATIVE ${SLICE_DIRECTORY}
        "*.ice"
    )

    ### Trick to publich .ice files into CMake's sources
    file(GLOB_RECURSE SLICE_DEFINITIONS "slice/**.ice")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.c "")
    add_library(JderobotInterfaces_Definitions ${CMAKE_CURRENT_BINARY_DIR}/dummy.c ${SLICE_DEFINITIONS})


    foreach(slice_rel_path ${SLICE_FILES})
        get_filename_component(slice_name ${slice_rel_path} NAME_WE)
        get_filename_component(slice_file_name ${slice_rel_path} NAME)
        get_filename_component(slice_rel_dir   ${slice_rel_path} PATH) # PATH for cmake <= 2.8.11
        set(slice_abs_path  ${SLICE_DIRECTORY}/${slice_rel_path}) # get_filename_component() relative to absolute relies in cmake_current_source_dir. Not valid fot his setup


        ### CPP
        if (build_interfaces_cpp)
            set(target_directory ${SLICE_CPP_DIR}/${slice_rel_dir})
            file(MAKE_DIRECTORY ${target_directory})
            execute_process(
                WORKING_DIRECTORY ${target_directory}
                COMMAND slice2cpp ${slice_abs_path} -I${SLICE_DIRECTORY} --output-dir ${target_directory} --include-dir ${slice_rel_dir}
                INPUT_FILE ${slice_abs_path}
            )
            #LIST(APPEND SLICE_CPP_GENERATED ${target_directory}/${slice_name}.h) # fetching .h files relies on include_directories()
            LIST(APPEND SLICE_CPP_GENERATED ${target_directory}/${slice_name}.cpp)
        endif()

    endforeach()


    unset(slice_abs_path)
    unset(slice_rel_dir)
    unset(slice_file_name)
    unset(slice_name)
    unset(target_directory)
    unset(SLICE_DIRECTORY)


    ### CPP
    if (build_interfaces_cpp)
        include_directories(${SLICE_CPP_DIR})
        add_library (JderobotInterfaces SHARED ${SLICE_CPP_GENERATED})
        target_link_libraries(JderobotInterfaces ${ice2_LIBRARIES})
    endif()

    set(ice_interfaces_INCLUDE_DIRS "${SLICE_CPP_DIR}" CACHE PATH "Find(ice_interfaces)")

    endif(NOT DEFINED SLICE_NEW_STYLE)

    set(${PROJECT_NAME}_LIBRARIES "${PROJECT_NAME}" CACHE STRINGS "Find(${PROJECT_NAME})")

endif()
