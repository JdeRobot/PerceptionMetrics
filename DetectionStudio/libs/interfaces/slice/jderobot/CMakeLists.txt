
# Check the ice binaries

unset(ICE_FILES CACHE)
FILE(GLOB ICE_FILES "slice/jderobot/*.ice")

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/
  ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/jderobot/
  ${ice_path_storm}
  )



FOREACH(currentSourceFile ${ICE_FILES})


	string(REGEX REPLACE ".*/(.*)" "\\1" new_source ${currentSourceFile})
	string(REGEX REPLACE ".*/(.*).ice" "\\1.cpp" new_source1 ${currentSourceFile})
	string(REGEX REPLACE ".*/(.*).ice" "\\1.h" new_source2 ${currentSourceFile})
	string(REGEX REPLACE ".*/(.*).ice" "\\1" new_source_name ${currentSourceFile})
	LIST(APPEND SOURCE_FILES cpp/jderobot/${new_source1})
	LIST(APPEND SOURCE_FILES cpp/jderobot/${new_source2})

	#MESSAGE(STATUS, "source1= ${new_source1}  source2= ${new_source2}  source=${new_source}")
	# C++
	execute_process(COMMAND slice2cpp -I${CMAKE_CURRENT_LIST_DIR}/.. -I${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/${new_source}
			WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../cpp/jderobot/
	)

	message(COMMAND slice2cpp -I${CMAKE_CURRENT_LIST_DIR}/.. -I${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/${new_source}
			WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../../cpp/jderobot/
			)

ENDFOREACH(currentSourceFile)
