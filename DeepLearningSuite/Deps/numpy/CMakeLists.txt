SET(CMAKE_MODULE_PATH
	${CMAKE_MODULE_PATH}
	"${CMAKE_CURRENT_LIST_DIR}/CMake"
)


find_package( PythonInterp REQUIRED )
find_package( PythonLibs REQUIRED )
find_package( NumPy REQUIRED )


include_directories ( ${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR} )
link_directories(${PYTHON_LIBRARIES})
