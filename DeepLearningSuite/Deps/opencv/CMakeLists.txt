FIND_PACKAGE(OpenCV NO_DEFAULT_PATH PATHS "/usr/local" "/usr" )

if(OpenCV_FOUND)

  if(OpenCV_VERSION VERSION_LESS "3.4")
    message(FATAL_ERROR "Minimum OpenCV Version Required is 3.4")
  endif(OpenCV_VERSION VERSION_LESS "3.4")
else()
  message(FATAL_ERROR "Error: Can't Find OpenCV")
endif(OpenCV_FOUND)
