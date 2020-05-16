FIND_PACKAGE(OpenCV NO_DEFAULT_PATH PATHS "/usr/local" "/usr" )

if(OpenCV_FOUND)

  if(OpenCV_VERSION VERSION_LESS "4.2")
    message(FATAL_ERROR "Minimum OpenCV Version Required is 4.2")
  endif(OpenCV_VERSION VERSION_LESS "4.2")
else()
  message(FATAL_ERROR "Error: Can't Find OpenCV")
endif(OpenCV_FOUND)
