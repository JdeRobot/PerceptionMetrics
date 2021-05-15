find_package(yaml-cpp REQUIRED)

if (YAML_CPP_INCLUDE_DIR)
	message("***YAML-CPP FOUND: ${YAML_CPP_INCLUDE_DIR}")
	list(APPEND DEPS libyaml-cpp0.5v5)
	list(APPEND DEPS_DEV libyaml-cpp-dev)
else()
	message ("*** YAML-CPP NOT FOUND")
endIF()
