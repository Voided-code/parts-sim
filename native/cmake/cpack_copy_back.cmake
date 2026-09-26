# CPack post-build step: copy the packages from the staging directory into the build directory.
foreach(f IN LISTS CPACK_PACKAGE_FILES)
  file(COPY "${f}" DESTINATION "${CPACK_PS_OUTPUT_DIR}")
  get_filename_component(name "${f}" NAME)
  message(STATUS "Package: ${CPACK_PS_OUTPUT_DIR}/${name}")
endforeach()
