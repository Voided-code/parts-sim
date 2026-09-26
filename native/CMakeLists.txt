cmake_minimum_required(VERSION 3.24)
project(PartsSim VERSION 0.3.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# OCCT exports its precompiled headers to consumers. Keep those Windows/system
# headers out of unrelated app and solver sources; OCCT itself still builds with PCH.
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PARTS_SIM_APP "Build the Qt desktop app" ON)
option(PARTS_SIM_GPU "Use wgpu-native for GPU compute" ON)
option(PARTS_SIM_STEP "Use OpenCascade for STEP/IGES import" ON)

if(MSVC)
  # sources are UTF-8 (°, µ, × in UI strings); no min/max macros from windows.h
  add_compile_options(/W3 /permissive- /Zc:__cplusplus /bigobj /utf-8)
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
  endif()
endif()

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
if(PARTS_SIM_GPU)
  # Homebrew's wgpu-native, or an unpacked release from github.com/gfx-rs/wgpu-native (WGPU_NATIVE_DIR)
  set(WGPU_NATIVE_DIR "$ENV{WGPU_NATIVE_DIR}" CACHE PATH "Unpacked wgpu-native release")
  find_path(WGPU_INCLUDE_DIR webgpu.h HINTS ${WGPU_NATIVE_DIR} PATHS /opt/homebrew/opt/wgpu-native /usr/local PATH_SUFFIXES include/webgpu include)
  # on Windows link the DLL's import library, not the static library
  find_library(WGPU_LIBRARY NAMES wgpu_native.dll.lib wgpu_native HINTS ${WGPU_NATIVE_DIR} PATHS /opt/homebrew/opt/wgpu-native /usr/local PATH_SUFFIXES lib)
  if(WIN32)
    find_file(WGPU_DLL wgpu_native.dll HINTS ${WGPU_NATIVE_DIR} PATH_SUFFIXES lib bin)
  endif()
  if(NOT WGPU_INCLUDE_DIR OR NOT WGPU_LIBRARY)
    message(WARNING "wgpu-native not found: solvers run on the CPU only")
    set(PARTS_SIM_GPU OFF)
  endif()
endif()
file(GLOB WGSL_SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.wgsl)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wgsl_sources.cpp
  COMMAND ${CMAKE_COMMAND} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/wgsl_sources.cpp -DDIR=${CMAKE_CURRENT_SOURCE_DIR}/shaders -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake
  DEPENDS ${WGSL_SHADERS} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_wgsl.cmake
  COMMENT "Embedding WGSL shaders"
  VERBATIM)
if(PARTS_SIM_STEP)
  find_package(OpenCASCADE CONFIG QUIET)
  if(NOT OpenCASCADE_FOUND)
    message(WARNING "OpenCascade not found: STEP/IGES import and CAD sample parts are disabled")
    set(PARTS_SIM_STEP OFF)
  endif()
endif()

# ---------------------------------------------------------------------------------------------
# Simulation engine (no UI): importers, voxelizer, structural, thermal and airflow solvers
add_library(partssim_core STATIC
  src/util/parallel.cpp
  src/util/xml.cpp
  src/util/zip.cpp
  src/util/json.cpp
  src/util/system.cpp
  src/core/mesh.cpp
  src/core/bvh.cpp
  src/core/voxelize.cpp
  src/core/materials.cpp
  src/core/importers.cpp
  src/core/solidworks.cpp
  src/core/cad_occt.cpp
  src/fea/hex8.cpp
  src/fea/voxel_fea.cpp
  src/fea/structural.cpp
  src/fea/static_study.cpp
  src/fea/eigen.cpp
  src/fea/explicit.cpp
  src/fea/nonlinear.cpp
  src/fea/fatigue.cpp
  src/fea/response.cpp
  src/fea/topology.cpp
  src/fea/thermal.cpp
  src/fea/study_runner.cpp
  src/cfd/lbm.cpp
  src/cfd/lbm_gpu.cpp
  src/cfd/airflow.cpp
  src/gpu/gpu.cpp
  src/gpu/gpu_fea.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/wgsl_sources.cpp
)
target_include_directories(partssim_core PUBLIC src)
target_link_libraries(partssim_core PUBLIC Threads::Threads ZLIB::ZLIB)
if(PARTS_SIM_GPU)
  target_compile_definitions(partssim_core PUBLIC PARTS_SIM_HAS_GPU)
  target_include_directories(partssim_core PUBLIC ${WGPU_INCLUDE_DIR})
  target_link_libraries(partssim_core PUBLIC ${WGPU_LIBRARY})
  if(APPLE)
    target_link_libraries(partssim_core PUBLIC "-framework Metal" "-framework QuartzCore" "-framework Foundation")
  endif()
endif()
if(PARTS_SIM_STEP)
  target_compile_definitions(partssim_core PUBLIC PARTS_SIM_HAS_OCCT)
  target_include_directories(partssim_core PUBLIC ${OpenCASCADE_INCLUDE_DIR})
  target_link_libraries(partssim_core PUBLIC TKernel TKMath TKBRep TKG2d TKG3d TKGeomBase TKGeomAlgo TKTopAlgo TKPrim TKBO
    TKOffset TKFillet TKShHealing TKMesh TKXSBase TKDESTEP TKDEIGES)
endif()

# ---------------------------------------------------------------------------------------------
# Validation tests (the same cases as the JavaScript suite)
enable_testing()
function(ps_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE partssim_core)
  target_compile_definitions(${name} PRIVATE PS_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_test(NAME ${name} COMMAND ${name})
endfunction()
ps_test(test_fea)
ps_test(test_model)
ps_test(test_gpu)
ps_test(test_studies)
ps_test(test_lbm)

# ---------------------------------------------------------------------------------------------
# Desktop app (Qt Widgets, 3D viewport through Qt RHI: Metal / Direct3D / Vulkan / OpenGL)
if(PARTS_SIM_APP)
  find_package(Qt6 6.7 REQUIRED COMPONENTS Core Gui Widgets ShaderTools)
  # Qt 6.8 exports GuiPrivate from Gui; newer Qt versions also provide a
  # separate package. Do not require that package on older supported Qt builds.
  if(NOT TARGET Qt6::GuiPrivate)
    find_package(Qt6 REQUIRED COMPONENTS GuiPrivate)
  endif()
  if(Qt6_FOUND)
    qt_standard_project_setup()
    set(CMAKE_AUTOMOC ON)
    qt_add_executable(PartsSim WIN32 MACOSX_BUNDLE
      src/app/main.cpp
      src/app/mainwindow.cpp
      src/app/viewport.cpp
      src/app/widgets.cpp
      src/app/jobs.cpp
      src/app/partpanel.cpp
      src/app/picker.cpp
      src/app/structuralpanel.cpp
      src/app/samples.cpp
      src/app/studies.cpp
      src/app/thermalpanel.cpp
      src/app/airflowpanel.cpp
    )
    if(WIN32)
      target_sources(PartsSim PRIVATE cmake/PartsSim.rc)
    endif()
    qt_add_shaders(PartsSim "viewport_shaders" PREFIX "/shaders" BASE shaders FILES shaders/mesh.vert shaders/mesh.frag)
    target_link_libraries(PartsSim PRIVATE partssim_core Qt6::Core Qt6::Gui Qt6::GuiPrivate Qt6::Widgets)
    target_compile_definitions(PartsSim PRIVATE PARTS_SIM_VERSION="${PROJECT_VERSION}")
    # window icon (all platforms)
    set(PS_ICON_PNG "${CMAKE_CURRENT_SOURCE_DIR}/../build/icon.png")
    if(EXISTS "${PS_ICON_PNG}")
      set_source_files_properties("${PS_ICON_PNG}" PROPERTIES QT_RESOURCE_ALIAS "icon.png")
      qt_add_resources(PartsSim "app_icon" PREFIX "/" FILES "${PS_ICON_PNG}")
    endif()
    set_target_properties(PartsSim PROPERTIES
      OUTPUT_NAME "Parts Sim"
      MACOSX_BUNDLE_BUNDLE_NAME "Parts Sim"
      MACOSX_BUNDLE_GUI_IDENTIFIER "org.partssim.app"
      MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
      MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
      MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist.in")
    if(UNIX AND NOT APPLE)
      # Linux convention: a lowercase command without spaces (the desktop entry shows "Parts Sim")
      set_target_properties(PartsSim PROPERTIES OUTPUT_NAME "parts-sim" INSTALL_RPATH "$ORIGIN/../lib")
    endif()
    if(APPLE AND EXISTS "${PS_ICON_PNG}")
      # Finder icon: an .icns built from the same PNG
      set(PS_ICONSET "${CMAKE_CURRENT_BINARY_DIR}/PartsSim.iconset")
      set(PS_ICNS "${CMAKE_CURRENT_BINARY_DIR}/PartsSim.icns")
      set(PS_ICON_CMDS)
      foreach(size 16 32 128 256 512)
        math(EXPR size2 "${size} * 2")
        list(APPEND PS_ICON_CMDS
          COMMAND sips -z ${size} ${size} "${PS_ICON_PNG}" --out "${PS_ICONSET}/icon_${size}x${size}.png" > /dev/null
          COMMAND sips -z ${size2} ${size2} "${PS_ICON_PNG}" --out "${PS_ICONSET}/icon_${size}x${size}@2x.png" > /dev/null)
      endforeach()
      add_custom_command(OUTPUT "${PS_ICNS}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PS_ICONSET}"
        ${PS_ICON_CMDS}
        COMMAND iconutil -c icns "${PS_ICONSET}" -o "${PS_ICNS}"
        DEPENDS "${PS_ICON_PNG}"
        COMMENT "Building the app icon")
      target_sources(PartsSim PRIVATE "${PS_ICNS}")
      set_source_files_properties("${PS_ICNS}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
    endif()

    # ---- install and package: Qt, OpenCascade and wgpu libraries go into the app ----
    install(TARGETS PartsSim BUNDLE DESTINATION . RUNTIME DESTINATION bin)
    qt_generate_deploy_app_script(TARGET PartsSim OUTPUT_SCRIPT PS_DEPLOY_SCRIPT NO_UNSUPPORTED_PLATFORM_ERROR)
    if(APPLE)
      # codesign rejects extended attributes (Finder info, iCloud / file-provider tags) anywhere in the
      # bundle: strip them before macdeployqt signs, then sign the finished app again (ad hoc)
      set(PS_BUNDLE "\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/Parts Sim.app")
      install(CODE "execute_process(COMMAND xattr -cr \"${PS_BUNDLE}\")")
      install(SCRIPT "${PS_DEPLOY_SCRIPT}")
      install(CODE "
        execute_process(COMMAND xattr -cr \"${PS_BUNDLE}\")
        execute_process(COMMAND codesign --force --deep --sign - \"${PS_BUNDLE}\" RESULT_VARIABLE rc)
        if(NOT rc EQUAL 0)
          message(FATAL_ERROR \"Could not sign ${PS_BUNDLE}\")
        endif()
        execute_process(COMMAND codesign --verify --deep --strict \"${PS_BUNDLE}\" RESULT_VARIABLE rc)
        if(NOT rc EQUAL 0)
          message(FATAL_ERROR \"The signature of ${PS_BUNDLE} does not verify\")
        endif()")
    elseif(WIN32)
      # Qt's DLLs and plugins (windeployqt), then everything else the program loads: OpenCascade,
      # wgpu-native, zlib, and the Microsoft C++ runtime, so it runs on a PC without them
      install(SCRIPT "${PS_DEPLOY_SCRIPT}")
      set(PS_DLL_DIRS ${OpenCASCADE_BINARY_DIR} "${ZLIB_INCLUDE_DIR}/../bin")
      if(WGPU_DLL)
        get_filename_component(PS_WGPU_DLL_DIR "${WGPU_DLL}" DIRECTORY)
        list(APPEND PS_DLL_DIRS "${PS_WGPU_DLL_DIR}")
      endif()
      install(CODE "
        file(GET_RUNTIME_DEPENDENCIES
          EXECUTABLES \"\${CMAKE_INSTALL_PREFIX}/bin/Parts Sim.exe\"
          RESOLVED_DEPENDENCIES_VAR deps UNRESOLVED_DEPENDENCIES_VAR missing
          DIRECTORIES ${PS_DLL_DIRS}
          PRE_EXCLUDE_REGEXES \"^api-ms-\" \"^ext-ms-\" \"^[Qq]t6\"
          POST_EXCLUDE_REGEXES \".*[Ss][Yy][Ss][Tt][Ee][Mm]32/.*\")
        foreach(dll IN LISTS deps)
          file(INSTALL \"\${dll}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/bin\")
        endforeach()
        if(missing)
          message(FATAL_ERROR \"Libraries not found for the package: \${missing}\")
        endif()")
      set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION bin)
      include(InstallRequiredSystemLibraries)
    else()
      # Linux: the AppImage (linuxdeploy, see .github/workflows/native.yml) bundles Qt and the rest
      install(FILES linux/parts-sim.desktop DESTINATION share/applications)
      install(FILES ../build/icon.svg DESTINATION share/icons/hicolor/scalable/apps RENAME parts-sim.svg)
    endif()
    set(CPACK_PACKAGE_NAME "Parts Sim")
    set(CPACK_PACKAGE_VENDOR "Parts Sim")
    set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
    set(CPACK_PACKAGE_FILE_NAME "Parts-Sim-${PROJECT_VERSION}-native-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")
    if(APPLE)
      set(CPACK_GENERATOR "DragNDrop")
      set(CPACK_DMG_VOLUME_NAME "Parts Sim")
      # Stage and sign in the temp directory: folders synced by iCloud (like Documents) put Finder
      # attributes back on the app bundle, and codesign refuses bundles that carry them. The finished
      # .dmg is copied back into the build directory.
      set(PS_TMP "$ENV{TMPDIR}")
      if(NOT PS_TMP)
        set(PS_TMP "/tmp")
      endif()
      set(CPACK_PACKAGE_DIRECTORY "${PS_TMP}/parts-sim-cpack")
      set(CPACK_PS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
      set(CPACK_POST_BUILD_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpack_copy_back.cmake")
    elseif(WIN32)
      set(CPACK_GENERATOR "ZIP;NSIS")
      set(CPACK_NSIS_DISPLAY_NAME "Parts Sim")
      set(CPACK_NSIS_PACKAGE_NAME "Parts Sim")
      set(CPACK_PACKAGE_INSTALL_DIRECTORY "Parts Sim")
      set(CPACK_PACKAGE_EXECUTABLES "Parts Sim" "Parts Sim")
      set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PartsSim.ico")
      set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PartsSim.ico")
      set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\Parts Sim.exe")
      set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    else()
      set(CPACK_GENERATOR "TGZ")
    endif()
    include(CPack)
  endif()
endif()
