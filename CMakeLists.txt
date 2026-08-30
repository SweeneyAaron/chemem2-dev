cmake_minimum_required(VERSION 3.18)
project(ChemEM LANGUAGES CXX)

# ---- Policies / pybind11 behavior ----
cmake_policy(SET CMP0148 NEW)   # FindPython improvements (used by pybind11)
cmake_policy(SET CMP0167 NEW)   # Avoid FindBoost deprecation warnings
set(PYBIND11_FINDPYTHON ON)

# ---- Build options ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Output dirs (your original defaults) ----
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()
if(NOT DEFINED CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()
foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${cfg} CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CFG} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endforeach()

# Python modules should land directly in the ChemEM package
set(CHEMEM_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ChemEM CACHE PATH "ChemEM package directory")
message(STATUS "ChemEM package directory: ${CHEMEM_PACKAGE_DIR}")
message(STATUS "Compiled modules will be placed in: ${CHEMEM_PACKAGE_DIR}")
if(NOT EXISTS ${CHEMEM_PACKAGE_DIR})
  message(FATAL_ERROR "ChemEM package directory not found: ${CHEMEM_PACKAGE_DIR}")
endif()

# ---- Source layout (real paths) ----
set(PROJ_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/ChemEM/cpp)
set(PROJ_DOCK_DIR ${PROJ_SRC_DIR}/docking_refactor)
set(PROJ_DOCK_V2_DIR ${PROJ_SRC_DIR}/docking_v2)   # experimental engine, built alongside the baseline
if(NOT EXISTS ${PROJ_SRC_DIR})
  message(FATAL_ERROR "C++ source directory not found: ${PROJ_SRC_DIR}")
endif()
if(NOT EXISTS ${PROJ_DOCK_DIR})
  message(FATAL_ERROR "Docking source directory not found: ${PROJ_DOCK_DIR}")
endif()

# ---- Conda host prefix (very important in conda-build) ----
# In conda-build, the package's *host* prefix is $PREFIX. Use that.
if(DEFINED ENV{PREFIX})
  set(CONDA_HOST_PREFIX "$ENV{PREFIX}")
  message(STATUS "Using conda-build HOST prefix: ${CONDA_HOST_PREFIX}")
  list(PREPEND CMAKE_PREFIX_PATH "${CONDA_HOST_PREFIX}")
elseif(DEFINED ENV{CONDA_PREFIX})
  set(CONDA_HOST_PREFIX "$ENV{CONDA_PREFIX}")
  message(STATUS "Using CONDA_PREFIX: ${CONDA_HOST_PREFIX}")
  list(PREPEND CMAKE_PREFIX_PATH "${CONDA_HOST_PREFIX}")
else()
  message(FATAL_ERROR "No conda prefix detected. Activate an environment or run under conda-build.")
endif()

# ---- macOS: keep RPATHs relocatable ----
# Without this, CMake defaults to CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON and
# bakes the absolute build-env path (e.g.
# /Users/aaron.sweeney/anaconda3/envs/chem111/lib) into every .so as an
# LC_RPATH. conda-build then has to rewrite it with install_name_tool,
# which invalidates the linker's ad-hoc code signature on macOS and
# produces the `zsh: killed` / SIGKILL on import.
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)

  # THE key line: do not inherit link-time library dirs into install RPATH.
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

  # Build-tree RPATH: point at the active conda env so in-tree imports
  # (e.g. the post-build import tests in conda_build_commands.sh) still
  # find RDKit/OpenMP before the package is installed.
  set(CMAKE_BUILD_RPATH "${CONDA_HOST_PREFIX}/lib")

  # Install RPATH: @loader_path-relative so the installed .so finds
  # $PREFIX/lib regardless of where the target env lives on disk.
  # Installed layout: $PREFIX/lib/python3.11/site-packages/ChemEM/<mod>.so
  # Target dylibs:    $PREFIX/lib/
  # -> four levels up from the .so lands at $PREFIX/lib.
  set(CMAKE_INSTALL_RPATH "@loader_path/../../../..")

  # Stripping a signed Mach-O also invalidates its signature. Let
  # conda-build handle any binary post-processing.
  set(CMAKE_INSTALL_DO_STRIP OFF)
endif()

# ---- Dependencies (no RDKit find_package!) ----
find_package(Python REQUIRED COMPONENTS Development.Module Interpreter NumPy)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP QUIET COMPONENTS CXX)

# Optional: Boost headers if present
find_package(Boost CONFIG QUIET)
if(NOT Boost_FOUND)
  # try headers-only detection through include dir probe (as in your original)
  foreach(_bp "${CONDA_HOST_PREFIX}/include" "${CONDA_HOST_PREFIX}/Library/include")
    if(EXISTS "${_bp}/boost/config.hpp")
      set(BOOST_INCLUDE_DIR "${_bp}")
      message(STATUS "Found Boost headers in: ${_bp}")
      break()
    endif()
  endforeach()
endif()

# ---- Platform-specific lib/name pieces ----
if(WIN32)
  set(LIB_PREFIX "")
  set(LIB_SUFFIX ".lib")
  set(SHARED_LIB_SUFFIX ".dll")
elseif(APPLE)
  set(LIB_PREFIX "lib")
  set(LIB_SUFFIX ".dylib")
  set(SHARED_LIB_SUFFIX ".dylib")
else() # Linux
  set(LIB_PREFIX "lib")
  set(LIB_SUFFIX ".so")
  set(SHARED_LIB_SUFFIX ".so")
endif()

# ---- RDKit include & lib dirs (manual) ----
set(RDKIT_INCLUDE_DIRS
  ${CONDA_HOST_PREFIX}/include
  ${CONDA_HOST_PREFIX}/include/rdkit
  ${Python_NumPy_INCLUDE_DIRS}   # ensure NumPy headers are visible
)
if(DEFINED BOOST_INCLUDE_DIR)
  list(APPEND RDKIT_INCLUDE_DIRS "${BOOST_INCLUDE_DIR}")
endif()

if(WIN32)
  set(RDKIT_LIBRARY_DIRS
    ${CONDA_HOST_PREFIX}/lib
    ${CONDA_HOST_PREFIX}/Library/lib
    ${CONDA_HOST_PREFIX}/Library/bin
  )
else()
  set(RDKIT_LIBRARY_DIRS ${CONDA_HOST_PREFIX}/lib)
endif()

# ---- Manual RDKit library resolution (your method) ----
set(RDKIT_LIBRARIES)
set(RDKIT_LIB_NAMES
  GraphMol SmilesParse SubstructMatch FileParsers
  DistGeomHelpers DistGeometry ForceField MolAlign
)

foreach(_lib ${RDKIT_LIB_NAMES})
  set(RDKIT_${_lib}_LIBRARY RDKIT_${_lib}_LIBRARY-NOTFOUND)

  # Try different naming conventions
  set(_lib_names
    RDKit${_lib}
    ${_lib}
    libRDKit${_lib}
    lib${_lib}
  )

  foreach(_lib_name ${_lib_names})
    foreach(_lib_dir ${RDKIT_LIBRARY_DIRS})
      # Try different extensions (shared first, then static)
      foreach(_ext ${LIB_SUFFIX} ${SHARED_LIB_SUFFIX} .a)
        set(_full_path "${_lib_dir}/${LIB_PREFIX}${_lib_name}${_ext}")
        if(EXISTS "${_full_path}")
          set(RDKIT_${_lib}_LIBRARY "${_full_path}")
          break()
        endif()
      endforeach()
      if(RDKIT_${_lib}_LIBRARY)
        break()
      endif()
    endforeach()
    if(RDKIT_${_lib}_LIBRARY)
      break()
    endif()
  endforeach()

  if(RDKIT_${_lib}_LIBRARY)
    list(APPEND RDKIT_LIBRARIES "${RDKIT_${_lib}_LIBRARY}")
    message(STATUS "Found RDKit library: ${RDKIT_${_lib}_LIBRARY}")
  else()
    message(WARNING "Could not find RDKit library: ${_lib}")
  endif()
endforeach()

if(NOT RDKIT_LIBRARIES)
  message(FATAL_ERROR "No RDKit libraries found in ${RDKIT_LIBRARY_DIRS}")
endif()

# ---- Common compile flags ----
if(MSVC)
  add_compile_options(/W3)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra)
endif()

# ==================== modules ====================

# ---- docking ----
pybind11_add_module(docking
  ${PROJ_DOCK_DIR}/bindings.cpp
  ${PROJ_DOCK_DIR}/PreComputedData.cpp
  ${PROJ_DOCK_DIR}/Scorers.cpp
  ${PROJ_DOCK_DIR}/ScoringFunctions.cpp
  ${PROJ_DOCK_DIR}/SearchFunctions.cpp 
  ${PROJ_DOCK_DIR}/nealderMead.cpp 
  
)

target_include_directories(docking PRIVATE
  ${PROJ_SRC_DIR}
  ${PROJ_DOCK_DIR}
  ${PROJ_DOCK_DIR}/LBFGSpp/include
  ${EIGEN3_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
  ${RDKIT_INCLUDE_DIRS}
)
target_link_libraries(docking PRIVATE
  ${RDKIT_LIBRARIES}
  Eigen3::Eigen
  $<$<BOOL:${Boost_FOUND}>:Boost::boost>
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
set_target_properties(docking PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
  OUTPUT_NAME "docking"
)
install(TARGETS docking
  LIBRARY DESTINATION ChemEM
  RUNTIME DESTINATION ChemEM
  ARCHIVE DESTINATION ChemEM
)

# ---- docking_v2 (experimental, sampling-efficient engine; fully isolated) ----
# Separate .so with its own PYBIND11_MODULE(docking_v2). Building this NEVER
# recompiles the baseline `docking` target above. Optional: skipped cleanly if
# the source tree is absent, so the baseline build is unaffected either way.
if(EXISTS ${PROJ_DOCK_V2_DIR})
  pybind11_add_module(docking_v2
    ${PROJ_DOCK_V2_DIR}/bindings.cpp
    ${PROJ_DOCK_V2_DIR}/PreComputedData.cpp
    ${PROJ_DOCK_V2_DIR}/Scorers.cpp
    ${PROJ_DOCK_V2_DIR}/ScoringFunctions.cpp
    ${PROJ_DOCK_V2_DIR}/SearchFunctions.cpp
    ${PROJ_DOCK_V2_DIR}/nealderMead.cpp
  )
  target_include_directories(docking_v2 PRIVATE
    ${PROJ_SRC_DIR}
    ${PROJ_DOCK_V2_DIR}
    ${PROJ_DOCK_V2_DIR}/LBFGSpp/include
    ${EIGEN3_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${RDKIT_INCLUDE_DIRS}
  )
  target_link_libraries(docking_v2 PRIVATE
    ${RDKIT_LIBRARIES}
    Eigen3::Eigen
    $<$<BOOL:${Boost_FOUND}>:Boost::boost>
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
  )
  set_target_properties(docking_v2 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
    OUTPUT_NAME "docking_v2"
  )
  install(TARGETS docking_v2
    LIBRARY DESTINATION ChemEM
    RUNTIME DESTINATION ChemEM
    ARCHIVE DESTINATION ChemEM
  )
  message(STATUS "docking_v2 experimental engine: ENABLED")
else()
  message(STATUS "docking_v2 experimental engine: source tree absent, skipping")
endif()

# ---- grid_maps ----
pybind11_add_module(grid_maps
  ${PROJ_SRC_DIR}/grid_maps.cpp
)
target_include_directories(grid_maps PRIVATE
  ${PROJ_SRC_DIR}
  ${Python_NumPy_INCLUDE_DIRS}
)
target_link_libraries(grid_maps PRIVATE
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
set_target_properties(grid_maps PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
  OUTPUT_NAME "grid_maps"
)
install(TARGETS grid_maps
  LIBRARY DESTINATION ChemEM
  RUNTIME DESTINATION ChemEM
  ARCHIVE DESTINATION ChemEM
)

# ---- ligand_fitting ----
pybind11_add_module(ligand_fitting
  ${PROJ_SRC_DIR}/ligand_fitting.cpp
)
target_include_directories(ligand_fitting PRIVATE
  ${PROJ_SRC_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
  ${Python_NumPy_INCLUDE_DIRS}
)
target_link_libraries(ligand_fitting PRIVATE
  Eigen3::Eigen
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
set_target_properties(ligand_fitting PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CHEMEM_PACKAGE_DIR}
  OUTPUT_NAME "ligand_fitting"
)
install(TARGETS ligand_fitting
  LIBRARY DESTINATION ChemEM
  RUNTIME DESTINATION ChemEM
  ARCHIVE DESTINATION ChemEM
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ChemEM/
  DESTINATION ChemEM
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.json"
    PATTERN "*.pkl"
    PATTERN "*.txt"
)

# ---- convenience target ----
set(CHEMEM_MODULE_TARGETS docking grid_maps ligand_fitting)
if(TARGET docking_v2)
  list(APPEND CHEMEM_MODULE_TARGETS docking_v2)
endif()
add_custom_target(chemem_modules ALL
  DEPENDS ${CHEMEM_MODULE_TARGETS}
  COMMENT "Building all ChemEM modules"
)

# ---- Summary ----
message(STATUS "=== Build Configuration Summary ===")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "ChemEM package: ${CHEMEM_PACKAGE_DIR}")
message(STATUS "Modules output: ${CHEMEM_PACKAGE_DIR}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "==================================")
