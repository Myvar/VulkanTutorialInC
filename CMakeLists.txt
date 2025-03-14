cmake_minimum_required(VERSION 3.26)

project(vkt C CXX)

include(FetchContent)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS OFF)
set(CXX "/usr/bin/g++")

set(GLSLC "glslc")

function (target_shaders target shaders)
  set(SHADERS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
  file(MAKE_DIRECTORY "${SHADERS_BUILD_DIR}")
  foreach (SHADER_PATH ${SHADERS})
    get_filename_component(SHADER_FILENAME "${SHADER_PATH}" NAME)
    set(SHADER_SPIRV_PATH "${SHADERS_BUILD_DIR}/${SHADER_FILENAME}.spv")
    set(DEPFILE "${SHADER_SPIRV_PATH}.d")
    add_custom_command(
          COMMENT "Building ${SHADER_FILENAME}"
          OUTPUT "${SHADER_SPIRV_PATH}"
          COMMAND ${GLSLC} "${SHADER_PATH}" -o "${SHADER_SPIRV_PATH}" -MD -MF ${DEPFILE} -g
          DEPENDS "${SHADER_PATH}"
          DEPFILE "${DEPFILE}"
        )
    list(APPEND SPIRV_BINARY_FILES ${SHADER_SPIRV_PATH})
  endforeach()

  set(shaders_target_name "${target}_build_shaders")
  add_custom_target(${shaders_target_name}
      DEPENDS ${SPIRV_BINARY_FILES}
    )
  add_dependencies(${target} ${shaders_target_name})
  file(COPY assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ )
endfunction()


file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS
"${CMAKE_CURRENT_LIST_DIR}/src/*.c")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(vkt)

set(SHADERS
    min.vert
    min.frag
)

get_target_property(VKT_SOURCE_DIR vkt SOURCE_DIR)
set(VKT_SHADERS_DIR "${VKT_SOURCE_DIR}/shaders/")
list(TRANSFORM SHADERS PREPEND "${VKT_SHADERS_DIR}")

target_shaders(vkt ${SHADERS})

target_sources(vkt PRIVATE
    ${PROJECT_SOURCES}
)
target_link_libraries(vkt PRIVATE glfw vulkan cglm assimp)

