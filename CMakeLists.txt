cmake_minimum_required(VERSION 3.28)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

project(desktop_voice_orchestrator VERSION 0.1.0 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "desktop-voice-orchestrator currently targets Windows only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DVO_ENABLE_SHERPA "Enable the sherpa-onnx KWS/VAD adapter" ON)
option(DVO_BUILD_TESTS "Build unit and integration tests" ON)

add_custom_target(fetch_models
  COMMAND ${CMAKE_COMMAND}
          -DDVO_MODEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/models
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchModels.cmake
  COMMENT "Download and verify pinned KWS/VAD models"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

find_package(Boost REQUIRED COMPONENTS system)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)

add_library(dvo_core STATIC
  src/config.cpp
  src/hash.cpp
  src/ring_buffer.cpp
  src/segmenter.cpp
  src/preprocessor.cpp
  src/inference.cpp
  src/wav_file.cpp
  src/recorder.cpp
  src/replay.cpp
  src/debug_server.cpp
  src/runtime.cpp
  src/platform/windows/wasapi_capture.cpp
)

target_include_directories(dvo_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(dvo_core PUBLIC
  UNICODE
  _UNICODE
  WIN32_LEAN_AND_MEAN
  NOMINMAX
  DVO_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(dvo_core PRIVATE /W4 /permissive- /utf-8 /EHsc)
target_link_libraries(dvo_core
  PUBLIC
    Boost::system
    nlohmann_json::nlohmann_json
    tomlplusplus::tomlplusplus
  PRIVATE
    avrt
    ole32
    propsys
    runtimeobject
    bcrypt
)

if(DVO_ENABLE_SHERPA)
  dvo_configure_sherpa(dvo_core)
else()
  target_compile_definitions(dvo_core PUBLIC DVO_HAS_SHERPA=0)
endif()

add_executable(voice_frontend src/main.cpp)
target_compile_options(voice_frontend PRIVATE /W4 /permissive- /utf-8 /EHsc)
target_link_libraries(voice_frontend PRIVATE dvo_core)
target_link_libraries(voice_frontend PRIVATE shell32)

add_custom_command(TARGET voice_frontend POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/web"
          "$<TARGET_FILE_DIR:voice_frontend>/web"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/config"
          "$<TARGET_FILE_DIR:voice_frontend>/config"
)

include(CTest)
if(BUILD_TESTING AND DVO_BUILD_TESTS)
  find_package(Catch2 3 CONFIG REQUIRED)
  add_executable(dvo_tests
    tests/test_ring_buffer.cpp
    tests/test_segmenter.cpp
    tests/test_config.cpp
    tests/test_recording.cpp
    tests/test_inference.cpp
    tests/test_audio.cpp
  )
  target_compile_options(dvo_tests PRIVATE /W4 /permissive- /utf-8 /EHsc)
  target_link_libraries(dvo_tests PRIVATE dvo_core Catch2::Catch2WithMain)
  include(Catch)
  catch_discover_tests(dvo_tests)
endif()
