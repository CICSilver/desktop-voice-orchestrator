include(FetchContent)

function(dvo_configure_sherpa target)
  set(SHERPA_ARCHIVE "sherpa-onnx-v1.13.2-win-x64-shared-MD-Release-no-tts-lib.tar.bz2")
  set(SHERPA_ROOT_NAME "sherpa-onnx-v1.13.2-win-x64-shared-MD-Release-no-tts-lib")

  FetchContent_Declare(
    sherpa_onnx_binary
    URL "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.2/${SHERPA_ARCHIVE}"
    URL_HASH "SHA256=6ddd96bd875349b0580d0bbfd70fb08694ad1b7ef9f02966005aec5c7824b700"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(sherpa_onnx_binary)

  set(SHERPA_HEADER_DIR "${FETCHCONTENT_BASE_DIR}/sherpa-headers/include")
  set(SHERPA_HEADER "${SHERPA_HEADER_DIR}/sherpa-onnx/c-api/c-api.h")
  if(NOT EXISTS "${SHERPA_HEADER}")
    file(MAKE_DIRECTORY "${SHERPA_HEADER_DIR}/sherpa-onnx/c-api")
    file(DOWNLOAD
      "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.2/sherpa-onnx/c-api/c-api.h"
      "${SHERPA_HEADER}"
      EXPECTED_HASH "SHA256=437b1279047877167d8fadc74a60d47f3df514d703fdac1c1b6851da9bc2fdb4"
      TLS_VERIFY ON
    )
  endif()

  set(SHERPA_LIB_DIR "${sherpa_onnx_binary_SOURCE_DIR}/lib")
  add_library(sherpa_onnx_c_api SHARED IMPORTED GLOBAL)
  set_target_properties(sherpa_onnx_c_api PROPERTIES
    IMPORTED_IMPLIB "${SHERPA_LIB_DIR}/sherpa-onnx-c-api.lib"
    IMPORTED_LOCATION "${SHERPA_LIB_DIR}/sherpa-onnx-c-api.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${SHERPA_HEADER_DIR}"
  )
  target_link_libraries(${target} PUBLIC sherpa_onnx_c_api)
  target_compile_definitions(${target} PUBLIC DVO_HAS_SHERPA=1)

  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SHERPA_LIB_DIR}/sherpa-onnx-c-api.dll"
            "${SHERPA_LIB_DIR}/onnxruntime.dll"
            "${SHERPA_LIB_DIR}/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:${target}>"
  )
endfunction()
