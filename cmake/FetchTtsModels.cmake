cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED DVO_MODEL_DIR)
  set(DVO_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../models")
endif()

file(MAKE_DIRECTORY "${DVO_MODEL_DIR}")
set(DOWNLOAD_DIR "${DVO_MODEL_DIR}/_downloads")
file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")

function(dvo_verify_model_file path label)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Pinned model file is missing: ${label} (${path})")
  endif()
endfunction()

set(KOKORO_ARCHIVE "${DOWNLOAD_DIR}/kokoro-int8-multi-lang-v1_1.tar.bz2")
set(KOKORO_DIR "${DVO_MODEL_DIR}/kokoro-int8-multi-lang-v1_1")
if(NOT EXISTS "${KOKORO_DIR}/model.int8.onnx" OR
   NOT EXISTS "${KOKORO_DIR}/voices.bin" OR
   NOT EXISTS "${KOKORO_DIR}/tokens.txt" OR
   NOT EXISTS "${KOKORO_DIR}/espeak-ng-data")
  file(DOWNLOAD
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2"
    "${KOKORO_ARCHIVE}"
    EXPECTED_HASH "SHA256=259c9c8bc906d6078e4e0570421885dabb68b539c2f970fcbb890a58eeaf3e9c"
    SHOW_PROGRESS TLS_VERIFY ON
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xjf "${KOKORO_ARCHIVE}"
    WORKING_DIRECTORY "${DVO_MODEL_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()

dvo_verify_model_file("${KOKORO_DIR}/model.int8.onnx" "Kokoro model")
dvo_verify_model_file("${KOKORO_DIR}/voices.bin" "Kokoro voices")
dvo_verify_model_file("${KOKORO_DIR}/tokens.txt" "Kokoro tokens")
dvo_verify_model_file("${KOKORO_DIR}/espeak-ng-data" "Kokoro espeak data")

message(STATUS "Kokoro TTS models verified in ${KOKORO_DIR}")
