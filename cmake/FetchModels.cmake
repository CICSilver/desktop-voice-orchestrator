cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED DVO_MODEL_DIR)
  set(DVO_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../models")
endif()

file(MAKE_DIRECTORY "${DVO_MODEL_DIR}")
set(DOWNLOAD_DIR "${DVO_MODEL_DIR}/_downloads")
file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")

set(KWS_ARCHIVE "${DOWNLOAD_DIR}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2")
set(KWS_DIR "${DVO_MODEL_DIR}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
if(NOT EXISTS "${KWS_DIR}/encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx")
  file(DOWNLOAD
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"
    "${KWS_ARCHIVE}"
    EXPECTED_HASH "SHA256=68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
    SHOW_PROGRESS TLS_VERIFY ON
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xjf "${KWS_ARCHIVE}"
    WORKING_DIRECTORY "${DVO_MODEL_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()

set(VAD_MODEL "${DVO_MODEL_DIR}/silero_vad.int8.onnx")
if(NOT EXISTS "${VAD_MODEL}")
  file(DOWNLOAD
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx"
    "${VAD_MODEL}"
    EXPECTED_HASH "SHA256=c36d490aff5ab924ca6c7aeec4d8f6bd3d22db6fa17611b9c5b17eae58ac3a20"
    SHOW_PROGRESS TLS_VERIFY ON
  )
endif()

message(STATUS "Pinned models are ready in ${DVO_MODEL_DIR}")
