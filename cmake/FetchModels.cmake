cmake_minimum_required(VERSION 3.28)

if(NOT DEFINED DVO_MODEL_DIR)
  set(DVO_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../models")
endif()

file(MAKE_DIRECTORY "${DVO_MODEL_DIR}")
set(DOWNLOAD_DIR "${DVO_MODEL_DIR}/_downloads")
file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")

function(dvo_verify_model_file path sha256 label)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Pinned model file is missing: ${label} (${path})")
  endif()
  file(SHA256 "${path}" actual_sha256)
  if(NOT "${actual_sha256}" STREQUAL "${sha256}")
    message(FATAL_ERROR
      "Pinned model hash mismatch: ${label}; expected ${sha256}, got ${actual_sha256}")
  endif()
endfunction()

set(KWS_ARCHIVE "${DOWNLOAD_DIR}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2")
set(KWS_DIR "${DVO_MODEL_DIR}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
if(NOT EXISTS "${KWS_DIR}/encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx" OR
   NOT EXISTS "${KWS_DIR}/decoder-epoch-13-avg-2-chunk-16-left-64.onnx" OR
   NOT EXISTS "${KWS_DIR}/joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx" OR
   NOT EXISTS "${KWS_DIR}/tokens.txt" OR
   NOT EXISTS "${KWS_DIR}/en.phone")
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
dvo_verify_model_file(
  "${KWS_DIR}/encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
  "408bbd740838c42d5bf6d1c5b80b3c88b616c7860b92d980328b5b068c76ae48"
  "KWS encoder")
dvo_verify_model_file(
  "${KWS_DIR}/decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
  "63a22dd60f40fff082ac3e09afa507f6787da36df76ded2fbe145fa233e22c21"
  "KWS decoder")
dvo_verify_model_file(
  "${KWS_DIR}/joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
  "190d4067b4cc20b72a42a1916e69d92052000fb7051a427ebb1bc72a69207dc1"
  "KWS joiner")
dvo_verify_model_file(
  "${KWS_DIR}/tokens.txt"
  "2d3f32311f9b692b964da3c90e830258d3e78e013cb0c992dbfb15cd5a1a71b0"
  "KWS tokens")
dvo_verify_model_file(
  "${KWS_DIR}/en.phone"
  "f7000ec3a90544c0c7c16090d8951779c2b322e14dad5006290f498567d439ea"
  "KWS lexicon")

set(VAD_MODEL "${DVO_MODEL_DIR}/silero_vad.int8.onnx")
if(NOT EXISTS "${VAD_MODEL}")
  file(DOWNLOAD
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx"
    "${VAD_MODEL}"
    EXPECTED_HASH "SHA256=c36d490aff5ab924ca6c7aeec4d8f6bd3d22db6fa17611b9c5b17eae58ac3a20"
    SHOW_PROGRESS TLS_VERIFY ON
  )
endif()
dvo_verify_model_file(
  "${VAD_MODEL}"
  "c36d490aff5ab924ca6c7aeec4d8f6bd3d22db6fa17611b9c5b17eae58ac3a20"
  "Silero VAD")

# The generic FunASR ONNX export in csukuangfj/streaming-paraformer-zh does
# not carry the metadata required by sherpa-onnx's OnlineParaformer loader.
# Keep the compatible re-export in a distinct directory so the two encoders
# cannot be selected accidentally.
set(ASR_DIR "${DVO_MODEL_DIR}/sherpa-onnx-streaming-paraformer-zh")
file(MAKE_DIRECTORY "${ASR_DIR}")
set(ASR_BASE_URL
  "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-zh/resolve/2a7f71bb58885c1b522ed4e683abd397355d9fc4")

function(dvo_fetch_asr_file filename sha256)
  if(NOT EXISTS "${ASR_DIR}/${filename}")
    file(DOWNLOAD
      "${ASR_BASE_URL}/${filename}"
      "${ASR_DIR}/${filename}"
      EXPECTED_HASH "SHA256=${sha256}"
      SHOW_PROGRESS TLS_VERIFY ON
    )
  else()
    dvo_verify_model_file("${ASR_DIR}/${filename}" "${sha256}" "ASR ${filename}")
  endif()
endfunction()

dvo_fetch_asr_file("encoder.int8.onnx"
  "81a70226a8934e6ed92aa1d4fc486b428b5398e2f2619ed4897b7294cab90e9a")
dvo_fetch_asr_file("decoder.int8.onnx"
  "f3cca9f77bb9d93c8fcbfb63ae617b6b1ee96818df3aa3b151c40658fe38594f")
dvo_fetch_asr_file("tokens.txt"
  "59aba8873a2ed1e122c25fee421e25f283b63290efbde85c1f01a853d83cb6e6")

# Non-streaming final decoders (see [asr_final] / [asr_fallback] in manifest).
function(dvo_fetch_release_model name archive_sha256 model_sha256 tokens_sha256)
  set(dir "${DVO_MODEL_DIR}/${name}")
  if(NOT EXISTS "${dir}/model.int8.onnx" OR NOT EXISTS "${dir}/tokens.txt")
    set(archive "${DOWNLOAD_DIR}/${name}.tar.bz2")
    file(DOWNLOAD
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${name}.tar.bz2"
      "${archive}"
      EXPECTED_HASH "SHA256=${archive_sha256}"
      SHOW_PROGRESS TLS_VERIFY ON
    )
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xjf "${archive}"
      WORKING_DIRECTORY "${DVO_MODEL_DIR}"
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
  dvo_verify_model_file("${dir}/model.int8.onnx" "${model_sha256}" "${name} model")
  dvo_verify_model_file("${dir}/tokens.txt" "${tokens_sha256}" "${name} tokens")
endfunction()

dvo_fetch_release_model("sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17"
  "257936ea9a64cbe33200274e6367fc26d373ff6ca58b996b15108ffd6b9f6148"
  "9dc6e72aa8bc6f5966cf2857a0ce3a425b1d72e91500e147d66329f407c017a1"
  "2db4bb25d046e8849e336c9465e248e1694914d996c58c71bdeca18cfb722992")
dvo_fetch_release_model("sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
  "f3ad1814fea34c407eab0cc3df6f6b625419ac9a60d8aebd8efe772a8e85ef67"
  "e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07"
  "6fed8c6c248516f38e7faa19404b57413e8ce259f1cbc1fa4aebc86eac32fdfd")

message(STATUS "Pinned models are ready in ${DVO_MODEL_DIR}")
