# Project-local compatibility overlay for the exact WebRTC port pinned by the
# manifest baseline. Keep the upstream port implementation intact, but inject
# one guarded source rewrite after its Chromium build repository is fetched.

set(DVO_WEBRTC_UPSTREAM_PORT "${VCPKG_ROOT_DIR}/ports/webrtc")
set(DVO_WEBRTC_EXPECTED_PORTFILE_SHA256
    "5e9381069a814bfdc660fead10f98385e4ab1c4e9cdc7f9ee522366426271ec6")

if(NOT EXISTS "${DVO_WEBRTC_UPSTREAM_PORT}/portfile.cmake")
    message(FATAL_ERROR
        "The pinned built-in WebRTC port was not found under '${DVO_WEBRTC_UPSTREAM_PORT}'.")
endif()
file(SHA256 "${DVO_WEBRTC_UPSTREAM_PORT}/portfile.cmake" DVO_WEBRTC_PORTFILE_SHA256)
if(NOT DVO_WEBRTC_PORTFILE_SHA256 STREQUAL DVO_WEBRTC_EXPECTED_PORTFILE_SHA256)
    message(FATAL_ERROR
        "The project overlay expects vcpkg WebRTC 2026-03-17#1 from baseline "
        "f87344cac03158cbf1467264565f1fd36b382a24, but VCPKG_ROOT contains a "
        "different portfile (${DVO_WEBRTC_PORTFILE_SHA256}). Use the pinned "
        "vcpkg checkout or update and re-audit this overlay explicitly.")
endif()

set(DVO_WEBRTC_DELEGATE_PORT "${CURRENT_BUILDTREES_DIR}/dvo-webrtc-upstream-port")
file(MAKE_DIRECTORY "${DVO_WEBRTC_DELEGATE_PORT}")
file(COPY "${DVO_WEBRTC_UPSTREAM_PORT}/" DESTINATION "${DVO_WEBRTC_DELEGATE_PORT}")
file(READ "${DVO_WEBRTC_UPSTREAM_PORT}/portfile.cmake" DVO_WEBRTC_PORTFILE)

set(DVO_WEBRTC_INJECTION_POINT [=[fetch_declared_webrtc_repos("${SOURCE_PATH}")]=])
set(DVO_WEBRTC_SDK_COMPAT_INJECTION [=[

# Chromium's pinned build config assumes the 10.0.26100 SDK, where
# NTDDI_WIN11_GE exists. The port intentionally substitutes the installed SDK
# version, and 10.0.22621 only defines WDK_NTDDI_VERSION/NTDDI_WIN10_NI. Leaving
# the unknown identifier makes NTDDI_VERSION evaluate as zero and Windows SDK
# headers omit FILE_INFO_BY_HANDLE_CLASS before fileapi.h uses it.
if(WEBRTC_TARGET_IS_WINDOWS)
    vcpkg_replace_string(
        "${SOURCE_PATH}/build/config/win/BUILD.gn"
        "NTDDI_VERSION=NTDDI_WIN11_GE"
        "NTDDI_VERSION=WDK_NTDDI_VERSION"
    )
endif()
]=])

string(FIND "${DVO_WEBRTC_PORTFILE}" "${DVO_WEBRTC_INJECTION_POINT}"
       DVO_WEBRTC_INJECTION_POSITION)
if(DVO_WEBRTC_INJECTION_POSITION EQUAL -1)
    message(FATAL_ERROR "The expected WebRTC port injection point was not found.")
endif()
string(REPLACE
    "${DVO_WEBRTC_INJECTION_POINT}"
    "${DVO_WEBRTC_INJECTION_POINT}${DVO_WEBRTC_SDK_COMPAT_INJECTION}"
    DVO_WEBRTC_PORTFILE
    "${DVO_WEBRTC_PORTFILE}")

# Desktop Voice Orchestrator consumes only APM/AEC. SDK 10.0.22621 predates
# IGraphicsCaptureSession6, so do not compile the unrelated Windows Graphics
# Capture backend that requires the 10.0.26100 contracts.
set(DVO_WEBRTC_GN_ARGUMENT_POINT [=[        "rtc_use_h264=false"]=])
string(FIND "${DVO_WEBRTC_PORTFILE}" "${DVO_WEBRTC_GN_ARGUMENT_POINT}"
       DVO_WEBRTC_GN_ARGUMENT_POSITION)
if(DVO_WEBRTC_GN_ARGUMENT_POSITION EQUAL -1)
    message(FATAL_ERROR "The expected WebRTC GN argument injection point was not found.")
endif()
string(REPLACE
    "${DVO_WEBRTC_GN_ARGUMENT_POINT}"
    "${DVO_WEBRTC_GN_ARGUMENT_POINT}\n        \"rtc_enable_win_wgc=false\""
    DVO_WEBRTC_PORTFILE
    "${DVO_WEBRTC_PORTFILE}")
file(WRITE "${DVO_WEBRTC_DELEGATE_PORT}/portfile.cmake" "${DVO_WEBRTC_PORTFILE}")

# vcpkg_from_git resolves relative PATCHES against CURRENT_PORT_DIR rather than
# the lexical directory of an included portfile.
set(DVO_WEBRTC_OVERLAY_PORT_DIR "${CURRENT_PORT_DIR}")
set(CURRENT_PORT_DIR "${DVO_WEBRTC_DELEGATE_PORT}")
include("${DVO_WEBRTC_DELEGATE_PORT}/portfile.cmake")
set(CURRENT_PORT_DIR "${DVO_WEBRTC_OVERLAY_PORT_DIR}")
