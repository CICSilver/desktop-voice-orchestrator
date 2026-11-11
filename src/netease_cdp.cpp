#include "dvo/netease_cdp.h"

#include <boost/asio.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast.hpp>

#include <exception>

namespace dvo {
namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = asio::ip::tcp;

// Finds the client's dva store helper (getStore/getDispatch) through webpack's
// module cache. Module ids change between client builds, so it is looked up
// by shape, and the result is cached on the page.
constexpr std::string_view kStorePrelude = R"JS(
const dvoStore = () => {
  const cached = window.__dvoStoreTool;
  if (cached && cached.inited) return cached;
  if (!window.__dvoRequire) {
    webpackJsonp.push([['dvo-require'], {'dvo-require': (module, exports, require) => {
      window.__dvoRequire = require;
    }}, [['dvo-require']]]);
  }
  for (const m of Object.values(window.__dvoRequire.c)) {
    const e = m && m.exports;
    if (!e) continue;
    for (const v of [e, e.a, e.default]) {
      if (v && typeof v.getDispatch === 'function' && typeof v.getStore === 'function' && v.inited) {
        window.__dvoStoreTool = v;
        return v;
      }
    }
  }
  throw new Error('NetEase store not found');
};
const dvoSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
)JS";

}  // namespace

std::optional<std::string> netease_cdp_page_path(const nlohmann::json& targets) {
  if (!targets.is_array()) return std::nullopt;
  for (const auto& target : targets) {
    if (!target.is_object() || target.value("type", "") != "page") continue;
    if (!target.value("url", "").starts_with("orpheus://orpheus/pub/app.html")) continue;
    const auto url = target.value("webSocketDebuggerUrl", "");
    const auto path = url.find("/devtools/");
    if (path != std::string::npos) return url.substr(path);
  }
  return std::nullopt;
}

nlohmann::json netease_cdp_result(const nlohmann::json& reply) {
  if (const auto error = reply.find("error"); error != reply.end()) {
    throw NeteaseCdpError(NeteaseCdpError::Kind::protocol, "DevTools error: " + error->dump());
  }
  const auto result = reply.find("result");
  if (result == reply.end() || !result->is_object()) {
    throw NeteaseCdpError(NeteaseCdpError::Kind::protocol, "DevTools reply has no result");
  }
  if (const auto details = result->find("exceptionDetails"); details != result->end()) {
    std::string description = details->value("text", "exception");
    if (const auto exception = details->find("exception");
        exception != details->end() && exception->is_object()) {
      description = exception->value("description", description);
    }
    throw NeteaseCdpError(NeteaseCdpError::Kind::script, "NetEase page script failed: " + description);
  }
  const auto value = result->find("result");
  if (value == result->end() || !value->is_object()) return nullptr;
  return value->value("value", nlohmann::json(nullptr));
}

nlohmann::json netease_cdp_evaluate(std::uint16_t port, std::string_view expression,
                                    std::chrono::steady_clock::time_point deadline,
                                    std::stop_token stop) {
  const tcp::endpoint endpoint(asio::ip::address_v4::loopback(), port);
  const auto host = "127.0.0.1:" + std::to_string(port);
  const auto request_text =
      nlohmann::json{{"id", 1},
                     {"method", "Runtime.evaluate"},
                     {"params",
                      {{"expression", std::string(expression)},
                       {"awaitPromise", true},
                       {"returnByValue", true}}}}
          .dump();
  // Declared before the io_context so that they outlive the coroutine frame,
  // which the io_context destroys if it has to be abandoned.
  nlohmann::json reply;
  std::exception_ptr failure;
  bool finished{};
  asio::io_context io;
  asio::co_spawn(
      io,
      [&]() -> asio::awaitable<void> {
        std::string path;
        {
          beast::tcp_stream stream(io);
          co_await stream.async_connect(endpoint, asio::use_awaitable);
          http::request<http::empty_body> request{http::verb::get, "/json", 11};
          request.set(http::field::host, host);
          co_await http::async_write(stream, request, asio::use_awaitable);
          beast::flat_buffer buffer;
          http::response<http::string_body> response;
          co_await http::async_read(stream, buffer, response, asio::use_awaitable);
          const auto page = netease_cdp_page_path(
              nlohmann::json::parse(response.body(), nullptr, false));
          if (!page) {
            throw NeteaseCdpError(NeteaseCdpError::Kind::starting,
                                  "the NetEase client's main page is not open yet");
          }
          path = *page;
        }
        websocket::stream<beast::tcp_stream> ws(io);
        co_await beast::get_lowest_layer(ws).async_connect(endpoint, asio::use_awaitable);
        co_await ws.async_handshake(host, path, asio::use_awaitable);
        co_await ws.async_write(asio::buffer(request_text), asio::use_awaitable);
        for (;;) {
          beast::flat_buffer buffer;
          co_await ws.async_read(buffer, asio::use_awaitable);
          auto message =
              nlohmann::json::parse(beast::buffers_to_string(buffer.data()), nullptr, false);
          if (message.is_object() && message.value("id", 0) == 1) {
            reply = std::move(message);
            break;
          }
        }
        finished = true;
        boost::system::error_code ignored;
        co_await ws.async_close(websocket::close_code::normal,
                                asio::redirect_error(asio::use_awaitable, ignored));
      },
      [&](std::exception_ptr error) { failure = error; });

  while (!io.stopped()) {
    if (stop.stop_requested()) {
      throw NeteaseCdpError(NeteaseCdpError::Kind::cancelled, "NetEase control was cancelled");
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw NeteaseCdpError(NeteaseCdpError::Kind::timeout, "NetEase control timed out");
    }
    io.run_for(std::chrono::milliseconds(20));
  }
  if (failure && !finished) {
    try {
      std::rethrow_exception(failure);
    } catch (const NeteaseCdpError&) {
      throw;
    } catch (const boost::system::system_error& error) {
      if (error.code() == asio::error::connection_refused) {
        throw NeteaseCdpError(NeteaseCdpError::Kind::unavailable,
                              "no NetEase DevTools channel on " + host);
      }
      throw NeteaseCdpError(NeteaseCdpError::Kind::protocol, error.what());
    }
  }
  return netease_cdp_result(reply);
}

std::string netease_ready_script() {
  return std::string("(async () => {") + std::string(kStorePrelude) + R"JS(
  const s = dvoStore().getStore();
  return {ok: true, mode: s.playing.playingMode};
})())JS";
}

std::string netease_play_daily_script() {
  return std::string("(async () => {") + std::string(kStorePrelude) + R"JS(
  const tool = dvoStore();
  const state = () => {
    const s = tool.getStore();
    const current = s.playing.curPlaying || {};
    return {scene: current.scene || '', id: String(current.resourceId || ''),
            name: (current.track || {}).name || '', playing: s.playing.playingState,
            queue: (s.playingList.curPlayingList || []).length};
  };
  const before = state();
  await tool.getDispatch()({type: 'async:action/doAction', payload: {
    actionId: 'playAll',
    data: {resource: {}, resourceType: 'dailyRecommend',
           from: {sourceData: {id: ''}, scene: 'dailyRecommend', trialMode: 'dailyRecommend',
                  pldScene: 'musicDesktop'}}}});
  for (let i = 0; i < 30; i++) {
    const now = state();
    if (now.scene === 'dailyRecommend' && now.queue > 0 &&
        (now.id !== before.id || before.scene === 'dailyRecommend')) {
      return {ok: true, ...now};
    }
    await dvoSleep(100);
  }
  return {ok: false, ...state()};
})())JS";
}

std::string netease_play_mode_script(std::string_view mode) {
  return std::string("(async () => {") + std::string(kStorePrelude) +
         "const target = " + nlohmann::json(std::string(mode)).dump() + ";" + R"JS(
  const tool = dvoStore();
  const mode = () => tool.getStore().playing.playingMode;
  const before = mode();
  if (before === target) return {ok: true, changed: false, mode: before};
  await tool.getDispatch()({type: 'playing/switchPlayingMode',
                            payload: {playingMode: target, triggerScene: 'sysTray',
                                      HeartBeatFlage: false}});
  for (let i = 0; i < 20 && mode() !== target; i++) await dvoSleep(50);
  return {ok: mode() === target, changed: true, before, mode: mode()};
})())JS";
}

std::string netease_search_script(MediaSearchType type, std::string_view keyword,
                                  int limit) {
  const int search_type = type == MediaSearchType::playlist ? 1000 : 1;
  return std::string("(async () => {") +
         "const kw = " + nlohmann::json(std::string(keyword)).dump() + ";" +
         "const st = " + std::to_string(search_type) + ";" +
         "const lim = " + std::to_string(limit > 0 ? limit : 20) + ";" + R"JS(
  const urls = [
    'https://music.163.com/api/cloudsearch/pc?s=' + encodeURIComponent(kw) + '&type=' + st + '&offset=0&limit=' + lim,
    'https://music.163.com/api/search/get/web?csrf_token=&s=' + encodeURIComponent(kw) + '&type=' + st + '&offset=0&limit=' + lim
  ];
  for (const url of urls) {
    try {
      const resp = await fetch(url, {method: 'POST', credentials: 'include'});
      if (!resp.ok) continue;
      const body = await resp.json();
      if (body && typeof body === 'object' && body.result) {
        return {ok: true, body};
      }
    } catch (_) {}
  }
  return {ok: false};
})())JS";
}

std::string netease_play_item_script(MediaSearchType type,
                                     std::string_view item_id) {
  const bool is_playlist = type == MediaSearchType::playlist;
  return std::string("(async () => {") + std::string(kStorePrelude) +
         "const strId = " + nlohmann::json(std::string(item_id)).dump() + ";" +
         "const isPlaylist = " + (is_playlist ? "true" : "false") + ";" + R"JS(
  const numId = Number(strId);
  const tool = dvoStore();
  const dispatch = tool.getDispatch();
  const state = () => {
    const s = tool.getStore();
    const current = (s.playing && s.playing.curPlaying) || {};
    const track = current.track || current || {};
    return {
      scene: current.scene || '',
      resourceId: String(current.resourceId || ''),
      trackId: String(track.id || current.id || ''),
      name: track.name || '',
      playing: s.playing ? s.playing.playingState : 0,
      queue: ((s.playingList && s.playingList.curPlayingList) || []).length
    };
  };
  if (isPlaylist) {
    await dispatch({
      type: 'async:action/doAction',
      payload: {
        actionId: 'playAll',
        data: {
          resource: {id: numId},
          resourceType: 'playlist',
          from: {sourceData: {id: strId}, scene: 'playlist', pldScene: 'musicDesktop'}
        }
      }
    });
    for (let i = 0; i < 30; i++) {
      const now = state();
      if (now.resourceId === strId && now.queue > 0) {
        return {ok: true, ...now};
      }
      await dvoSleep(100);
    }
    return {ok: false, ...state()};
  }

  let trackObj = null;
  try {
    const detailUrl = 'https://music.163.com/api/v3/song/detail?id=' + encodeURIComponent(strId) +
                      '&c=' + encodeURIComponent(JSON.stringify([{id: numId}]));
    const resp = await fetch(detailUrl, {method: 'POST', credentials: 'include'});
    if (resp.ok) {
      const data = await resp.json();
      if (data && Array.isArray(data.songs) && data.songs.length > 0) {
        trackObj = data.songs[0];
      }
    }
  } catch (_) {}

  try {
    await dispatch({
      type: 'async:action/doAction',
      payload: {
        actionId: 'play',
        data: {
          resource: trackObj || {id: numId},
          resourceType: 'song',
          from: {sourceData: {id: strId}, scene: 'search', pldScene: 'musicDesktop'}
        }
      }
    });
  } catch (_) {}

  for (let i = 0; i < 15; i++) {
    const now = state();
    if (now.trackId === strId) return {ok: true, ...now};
    await dvoSleep(100);
  }

  if (trackObj) {
    for (const actionType of ['playing/play', 'async:playing/play']) {
      try {
        await dispatch({type: actionType, payload: {track: trackObj, id: numId}});
      } catch (_) {}
      for (let i = 0; i < 10; i++) {
        const now = state();
        if (now.trackId === strId) return {ok: true, ...now};
        await dvoSleep(100);
      }
    }
  }

  return {ok: false, ...state()};
})())JS";
}

}  // namespace dvo
