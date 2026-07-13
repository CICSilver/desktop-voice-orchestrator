#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include "dvo/audio_types.h"

namespace dvo {

class FloatWavWriter {
 public:
  FloatWavWriter() = default;
  ~FloatWavWriter();
  FloatWavWriter(const FloatWavWriter&) = delete;
  FloatWavWriter& operator=(const FloatWavWriter&) = delete;

  void open(const std::filesystem::path& path, AudioFormat format);
  void write(std::span<const float> interleaved_samples);
  void close();
  [[nodiscard]] bool is_open() const { return stream_.is_open(); }
  [[nodiscard]] std::uint64_t frames_written() const { return frames_written_; }
  [[nodiscard]] AudioFormat format() const { return format_; }

 private:
  void write_header();
  std::fstream stream_;
  AudioFormat format_{};
  std::uint64_t frames_written_{};
};

class FloatWavReader {
 public:
  explicit FloatWavReader(const std::filesystem::path& path);
  [[nodiscard]] AudioFormat format() const { return format_; }
  [[nodiscard]] std::uint64_t frame_count() const { return frame_count_; }
  [[nodiscard]] std::vector<float> read_frames(std::uint64_t offset, std::uint64_t count);

 private:
  std::ifstream stream_;
  AudioFormat format_{};
  std::uint64_t data_offset_{};
  std::uint64_t frame_count_{};
};

}  // namespace dvo
