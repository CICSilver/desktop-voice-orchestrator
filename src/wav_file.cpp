#include "dvo/wav_file.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace dvo {
namespace {

template <typename T>
void write_value(std::ostream& stream, T value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
T read_value(std::istream& stream) {
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error("unexpected end of WAV file");
  return value;
}

}  // namespace

FloatWavWriter::~FloatWavWriter() { close(); }

void FloatWavWriter::open(const std::filesystem::path& path, AudioFormat format) {
  close();
  if (format.sample_rate == 0 || format.channels == 0) throw std::invalid_argument("invalid WAV format");
  std::filesystem::create_directories(path.parent_path());
  stream_.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  if (!stream_) throw std::runtime_error("cannot create WAV: " + path.string());
  format_ = format;
  frames_written_ = 0;
  write_header();
}

void FloatWavWriter::write(std::span<const float> samples) {
  if (!stream_) throw std::runtime_error("WAV writer is not open");
  if (samples.size() % format_.channels != 0) throw std::invalid_argument("unaligned interleaved WAV samples");
  stream_.seekp(0, std::ios::end);
  stream_.write(reinterpret_cast<const char*>(samples.data()),
                static_cast<std::streamsize>(samples.size_bytes()));
  if (!stream_) throw std::runtime_error("failed to write WAV samples");
  frames_written_ += samples.size() / format_.channels;
}

void FloatWavWriter::close() {
  if (!stream_.is_open()) return;
  write_header();
  stream_.close();
}

void FloatWavWriter::write_header() {
  const auto data_bytes_64 = frames_written_ * format_.channels * sizeof(float);
  if (data_bytes_64 > 0xffffffffULL - 36) throw std::runtime_error("WAV exceeds RIFF 32-bit size limit");
  const auto data_bytes = static_cast<std::uint32_t>(data_bytes_64);
  const std::uint16_t format_tag = 3;  // IEEE float
  const std::uint16_t bits_per_sample = 32;
  const std::uint16_t block_align = static_cast<std::uint16_t>(format_.channels * sizeof(float));
  const std::uint32_t byte_rate = format_.sample_rate * block_align;
  const std::uint32_t fmt_size = 16;
  const std::uint32_t riff_size = 36 + data_bytes;

  stream_.seekp(0, std::ios::beg);
  stream_.write("RIFF", 4);
  write_value(stream_, riff_size);
  stream_.write("WAVEfmt ", 8);
  write_value(stream_, fmt_size);
  write_value(stream_, format_tag);
  write_value(stream_, format_.channels);
  write_value(stream_, format_.sample_rate);
  write_value(stream_, byte_rate);
  write_value(stream_, block_align);
  write_value(stream_, bits_per_sample);
  stream_.write("data", 4);
  write_value(stream_, data_bytes);
  stream_.flush();
}

FloatWavReader::FloatWavReader(const std::filesystem::path& path) {
  stream_.open(path, std::ios::binary);
  if (!stream_) throw std::runtime_error("cannot open WAV: " + path.string());
  std::array<char, 4> id{};
  stream_.read(id.data(), 4);
  if (std::string_view(id.data(), 4) != "RIFF") throw std::runtime_error("not a RIFF WAV file");
  (void)read_value<std::uint32_t>(stream_);
  stream_.read(id.data(), 4);
  if (std::string_view(id.data(), 4) != "WAVE") throw std::runtime_error("not a WAVE file");

  std::uint16_t tag{};
  std::uint16_t bits{};
  std::uint32_t data_size{};
  while (stream_ && data_offset_ == 0) {
    stream_.read(id.data(), 4);
    const auto size = read_value<std::uint32_t>(stream_);
    const auto next = static_cast<std::uint64_t>(stream_.tellg()) + size + (size & 1U);
    if (std::string_view(id.data(), 4) == "fmt ") {
      tag = read_value<std::uint16_t>(stream_);
      format_.channels = read_value<std::uint16_t>(stream_);
      format_.sample_rate = read_value<std::uint32_t>(stream_);
      (void)read_value<std::uint32_t>(stream_);
      (void)read_value<std::uint16_t>(stream_);
      bits = read_value<std::uint16_t>(stream_);
    } else if (std::string_view(id.data(), 4) == "data") {
      data_offset_ = static_cast<std::uint64_t>(stream_.tellg());
      data_size = size;
    }
    stream_.seekg(static_cast<std::streamoff>(next), std::ios::beg);
  }
  if (tag != 3 || bits != 32 || format_.channels == 0 || data_offset_ == 0) {
    throw std::runtime_error("only 32-bit float WAV is supported for session replay");
  }
  frame_count_ = data_size / (format_.channels * sizeof(float));
}

std::vector<float> FloatWavReader::read_frames(std::uint64_t offset, std::uint64_t count) {
  if (offset > frame_count_) throw std::out_of_range("WAV frame offset is out of range");
  count = std::min(count, frame_count_ - offset);
  std::vector<float> result(static_cast<std::size_t>(count * format_.channels));
  const auto byte_offset = data_offset_ + offset * format_.channels * sizeof(float);
  stream_.clear();
  stream_.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
  stream_.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(result.size() * sizeof(float)));
  if (!stream_ && !result.empty()) throw std::runtime_error("failed to read WAV frames");
  return result;
}

}  // namespace dvo
