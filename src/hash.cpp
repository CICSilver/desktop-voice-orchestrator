#include "dvo/hash.h"

#include <Windows.h>
#include <bcrypt.h>

#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace dvo {
namespace {

void check(NTSTATUS status, const char* operation) {
  if (status != 0) throw std::runtime_error(std::string(operation) + " failed");
}

class AlgorithmHandle {
 public:
  ~AlgorithmHandle() { if (value) BCryptCloseAlgorithmProvider(value, 0); }
  BCRYPT_ALG_HANDLE value{};
};

class HashHandle {
 public:
  ~HashHandle() { if (value) BCryptDestroyHash(value); }
  BCRYPT_HASH_HANDLE value{};
};

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot hash file: " + path.string());

  AlgorithmHandle algorithm;
  check(BCryptOpenAlgorithmProvider(&algorithm.value, BCRYPT_SHA256_ALGORITHM, nullptr, 0),
        "BCryptOpenAlgorithmProvider");
  DWORD object_size{};
  DWORD digest_size{};
  DWORD returned{};
  check(BCryptGetProperty(algorithm.value, BCRYPT_OBJECT_LENGTH,
                          reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size), &returned, 0),
        "BCryptGetProperty(BCRYPT_OBJECT_LENGTH)");
  check(BCryptGetProperty(algorithm.value, BCRYPT_HASH_LENGTH,
                          reinterpret_cast<PUCHAR>(&digest_size), sizeof(digest_size), &returned, 0),
        "BCryptGetProperty(BCRYPT_HASH_LENGTH)");

  std::vector<UCHAR> object(object_size);
  std::vector<UCHAR> digest(digest_size);
  HashHandle hash;
  check(BCryptCreateHash(algorithm.value, &hash.value, object.data(), object_size,
                         nullptr, 0, 0), "BCryptCreateHash");
  std::array<char, 64 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      check(BCryptHashData(hash.value, reinterpret_cast<PUCHAR>(buffer.data()),
                           static_cast<ULONG>(count), 0), "BCryptHashData");
    }
  }
  if (!input.eof()) throw std::runtime_error("failed while hashing file: " + path.string());
  check(BCryptFinishHash(hash.value, digest.data(), digest_size, 0), "BCryptFinishHash");

  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (const auto byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

}  // namespace dvo
