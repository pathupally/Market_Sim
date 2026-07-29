#include "marketforge/core/mapped_file.hpp"

#include <cerrno>
#include <cstdint>
#include <limits>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace marketforge {

Result<MappedFile>
MappedFile::open_read_only(const std::filesystem::path& path) noexcept {
  const int file_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (file_descriptor < 0) {
    return Result<MappedFile>::failure(Status::failure(
        ErrorCode::io_error, static_cast<std::uint32_t>(errno)));
  }

  struct stat file_status {};
  if (::fstat(file_descriptor, &file_status) != 0 ||
      !S_ISREG(file_status.st_mode) || file_status.st_size < 0) {
    const int saved_errno = errno;
    static_cast<void>(::close(file_descriptor));
    return Result<MappedFile>::failure(Status::failure(
        ErrorCode::io_error, static_cast<std::uint32_t>(saved_errno)));
  }

  const auto unsigned_size = static_cast<std::uint64_t>(file_status.st_size);
  if (unsigned_size > std::numeric_limits<std::size_t>::max()) {
    static_cast<void>(::close(file_descriptor));
    return Result<MappedFile>::failure(
        Status::failure(ErrorCode::resource_limit));
  }
  const auto size = static_cast<std::size_t>(unsigned_size);
  if (size == 0) {
    static_cast<void>(::close(file_descriptor));
    return Result<MappedFile>::success(MappedFile{nullptr, 0});
  }

  void* const mapping =
      ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
  const int saved_errno = errno;
  static_cast<void>(::close(file_descriptor));
  if (mapping == MAP_FAILED) {
    return Result<MappedFile>::failure(Status::failure(
        ErrorCode::io_error, static_cast<std::uint32_t>(saved_errno)));
  }

  return Result<MappedFile>::success(
      MappedFile{static_cast<const std::byte*>(mapping), size});
}

MappedFile::~MappedFile() { reset(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void MappedFile::reset() noexcept {
  if (data_ != nullptr) {
    static_cast<void>(::munmap(const_cast<std::byte*>(data_), size_));
  }
  data_ = nullptr;
  size_ = 0;
}

} // namespace marketforge
