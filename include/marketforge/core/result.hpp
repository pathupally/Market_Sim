#pragma once

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

#include "marketforge/core/status.hpp"

namespace marketforge {

template <typename T> class [[nodiscard]] Result {
  static_assert(!std::is_same_v<std::remove_cv_t<T>, Status>);

public:
  [[nodiscard]] static Result success(T value) {
    return Result(std::in_place_index<0>, std::move(value));
  }

  [[nodiscard]] static Result failure(Status status) {
    assert(!status.ok());
    return Result(std::in_place_index<1>, status);
  }

  [[nodiscard]] bool has_value() const noexcept {
    return storage_.index() == 0;
  }

  [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }

  [[nodiscard]] T& value() & {
    assert(has_value());
    return std::get<0>(storage_);
  }

  [[nodiscard]] const T& value() const& {
    assert(has_value());
    return std::get<0>(storage_);
  }

  [[nodiscard]] T&& value() && {
    assert(has_value());
    return std::get<0>(std::move(storage_));
  }

  [[nodiscard]] Status status() const noexcept {
    return has_value() ? Status::success() : std::get<1>(storage_);
  }

private:
  template <std::size_t Index, typename Value>
  explicit Result(std::in_place_index_t<Index> index, Value&& value)
      : storage_(index, std::forward<Value>(value)) {}

  std::variant<T, Status> storage_;
};

} // namespace marketforge
