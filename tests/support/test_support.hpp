#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace marketforge::test {

using TestFunction = void (*)();

struct TestCase {
  std::string_view name;
  TestFunction function;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

class Registrar {
public:
  Registrar(std::string_view name, TestFunction function) {
    registry().push_back(TestCase{name, function});
  }
};

class Failure final : public std::exception {};

inline void check(const bool condition, const char* expression,
                  const char* file, const int line) {
  if (condition) {
    return;
  }
  std::cerr << file << ':' << line << ": check failed: " << expression << '\n';
  throw Failure{};
}

inline void check_near(const double left, const double right,
                       const double tolerance, const char* left_expression,
                       const char* right_expression, const char* file,
                       const int line) {
  if (std::isfinite(left) && std::isfinite(right) &&
      std::abs(left - right) <= tolerance) {
    return;
  }
  std::cerr << file << ':' << line << ": check failed: " << left_expression
            << " ~= " << right_expression << " (" << left << " vs " << right
            << ", tolerance " << tolerance << ")\n";
  throw Failure{};
}

} // namespace marketforge::test

#define MF_TEST(name)                                                          \
  static void name();                                                          \
  static const ::marketforge::test::Registrar registrar_##name(#name, &name);  \
  static void name()

#define MF_CHECK(expression)                                                   \
  ::marketforge::test::check(static_cast<bool>(expression), #expression,       \
                             __FILE__, __LINE__)

#define MF_CHECK_EQ(lhs, rhs) MF_CHECK((lhs) == (rhs))
#define MF_CHECK_NE(lhs, rhs) MF_CHECK((lhs) != (rhs))

#define MF_CHECK_NEAR(lhs, rhs, tolerance)                                     \
  ::marketforge::test::check_near(                                             \
      static_cast<double>(lhs), static_cast<double>(rhs),                      \
      static_cast<double>(tolerance), #lhs, #rhs, __FILE__, __LINE__)
