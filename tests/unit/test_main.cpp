#include "test_support.hpp"

#include <cstddef>
#include <iostream>

int main() {
  std::size_t failures = 0;
  for (const auto& test : marketforge::test::registry()) {
    try {
      test.function();
      std::cout << "[pass] " << test.name << '\n';
    } catch (const marketforge::test::Failure&) {
      ++failures;
      std::cout << "[fail] " << test.name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[fail] " << test.name
                << ": unexpected exception: " << error.what() << '\n';
    } catch (...) {
      ++failures;
      std::cerr << "[fail] " << test.name << ": unknown exception\n";
    }
  }

  std::cout << marketforge::test::registry().size() << " tests, " << failures
            << " failures\n";
  return failures == 0 ? 0 : 1;
}
