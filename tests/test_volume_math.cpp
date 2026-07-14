#include <limits>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dvo/volume_math.h"

TEST_CASE("Relative endpoint volume uses slider percentage points") {
  const auto up = dvo::calculate_volume_adjustment(0.50, 5);
  CHECK(up.requested == Catch::Approx(0.55));
  CHECK(up.target == Catch::Approx(0.55));
  CHECK_FALSE(up.clamped);
  CHECK_FALSE(up.noop);

  const auto down = dvo::calculate_volume_adjustment(0.50, -5);
  CHECK(down.target == Catch::Approx(0.45));
  CHECK_FALSE(down.clamped);
}

TEST_CASE("Relative endpoint volume clamps at system boundaries") {
  const auto upper = dvo::calculate_volume_adjustment(0.98, 5);
  CHECK(upper.requested == Catch::Approx(1.03));
  CHECK(upper.target == Catch::Approx(1.0));
  CHECK(upper.clamped);
  CHECK_FALSE(upper.noop);

  const auto lower = dvo::calculate_volume_adjustment(0.02, -5);
  CHECK(lower.target == Catch::Approx(0.0));
  CHECK(lower.clamped);

  const auto upper_noop = dvo::calculate_volume_adjustment(1.0, 5);
  CHECK(upper_noop.noop);
  CHECK(upper_noop.clamped);
  const auto lower_noop = dvo::calculate_volume_adjustment(0.0, -5);
  CHECK(lower_noop.noop);
  CHECK(lower_noop.clamped);
}

TEST_CASE("Relative endpoint volume rejects invalid inputs") {
  CHECK_THROWS_AS(dvo::calculate_volume_adjustment(-0.1, 5), std::invalid_argument);
  CHECK_THROWS_AS(dvo::calculate_volume_adjustment(1.1, -5), std::invalid_argument);
  CHECK_THROWS_AS(dvo::calculate_volume_adjustment(
                      std::numeric_limits<double>::quiet_NaN(), 5),
                  std::invalid_argument);
  CHECK_THROWS_AS(dvo::calculate_volume_adjustment(0.5, 0), std::invalid_argument);
  CHECK_THROWS_AS(dvo::calculate_volume_adjustment(0.5, 101), std::invalid_argument);
}
