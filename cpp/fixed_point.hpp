// Qm.n fixed-point helpers: saturation, round-to-nearest shifts, multiply.
#pragma once
#include <cmath>
#include <cstdint>

namespace fx {
inline int32_t sat32(int64_t v) {
    return v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : (int32_t)v);
}
inline int32_t sat_add(int32_t a, int32_t b) { return sat32((int64_t)a + b); }
// Round-to-nearest (ties toward +inf) arithmetic right shift; s <= 0 shifts left.
inline int64_t rshift_round(int64_t v, int s) {
    return s <= 0 ? v * (int64_t(1) << -s) : (v + (int64_t(1) << (s - 1))) >> s;
}
// (a * b) in Q.fl -> Q.fl, rounded and saturated.
inline int32_t mul_q(int32_t a, int32_t b, int fl) { return sat32(rshift_round((int64_t)a * b, fl)); }
// Offline only (LUT/threshold construction): real -> Q.fl, round-to-nearest, saturated.
inline int32_t to_q(double x, int fl) { return sat32(std::llround(std::ldexp(x, fl))); }
}  // namespace fx
