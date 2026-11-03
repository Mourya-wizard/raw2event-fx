// Bit-accurate fixed-point raw-first event generator.
// event_fx --in frames.bin [--w W --h H] --fl 12 --lut-bits 12 --interp 1 [--c 0.25] [--out events.bin]
// Input frames.bin: uint32 N,W,H; int32 t_us[N]; uint16 raw[N][H][W] (RGGB, 10-bit, black 64).
// Output events.bin: packed {uint16 x, uint16 y, int32 t_us, int8 pol} (9 bytes), half-res coords.
// Prints one line of key=value stats incl. per-stage times (std::chrono, single thread, I/O excluded).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "fixed_point.hpp"

#pragma pack(push, 1)
struct Event { uint16_t x, y; int32_t t; int8_t p; };
#pragma pack(pop)
static_assert(sizeof(Event) == 9, "packed event");

using clk = std::chrono::steady_clock;
static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }

int main(int argc, char** argv) {
    std::string in, out;
    int fl = 12, lut_bits = 12, interp = 1, black = 64, W_arg = 0, H_arg = 0;
    double c_real = 0.25;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--in") in = v; else if (k == "--out") out = v;
        else if (k == "--fl") fl = std::atoi(v.c_str()); else if (k == "--lut-bits") lut_bits = std::atoi(v.c_str());
        else if (k == "--interp") interp = std::atoi(v.c_str()); else if (k == "--c") c_real = std::atof(v.c_str());
        else if (k == "--black") black = std::atoi(v.c_str());
        else if (k == "--w") W_arg = std::atoi(v.c_str()); else if (k == "--h") H_arg = std::atoi(v.c_str());
        else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (in.empty() || fl < 1 || fl > 24 || lut_bits < 1 || lut_bits > 12) { std::fprintf(stderr, "bad args\n"); return 2; }

    FILE* f = std::fopen(in.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", in.c_str()); return 1; }
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3) { std::fprintf(stderr, "bad header\n"); return 1; }
    const int N = (int)hdr[0], W = (int)hdr[1], H = (int)hdr[2], w = W / 2, h = H / 2;
    if ((W_arg && W_arg != W) || (H_arg && H_arg != H) || N < 2 || W % 2 || H % 2) {
        std::fprintf(stderr, "size mismatch: file %dx%d n=%d\n", W, H, N); return 1;
    }
    std::vector<int32_t> ts(N);
    std::vector<uint16_t> raw((size_t)N * W * H);
    if (std::fread(ts.data(), 4, N, f) != (size_t)N || std::fread(raw.data(), 2, raw.size(), f) != raw.size()) {
        std::fprintf(stderr, "truncated input\n"); return 1;
    }
    std::fclose(f);

    // ln LUT over the 12-bit cell sum, output Q.fl. Offline construction may use double.
    const int shift = 12 - lut_bits, mask = (1 << shift) - 1;
    std::vector<int32_t> lut((1 << lut_bits) + 1);
    for (int i = 0; i <= (1 << lut_bits); ++i) {
        double s = interp ? (double)(i << shift) : (double)((i << shift) + ((1 << shift) >> 1));  // nodes vs bin centres
        lut[i] = fx::to_q(std::log(s < 1 ? 1.0 : s), fl);
    }
    const int32_t C = fx::to_q(c_real, fl);
    if (C <= 0) { std::fprintf(stderr, "threshold underflows at fl=%d\n", fl); return 1; }

    std::vector<uint16_t> sum((size_t)w * h);
    std::vector<int32_t> L((size_t)w * h), Lref((size_t)w * h);
    std::vector<Event> ev;
    ev.reserve(1 << 20);
    clk::duration t_bin{}, t_log{}, t_emit{};

    for (int k = 0; k < N; ++k) {
        const uint16_t* fr = raw.data() + (size_t)k * W * H;
        auto t0 = clk::now();
        for (int y = 0; y < h; ++y) {  // stage 1: 2x2 RGGB bin, black-subtracted integer sum (<= 12 bit)
            const uint16_t* r0 = fr + (size_t)(2 * y) * W; const uint16_t* r1 = r0 + W;
            for (int x = 0; x < w; ++x) {
                int s = 0;
                for (int v : {r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]}) s += v > black ? v - black : 0;
                sum[(size_t)y * w + x] = (uint16_t)(s > 4095 ? 4095 : s);
            }
        }
        auto t1 = clk::now();
        for (size_t i = 0; i < sum.size(); ++i) {  // stage 2: LUT log (+ optional linear interpolation)
            int s = sum[i], j = s >> shift;
            L[i] = interp ? lut[j] + (int32_t)fx::rshift_round((int64_t)(lut[j + 1] - lut[j]) * (s & mask), shift) : lut[j];
        }
        auto t2 = clk::now();
        if (k == 0) {
            Lref = L;
        } else {  // stage 3: event emission, integer only
            const int32_t ta = ts[k - 1], dt = ts[k] - ts[k - 1];
            for (size_t i = 0; i < L.size(); ++i) {
                int32_t d = L[i] - Lref[i];
                int64_t a = d < 0 ? -(int64_t)d : d;
                int32_t n = (int32_t)(a / C);
                if (!n) continue;
                int8_t p = d > 0 ? 1 : -1;
                for (int32_t j = 1; j <= n; ++j)  // t_j = ta + dt * j*C / |dL|, round-to-nearest
                    ev.push_back({(uint16_t)(i % w), (uint16_t)(i / w), ta + (int32_t)(((int64_t)dt * j * C + a / 2) / a), p});
                Lref[i] = fx::sat_add(Lref[i], p * n * C);
            }
        }
        auto t3 = clk::now();
        t_bin += t1 - t0; t_log += t2 - t1; t_emit += t3 - t2;
    }

    if (!out.empty()) {
        FILE* o = std::fopen(out.c_str(), "wb");
        if (!o || std::fwrite(ev.data(), sizeof(Event), ev.size(), o) != ev.size()) { std::fprintf(stderr, "write failed\n"); return 1; }
        std::fclose(o);
    }
    const double tot = secs(t_bin + t_log + t_emit), px = (double)N * W * H;
    std::printf("frames=%d W=%d H=%d events=%zu C_int=%d bin_s=%.6f log_s=%.6f emit_s=%.6f total_s=%.6f "
                "ns_per_rawpx=%.4f mev_per_s=%.3f fps=%.1f\n",
                N, W, H, ev.size(), C, secs(t_bin), secs(t_log), secs(t_emit), tot,
                tot * 1e9 / px, ev.size() / tot / 1e6, N / tot);
    return 0;
}
