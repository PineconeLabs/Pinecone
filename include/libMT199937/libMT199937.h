#include <stdint.h>
#include <time.h>

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

void init_MT199937(uint32_t seed);
uint32_t MT199937_int32(void);
uint32_t get_seed(void);

#ifdef LIBMT19937_IMPLEMENTATION

static uint32_t mt[N];
static int mti = N + 1;

void init_MT199937(uint32_t seed) {
    mt[0] = seed;
    for (mti = 1; mti < N; mti++) {
        mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
        mt[mti] &= 0xffffffffUL;
    }
}

uint32_t MT199937_int32() {
    uint32_t y;
    static uint32_t mag01[2] = {0x0UL, MATRIX_A};

    if (mti >= N) {
        int kk;

        for (kk = 0; kk < N - M; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < N - 1; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }

    y = mt[mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

uint32_t get_seed() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint32_t)(ts.tv_nsec ^ ts.tv_sec);
}

#endif // LIBMT19937_IMPLEMENTATION
