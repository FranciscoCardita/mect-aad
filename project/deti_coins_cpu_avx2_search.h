#ifndef DETI_COINS_CPU_AVX2_SEARCH
#define DETI_COINS_CPU_AVX2_SEARCH

#include "md5_cpu_avx2.h"

#define N_LANES 8

static void deti_coins_cpu_avx2_search() {
  {

    u32_t coins[13][N_LANES] __attribute__((aligned(4 * N_LANES)));
    u32_t hashes[4][N_LANES] __attribute__((aligned(4 * N_LANES)));
    union {
      u32_t coin_as_ints[13];
      u08_t coin_as_chars[13 * 4 + 1];
    } coin_template;
    u32_t n, t, lane, coin[13u], v1, v2;
    u64_t n_attempts, n_coins;

    //
    // setup
    //
    t = (u32_t)time(NULL) % 10000u;
    if ((n = snprintf((char *)coin_template.coin_as_chars, 53,
                      "DETI coin 0 [%04u]                                 \n",
                      t)) != 52) {
      fprintf(stderr, "not 52 bytes, but %u\n", n);
      exit(1);
    }

    for (lane = 0; lane < N_LANES; lane++) {
      for (n = 0; n < 13; n++) {
        coins[n][lane] = coin_template.coin_as_ints[n];
      }
      coin_template.coin_as_ints[2] += 1 << 16;
    }

    //
    // find DETI coins
    //
    v1 = v2 = 0x20202020;
    for (n_attempts = n_coins = 0ul; stop_request == 0; n_attempts += N_LANES) {
      next_value_to_try(v1);
      for (lane = 0; lane < N_LANES; lane++) {
        coins[11][lane] = v1;
      }

      if (v1 == 0x20202020) {
        next_value_to_try(v2);
        for (lane = 0; lane < N_LANES; lane++) {
          coins[10][lane] = v2;
        }
      }

      md5_cpu_avx2((v8hi *)coins, (v8hi *)hashes);

      for (lane = 0; lane < N_LANES; lane++) {
        //
        // check if hash is a DETI coin
        //
        if (hashes[3][lane] == 0) {

          for (n = 0; n < 13; n++) {
            coin[n] = coins[n][lane];
          }
          save_deti_coin(coin);
          n_coins++;
        }
      }
    }
    STORE_DETI_COINS();
    printf("deti_coins_cpu_avx2_search: %lu DETI coin%s found in %lu attempt%s "
           "(expected %.2f coins)\n",
           n_coins, (n_coins == 1ul) ? "" : "s", n_attempts,
           (n_attempts == 1ul) ? "" : "s",
           (double)n_attempts / (double)(1ul << 32));
  }
}

#undef N_LANES
#endif
