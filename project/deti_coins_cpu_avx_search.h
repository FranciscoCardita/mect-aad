#ifndef DETI_COINS_CPU_AVX_SEARCH
#define DETI_COINS_CPU_AVX_SEARCH

#include "deti_coins_vault.h"
#include "md5_cpu_avx.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define N_LANES 4

typedef uint32_t u32_t;
typedef uint64_t u64_t;
typedef uint8_t u08_t;

static void deti_coins_cpu_avx_search() {
  u32_t interleaved_coins[13 * N_LANES] __attribute__((aligned(16)));
  u32_t interleaved_hashes[4 * N_LANES] __attribute__((aligned(16)));
  u08_t *bytes[N_LANES];
  u64_t n_attempts, n_coins, row, idx;
  u32_t lane, coin[13];

  for (row = 0; row < 13; row++) {
    for (lane = 0; lane < N_LANES; lane++) {
      if (row == 0)
        interleaved_coins[row * N_LANES + lane] = 0x49544544u; // "ITED"
      else if (row == 1)
        interleaved_coins[row * N_LANES + lane] = 0x696F6320u; // "ioc "
      else if (row == 2)
        interleaved_coins[row * N_LANES + lane] =
            0x3041206Eu + (lane << 24); // "0A n"
      else if (row == 12)
        interleaved_coins[row * N_LANES + lane] = 0x0A202020u; // "\n   "
      else
        interleaved_coins[row * N_LANES + lane] = 0x20202020u; // spaces
    }
  }

  for (lane = 0; lane < N_LANES; lane++) {
    bytes[lane] = (u08_t *)&interleaved_coins[3 * N_LANES + lane];
  }

  for (n_attempts = n_coins = 0; stop_request == 0; n_attempts += N_LANES) {
    md5_cpu_avx((v4si *)interleaved_coins, (v4si *)interleaved_hashes);

    for (lane = 0; lane < N_LANES; lane++) {
      for (row = 0; row < 13; row++) {
        coin[row] = interleaved_coins[row * N_LANES + lane];
      }

      if (interleaved_hashes[3 * N_LANES + lane] == 0x00000000) {
        save_deti_coin(coin);
        n_coins++;
      }
    }

    for (lane = 0; lane < N_LANES; lane++) {
      u08_t *field = bytes[lane];
      for (idx = 0; idx < 13 * 4 - 1 && field[idx] == (u08_t)126; idx++) {
        field[idx] = 0x20;
      }
      if (idx < 13 * 4 - 1) {
        field[idx]++;
      }
    }
  }

  STORE_DETI_COINS();
  printf("deti_coins_cpu_avx_search: %lu DETI coin%s found in %lu attempt%s "
         "(expected %.2f coins)\n",
         n_coins, (n_coins == 1ul) ? "" : "s", n_attempts,
         (n_attempts == 1ul) ? "" : "s", (double)n_attempts / (1ul << 32));
}

#undef N_LANES
#endif
