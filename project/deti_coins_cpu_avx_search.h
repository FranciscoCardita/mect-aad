#ifndef DETI_COINS_CPU_AVX_SEARCH
#define DETI_COINS_CPU_AVX_SEARCH

#include "md5_cpu_avx2.h"
#include <stdint.h>
#include <stdio.h>

static void deti_coins_cpu_avx_search(u32_t n_random_words) {
  if (n_random_words < 1u || n_random_words > 9u) {
    fprintf(stderr, "Error: n_random_words must be between 1 and 9.\n");
    return;
  }

  u32_t idx, coin[9 * 4u] __attribute__((aligned(16))), hash[4u * 8u];
  u64_t n_attempts, n_coins;
  u08_t *bytes[8];
  u32_t lanes;

  for (lanes = 0u; lanes < 8u; lanes++) {
    bytes[lanes] = (u08_t *)&coin[lanes * n_random_words * 4u];

    bytes[lanes][0u] = 'D';
    bytes[lanes][1u] = 'E';
    bytes[lanes][2u] = 'T';
    bytes[lanes][3u] = 'I';
    bytes[lanes][4u] = ' ';
    bytes[lanes][5u] = 'c';
    bytes[lanes][6u] = 'o';
    bytes[lanes][7u] = 'i';
    bytes[lanes][8u] = 'n';
    bytes[lanes][9u] = ' ';

    for (idx = 10u; idx < n_random_words * 4u - 1u; idx++) {
      bytes[lanes][idx] = ' ';
    }

    bytes[lanes][n_random_words * 4u - 1u] = '\n';
  }

  n_attempts = n_coins = 0ul;

  while (stop_request == 0) {
    md5_cpu_avx((v4si *)coin, (v4si *)hash);

    for (lanes = 0; lanes < 8; lanes++) {
      printf("Lane %u: ", lanes);
      for (idx = 0; idx < n_random_words * 4; idx++) {
        printf("0x%02X ", bytes[lanes][idx]);
      }
      printf("\n");
    }

    for (lanes = 0u; lanes < 8u; lanes++) {
      if (hash[4u * lanes + 3u] == 0u) {
        save_deti_coin(&coin[n_random_words * lanes]);
        n_coins++;
      }
    }

    for (lanes = 0u; lanes < 8u; lanes++) {
      for (idx = 10u;
           idx < n_random_words * 4u - 1u && bytes[lanes][idx] == (u08_t)126;
           idx++) {
        bytes[lanes][idx] = ' ';
      }
      if (idx < n_random_words * 4u - 1u) {
        bytes[lanes][idx]++;
      }
    }

    n_attempts += 8ul;
  }

  STORE_DETI_COINS();

  printf("deti_coins_cpu_avx_search: %lu DETI coin%s found in %lu attempt%s "
         "(expected %.2f coins)\n",
         n_coins, (n_coins == 1ul) ? "" : "s", n_attempts,
         (n_attempts == 1ul) ? "" : "s",
         (double)n_attempts / (double)(1ul << 32));
}

#endif
