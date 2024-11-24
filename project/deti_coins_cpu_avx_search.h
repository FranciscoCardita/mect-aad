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

  __attribute__((aligned(32))) u32_t coin[13u * 8u];
  __attribute__((aligned(32))) u32_t hash[4u * 8u];
  u64_t n_attempts = 0, n_coins = 0;
  u32_t lanes, idx;

  for (lanes = 0u; lanes < 8u; lanes++) {
    u08_t *bytes = (u08_t *)&coin[13u * lanes];
    bytes[0u] = 'D';
    bytes[1u] = 'E';
    bytes[2u] = 'T';
    bytes[3u] = 'I';
    bytes[4u] = ' ';
    bytes[5u] = 'c';
    bytes[6u] = 'o';
    bytes[7u] = 'i';
    bytes[8u] = 'n';
    bytes[9u] = ' ';
    for (idx = 10u; idx < 13u * 4u - 1u; idx++) {
      bytes[idx] = ' ';
    }
    bytes[13u * 4u - 1u] = '\n'; // Mandatory termination
  }

  while (stop_request == 0) {
    md5_cpu_avx2((v8hi *)coin, (v8hi *)hash);

    for (lanes = 0u; lanes < 8u; lanes++) {
      printf("Coin %u: Hash = %08x %08x %08x %08x\n", lanes, hash[4u * lanes],
             hash[4u * lanes + 1u], hash[4u * lanes + 2u],
             hash[4u * lanes + 3u]);
    }

    // Check results for each lane
    for (lanes = 0u; lanes < 8u; lanes++) {
      // Validate the last 32 bits (hash[3] for each lane)
      if ((hash[4u * lanes + 3u] & 0xFFFFFFFFu) == 0u) { // Check last word
        save_deti_coin(&coin[13u * lanes]);
        n_coins++;
      }
    }

    // Increment the random portion of each lane's coin
    for (lanes = 0u; lanes < 8u; lanes++) {
      u08_t *bytes = (u08_t *)&coin[13u * lanes];
      for (idx = 10u; idx < n_random_words * 4u && bytes[idx] == (u08_t)126;
           idx++) {
        bytes[idx] = ' '; // Reset to the first character
      }
      if (idx < n_random_words * 4u) {
        bytes[idx]++;
      }
    }

    n_attempts += 8u; // Increment attempts by 8 (one per lane)
  }

  STORE_DETI_COINS();

  printf("deti_coins_cpu_avx_search: %lu DETI coin%s found in %lu attempt%s "
         "(expected %.2f coins)\n",
         n_coins, (n_coins == 1ul) ? "" : "s", n_attempts,
         (n_attempts == 1ul) ? "" : "s",
         (double)n_attempts / (double)(1ul << 32));
}

#endif
