#ifndef DETI_COINS_CPU_AVX_SEARCH
#define DETI_COINS_CPU_AVX_SEARCH

#include "md5_cpu_avx.h"
#include <stdint.h>
#include <stdio.h>

static void deti_coins_cpu_avx_search(u32_t n_random_words) {
  if (n_random_words < 1u || n_random_words > 9u) {
    fprintf(stderr, "Error: n_random_words must be between 1 and 9.\n");
    return;
  }

  __attribute__(( aligned(16))) u32_t coin[13u * 4u]; // Interleaved coins for 4 lanes
  __attribute__((aligned(16))) u32_t hash[4u * 4u]; // Hashes for 4 lanes
  u64_t n_attempts = 0ul;
  u64_t n_coins = 0ul;
  u32_t lanes, idx;

  for (lanes = 0u; lanes < 4u; lanes++) {
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
    bytes[51u] = '\n';

    for (idx = 10u; idx < 10u + n_random_words * 4u; idx++) {
      bytes[idx] = (u08_t)(32 + lanes);
    }

    for (; idx < 13u * 4u - 1u; idx++) {
      bytes[idx] = ' ';
    }
  }

  while (stop_request == 0) {
    md5_cpu_avx((v4si *)coin, (v4si *)hash);

    for (lanes = 0u; lanes < 4u; lanes++) {
      u32_t *h = &hash[4u * lanes];
      u32_t *c = &coin[13u * lanes];

      if (h[3u] == 0u) {
        fprintf(stdout,
                "deti_coins_cpu_avx_search: Found a DETI coin (Lane %u)\n",
                lanes);

        // ---------DEBUG---------------
        u08_t *coin_bytes = (u08_t *)c; // Interpret the coin as an array of bytes
        fprintf(stdout, "Coin (ASCII): \"");
        for (int i = 0; i < 13u * 4u; i++) {
          if (coin_bytes[i] >= 32 && coin_bytes[i] <= 126) {
            fprintf(stdout, "%c", coin_bytes[i]); // Printable ASCII
          } else {
            fprintf(stdout, "."); // Non-printable characters as '.'
          }
        }
        fprintf(stdout, "\"\n");
        // Print the coin as hexadecimal values
        fprintf(stdout, "Coin (Hex): [ ");
        for (int i = 0; i < 13u * 4u; i++) {
          fprintf(stdout, "%02X ", coin_bytes[i]);
        }
        fprintf(stdout, "]\n");
        // Print the hash in hexadecimal
        fprintf(stdout, "Hash: [ %08X %08X %08X %08X ]\n", h[0], h[1], h[2],
                h[3]);
        fprintf(stdout, "\n");
        // -----------------------------

        save_deti_coin(c);
        n_coins++;
      }
    }

    for (lanes = 0u; lanes < 4u; lanes++) {
      u08_t *bytes = (u08_t *)&coin[13u * lanes];
      for (idx = 10u; idx < n_random_words * 4u && bytes[idx] == (u08_t)126;
           idx++) {
        bytes[idx] = ' ';
      }
      if (idx < n_random_words * 4u) {
        bytes[idx]++;
      }
    }

    n_attempts += 4u;
  }

  STORE_DETI_COINS();

  printf("deti_coins_cpu_avx_search: %lu DETI coin%s found in %lu attempt%s "
         "(expected %.2f coins)\n",
         n_coins, (n_coins == 1ul) ? "" : "s", n_attempts,
         (n_attempts == 1ul) ? "" : "s",
         (double)n_attempts / (double)(1ul << 32));
}

#endif
