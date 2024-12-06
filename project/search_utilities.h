#define next_value_to_try(v)                                                   \
  do {                                                                         \
    (v)++;                                                                     \
    if (((v) & 0xFF) == 0x7F) {                                                \
      (v) += 0xA1;                                                             \
      if ((((v) >> 8) & 0xFF) == 0x7F) {                                       \
        (v) += 0xA1 << 8;                                                      \
        if ((((v) >> 16) & 0xFF) == 0x7F) {                                    \
          (v) += 0xA1 << 16;                                                   \
          if ((((v) >> 24) & 0xFF) == 0x7F) {                                  \
            (v) += 0xA1 << 24;                                                 \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

#define random_value_to_try_ascii()                                            \
  ({                                                                           \
    u32_t value = 0;                                                           \
    for (int i = 0; i < 4; i++) {                                              \
      value |= ((u32_t)(0x20 + (rand() % (0x7E - 0x20 + 1)))) << (i * 8);      \
    }                                                                          \
    value;                                                                     \
  })
