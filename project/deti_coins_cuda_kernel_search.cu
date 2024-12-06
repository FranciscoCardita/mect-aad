//
// Tomás Oliveira e Silva,  October 2024
//
// Arquiteturas de Alto Desempenho 2024/2025
//
// MD5 hash CUDA kernel code
//
// md5_cuda_kernel() --- each thread computes the MD5 hash of one message
//
// do not use this directy to search for DETI coins!
//

//
// needed stuff
//

typedef unsigned int u32_t;
typedef union {
    u32_t coin_as_ints[13]; // 13 4-byte integers (52 bytes)
    char coin_as_chars[52]; // 52 characters
} coin_template;


#include "md5.h"

//
// the nvcc compiler stores x[] and state[] in registers (constant indices!)
//
// global thread number: n = threadIdx.x + blockDim.x * blockIdx.x

__device__ __host__ inline void init_deti_coin(coin_template *coin) {
	for (int i = 0; i < sizeof(coin->coin_as_chars); i++) {
		coin->coin_as_chars[i] = 0;
	}
	const char *prefix = "DETI coin ";
	for (int i = 0; i < 10; i++) {
		coin->coin_as_chars[i] = prefix[i];
	}
	for (int i = 10; i < 51; i++) {
		coin->coin_as_chars[i] = ' ';
	}
	coin->coin_as_chars[51] = '\n';
}

extern "C" __global__ __launch_bounds__(128,1) void deti_coins_cuda_kernel_search(u32_t *storage_area,u32_t v1, u32_t v2)
{	
  coin_template coin;
  u32_t n,a,b,c,d,state[4],x[16],hash[4];

  //
  // get the global thread number
  //
  n = (u32_t)threadIdx.x + (u32_t)blockDim.x * (u32_t)blockIdx.x;
  // coin[ 0] = 0x49544544u;
  // coin[ 1] = 0x696f6320u;
  // coin[ 2] = 0x6e20206eu;
  // coin[ 3] = 0x20202020u;
  // coin[ 4] = 0x20202020u;
  // coin[ 5] = 0x20202020u;
  // coin[ 6] = 0x20202020u;
  // coin[ 7] = 0x20202020u;
  // coin[ 8] = 0x20202020u;
  // coin[ 9] = 0x20202020u;
  // coin[10] = v1;
  // coin[11] = v2;
  // coin[12] = 0x20202020u;
  // coin[51] = '\n';

  init_deti_coin(&coin);
  coin.coin_as_ints[10] = v1;
  coin.coin_as_ints[11] = v2;

  coin.coin_as_ints[ 4] += (n % 64) <<  0; n /= 64;
  coin.coin_as_ints[ 4] += (n % 64) <<  8; n /= 64;
  coin.coin_as_ints[ 4] += (n % 64) << 16; n /= 64;
  coin.coin_as_ints[ 4] += (n % 64) << 24;

  for(n = 0; n < 64; n++) {
  //
  // compute MD5 hash
  //
	# define C(c)         (c)
	# define ROTATE(x,n)  (((x) << (n)) | ((x) >> (32 - (n))))
	# define DATA(idx)    coin.coin_as_ints[idx]
	# define HASH(idx)    hash[idx]
	# define STATE(idx)   state[idx]
	# define X(idx)       x[idx]
  	CUSTOM_MD5_CODE();
	if (hash[3] == 0) {
		u32_t n = atomicAdd(storage_area, 13);
		if (n + 13 <= 1024) {
			for (u32_t i = 0; i < 13; i++) {
				storage_area[n + i] = coin.coin_as_ints[i];
			}
		}
	}
	coin.coin_as_ints[12] += 1 << 16;
  }
//# undef C
//# undef ROTATE
//# undef DATA
//# undef HASH
//# undef STATE
//# undef X
}
