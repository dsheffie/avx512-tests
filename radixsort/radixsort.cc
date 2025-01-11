#include <x86intrin.h>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <sys/time.h>

std::ostream &operator<<(std::ostream &out, const __m512i &v) {
  int32_t arr[16] = {0};
  _mm512_storeu_epi32(arr, v);
  for(int i = 0; i < 16; i++) {
    out << arr[i];
    if(i != 15)
      out << ",";
  }
  return out;
}


void scalar_radixsort(uint32_t *arr, uint32_t *tmp, uint32_t n) {
  uint32_t *in = arr, *out = tmp;
  
  for(uint32_t i = 0; i < 32; i++) {
    uint32_t ns = 0, off_s = 0, off_ns = 0;
    /* count not-set bits */
    for(uint32_t j = 0; j < n; j++) {
      uint32_t b = (in[j] >> i) & 0x1;
      ns += (b==0);
    }
    for(uint32_t j = 0; j < n; j++) {
      uint32_t b = (in[j] >> i) & 0x1;
      if(b) {
	out[ns+off_s] = in[j];
	off_s++;
      }
      else {
	out[off_ns] = in[j];
	off_ns++;
      }
    }
    std::swap(in, out);
  }

  if(arr != in) {
    memcpy(arr, in, sizeof(uint32_t)*n);
  }
}

void avx512_radixsort(uint32_t *arr, uint32_t *tmp, uint32_t n) {
  uint32_t *in = arr, *out = tmp;
  
  for(uint32_t i = 0; i < 32; i++) {
    uint32_t ns = 0, off_s = 0, off_ns = 0;
    /* count not-set bits */

    __m512i vidx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    __m512i v_sum = _mm512_set1_epi32(0);
    for(uint32_t j = 0; j < n; j+= 16) {
      __mmask16 k = _mm512_cmp_epi32_mask(vidx, _mm512_set1_epi32(n), _MM_CMPINT_LT);
      vidx = _mm512_add_epi32(vidx, _mm512_set1_epi32(16));
      __m512i v_in = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), k, &in[j]);
      v_in = _mm512_srl_epi32 (v_in, _mm_set1_epi64x(i));
      v_in = _mm512_and_epi32(v_in, _mm512_set1_epi32(1));
      v_sum =  _mm512_add_epi32(v_sum, v_in);
    }
    ns = n - _mm512_reduce_add_epi32(v_sum);
    
    vidx = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    
    for(uint32_t j = 0; j < n; j+=16) {
      __mmask16 k = _mm512_cmp_epi32_mask(vidx, _mm512_set1_epi32(n), _MM_CMPINT_LT);
      vidx = _mm512_add_epi32(vidx, _mm512_set1_epi32(16));
      __m512i v_in = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), k, &in[j]);
      
      __m512i v_b = _mm512_srl_epi32 (v_in, _mm_set1_epi64x(i));
      v_b = _mm512_and_epi32(v_b, _mm512_set1_epi32(1));

      __mmask16 ks = _mm512_cmp_epi32_mask(v_b, _mm512_set1_epi32(1), _MM_CMPINT_EQ);
      __mmask16 kc = _mm512_cmp_epi32_mask(v_b, _mm512_set1_epi32(0), _MM_CMPINT_EQ);      
      ks = _kand_mask16(k, ks);
      kc = _kand_mask16(k, kc);

      _mm512_mask_compressstoreu_epi32(&out[ns+off_s], ks, v_in);
      _mm512_mask_compressstoreu_epi32(&out[off_ns], kc, v_in);

      off_s  += __builtin_popcount(_mm512_mask2int(ks));
      off_ns += __builtin_popcount(_mm512_mask2int(kc));
    }
    std::swap(in, out);
  }

  if(arr != in) {
    memcpy(arr, in, sizeof(uint32_t)*n);
  }
}



void shuffle(uint32_t *arr, uint32_t n) {
  for(uint32_t i=0;i<n;i++) {
    uint32_t j = i + rand() % (n-i);
    std::swap(arr[i], arr[j]);
  }
}

bool issorted(uint32_t *arr, uint32_t n) {
  for(uint32_t i=1;i<n;i++) {
    if(arr[i-1]>arr[i]) 
      return false;
  }
  return true;
}

static inline double timeval_to_sec(struct timeval &t) {
  return t.tv_sec + 1e-6 * static_cast<double>(t.tv_usec);
}

static double timestamp() {
  struct timeval t;
  gettimeofday(&t, nullptr);
  return timeval_to_sec(t);
}

int main(int argc, char *argv[]) {
  double t,ts,tv;
  uint32_t n = 1U<<28;
  uint32_t *arr = new uint32_t[n];
  uint32_t *tmp = new uint32_t[n];
  for(uint32_t i = 0; i < n; i++) {
    arr[i] = i;
  }
  std::cout << n << " keys\n";
  shuffle(arr, n);
  t = timestamp();
  avx512_radixsort(arr, tmp, n);
  tv = timestamp() - t;
  std::cout << "issorted() = " << issorted(arr, n) << "\n";
  std::cout << "avx512 took " << tv << " seconds\n";

  
  shuffle(arr, n);
  t = timestamp();
  scalar_radixsort(arr, tmp, n);
  ts = timestamp() - t;
  std::cout << "issorted() = " << issorted(arr, n) << "\n";
  std::cout << "scalar took " << ts << " seconds\n";

  std::cout << (ts/tv) << " speedup with avx512\n";
  
  delete [] arr;
  delete [] tmp;
  return 0;
}
