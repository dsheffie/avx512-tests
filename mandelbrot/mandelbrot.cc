#include <x86intrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sys/time.h>

std::ostream &operator<<(std::ostream &out, const __m512i &v) {
  int32_t arr[16] = {~0};
  _mm512_storeu_epi32(arr, v);
  for(int i = 0; i < 16; i++) {
    out << arr[i];
    if(i != 15)
      out << ",";
  }
  return out;
}
std::ostream &operator<<(std::ostream &out, const __m512 &v) {
  float arr[16] = {0.0};
  _mm512_storeu_ps(arr, v);
  for(int i = 0; i < 16; i++) {
    out << arr[i];
    if(i != 15)
      out << ",";
  }
  return out;
}


void scalar_mandelbrot(uint32_t ydim, uint32_t xdim, uint32_t *img, uint32_t maxiter = 256) {
  const float x0 = -2.0f, x1 = 1.0f;
  const float y0 = -1.0f, y1 = 1.0f;
  const float dx = (x1 - x0) / static_cast<float>(xdim);
  const float dy = (y1 - y0) / static_cast<float>(ydim);
    
  for(uint32_t y = 0; y < ydim; y++) {
    for(uint32_t x = 0; x < xdim; x++) {
      float xx = x0 + x * dx;
      float yy = y0 + y * dy;
      float z_re = xx, z_im = yy;
      uint32_t k;
      for (k = 0; k < maxiter; ++k) {
	float d = z_re * z_re + z_im * z_im;
        if (d > 4.0f) {
	  break;
	}
        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = xx + new_re;
        z_im = yy + new_im;
      }
      img[y*xdim+x] = k;
    }
  }
 done:
  return;
}

void avx512_mandelbrot(uint32_t ydim, uint32_t xdim, uint32_t *img, uint32_t maxiter = 256) {
  const float x0 = -2.0f, x1 = 1.0f;
  const float y0 = -1.0f, y1 = 1.0f;
  const float dx = (x1 - x0) / static_cast<float>(xdim);
  const float dy = (y1 - y0) / static_cast<float>(ydim);
  static const uint32_t VL = 16;

  __m512 v_idx = _mm512_set_ps(15.0f,14.0f,13.0f,12.0f,
			       11.0f,10.0f,9.0f,8.0f,
			       7.0f,6.0f,5.0f,4,
			       3.0f,2.0f,1.0f,0.0f);

  const __m512 v_dx = _mm512_set1_ps(dx);
  const __m512 v_x0 = _mm512_set1_ps(x0);
  for(uint32_t y = 0; y < ydim; y++) {
    float yy = y0 + y * dy;
    __m512 v_yy = _mm512_set1_ps(yy);

    __m512 v_x = v_idx;
    for(uint32_t x = 0; x < xdim; x+=VL) {
      float xx = x0 + x * dx;
      __m512 v_xx = _mm512_add_ps(v_x0, _mm512_mul_ps(v_x, v_dx));

      __m512i v_k = _mm512_set1_epi32(0);
      __m512 v_z_re = v_xx, v_z_im =  v_yy;
      
      for (uint32_t k = 0; k < maxiter; ++k) {
	__m512 v_z_re2 = _mm512_mul_ps(v_z_re, v_z_re);
	__m512 v_z_im2 = _mm512_mul_ps(v_z_im, v_z_im);
	__m512 v_dist  = _mm512_add_ps(v_z_re2, v_z_im2);

	__mmask16 p = _mm512_cmp_ps_mask(v_dist, _mm512_set1_ps(4.0f), _MM_CMPINT_LT);
	/* no active lanes */
	if(_mm512_mask2int(p) == 0) {
	  break;
	}
	__m512 v_new_re = _mm512_sub_ps(v_z_re2, v_z_im2);
	__m512 v_new_im = _mm512_mul_ps(_mm512_set1_ps(2.0f), _mm512_mul_ps(v_z_re, v_z_im));
	v_z_re = _mm512_mask_add_ps(v_z_re, p, v_xx, v_new_re);
	v_z_im = _mm512_mask_add_ps(v_z_im, p, v_yy, v_new_im);
	v_k = _mm512_mask_add_epi32(v_k, p, v_k, _mm512_set1_epi32(1));
      }
      _mm512_storeu_epi32(&img[y*xdim+x], v_k);
      v_x = _mm512_add_ps(v_x, _mm512_set1_ps(16.0f));      
    }
  }
}



/* Write a PPM image file with the image of the Mandelbrot set */
static void writePPM(uint32_t *buf, uint32_t width, uint32_t height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    if (!fp) {
        printf("Couldn't open a file '%s'\n", fn);
        exit(-1);
    }
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (uint32_t i = 0; i < width * height; ++i) {
      char c = (buf[i] & 0x1) ? (char)240 : 20;
      for (uint32_t j = 0; j < 3; ++j) {
	fputc(c, fp);
      }
    }
    fclose(fp);
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
  uint32_t ydim = (1U<<15), xdim = (1U<<15);
  uint32_t *scalar_img = new uint32_t[ydim*xdim];
  uint32_t *avx512_img = new uint32_t[ydim*xdim];  

  double t0,t1;
  t0 = timestamp();
  scalar_mandelbrot(ydim, xdim, scalar_img);
  t0 = timestamp() - t0;

  t1 = timestamp();
  avx512_mandelbrot(ydim, xdim, avx512_img);
  t1 = timestamp() - t1;

  std::cout << "scalar  = " << t0 << " sec\n";
  std::cout << "avx512  = " << t1 << " sec\n";
  std::cout << "speedup = " << (t0/t1) << "x\n";
  
  writePPM(scalar_img, xdim, ydim, "scalar.ppm");
  writePPM(avx512_img, xdim, ydim, "avx512.ppm");
  
  delete [] scalar_img;
  delete [] avx512_img;
  return 0;
}
