#include <x86intrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

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
        if (z_re * z_re + z_im * z_im > 4.)
	  break;
        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = xx + new_re;
        z_im = yy + new_im;
      }
      img[y*xdim+x] = k;
    }
  }
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

  __m512 v_dx = _mm512_set1_ps(dx);
  __m512 v_x0 = _mm512_set1_ps(x0);
  
  for(uint32_t y = 0; y < ydim; y++) {
    float yy = y0 + y * dy;
    __m512 v_yy = _mm512_set1_ps(yy);

    __m512 v_x = v_idx;
    for(uint32_t x = 0; x < xdim; x+=VL) {
      float xx = x0 + x * dx;
      __m512 v_xx = _mm512_add_ps(v_x0, _mm512_mul_ps(v_x, v_dx));
      v_x = _mm512_add_ps(v_x, _mm512_set1_ps(1.0f));

      __m512i v_k = _mm512_set1_epi32(0);
      __m512 v_z_re = v_xx, v_z_im =  v_yy;
      
      for (uint32_t k = 0; k < maxiter; ++k) {
	__m512 v_z_re2 = _mm512_mul_ps(v_z_re, v_z_re);
	__m512 v_z_im2 = _mm512_mul_ps(v_z_im, v_z_im);
	__m512 v_dist  = _mm512_add_ps(v_z_re2, v_z_im2);

	__mmask16 p = _mm512_cmp_ps_mask(v_dist, _mm512_set1_ps(4.0f), _MM_CMPINT_LT);
	/* no active lanes */
	if(__builtin_popcount(_mm512_mask2int(p)) == 0) {
	  break;
	}
	__m512 v_new_re = _mm512_sub_ps(v_z_re2, v_z_im2);
	__m512 v_new_im = _mm512_mul_ps(_mm512_set1_ps(2.0f), _mm512_mul_ps(v_z_re, v_z_im));
	v_z_re = _mm512_mask_add_ps(v_z_re, p, v_xx, v_new_re);
	v_z_im = _mm512_mask_add_ps(v_z_im, p, v_yy, v_new_im);
	v_k = _mm512_mask_add_epi32(v_k, p, v_k, _mm512_set1_epi32(1));
      }
      _mm512_storeu_epi32(&img[y*xdim+x], v_k);
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

int main(int argc, char *argv[]) {
  uint32_t ydim = 1024, xdim = 1024;
  uint32_t *img = new uint32_t[ydim*xdim];

  avx512_mandelbrot(ydim, xdim, img);
  writePPM(img, xdim, ydim, "avx512.ppm");
  
  delete [] img;
  return 0;
}
