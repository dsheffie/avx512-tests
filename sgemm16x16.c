#define _GNU_SOURCE
#include <time.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <x86intrin.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <linux/unistd.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sched.h>

const int64_t n_tries = 1L<<28;

void sgemm(float *A, float *B, float *Y, int n) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      for(int k = 0; k < n; k++) {
	Y[i*n+j] += A[i*n+k] * B[k*n+j];
      }
    }
  }
}

void sgemm16x16(float *A, float *B, float *Y) {
  __m512 vB[16], vY[16];
  __m512 vA;

#define XS(x) {}
  
  for(int i = 0; i < 16; i++) {
    vY[i] = _mm512_loadu_ps(&Y[i*16]);
    vB[i] = _mm512_loadu_ps(&B[i*16]);
  }

#define FMA_STEP(k,i) {					\
    vA = _mm512_set1_ps(A[i*16+k]);			\
    vY[i] = _mm512_fmadd_ps(vB[k], vA, vY[i]);		\
  }
  
#define DO_STEP(k) {				\
    FMA_STEP(k,0);				\
    FMA_STEP(k,1);				\
    FMA_STEP(k,2);				\
    FMA_STEP(k,3);				\
    FMA_STEP(k,4);				\
    FMA_STEP(k,5);				\
    FMA_STEP(k,6);				\
    FMA_STEP(k,7);				\
    FMA_STEP(k,8);				\
    FMA_STEP(k,9);				\
    FMA_STEP(k,10);				\
    FMA_STEP(k,11);				\
    FMA_STEP(k,12);				\
    FMA_STEP(k,13);				\
    FMA_STEP(k,14);				\
    FMA_STEP(k,15);				\
      }

  //for(int64_t cnt = 0; cnt < n_tries; cnt++) {
    DO_STEP(0);
    DO_STEP(1);
    DO_STEP(2);
    DO_STEP(3);  
    DO_STEP(4);
    DO_STEP(5);
    DO_STEP(6);
    DO_STEP(7);  
    DO_STEP(8);
    DO_STEP(9);
    DO_STEP(10);
    DO_STEP(11);
    DO_STEP(12);
    DO_STEP(13);
    DO_STEP(14);
    DO_STEP(15);
    //}
  
  for(int i = 0; i < 16; i++) {
    _mm512_storeu_ps(&Y[i*16], vY[i]);
  }
  
 
}


int main() {
  struct perf_event_attr pe;
  int fd, cpu;
  uint64_t t0,t1;
  float *A, *B, *Y0, *Y1;
  A = (float*)malloc(sizeof(float)*16*16);
  B = (float*)malloc(sizeof(float)*16*16);
  Y0 = (float*)malloc(sizeof(float)*16*16);
  Y1 = (float*)malloc(sizeof(float)*16*16);

  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(pe);
  pe.disabled = 0;
  cpu = sched_getcpu();
  pe.config = PERF_COUNT_HW_CPU_CYCLES;  
  fd = syscall(__NR_perf_event_open, &pe, 0, cpu, -1, 0);

  
  for(int i = 0; i < (16*16); i++) {
    Y0[i] = Y1[i] = drand48();
  }
  
  for(int i = 0; i < (16*16); i++) {
    A[i] = drand48();
    B[i] = drand48();
  }


  
  read(fd, &t0, sizeof(t0));
  for(int i = 0; i < n_tries; i++) {
    sgemm16x16(A,B,Y0);
  }
  read(fd, &t1, sizeof(t1));  
  long long flops = 2L*16L*16L*16L*n_tries;
  t1 -= t0;  
  printf("oflops = %lld, cycles = %lu, flops per cycle %g\n",
	 flops, t1, ((double)flops)/t1);

  read(fd, &t0, sizeof(t0));
  for(int64_t i = 0; i < n_tries; i++) {
    sgemm(A,B,Y1,16);
  }
  read(fd, &t1, sizeof(t1));    
  t1 -= t0;  
  printf("nflops = %lld, cycles = %lu, flops per cycle %g\n",
	 flops, t1, ((double)flops)/t1);

  for(int i = 0; i < (16*16); i++) {
    double err = Y0[i]-Y1[i];
    err *= err;
    if(err >= 1e-6) {
      printf("err = %g, %g, %g\n", err, Y0[i], Y1[i]);
    }
  }
  
  free(A);
  free(B);
  free(Y0);
  free(Y1);
  close(fd);
  return 0;
}
