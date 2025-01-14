#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>

#define TRIES 4

typedef void (*sorter_t)(uint32_t *arr, uint32_t n);

uint32_t issorted(uint32_t *arr, uint32_t n) {
  for(uint32_t i=1;i<n;i++) {
    if(arr[i-1]>arr[i]) 
      return 0;
  }
  return 1;
}

void shuffle(uint32_t *arr, uint32_t n) {
  uint32_t x = 1;
  for(uint32_t i=0;i<n;i++) {
    uint32_t j = i + x % (n-i);
    uint32_t t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
  }
}

double timestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + ((double)tv.tv_usec)*1e-6;
}

template <typename T>
static inline void swap(T &a, T &b) {
  T t = a;a = b;b = t;
}

uint32_t partition(uint32_t *arr, uint32_t n) {
  uint32_t d=0, k = rand() % n;
  uint32_t p = arr[k];
  swap(arr[k], arr[n-1]);
  
  for(uint32_t i=0;i<(n-1);i++) {
    if(arr[i] < p) {
      swap(arr[i], arr[d]);
      d++;
    }
  }
  swap(arr[n-1], arr[d]);
  return d;
}

uint32_t hoare_partition(uint32_t *arr, uint32_t n) {
  uint32_t p = arr[0];
  int32_t i = 1, j = n-1;
   
  while(1) {
    /* scan from the left finding the first value larger than the pivot */
    while((i < n) and (arr[i] <= p)) {
      i++;
    }
    
    while((j > 0) and arr[j] >= p) {
      j--;
    } 


    if(i < j) {
      swap(arr[i], arr[j]);
    }
    else {
      swap(arr[0], arr[j]);
      return j;
    }
	 

  }
  return ~0U;
}

void inssort(uint32_t *arr, uint32_t n) {
  for(uint32_t i=1;i<n;i++) {
    uint32_t j=i;
    while((j > 0) && (arr[j-1] > arr[j])) {
      uint32_t t = arr[j-1];
      arr[j-1] = arr[j];
      arr[j] = t;
      j--;
    }
  }
}

void quicksort(uint32_t *arr, uint32_t n) {
  uint32_t d;
  if(n <= 2) { 
    inssort(arr, n);
    return;
  }
  d = hoare_partition(arr, n);
  quicksort(arr, d);
  quicksort(arr+d+1, n-d-1);
}

static const sorter_t sorts[] = {quicksort};
static const char *names[] = {"quicksort"};
static const uint32_t nsorts = sizeof(sorts) / sizeof(sorts[0]);


//#define QUADSORTS 1
int main() {
  uint32_t i,j,n = 10000;
  uint32_t *arr, *bak;


  srand(time(NULL));
  arr = (uint32_t*)malloc(sizeof(uint32_t)*n);
  bak = (uint32_t*)malloc(sizeof(uint32_t)*n);

  for(i=0;i<n;i++) {
    arr[i] = i;
  }
  shuffle(arr, n);

  memcpy(bak, arr, sizeof(uint32_t)*n);

  printf("n=%u\n", n);
  for(i=0;i<nsorts;i++) {
    double avg = 0.0;
    for(j=0;j<TRIES;j++) {
      memcpy(arr, bak, sizeof(uint32_t)*n);
      double t = timestamp();
      sorts[i](arr, n);
      t = timestamp()-t;
      assert(issorted(arr,n));
      avg += t;
    }
    avg /= (double)TRIES;
    printf("%s %g sec on avg\n", names[i], avg);
  }
  

  free(arr);
  free(bak);
  return 0;
}
