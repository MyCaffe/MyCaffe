#ifndef __PRIMITIVES_H__
#define __PRIMITIVES_H__

#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>


class Primitives
{
	float* m_gpu_workspace;
	float* m_host_workspace;
	size_t m_nMaxWorkspace;
	cublasHandle_t m_cublas;

public:
	Primitives()
	{
		m_gpu_workspace = NULL;
		m_host_workspace = NULL;
		m_nMaxWorkspace = 0;
		m_cublas = NULL;
	}

	~Primitives();

	long Initialize(size_t nMaxWorkspace, int nDim, int nNumLayers, int nNumHeads, int nSeqLen);

	void ToHost(size_t n, float* o, float* x, bool bSync = true);
	void Print(const char* name, float* x, size_t n, long long nLayer = -1);

	int denan(int n, float* o, float* x, bool bSync = true);
	void add(size_t n, float* o, float* x, bool bSync = true);
	void rmsnorm(int n, float* o, float* x, float* weight, bool bSync = true);
	void rope(float* q, float* k, float* f_real, float* f_imag, int num_heads, int head_size, int nFreqOffset, bool bSync = true);
	void matmul(float* xout, float* x, float* w, int n, int d, bool bSync = true);
	void siglu(size_t n, float* o, float* x1, float* x2, bool bSync = true);
	void softmax(int n, float* x, bool bSync = true);

	void attention(float* o, float* q, float* key_cache, float* value_cache, int num_heads, int head_size, size_t loff, int pos, int seq_len, int kv_dim, int kv_mul, bool bSync = true);
};

#endif // __ATTENTION_H__