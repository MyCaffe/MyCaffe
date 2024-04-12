// Portions from https://github.com/rogerallen/llama2.cu
// Portions from https://github.com/ankan-ban/llama2.cu
#include "primitives.h"

#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

const int NUM_THREADS = 256;

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

const int num_threads_lrg = 1024;
const int num_threads_med = 256;

//------------------------------------------------------------------------
// CUDA kernels
//------------------------------------------------------------------------

__global__ void denan_kernel(size_t n, float* o, float* x, float* count)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        o[i] = x[i];
        count[i] = 0.0;

        if (isinf(o[i]) || isnan(o[i]))
        {
            o[i] = 0;
			count[i] = 1;
        }
    }
}

__global__ void exp_kernel(size_t n, float fmax, float* o, float* x)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        o[i] = expf(x[i] - fmax);
    }
}

__global__ void add_kernel(size_t n, float* o, float* x1, float* x2)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        o[i] = x1[i] + x2[i];
    }
}

__global__ void mul_kernel(size_t n, float* o, float* x1, float* x2)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        o[i] = x1[i] * x2[i];
    }
}

__global__ void div_scalar_kernel(size_t n, float scalar, float* o, float* x)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        o[i] = x[i] / scalar;
    }
}

__global__ void norm_kernel(size_t n, float ss, float* o, float* x, float* weight)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        o[i] = weight[i] * (ss * x[i]);
    }
}

__global__ void add_mul_scalar_kernel(size_t n, float* scalar, float* o, float* x)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        o[i] = o[i] + (x[i] * scalar[0]);
    }
}

__global__ void silu_kernel(size_t n, float* o, float* x1, float* x2)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float val = x1[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= x2[i];
        o[i] = val;
    }
}

// Each block processes a single head
__global__ void rope_kernel(float* sq, float* sk, float* f_real, float* f_imag, int num_heads, int head_size)
{
    int h = blockIdx.x;
    float* q = sq + h * head_size;
    float* k = sk + h * head_size;

    int i = threadIdx.x * 2;
    float q0 = q[i];
    float q1 = q[i + 1];
    float k0 = k[i];
    float k1 = k[i + 1];
    float fcr = f_real[i / 2];
    float fci = f_imag[i / 2];
    q[i] = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;
    k[i] = k0 * fcr - k1 * fci;
    k[i + 1] = k0 * fci + k1 * fcr;
}

__device__ void softmax_gpu(float* __restrict__ x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_lrg>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}

__global__ void multi_head_attention_kernel(int pos, int seq_len, float* sq, float* satt, float* sxb, float* key_cache, float* value_cache, int kv_dim, int kv_mul, int head_size, int loff) 
{
    int h = blockIdx.x;
    // get the query vector for this head
    float* q = sq + h * head_size;
    // attention scores for this head
    float* att = satt + h * seq_len;
    // iterate over all timesteps, including the current one 
    // In CUDA, each thread does a small portion of the calc
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        float* k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    // above was this threads portion of the iteration.  wait for all threads to finish
    __syncthreads();

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax_gpu(att, pos + 1);
    __syncthreads();

    // weighted sum of the values, store back into xb
    // NOTE: by swapping the order of the for loops (vs. C) a simpler
    // version of the code accomplishes the same task and fits more
    // naturally with the CUDA way of subdividing the problem.
    float* xb = sxb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            val += a * v[i];
        }
        xb[i] = val;
    }
}

//------------------------------------------------------------------------
// CUDA functions
//------------------------------------------------------------------------

Primitives::~Primitives()
{
    if (m_gpu_workspace)
    {
		cudaFree(m_gpu_workspace);
		m_gpu_workspace = NULL;
	}

    if (m_host_workspace)
    {
		free(m_host_workspace);
		m_host_workspace = NULL;
	}

    if (m_cublas)
    {
		cublasDestroy(m_cublas);
		m_cublas = NULL;
	}
}

long Primitives::Initialize(size_t nMaxWorkspace, int nDim, int nNumLayers, int nNumHeads, int nSeqLen)
{
    LONG lErr;
	m_nMaxWorkspace = nMaxWorkspace;

    if (m_nMaxWorkspace > 0)
    {
        if (lErr = cudaMalloc((void**)&m_gpu_workspace, m_nMaxWorkspace * sizeof(float)))
            return lErr;

		m_host_workspace = (float*)malloc(m_nMaxWorkspace * sizeof(float));
        if (m_host_workspace == NULL)
            return ERROR_OUTOFMEMORY;
	}

    if (lErr = cublasCreate(&m_cublas))
		return lErr;

    return 0;
}

void Primitives::ToHost(size_t n, float* o, float* x, bool bSync)
{
    long lErr;

    if (lErr = cudaMemcpy(o, x, n * sizeof(float), cudaMemcpyDeviceToHost))
        throw lErr;
}


int Primitives::denan(int n, float* o, float* x, bool bSync)
{
    LONG lErr;

	denan_kernel << <divUp(n, NUM_THREADS), NUM_THREADS >> > (n, o, x, m_gpu_workspace);
	if (lErr = cudaStreamSynchronize(0))
		throw lErr;

    if (lErr = cublasSasum(m_cublas, n, m_gpu_workspace, 1, m_host_workspace))
        throw lErr;

    int nCount = (int)m_host_workspace[0];

    if (bSync)
    {
		if (lErr = cudaStreamSynchronize(0))
			return lErr;
	}

	return nCount;
}

void Primitives::rmsnorm(int n, float* o, float* x, float* weight, bool bSync)
{
    LONG lErr;

    mul_kernel << <divUp(n, NUM_THREADS), NUM_THREADS >> > (n, m_gpu_workspace, x, x);
    if (lErr = cudaStreamSynchronize(0))
        throw lErr;

    thrust::device_ptr<float> d_ptr2 = thrust::device_pointer_cast(m_gpu_workspace);
    float fSum = thrust::reduce(d_ptr2, d_ptr2 + n, 0.0f, thrust::plus<float>());

    float ss = fSum / n;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    norm_kernel << <divUp(n, NUM_THREADS), NUM_THREADS >> > (n, ss, o, x, weight);

    if (bSync)
    {
        if (lErr = cudaStreamSynchronize(0))
            throw lErr;
    }
}


void Primitives::rope(float* q, float* k, float* f_real, float* f_imag, int num_heads, int head_size, int nFreqOffset, bool bSync)
{
    f_real += nFreqOffset;
    f_imag += nFreqOffset;

    rope_kernel << <num_heads, head_size / 2 >> > (q, k, f_real, f_imag, num_heads, head_size);

    if (bSync)
    {
        long lErr;
        if (lErr = cudaStreamSynchronize(0))
            throw lErr;
    }
}


void Primitives::matmul(float* xout, float* x, float* w, int n, int d, bool bSync) {
    // W (d,n) @ x (n,) -> xout (d,)
    // W is stored in this order: (n=0,d=0), (n=1,d=0), (n=2,d=0), ... 
    // so W is n x d in cublas terms & we'll need to transpose.
    // Sgemv does y = alpha * op(A) * x + beta * y (modifying y)
    //   where op can transpose the matrix A
    // Translating to our local vars, that is
    // xout = 1.0*op(w)*x + 0.0*xout
    float alpha = 1.0f;
    float beta = 0.0f; // when this is 0, xout will not be used for input
    cublasSgemv(m_cublas, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, xout, 1);

    if (bSync)
    {
        long lErr;
        if (lErr = cudaStreamSynchronize(0))
            throw lErr;
    }
}

void Primitives::add(size_t n, float* o, float* x, bool bSync) 
{
    add_kernel << <divUp(n, NUM_THREADS), NUM_THREADS >> > (n, o, o, x);

    if (bSync)
    {
        long lErr;
        if (lErr = cudaStreamSynchronize(0))
            throw lErr;
    }
}

void Primitives::siglu(size_t n, float* o, float* x1, float* x2, bool bSync) 
{
    silu_kernel << <divUp(n, NUM_THREADS), NUM_THREADS >> > (n, o, x1, x2);

    if (bSync)
    {
        long lErr;
        if (lErr = cudaStreamSynchronize(0))
            throw lErr;
    }
}

void Primitives::softmax(int n, float* x, bool bSync)
{
    LONG lErr;

    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(x);
    thrust::device_ptr<float> max_ptr = thrust::max_element(d_ptr, d_ptr + n);
    float fMax = max_ptr[0];

    exp_kernel << <divUp(n, NUM_THREADS), NUM_THREADS >> > (n, fMax, x, x);
    if (lErr = cudaStreamSynchronize(0))
        throw lErr;

    thrust::device_ptr<float> d_ptr2 = thrust::device_pointer_cast(x);
    float fSum = thrust::reduce(d_ptr2, d_ptr2 + n, 0.0f, thrust::plus<float>());

    div_scalar_kernel << <divUp(n, NUM_THREADS), NUM_THREADS >> > (n, fSum, x, x);

    if (bSync)
    {
        long lErr;
        if (lErr = cudaStreamSynchronize(0))
            throw lErr;
    }
}

void Primitives::attention(float* xb, float* q, float* key_cache, float* value_cache, int num_heads, int head_size, size_t loff, int pos, int seq_len, int kv_dim, int kv_mul, bool bSync)
{
    long lErr;

    float* att = m_gpu_workspace;
    multi_head_attention_kernel << <num_heads, num_threads_lrg >> > (pos, seq_len, q, att, xb, key_cache, value_cache, kv_dim, kv_mul, head_size, loff);

    if (bSync)
    {
        if (lErr = cudaStreamSynchronize(0))
            throw lErr;
    }
}

// end