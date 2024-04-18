//=============================================================================
//	FILE:	attn.h
//
//	DESC:	This file manages the ATTN functinality (requires cuDnn 8.0+)
//=============================================================================
#ifndef __ATTN_CU__
#define __ATTN_CU__

#include "util.h"
#include "math.h"

//=============================================================================
//	Flags
//=============================================================================

//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Memory;

template <class T>
class blob
{
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	long m_hDesc;
	int m_nGpuID;
	bool m_bOwner;
	long m_hData;
	T* m_data;
	long m_hDiff;
	T* m_diff;
	int m_nCount;
	int m_nN;
	int m_nC;
	int m_nH;
	int m_nW;

public:
	inline blob(Memory<T>* pMem = NULL, Math<T>* pMath = NULL, int nGpuID = 0)
	{
		m_nGpuID = nGpuID;
		m_pMem = pMem;
		m_pMath = pMath;
		m_bOwner = false;
		m_hDesc = NULL;
		m_hData = NULL;
		m_data = NULL;
		m_hDiff = NULL;
		m_diff = NULL;
		m_nCount = 0;
		m_nN = 0;
		m_nC = 0;
		m_nH = 0;
		m_nW = 0;
	}

	inline ~blob()
	{
		cleanup();
	}

	void cleanup();

	LONG create(Memory<T>* pMem, Math<T>* pMath, int nGpuID, int nN, int nC, int nH, int nW, bool bData = true, bool bDiff = false, bool bCreateDesc = false);
	LONG set(long hData, long hDiff, int nN, int nC, int nH, int nW);
	LONG copy_from(blob<T>& src, bool bDiff = false);

	LONG load_from_file(const char* strFile, bool bDiff = false)
	{
		LONG lErr;
		long h = bDiff ? m_hDiff : m_hData;
		MemoryItem* pItem;

		if (lErr = m_pMem->GetMemory(h, &pItem))
			return lErr;

		return m_pMath->load_from_file(pItem, (char*)strFile, m_nCount);
	}

	LONG save_to_file(const char* strFile, bool bDiff = false)
	{
		LONG lErr;
		long h = bDiff ? m_hDiff : m_hData;
		MemoryItem* pItem;

		if (lErr = m_pMem->GetMemory(h, &pItem))
			return lErr;

		return m_pMath->save_to_file(pItem, (char*)strFile, m_nCount);
	}

	LONG compare_to_file(const char* strFile, bool bDiff = false)
	{
		LONG lErr;
		long h = bDiff ? m_hDiff : m_hData;
		MemoryItem* pItem;

		if (lErr = m_pMem->GetMemory(h, &pItem))
			return lErr;

		return m_pMath->compare_to_file(pItem, (char*)strFile, m_nCount);
	}

	float* GetHostDataAsFloat(bool bDiff = false)
	{
		long h = bDiff ? m_hDiff : m_hData;
		MemoryItem* pItem;

		if (m_pMem->GetMemory(h, &pItem))
			return NULL;

		return pItem->GetHostDataAsFloat();
	}

	inline T* data()
	{
		return m_data;
	}

	inline T* diff()
	{
		return m_diff;
	}

	inline long hdata()
	{
		return m_hData;
	}

	inline long hdiff()
	{
		return m_hDiff;
	}

	inline long desc()
	{
		return m_hDesc;
	}

	inline int count()
	{
		return m_nCount;
	}

	inline int n()
	{
		return m_nN;
	}

	inline int c()
	{
		return m_nC;
	}

	inline int h()
	{
		return m_nH;
	}

	inline int w()
	{
		return m_nW;
	}

	inline long compare_sizes(blob<T>& b, bool bOuterOnly = false, bool bTransposed = false)
	{
		if (m_nN != b.n() || m_nC != b.c())
			return ERROR_ATTN_INCOMPATIBLE_BLOB_SIZE;

		if (bOuterOnly)
			return 0;

		if (bTransposed)
		{
			if (m_nH != b.w() || m_nW != b.h())
				return ERROR_ATTN_INCOMPATIBLE_BLOB_SIZE;
		}
		else
		{
			if (m_nH != b.h() || m_nW != b.w())
				return ERROR_ATTN_INCOMPATIBLE_BLOB_SIZE;
		}

		return 0;
	
	}

	inline long reshape(int nN, int nC, int nH, int nW)
	{
		m_nN = nN;
		m_nC = nC;
		m_nH = nH;
		m_nW = nW;
		m_nCount = nN * nC * nH * nW;

		if (m_hDesc != 0)
			return set_tensor_desc(m_hDesc, nN, nC, nH, nW);

		return 0;
	}

	inline long reshape_like(blob<T>& b)
	{
		m_nN = b.n();
		m_nC = b.c();
		m_nH = b.h();
		m_nW = b.w();
		m_nCount = b.count();

		return 0;
	}

	LONG scale(T dfScale, bool bDiff)
	{
		T* x = (bDiff) ? m_diff : m_data;
		return m_pMath->scal(m_nCount, dfScale, x);
	}

	LONG transpose_hw(blob<T>& blobSrc, bool bDiff = false, bool bReshape = false);
	LONG transpose_hw2(blob<T>& blobSrc, bool bDiff = false, bool bReshape = false);
	LONG softmax_fwd(long hCuda, blob<T>& blobA);
	LONG softmax_bwd(long hCuda, blob<T>& blobA);
	LONG apply_dropout_fwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize);
	LONG apply_dropout_bwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize);
	LONG apply_mask(blob<T>& blobMask);
	LONG apply_mask_batch(blob<T>& blobMask);
	LONG matmul(blob<T>& blobA, blob<T>& blobB, double dfScale = 1.0, bool bAdiff = false, bool bBdiff = false, bool bCdiff = false, bool bTransA = false, bool bTransB = false);
	LONG matmulgrad(blob<T>& blobA, blob<T>& blobB, blob<T>& blobWork, double dfScale = 1.0);

	LONG matmul(int nOuterCount, int m, int n, int k, long hA, long hB, long hC, double dfScale = 1.0, bool bTransA = false, bool bTransB = false);

	inline long matmul(int nOuterCount, int m, int n, int k, T* a, T* b, T* c, double dfScale = 1.0, bool bTransA = false, bool bTransB = false)
	{
		int ldb = (bTransB) ? k : n;
		int lda = (bTransA) ? m : k;
		int ldc = n;
		int strideb = k * n;
		int stridea = m * k;
		int stridec = m * n;

		return m_pMath->gemm2(bTransB, bTransA, n, m, k, T(dfScale), b, a, T(0.0), c, ldb, lda, ldc, strideb, stridea, stridec, nOuterCount);
	}

	inline long set_tensor_desc(long hDesc, int n, int c, int h, int w)
	{
		n = n * c * h;
		c = w;
		h = 1;
		w = 1;

		bool bHalf = false;
		int wStride = 1;
		int hStride = w * wStride;
		int cStride = h * hStride;
		int nStride = c * cStride;

		return m_pMem->SetTensorDesc(hDesc, n, c, h, w, nStride, cStride, hStride, wStride, bHalf);
	}
};

//-----------------------------------------------------------------------------
//	ATTN Handle Class
//
//	This class stores the ATTN description information.
//-----------------------------------------------------------------------------
template <class T>
class attnHandle
{
	int m_nGpuID;
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	int m_nBatch;
	int m_nBlockSize;
	int m_nHeads;
	int m_nSize;
	double m_dfScale;

	bool m_bTraining;
	double m_dfDropout;
	unsigned long long m_lSeed;
	size_t m_stateSize;
	void* m_states;

	blob<T> m_blobWork;
	blob<T> m_blobKt;
	blob<T> m_blobAttA;
	blob<T> m_blobAttB;

	cudnnDropoutDescriptor_t m_dropoutDesc;

public:
	
	attnHandle() : m_blobWork(), m_blobKt(), m_blobAttA(), m_blobAttB()
	{
		m_pMem = NULL;
		m_pMath = NULL;
		m_nBatch = 0;
		m_nBlockSize = 0;
		m_nHeads = 0;
		m_nSize = 0;
		m_dfScale = 0;
		m_bTraining = false;
		m_dfDropout = 0.0;
		m_lSeed = 0;
		m_stateSize = 0;
		m_states = NULL;
		m_dropoutDesc = NULL;
	}

	long Initialize(Memory<T>* pMem, Math<T>* pMath)
	{
#ifndef CUDNN_8
		return ERROR_ATTN_INCOMPATIBLE_CUDNN_VER;
#endif
		m_pMem = pMem;
		m_pMath = pMath;
		return 0;
	}

	long CleanUp();

	long Set(long hCuda, int nGpuID, bool bTraining, int nBatch, int nBlockSize, int nHeads, int nSize, float fDropout, unsigned long long lSeed);

	long Forward(long hCuda, int nBlockSize, long hQ, long hK, long hV, long hMask, long hY, bool bBatchMask);

	long Backward(long hCuda, long hQ, long hdQ, long hK, long hdK, long hV, long hdV, long hMask, long hY, long hdY);


	inline long matmulgrad(int nOuterCount, int nN, int nC, int nH, int nW, int m, int n, int k, T* a, T* b, T* c, double dfScale = 1.0, bool bTransA = false, bool bTransB = false)
	{
		LONG lErr;

		if (lErr = m_pMath->transpose_hw(nN, nC, nH, nW, b, m_work))
			return lErr;
	}

};


//=============================================================================
//	Inline Methods
//=============================================================================


#endif