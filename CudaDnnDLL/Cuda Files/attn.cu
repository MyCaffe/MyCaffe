//=============================================================================
//	FILE:	attn.cu
//
//	DESC:	This file implements the base class used to manage the the attn
//			functionality.
//=============================================================================

#include "util.h"
#include "memory.h"
#include "attn.h"


//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
void blob<T>::cleanup()
{
	if (m_bOwner)
	{
		if (m_hData != NULL)
			m_pMem->FreeMemory(m_hData);

		if (m_hDiff != NULL)
			m_pMem->FreeMemory(m_hDiff);
	}

	if (m_hDesc != NULL)
	{
		m_pMem->FreeTensorDesc(m_hDesc);
		m_hDesc = NULL;
	}

	m_pMem = NULL;
	m_pMath = NULL;
	m_nGpuID = 0;
	m_bOwner = false;
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

template void blob<double>::cleanup();
template void blob<float>::cleanup();


template <class T>
LONG blob<T>::create(Memory<T>* pMem, Math<T>* pMath, int nGpuID, int nN, int nC, int nH, int nW, bool bData = true, bool bDiff = false, bool bCreateDesc = false)
{
	LONG lErr;

	m_pMem = pMem;
	m_pMath = pMath;
	m_nGpuID = nGpuID;
	m_nCount = nN * nC * nH * nW;

	if (bData)
	{
		if (lErr = m_pMem->AllocMemory(m_nGpuID, false, m_nCount, NULL, 0, &m_hData))
			return lErr;

		MemoryItem* pData;

		if (lErr = pMath->GetData(m_hData, &pData))
			return lErr;

		m_data = (T*)pData->Data();
	}

	if (bDiff)
	{
		if (lErr = m_pMem->AllocMemory(m_nGpuID, false, m_nCount, NULL, 0, &m_hDiff))
		{
			cleanup();
			return lErr;
		}

		MemoryItem* pDiff;

		if (lErr = pMath->GetData(m_hDiff, &pDiff))
			return lErr;

		m_diff = (T*)pDiff->Data();
	}

	if (bCreateDesc)
	{
		if (lErr = m_pMem->CreateTensorDesc(&m_hDesc))
		{
			cleanup();
			return lErr;
		}

		if (lErr = set_tensor_desc(m_hDesc, nN, nC, nH, nW))
		{
			cleanup();
			return lErr;
		}
	}

	m_nN = nN;
	m_nC = nC;
	m_nH = nH;
	m_nW = nW;

	return 0;
}

template LONG blob<double>::create(Memory<double>* pMem, Math<double>* pMath, int nGpuID, int nN, int nC, int nH, int nW, bool bData = true, bool bDiff = false, bool bCreateDesc = false);
template LONG blob<float>::create(Memory<float>* pMem, Math<float>* pMath, int nGpuID, int nN, int nC, int nH, int nW, bool bData = true, bool bDiff = false, bool bCreateDesc = false);


template<class T>
LONG blob<T>::set(long hData, long hDiff, int nN, int nC, int nH, int nW)
{
	LONG lErr;

	m_bOwner = false;
	m_hData = hData;
	m_data = NULL;
	m_hDiff = hDiff;
	m_diff = NULL;
	m_nCount = nN * nC * nH * nW;
	m_nN = nN;
	m_nC = nC;
	m_nH = nH;
	m_nW = nW;

	if (m_hData != 0)
	{
		MemoryItem* pData;
		if (lErr = m_pMem->GetMemory(m_hData, &pData))
			return lErr;

		m_data = (T*)pData->Data();
	}

	if (m_hDiff != 0)
	{
		MemoryItem* pDiff;
		if (lErr = m_pMem->GetMemory(m_hDiff, &pDiff))
			return lErr;

		m_diff = (T*)pDiff->Data();
	}

	return 0;
}

template LONG blob<double>::set(long hData, long hDiff, int nN, int nC, int nH, int nW);
template LONG blob<float>::set(long hData, long hDiff, int nN, int nC, int nH, int nW);

template <class T>
LONG blob<T>::copy_from(blob<T>& src, bool bDiff)
{
	reshape(src.n(), src.c(), src.h(), src.w());

	long hDst = (bDiff) ? m_hDiff : m_hData;
	long hSrc = (bDiff) ? src.m_hDiff : src.m_hData;

	return m_pMath->copy(m_nCount, hSrc, hDst, 0, 0, 0);
}

template LONG blob<double>::copy_from(blob<double>& src, bool bDiff);
template LONG blob<float>::copy_from(blob<float>& src, bool bDiff);

template <class T>
LONG blob<T>::transpose_hw(blob<T>& blobSrc, bool bDiff = false, bool bReshape = false)
{
	LONG lErr;

	if (!bReshape && m_nCount != blobSrc.count())
		return ERROR_ATTN_INCOMPATIBLE_BLOB_SIZE;

	if (lErr = compare_sizes(blobSrc, true))
		return lErr;

	T* dst = (bDiff) ? m_diff : m_data;
	T* src = (bDiff) ? blobSrc.diff() : blobSrc.data();

	if (lErr = m_pMath->transpose_hw(blobSrc.n(), blobSrc.c(), blobSrc.h(), blobSrc.w(), src, dst))
		return lErr;

	if (bReshape)
		reshape(blobSrc.n(), blobSrc.c(), blobSrc.w(), blobSrc.h());

	return 0;
}

template LONG blob<double>::transpose_hw(blob<double>& blobSrc, bool bDiff = false, bool bReshape = false);
template LONG blob<float>::transpose_hw(blob<float>& blobSrc, bool bDiff = false, bool bReshape = false);

template <class T>
LONG blob<T>::transpose_hw2(blob<T>& blobSrc, bool bDiff = false, bool bReshape = false)
{
	LONG lErr;

	if (m_nCount != blobSrc.count())
		return ERROR_ATTN_INCOMPATIBLE_BLOB_SIZE;

	if (lErr = compare_sizes(blobSrc, true))
		return lErr;

	T* dst = (bDiff) ? m_diff : m_data;
	T* src = (bDiff) ? blobSrc.diff() : blobSrc.data();

	if (lErr = m_pMath->transpose_hw(blobSrc.n(), blobSrc.c(), blobSrc.h(), blobSrc.w(), src, dst))
		return lErr;

	if (bReshape)
		reshape(blobSrc.n(), blobSrc.c(), blobSrc.w(), blobSrc.h());

	return 0;
}

template LONG blob<double>::transpose_hw2(blob<double>& blobSrc, bool bDiff = false, bool bReshape = false);
template LONG blob<float>::transpose_hw2(blob<float>& blobSrc, bool bDiff = false, bool bReshape = false);


template <class T>
LONG blob<T>::softmax_fwd(long hCuda, blob<T>& blobA)
{
	LONG lErr;

	if (lErr = compare_sizes(blobA))
		return lErr;

	SoftmaxAlgorithm alg = SOFTMAX_ACCURATE;
	SoftmaxMode mode = SOFTMAX_MODE_INSTANCE;
	return m_pMem->SoftmaxForward(hCuda, alg, mode, T(1.0), m_hDesc, blobA.m_hData, T(0.0), m_hDesc, m_hData);
}

template LONG blob<double>::softmax_fwd(long hCuda, blob<double>& blobA);
template LONG blob<float>::softmax_fwd(long hCuda, blob<float>& blobA);

template <class T>
LONG blob<T>::softmax_bwd(long hCuda, blob<T>& blobA)
{
	LONG lErr;

	if (lErr = compare_sizes(blobA))
		return lErr;

	SoftmaxAlgorithm alg = SOFTMAX_ACCURATE;
	SoftmaxMode mode = SOFTMAX_MODE_INSTANCE;
	return m_pMem->SoftmaxBackward(hCuda, alg, mode, T(1.0), m_hDesc, m_hData, m_hDesc, m_hDiff, T(0.0), blobA.desc(), blobA.m_hDiff);
}

template LONG blob<double>::softmax_bwd(long hCuda, blob<double>& blobA);
template LONG blob<float>::softmax_bwd(long hCuda, blob<float>& blobA);


template <class T>
LONG blob<T>::apply_dropout_fwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize)
{
	LONG lErr;

	if (m_hDesc)
		return ERROR_ATTN_MISSING_DESCRIPTOR;

	cudnnTensorDescriptor_t attnDesc = m_pMem->GetTensorDesc(m_hDesc);
	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	if (lErr = cudnnDropoutForward(cuda, dropoutDesc, attnDesc, data(), attnDesc, data(), states, statesize))
		return lErr;

	return cudaStreamSynchronize(0);
}

template LONG blob<double>::apply_dropout_fwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize);
template LONG blob<float>::apply_dropout_fwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize);


template <class T>
LONG blob<T>::apply_dropout_bwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize)
{
	LONG lErr;

	if (m_hDesc)
		return ERROR_ATTN_MISSING_DESCRIPTOR;

	cudnnTensorDescriptor_t attnDesc = m_pMem->GetTensorDesc(m_hDesc);
	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	if (lErr = cudnnDropoutBackward(cuda, dropoutDesc, attnDesc, diff(), attnDesc, diff(), states, statesize))
		return lErr;

	return cudaStreamSynchronize(0);
}

template LONG blob<double>::apply_dropout_bwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize);
template LONG blob<float>::apply_dropout_bwd(long hCuda, cudnnDropoutDescriptor_t dropoutDesc, void* states, size_t statesize);


template <class T>
LONG blob<T>::apply_mask(blob<T>& blobMask)
{
	if (blobMask.n() != 1 || blobMask.c() != 1 || blobMask.h() != m_nH || blobMask.w() != m_nW)
		return ERROR_ATTN_INCOMPATIBLE_BLOB_SIZE;

	float fInf = -1e+29f;
	//return m_pMath->mask(count(), blobMask.count(), T(0.0), T(fInf), m_data, blobMask.data(), m_data);
	int nBatch = n();
	int nMaskDim = blobMask.count();
	return m_pMath->mask_batch(count(), nBatch, nMaskDim, T(0.0), T(fInf), m_data, blobMask.data(), m_data);
}

template LONG blob<double>::apply_mask(blob<double>& blobMask);
template LONG blob<float>::apply_mask(blob<float>& blobMask);


template <class T>
LONG blob<T>::matmul(blob<T>& blobA, blob<T>& blobB, double dfScale = 1.0, bool bAdiff = false, bool bBdiff = false, bool bCdiff = false, bool bTransA = false, bool bTransB = false)
{
	LONG lErr;

	if (lErr = compare_sizes(blobA, true, bTransA))
		return lErr;

	if (lErr = compare_sizes(blobB, true, bTransB))
		return lErr;

	int nOuter = blobA.n() * blobA.c();
	int nM = (bTransA) ? blobA.w() : blobA.h();
	int nN = (bTransB) ? blobB.h() : blobB.w();
	int nK = (bTransA) ? blobA.h() : blobA.w();

	T* a = (bAdiff) ? blobA.m_diff : blobA.data();
	T* b = (bBdiff) ? blobB.m_diff : blobB.m_data;
	T* c = (bCdiff) ? m_diff : m_data;

	if (lErr = matmul(nOuter, nM, nN, nK, a, b, c, T(dfScale), bTransA, bTransB))
		return lErr;

	return 0;

}

template LONG blob<double>::matmul(blob<double>& blobA, blob<double>& blobB, double dfScale = 1.0, bool bAdiff = false, bool bBdiff = false, bool bCdiff = false, bool bTransA = false, bool bTransB = false);
template LONG blob<float>::matmul(blob<float>& blobA, blob<float>& blobB, double dfScale = 1.0, bool bAdiff = false, bool bBdiff = false, bool bCdiff = false, bool bTransA = false, bool bTransB = false);

template <class T>
LONG blob<T>::matmul(int nOuterCount, int m, int n, int k, long hA, long hB, long hC, double dfScale = 1.0, bool bTransA = false, bool bTransB = false)
{
	int ldb = (bTransB) ? k : n;
	int lda = (bTransA) ? m : k;
	int ldc = n;
	int strideb = k * n;
	int stridea = m * k;
	int stridec = m * n;

	return m_pMath->gemm2(bTransB, bTransA, n, m, k, T(dfScale), hB, hA, T(0.0), hC, ldb, lda, ldc, strideb, stridea, stridec, nOuterCount);
}

template LONG blob<double>::matmul(int nOuterCount, int m, int n, int k, long hA, long hB, long hC, double dfScale = 1.0, bool bTransA = false, bool bTransB = false);
template LONG blob<float>::matmul(int nOuterCount, int m, int n, int k, long hA, long hB, long hC, double dfScale = 1.0, bool bTransA = false, bool bTransB = false);


template <class T>
LONG blob<T>::matmulgrad(blob<T>& blobA, blob<T>& blobB, blob<T>& blobWork, double dfScale = 1.0)
{
	LONG lErr;

	if (dfScale != 1.0)
	{
		if (lErr = scale(T(dfScale), true))
			return lErr;
	}

	if (lErr = blobA.matmul(*this, blobB, 1.0, true, false, true, false, true))
		return lErr;

	if (lErr = blobB.matmul(blobA, *this, 1.0, false, true, true, true, false))
		return lErr;

	return 0;
}

template LONG blob<double>::matmulgrad(blob<double>& blobA, blob<double>& blobB, blob<double>& blobWork, double dfScale = 1.0);
template LONG blob<float>::matmulgrad(blob<float>& blobA, blob<float>& blobB, blob<float>& blobWork, double dfScale = 1.0);


//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long attnHandle<T>::Set(long hCuda, int nGpuID, bool bTraining, int nBatch, int nBlockSize, int nHeads, int nSize, float fDropout, unsigned long long lSeed)
{
	LONG lErr;

	CleanUp();

	m_nGpuID = nGpuID;
	if (m_pMem == NULL)
		return ERROR_ATTN_NOT_INITIALIZED;

	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	m_bTraining = bTraining;
	m_dfDropout = fDropout;
	m_lSeed = lSeed;

	m_nBatch = nBatch;
	m_nBlockSize = nBlockSize;
	m_nHeads = nHeads;
	m_nSize = nSize;
	m_dfScale = 1.0 / sqrt((double)nSize);

	if (lErr = m_blobKt.create(m_pMem, m_pMath, nGpuID, nBatch, nHeads, nSize, nBlockSize, true, bTraining))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = m_blobAttA.create(m_pMem, m_pMath, nGpuID, nBatch, nHeads, nBlockSize, nBlockSize, true, bTraining, true))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = m_blobAttB.create(m_pMem, m_pMath, nGpuID, nBatch, nHeads, nBlockSize, nBlockSize, true, bTraining, true))
	{
		CleanUp();
		return lErr;
	}

	if (lErr = m_blobWork.create(m_pMem, m_pMath, nGpuID, nBatch, nHeads, nBlockSize, nBlockSize, true, bTraining))
	{
		CleanUp();
		return lErr;
	}

	//-----------------------------------------------------
	//	Setup Dropout
	//-----------------------------------------------------

	if (bTraining && fDropout > 0)
	{
		if (lErr = cudnnDropoutGetStatesSize(cuda, &m_stateSize))
		{
			CleanUp();
			return lErr;
		}

		if (lErr = cudaMalloc(&m_states, m_stateSize))
		{
			CleanUp();
			return lErr;
		}

		if (lErr = cudnnCreateDropoutDescriptor(&m_dropoutDesc))
		{
			CleanUp();
			return lErr;
		}

		if (lErr = cudnnSetDropoutDescriptor(m_dropoutDesc, cuda, (float)m_dfDropout, m_states, m_stateSize, m_lSeed))
		{
			CleanUp();
			return lErr;
		}
	}

	return cudaStreamSynchronize(0);
}

template long attnHandle<double>::Set(long hCuda, int nGpuID, bool bTraining, int nBatch, int nBlockSize, int nHeads, int nSize, float fDropout, unsigned long long lSeed);
template long attnHandle<float>::Set(long hCuda, int nGpuID, bool bTraining, int nBatch, int nBlockSize, int nHeads, int nSize, float fDropout, unsigned long long lSeed);

template <class T>
long attnHandle<T>::CleanUp()
{
	if (m_states != NULL)
	{
		cudaFree(m_states);
		m_states = NULL;
	}

	if (m_dropoutDesc != NULL)
	{
		cudnnDestroyDropoutDescriptor(m_dropoutDesc);
		m_dropoutDesc = NULL;
	}

	m_blobKt.cleanup();
	m_blobAttA.cleanup();
	m_blobAttB.cleanup();
	m_blobWork.cleanup();

	return 0;
}

template long attnHandle<double>::CleanUp();
template long attnHandle<float>::CleanUp();

template <class T>
long attnHandle<T>::Forward(long hCuda, int nBlockSize, long hQ, long hK, long hV, long hMask, long hY)
{
	LONG lErr;
	blob<T> blobQ(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobK(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobV(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobMask(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobY(m_pMem, m_pMath, m_nGpuID);

	if (m_pMem == NULL)
		return ERROR_ATTN_NOT_INITIALIZED;

	if (nBlockSize > m_nBlockSize)
		return ERROR_ATTN_INVALID_BLOCK_SIZE;

	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	if (lErr = blobQ.set(hQ, NULL, m_nBatch, m_nHeads, nBlockSize, m_nSize))
		return lErr;

	if (lErr = blobK.set(hK, NULL, m_nBatch, m_nHeads, nBlockSize, m_nSize))
		return lErr;

	if (lErr = blobV.set(hV, NULL, m_nBatch, m_nHeads, nBlockSize, m_nSize))
		return lErr;

	if (lErr = blobMask.set(hMask, NULL, 1, 1, nBlockSize, nBlockSize))
		return lErr;

	if (lErr = blobY.set(hY, NULL, m_nBatch, m_nHeads, nBlockSize, m_nSize))
		return lErr;

	// Transpose K -> Kt
	if (lErr = m_blobKt.transpose_hw(blobK, false, true))
		return lErr;

	// Matmul Qt @ Kt1 -> AttA
	m_blobAttA.reshape(m_blobAttA.n(), m_blobAttA.c(), nBlockSize, nBlockSize);
	if (lErr = m_blobAttA.matmul(blobQ, m_blobKt, m_dfScale, false, false, false))
		return lErr;

	// Apply mask to atention matrix.
	if (blobMask.data() != 0)
	{
		if (lErr = m_blobAttA.apply_mask(blobMask))
			return lErr;
	}

	// Softmax along the last dimension of AttA -> AttB
	m_blobAttB.reshape(m_blobAttB.n(), m_blobAttB.c(), nBlockSize, nBlockSize);
	if (lErr = m_blobAttB.softmax_fwd(hCuda, m_blobAttA))
		return lErr;

	// Apply dropout to AttB
	if (m_bTraining && m_dfDropout > 0)
	{
		if (lErr = m_blobAttB.apply_dropout_fwd(hCuda, m_dropoutDesc, m_states, m_stateSize))
			return lErr;
	}

	// Matmul AttB @ Vt -> Y
	if (lErr = blobY.matmul(m_blobAttB, blobV))
		return lErr;

	return cudaStreamSynchronize(0);
}

template long attnHandle<double>::Forward(long hCuda, int nBlockSize, long hQ, long hK, long hV, long hMask, long hY);
template long attnHandle<float>::Forward(long hCuda, int nBlockSize, long hQ, long hK, long hV, long hMask, long hY);

template <class T>
long attnHandle<T>::Backward(long hCuda, long hQ, long hdQ, long hK, long hdK, long hV, long hdV, long hMask, long hY, long hdY)
{
	LONG lErr;
	blob<T> blobQ(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobK(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobV(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobMask(m_pMem, m_pMath, m_nGpuID);
	blob<T> blobY(m_pMem, m_pMath, m_nGpuID);

	if (m_pMem == NULL)
		return ERROR_ATTN_NOT_INITIALIZED;

	cudnnHandle_t cuda = m_pMem->GetCuDNN(hCuda);

	if (lErr = blobQ.set(hQ, hdQ, m_nBatch, m_nHeads, m_nBlockSize, m_nSize))
		return lErr;

	if (lErr = blobK.set(hK, hdK, m_nBatch, m_nHeads, m_nBlockSize, m_nSize))
		return lErr;

	if (lErr = blobV.set(hV, hdV, m_nBatch, m_nHeads, m_nBlockSize, m_nSize))
		return lErr;

	if (lErr = blobMask.set(hMask, NULL, 1, 1, m_nBlockSize, m_nBlockSize))
		return lErr;

	if (lErr = blobY.set(hY, hdY, m_nBatch, m_nHeads, m_nBlockSize, m_nSize))
		return lErr;

	// MatmulGrad dAttB @ dVt <- dY
	if (lErr = blobY.matmulgrad(m_blobAttB, blobV, m_blobWork))
		return lErr;

//	m_pMem->SaveToNumpy("C:\\temp\\projects\\llama2\\llama2\\llama2\\test\\1.mycaffe.attb.grad.npy", m_blobAttB.hdiff(), m_blobAttB.n(), m_blobAttB.c(), m_blobAttB.h(), m_blobAttB.w());
//	m_pMem->SaveToNumpy("C:\\temp\\projects\\llama2\\llama2\\llama2\\test\\1.mycaffe.v.grad.npy", blobV.hdiff(), blobV.n(), blobV.c(), blobV.h(), blobV.w());

	// Apply dropout to dAttB
	if (m_bTraining && m_dfDropout > 0)
	{
		if (lErr = m_blobAttB.apply_dropout_bwd(hCuda, m_dropoutDesc, m_states, m_stateSize))
			return lErr;
	}

	// Softmax along the last dimension of AttA <- AttB
	if (lErr = m_blobAttB.softmax_bwd(hCuda, m_blobAttA))
		return lErr;

//	m_pMem->SaveToNumpy("C:\\temp\\projects\\llama2\\llama2\\llama2\\test\\2.mycaffe.atta.grad.npy", m_blobAttA.hdiff(), m_blobAttA.n(), m_blobAttA.c(), m_blobAttA.h(), m_blobAttA.w());

	// Matmul Qt @ dKt1 <- dAttA
	if (lErr = m_blobAttA.matmulgrad(blobQ, m_blobKt, m_blobWork, T(m_dfScale)))
		return lErr;

//	m_pMem->SaveToNumpy("C:\\temp\\projects\\llama2\\llama2\\llama2\\test\\3.mycaffe.q.grad.npy", blobQ.hdiff(), blobQ.n(), blobQ.c(), blobQ.h(), blobQ.w());
//	m_pMem->SaveToNumpy("C:\\temp\\projects\\llama2\\llama2\\llama2\\test\\3.mycaffe.kt.grad.npy", m_blobKt.hdiff(), m_blobKt.n(), m_blobKt.c(), m_blobKt.h(), m_blobKt.w());

	// Transpose dK <- dKt
	if (lErr = blobK.transpose_hw(m_blobKt, true))
		return lErr;

//	m_pMem->SaveToNumpy("C:\\temp\\projects\\llama2\\llama2\\llama2\\test\\4.mycaffe.q.grad.npy", blobK.hdiff(), blobK.n(), blobK.c(), blobK.h(), blobK.w());

	return cudaStreamSynchronize(0);
}

template long attnHandle<double>::Backward(long hCuda, long hQ, long hdQ, long hK, long hdK, long hV, long hdV, long hMask, long hY, long hdY);
template long attnHandle<float>::Backward(long hCuda, long hQ, long hdQ, long hK, long hdK, long hV, long hdV, long hMask, long hY, long hdY);

// end