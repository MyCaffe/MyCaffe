//=============================================================================
//	FILE:	Math.h
//
//	DESC:	This file implements the math operations performed on CUDA
//=============================================================================
#ifndef __MATH_CU__
#define __MATH_CU__

#include "util.h"
#include "memorycol.h"
#include "handlecol.h"


//=============================================================================
//	Flags
//=============================================================================

const int POOLING_METHOD_MAX = 0;
const int POOLING_METHOD_AVE = 1;
const int POOLING_METHOD_STO_TRAIN = 2;
const int POOLING_METHOD_STO_TEST = 3;

const int ORIENTATION_COLS = 0;
const int ORIENTATION_ROWS = 1;

const int TRANSPOSE_OP_ADD = 0;
const int TRANSPOSE_OP_MUL = 1;
const int TRANSPOSE_OP_DIV = 2;

const int AGGREGATION_SUM = 0;
const int AGGREGATION_MAX = 1;
const int AGGREGATION_MIN = 2;

const int DISTANCE_METHOD_HAMMING = 0;
const int DISTANCE_METHOD_EUCLIDEAN = 1;

const int MATH_NOP = 0;

const int MATH_ACOS = 1;
const int MATH_ACOSH = 2;
const int MATH_COS = 3;
const int MATH_COSH = 4;

const int MATH_ASIN = 10;
const int MATH_ASINH = 11;
const int MATH_SIN = 12;
const int MATH_SINH = 13;

const int MATH_ATAN = 20;
const int MATH_ATANH = 21;
const int MATH_TAN = 22;
const int MATH_TANH = 23;

const int MATH_CEIL = 30;
const int MATH_FLOOR = 31;
const int MATH_NEG = 32;
const int MATH_SIGN = 33;
const int MATH_SQRT = 34;

const int MEAN_ERROR_MSE = 1;
const int MEAN_ERROR_MAE = 2;


//=============================================================================
//	Forward References
//=============================================================================

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	Math Class
//
//	The device class implements the math operations.
//-----------------------------------------------------------------------------
template <class T>
class Math
{
	protected:
		int m_nDeviceID;
		Memory<T>* m_pMem;
		MemoryCollection* m_pMemCol;
		HandleCollection<MAX_HANDLES>* m_pStreamCol;
		cublasHandle_t m_cublas;
		curandGenerator_t m_curand;

		T get_random(T min, T max)
		{
			T fRange = max - min;
			T fRand = T(rand() / T(RAND_MAX));
			T fVal = (min + (fRange * fRand));

			return T(1.0) + T(fVal);
		}

		T get_brightness(T fBrightnessDelta)
		{
			return get_random(-fBrightnessDelta, fBrightnessDelta);
		}

		T get_contrast(T fContrastLower, T fContrastUpper)
		{
			T fContrast = get_random(fContrastLower, fContrastUpper);
			fContrast -= T(1.0);
			fContrast *= T(255.0);

			T fFactor = (T(259.0) * (fContrast + T(255.0))) / (T(255.0) * (T(259.0) - fContrast));

			return fFactor;
		}

		T get_saturation(T fSaturationLower, T fSaturationUpper)
		{
			T fGamma = get_random(fSaturationLower, fSaturationUpper);
			return T(1.0) / fGamma;
		}

	public:
		Math()
		{
			m_nDeviceID = 0;
			m_pMem = NULL;
			m_pMemCol = NULL;
			m_pStreamCol = NULL;
			m_cublas = NULL;
			m_curand = NULL;
		}

		~Math()
		{
		}

		cublasHandle_t GetCublasHandle()
		{
			return m_cublas;
		}

		void Connect(Memory<T>* pMem);

		void SetHandles(int nDeviceID, cublasHandle_t cublas, curandGenerator_t curand)
		{
			m_nDeviceID = nDeviceID;
			m_cublas = cublas;
			m_curand = curand;
		}

		long set(int nCount, T* pMem, T fVal);
		long set(int nCount, long hDst, T fVal, int nIdx, int nXOff = 0);
		long get(int nCount, long hSrc, int nIdx, T* pfOutput);
		long copy(int nCount, long hSrc, long hDst, int nSrcOffset, int nDstOffset, long hAsyncStream, int nSrcHalfSizeOverride = -1, int nDstHalfSizeOverride = -1);
		long copy_sim(int nCount, int nNum, int nDim, long hSrc1, long hSrc2, long hDst, long hSim, bool bInvert);
		long copy_fill(int n, int nDim, long hSrc, int nSrcOff, int nCount, long hDst);
		long sort(int nCount, long hY);
		long copy_batch(int n, int nNum, int nDim, long hSrcData, long hSrcLabel, int nDstCount, long hDstCache, long hWorkDevData, int nLabelStart, int nLabelCount, int nCacheSize, long hCacheHostCursors, long hWorkHostData);
		long copy_sequence(int nK, int nNum, int nDim, long hSrcData, long hSrcLabel, int nSrcCacheCount, long hSrcCache, int nLabelStart, int nLabelCount, int nCacheSize, long hCacheHostCursors, bool bOutputLabels, int nTopCount, long* rghTop, int* rgnTopCount, long hWorkHostData, bool bCombinePositiveAndNegative, int nRandomSeed);
		long copy_sequence(int n, long hSrc, int nSrcStep, int nSrcStartIdx, int nCopyCount, int nCopyDim, long hDst, int nDstStep, int nDstStartIdx, int nSrcSpatialDim, int nDstSpatialDim, int nSrcSpatialDimStartIdx, int nDstSpatialDimStartIdx,  int nSpatialDimCount);
		long copy_expand(int n, int nNum, int nDim, long hX, long hA);

		long gemm(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, __half* a, __half* b, T fBeta, __half* c, cudaStream_t stream = NULL);
		long gemm(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, T* a, T* b, T fBeta, T* c, cudaStream_t stream = NULL);
		long geam(bool bTransA, bool bTransB, int m, int n, T fAlpha, __half* a, __half* b, T fBeta, __half* c);
		long geam(bool bTransA, bool bTransB, int m, int n, T fAlpha, T* a, T* b, T fBeta, T* c);
		long gemv(bool bTransA, int m, int n, T fAlpha, T* a, T* x, T fBeta, T* y);

		long nrm2(int n, long hA, int nAOff, T* pfResult);
		long ger(int m, int n, T fAlpha, long hA, long hB, long hC, int nAoff = 0, int nBoff = 0, int nCoff = 0);

		long gemm(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, long hA, long hB, T fBeta, long hC, int nAOff, int nBOff, int nCOff, int nGroups, int nGroupAOff, int nGroupBOff, int nGroupCOff);
		long gemm2(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, long hA, long hB, T fBeta, long hC, int lda, int ldb, int ldc);
		long gemm2(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, long hA, long hB, T fBeta, long hC, int lda, int ldb, int ldc, int strideA, int nStrideB, int nStrideC, int nBatchCount);
		long geam(bool bTransA, bool bTransB, int m, int n, T fAlpha, long hA, long hB, T fBeta, long hC, int nAOff, int nBOff, int nCOff);
		long gemv(bool bTransA, int m, int n, T fAlpha, long hA, long hX, T fBeta, long hY, int nAOff, int nXOff, int nYOff);
		long axpy(int n, T fAlpha, long hX, long hY, int nXOff = 0, int nYOff = 0);
		long axpby(int n, T fAlpha, long hX, T fBeta, long hY);
		long scal(int n, T fAlpha, long hX, int nXOff = 0, long hAsyncStream = 0);
		long dot(int n, long hX, long hY, T* pOut, int nXOff = 0, int nYOff = 0);
		long asum(int n, long hX, T* pOut, int nXOff = 0);
		long asum(int n, T* x, T* pOut);
		long scale(int n, T fAlpha, long hX, long hY, int nXOff = 0, int nYOff = 0);
		long scale_to_range(int n, long hX, long hY, T fMin, T fMax);
		long erf(T fVal, T* fResult);
		long mask(int n, int nMaskDim, T fSearch, T fReplace, long hX, long hMask, long hY);
		long mask_batch(int n, int nBatch, int nMaskDim, T fSearch, T fReplace, long hX, long hMask, long hY);
		long interp2(int nC, long hX, int nX1, int nY1, int nH1, int nW1, int nH1B, int nW1B, long hY, int nX2, int nY2, int nH2, int nW2, int nH2B, int nW2B, bool bBwd);
		long add_scalar(int n, T fAlpha, long hY, int nYOff = 0);
		long add(int n, long hA, long hB, long hY, T fAlpha);
		long add(int n, T* a, T* b, T* c);
		long add2(int n, long hA, long hB, long hY, T fAlphaA, T fAlphaB, int AOff = 0, int nBOff = 0, int nYOff = 0);
		long add3(int n, long hA, long hB, long hC, long hY);

		long mulbsx(int n, long hA, int nAOff, long hX, int nXOff, int nChannels, int nSpatialDim, bool bTranspose, long hB, int nBOff);
		long divbsx(int n, long hA, int nAOff, long hX, int nXOff, int nChannels, int nSpatialDim, bool bTranspose, long hB, int nBOff);

		long sub(int n, long hA, long hB, long hY, int nAOff = 0, int nBOff = 0, int nYOff = 0, int nB = 0);
		long sub_and_dot(int n, int nN, int nLen, long hA, long hB, long hY, int nAOff = 0, int nBOff = 0, int nYOff = 0);
		long mul_scalar(int n, T fAlpha, long hY, int nYOff = 0);
		long mul(int n, long hA, long hB, long hY, int nAOff = 0, int nBOff = 0, int nYOff = 0);
		long div(int n, long hA, long hB, long hY);
		long abs(int n, long hA, long hY);
		long exp(int n, long hA, long hY, int nAOff = 0, int nYOff = 0, T fBeta = 1);
		long log(int n, long hA, long hY, T fBeta = 1, T fAlpha = 0);
		long powx(int n, long hA, T fAlpha, long hY, int nAOff = 0, int nYOff = 0);
		long sign(int n, long hX, long hY, int nXOff = 0, int nYOff = 0);
		long sqrt(int n, long hX, long hY);
		long sqrt_scale(int n, long hX, long hY);
		long reciprocol(int n, long hX, long hY);
		long student(int n, long hX, long hY);
		long logistic1(int n, long hX, long hY);
		long logistic2(int n, long hX, long hY);
		long compare_signs(int n, long hA, long hB, long hY);
		long max1(int n, long hA, long hB, long hY);
		long max_bwd2(int n, long hAdata, long hBdata, long hYdiff, long hAdiff, long hBdiff);
		long min1(int n, long hA, long hB, long hY);
		long maxval(int n, long hA, T* pOut, int nAOff = 0, long* plPos = NULL);
		long minval(int n, long hA, T* pOut, int nAOff = 0, long* plPos = NULL);
		long maxvalEx(int n, long hA, long hWork1, T* pOut, int nAOff = 0);
		long minvalEx(int n, long hA, long hWork1, T* pOut, int nAOff = 0);
		long minmaxval(int n, long hA, long hWork1, long hWork2, T* pMin, T* pMax, int nAOff = 0);
		long minmaxvec(int n, long hA, long hWork1, long hWork2, int nK, long hMin, long hMax, bool bNonZero);
		long transpose(int n, long hX, long hY, long hXCounts, long hYCounts, long hMapping, int nNumAxes, long hBuffer);
		long transpose_hw(int n, int c, int h, int w, long hSrc, long hDst);
		long transpose_hw(int n, int c, int h, int w, T* src, T* dst);
		long naninfval(int n, long hA, long hWork1, long hWork2, T* pNan, T* pInf, int nAOff = 0);
		long sumsq(int n, long hW, long hA, int nAOff, T* pOut);
		long sumsqdiff(int n, long hW, long hA, long hB, int nAOff, int nBOff, T* pOut);
		long sumsqdiff(int n, T* w, T* x, T* y, T* pOut, cudaStream_t stream = NULL);
		long sum(int n, int nOutNum, int nInNum, long hX, long hY);
		long width(int n, long hMean, long hMin, long hMax, T fAlpha, long hWidth);
		long contains_point(int n, long hMean, long hWidth, long hX, long hWork, T* pOut, int nXOff = 0);
		long denan(int n, long hX, T fReplacement);
		long set_bounds(int n, T fMin, T fMax, long hX);

		long channel_min(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, bool bReturnIdx);
		long channel_max(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, bool bReturnIdx);
		long channel_mean(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY);
		long channel_sub(int n, int nOutNum, int nChannels, int nInNum, long hA, long hX, long hY);
		long channel_sum(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, bool bSumAcrossChannels);
		long channel_div(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, int nMethod = 1);
		long channel_mul(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, int nMethod = 1);
		long channel_mulv(int n, int nOutNum, int nChannels, int nInNum, long hA, long hX, long hC);
		long channel_scale(int n, int nOutNum, int nChannels, int nInNum, long hX, long hA, long hY);
		long channel_dot(int n, int nOutNum, int nChannels, int nInNum, long hX, long hA, long hY);
		long channel_compare(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY);
		long channel_fill(int n, int nOutNum, int nChannels, int nInNum, long hX, int nLabelDim, long hLabels, long hY);
		long channel_fillfrom(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, int nDir);
		long channel_copy(int n, int nOutNum, int nChannels, int nBlocks, int nInNum, int nOffset, long hX, long hY, int nDir);
		long channel_add(int n, int nOutNum, int nChannels, int nBlocks, int nInNum, int nOffset, long hX, long hY, int nDir);
		long channel_copyall(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY);
		long channel_duplicate(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY);
		long channel_percentile(int n, int nOuterNum, int nChannels, int nInnerNum, long hX, long hY, T fPercentile);

		long im2col(long hDataIm, int nDataImOffset, int nChannels, int nHeight, int nWidth, int nKernelH, int nKernelW, int nPadH, int nPadW, int nStrideH, int nStrideW, int nDilationH, int nDilationW, long hDataCol, int nDataColOffset);
		long col2im(long hDataCol, int nDataColOffset, int nChannels, int nHeight, int nWidth, int nKernelH, int nKernelW, int nPadH, int nPadW, int nStrideH, int nStrideW, int nDilationH, int nDilationW, long hDataIm, int nDataImOffset);
		long im2col_nd(long hDataIm, int nDataImOffset, int nNumSpatialAxes, int nImCount, int nChannelAxis, long hImShape, long hColShape, long hKernelShape, long hPad, long hStride, long hDilation, long hDataCol, int nDataColOffset);
		long col2im_nd(long hDataCol, int nDataColOffset, int nNumSpatialAxes, int nColCount, int nChannelAxis, long hImShape, long hColShape, long hKernelShape, long hPad, long hStride, long hDilation, long hDataIm, int nDataImOffset);

		long rng_setseed(long lSeed);
		long rng_uniform(int n, T fMin, T fMax, long hY);
		long rng_gaussian(int n, T fMu, T fSigma, long hY);
		long rng_bernoulli(int n, T fNonZeroProb, long hY);
		long rng_uniform(int n, T fMin, T fMax, T* pMem);
		long rng_gaussian(int n, T fMu, T fSigma, T* pMem, size_t sz);

		long accuracy_fwd(int nCount, int nOuterNum, int nInnerNum, long hBtmData, long hBtmLabel, long hAccData, long hAccTotals, bool bIgnoreLabel, int nIgnoreLabel, bool bLastElementOnly, int nBatch);

		long batchreidx_fwd(int nCount, int nInnerDim, long hBottomData, long hPermutData, long hTopData);
		long batchreidx_bwd(int nCount, int nInnerDim, long hTopDiff, long hTopIdx, long hBegin, long hCounts, long hBottomDiff);

		long embed_fwd(int nCount, long hBottomData, long hWeight, int nM, int nN, int nK, long hTopData);
		long embed_bwd(int nCount, long hBottomData, long hTopDiff, int nM, int nN, int nK, long hWeightDiff);

		long pooling_fwd(int nMethod, int nCount, long hBottomData, int nNum, int nChannels, int h, int w, int hPooled, int wPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hTopData, long hMask, long hTopMask);
		long pooling_bwd(int nMethod, int nCount, long hTopDiff, int nNum, int nChannels, int h, int w, int hPooled, int wPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hBottomDiff, long hMask, long hTopMask);

		long unpooling_fwd(int nMethod, int nCount, long hBottomData, int nNum, int nChannels, int h, int w, int hUnPooled, int wUnPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hTopData, long hBottomMask);
		long unpooling_bwd(int nMethod, int nCount, long hTopDiff, int nNum, int nChannels, int h, int w, int hUnPooled, int wUnPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hBottomDiff, long hBottomMask);

		long clip_fwd(int nCount, long hBottomData, long hTopData, T fMin, T fMax);
		long clip_bwd(int nCount, long hTopDiff, long hBottomData, long hBottomDiff, T fMin, T fMax);

		long math_fwd(int nCount, long hBottomData, long hTopData, int nFunction);
		long math_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff, long hBottomData, int nFunction);

		long tanh_fwd(int nCount, long hBottomData, long hTopData);
		long tanh_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff);

		long mean_error_loss_bwd(int nCount, long hPredicted, long hTarget, long hBottomDiff, int nMeanErr);

		long mish_fwd(int nCount, long hBottomData, long hTopData, T fThreshold);
		long mish_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff, long hBottomData, T fThreshold, int nMethod);

		long gelu_fwd(int nCount, long hBottomData, long hTopData, bool bUseBertVersion);
		long gelu_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff, long hBottomData, bool bUseBertVersion);

		long serf_fwd(int nCount, long hBottomData, long hTopData, T fThreshold);
		long serf_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff, long hBottomData, T fThreshold);

		long sigmoid_fwd(int nCount, long hBottomData, long hTopData);
		long sigmoid_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff);

		long swish_bwd(int nCount, long hTopDiff, long hTopData, long hSigmoidOutputData, long hBottomDiff, T fBeta);

		long relu_fwd(int nCount, long hBottomData, long hTopData, T fNegativeSlope);
		long relu_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff, T fNegativeSlope);

		long elu_fwd(int nCount, long hBottomData, long hTopData, T fAlpha);
		long elu_bwd(int nCount, long hTopDiff, long hTopData, long hBottomData, long hBottomDiff, T fAlpha);

		long dropout_fwd(int nCount, long hBottomData, long hMask, unsigned int uiThreshold, T fScale, long hTopData);
		long dropout_bwd(int nCount, long hTopDiff, long hMask, unsigned int uiThreshold, T fScale, long hBottomDiff);

		long bnll_fwd(int nCount, long hBottomData, long hTopData);
		long bnll_bwd(int nCount, long hTopDiff, long hBottomData, long hBottomDiff);

		long prelu_fwd(int nCount, int nChannels, int nDim, long hBottomData, long hTopData, long hSlopeData, int nDivFactor);
		long prelu_bwd(int nCount, int nChannels, int nDim, long hTOpDiff, long hBottomData, long hBottomDiff, long hSlopeData, int nDivFactor);
		long prelu_bwd_param(int nCDim, int nNum, int nTopOffset, long hTopDiff, long hBottomData, long hBackBuffDiff);

		long nllloss_fwd(int nCount, long hProbData, long hLabels, long hLossData, int nOuterNum, int nDim, int nInnerNum, long hCounts, int nIgnoreLabel);
		long nllloss_bwd(int nCount, long hTopData, long hLabels, long hBottomDiff, int nOuterNum, int nDim, int nInnerNum, long hCounts, int nIgnoreLabel);
		
		long softmaxloss_fwd(int nCount, long hProbData, long hLabels, long hLossData, int nOuterNum, int nDim, int nInnerNum, long hCounts, int nIgnoreLabel);
		long softmaxloss_bwd(int nCount, long hTopData, long hLabels, long hBottomDiff, int nOuterNum, int nDim, int nInnerNum, long hCounts, int nIgnoreLabel);

		long min_fwd(int nCount, long hA, long hB, int nIdx, long hY, long hMask);
		long min_bwd(int nCount, long hX, int nIdx, long hMask, long hY);
		long max_fwd(int nCount, long hA, long hB, int nIdx, long hY, long hMask);
		long max_bwd(int nCount, long hX, int nIdx, long hMask, long hY);

		long crop_fwd(int nCount, int nNumAxes, long hSrcStrides, long hDstStrides, long hOffsets, long hBottomData, long hTopData);
		long crop_bwd(int nCount, int nNumAxes, long hSrcStrides, long hDstStrides, long hOffsets, long hBottomDiff, long hTopDiff);

		long concat_fwd(int nCount, long hBottomData, int nNumConcats, int nConcatInputSize, int nTopConcatAxis, int nBottomConcatAxis, int nOffsetConcatAxis, long hTopData);
		long concat_bwd(int nCount, long hTopDiff, int nNumConcats, int nConcatInputSize, int nTopConcatAxis, int nBottomConcatAxis, int nOffsetConcatAxis, long hBottomDiff);

		long slice_fwd(int nCount, long hBottomData, int nNumSlices, int nSliceInputSize, int nBottomSliceAxis, int nTopSliceAxis, int nOffsetSliceAxis, long hTopData);
		long slice_bwd(int nCount, long hTopDiff, int nNumSlices, int nSliceInputSize, int nBottomSliceAxis, int nTopSliceAxis, int nOffsetSliceAxis, long hBottomDiff);

		long tile_fwd(int nCount, long hBottomData, int nInnerDim, int nTiles, int nBottomTileAxis, long hTopData);
		long tile_bwd(int nCount, long hTopDiff, int nTileSize, int nTiles, int nBottomTileAxis, long hBottomDiff);

		long bias_fwd(int nCount, long hBottomData, long hBiasData, int nBiasDim, int nInnerDim, long hTopData);

		long scale_fwd(int nCount, long hX, long hScaleData, int nScaleDim, int nInnerDim, long hY, long hBiasData = 0);

		long threshold_fwd(int nCount, T fThreshold, long hX, long hY);

		long cll_bwd(int nCount, int nChannels, T fMargin, bool bLegacyVersion, T fAlpha, long hY, long hDiff, long hDistSq, long hBottomDiff);

		long smoothl1_fwd(int nCount, long hX, long hY);
		long smoothl1_bwd(int nCount, long hX, long hY);

		long permute(int nCount, long hX, bool bFwd, long hPermuteOrder, long hOldSteps, long hNewSteps, int nNumAxes, long hY);

		long gather_fwd(int nCount, long hX, long hY, int nAxis, int nDim, int nDimAtAxis, int nM, int nN, long hIdx);
		long gather_bwd(int nCount, long hX, long hY, int nAxis, int nDim, int nDimAtAxis, int nM, int nN, long hIdx);

		long lrn_fillscale(int nCount, long hBottomData, int nNum, int nChannels, int nHeight, int nWidth, int nSize, T fA, T fB, long hScaleData);
		long lrn_computeoutput(int nCount, long hBottomData, long hScaleData, T fA, long hTopData);
		long lrn_computediff(int nCount, long hBottomData, long hTopData, long hScaleData, long hTopDiff, int nNum, int nChannels, int nHeight, int nWidth, int nSize, T fB, T fA, long hBottomDiff);

		long lstm_fwd(int t, int nN, int nH, int nI, long hWeight_h, long hWeight_i, long hClipData, int nClipOffset, long hTopData, int nTopOffset, long hCellData, int nCellOffset, long hPreGateData, int nPreGateOffset, long hGateData, int nGateOffset, long hHT1Data, int nHT1Offset, long hCT1Data, int nCT1Offset, long hHtoGateData, long hContext, long hWeight_c, long hCtoGateData);
		long lstm_bwd(int t, int nN, int nH, int nI, T fClip, long hWeight_h, long hClipData, int nClipOffset, long hTopDiff, int nTopOffset, long hCellData, long hCellDiff, int nCellOffset, long hPreGateDiff, int nPreGateOffset, long hGateData, long hGateDiff, int nGateOffset, long hCT1Data, int nCT1Offset, long hDHT1Diff, int nDHT1Offset, long hDCT1Diff, int nDCT1Offset, long hHtoHData, long hContextDiff, long hWeight_c);

		long lstm_unit_fwd(int nCount, int nHiddenDim, int nXCount, long hX, long hX_acts, long hC_prev, long hCont, long hC, long hH);
		long lstm_unit_bwd(int nCount, int nHiddenDim, int nXCount, long hC_prev, long hX_acts, long hC, long hH, long hCont, long hC_diff, long hH_diff, long hC_prev_diff, long hX_acts_diff, long hX_diff);

		long coeff_sum_fwd(int nCount, int nDim, int nNumOffset, T fCoeff, long hCoeffData, long hBottom, long hTop);
		long coeff_sum_bwd(int nCount, int nDim, int nNumOffset, T fCoeff, long hCoeffData, long hTopDiff, long hBottomDiff);
		long coeff_sub_fwd(int nCount, int nDim, int nNumOffset, T fCoeff, long hCoeffData, long hBottom, long hTop);
		long coeff_sub_bwd(int nCount, int nDim, int nNumOffset, T fCoeff, long hCoeffData, long hTopDiff, long hBottomDiff);

		long sigmoid_cross_entropy_fwd(int nCount, long hInput, long hTarget, long hLoss, bool bHasIgnoreLabel, int nIgnoreLabel, long hCount);
		long sigmoid_cross_entropy_bwd(int nCount, int nIgnoreLabel, long hTarget, long hData);
		long softmax_cross_entropy_fwd(int nCount, long hProbData, long hTarget, long hLossDiff, long hLossData, int nOuterNum, int nDim, int nInnerNum, long hCounts, int nIgnoreLabel);
		long softmax_cross_entropy_bwd(int nCount, int nIgnoreLabel, long hTarget, long hData);

		long sgd_update(int nCount, long hNetParamDiff, long hHistoryData, T fMomentum, T fLearningRate);
		long nesterov_update(int nCount, long hNetParamDiff, long hHistoryData, T fMomentum, T fLearningRate);
		long adagrad_update(int nCount, long hNetParamDiff, long hHistoryData, T fDelta, T fLearningRate);
		long adadelta_update(int nCount, long hNetParamDiff, long hHistoryData1, long hHistoryData2, T fMomentum, T fDelta, T fLearningRate);
		long adam_update(int nCount, long hNetParamDiff, long hValM, long hValV, T fBeta1, T fBeta2, T fEpsHat, T fLearningRate, T fCorrection);
		long adamw_update(int nCount, long hNetParamDiff, long hValM, long hValV, T fBeta1, T fBeta2, T fEpsHat, T fLearningRate, T fDecayRate, long hNetParamData, int nStep);
		long rmsprop_update(int nCount, long hNetParamDiff, long hHistoryData, T fRmsDecay, T fDelta, T fLearningRate);

		long combine_data(int nCount, long hOriginal, long hUpdated, T fUpdatedPct, long hServer, T fServerPct, long hNewData);

		long mtx_set_diagonal(int nCount, int nRows, T fVal, long hData);
		long mtx_set_diagonal(int nCount, int nRows, long hDiagonal, T fScaleA, T fScaleB, long hData);
		long mtx_add_vector(int nOrientation, int nWidth, int nHeight, T fScale, long hA, long hB, long hY);
		long mtx_transpose_op(int nOp, int nWidth, int nHeight, long hA, long hB, long hY, T fScaleA, T fScaleB);
		long mtx_aggregate_cols(int nOp, int nWidth, int nHeight, long hA, long hY);
		long mtx_aggregate_rows(int nOp, int nWidth, int nHeight, long hA, long hOnes, long hY);
		long mtx_aggregate_rows_sum(int nWidth, int nHeight, T* a, T* ones, T* y);
		long mtx_transpose(int nWidth, int nHeight, long hA, long hY);
		long mtx_meancenter_by_column(int nWidth, int nHeight, long hA, long hB, long hY, bool bNormalize);
		long mtx_euclidean_dist(long hX, long hY, long hOut, int n, int d, int nStart, int nEnd);
		long mtx_mean(int nWidth, int nHeight, long hA, long hOnes, T fAlpha, long hY);
		long mtx_stdev(int nWidth, int nHeight, long hA, long hOnes, long hMean, long hWork, long hY);
		long mtx_correlation(int nWidth, int nHeight, long hA, long hOnes, long hMean, long hStdev, long hWork, long hY);

		// hC(k rows x m cols) = hA(m rows x n cols) * hB(n rows x k cols)
		long mtx_dot(int m, int n, int k, long hA, long hB, long hC);

		long tsne_update(unsigned int n, T fMomentum, T fLearningRate, long hdY, long huY, long hGains, long hY, T fGainFactor1, T fGainFactor2);
		long tsne_update_grad(unsigned int n, long hPosF, long hNegF, T fSumQ, long hdC);
		long tsne_compute_squared_euclidean_distance(unsigned int n, unsigned int d, long hW, long hX, T* pDD_on_host);
		long tsne_compute_squared_euclidean_distance(unsigned int n, unsigned int d, T* pX_on_host, T* pDD_on_host);
		long tsne_compute_q_matrix(unsigned int n, T* pDD_on_host, T* pQ_on_host, T* pfSumQ);
		long tsne_compute_exact_gradient(unsigned int n, unsigned int d, T* pY_on_host, T* pP_on_host, T* pQ_on_host, T* pdC_on_host, T fSumQ);
		long tsne_compute_exact_error(unsigned int n, long hP, long hQ, long hY);
		long tsne_symmetrize_matrix(unsigned int n, long hRowP, long hColP, long hValP, unsigned int* pnRowCount);
		long tsne_compute_knn_bounds(unsigned int n, long hData, T fPctInCircle, T* pfMinX, T* pfMinY, T* pfMaxX, T* pfMaxY);

		long gaussian_blur(int n, int c, int h, int w, T fSigma, long hX, long hY);
		long calc_dft(int n, long hX, int m, long hY);
		long hamming_diff(int n, T fThreshold, long hA, long hB, long hY, int nOffA = 0, int nOffB = 0, int nOffY = 0);
		long calc_batch_dist(int nDistMethod, T fThreshold, int nItemDim, long hS, long hT, long hW, const int nDim0, const int nDim1, LONGLONG* rgOffsets, T* rgDist);
};


//=============================================================================
//	Inline Methods
//=============================================================================

#endif // __DEVICE_CU__