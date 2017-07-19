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

		long set(int nCount, long hDst, T fVal, int nIdx, int nXOff = 0);
		long get(int nCount, long hSrc, int nIdx, T* pfOutput);
		long copy(int nCount, long hSrc, long hDst, int nSrcOffset, int nDstOffset, long hAsyncStream);

		long gemm(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, T* a, T* b, T fBeta, T* c);
		long gemv(bool bTransA, int m, int n, T fAlpha, T* a, T* x, T fBeta, T* y);

		long nrm2(int n, long hA, int nAOff, T* pfResult);
		long ger(int m, int n, T fAlpha, long hA, long hB, long hC, int nAoff = 0, int nBoff = 0, int nCoff = 0);

		long gemm(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, long hA, long hB, T fBeta, long hC, int nAOff, int nBOff, int nCOff);
		long gemm2(bool bTransA, bool bTransB, int m, int n, int k, T fAlpha, long hA, long hB, T fBeta, long hC, int lda, int ldb, int ldc);
		long gemv(bool bTransA, int m, int n, T fAlpha, long hA, long hX, T fBeta, long hY, int nAOff, int nXOff, int nYOff);
		long axpy(int n, T fAlpha, long hX, long hY, int nXOff = 0, int nYOff = 0);
		long axpby(int n, T fAlpha, long hX, T fBeta, long hY);
		long scal(int n, T fAlpha, long hX, int nXOff = 0, long hAsyncStream = 0);
		long dot(int n, long hX, long hY, T* pOut, int nXOff = 0, int nYOff = 0);
		long asum(int n, long hX, T* pOut, int nXOff = 0);
		long asum(int n, T* x, T* pOut);
		long scale(int n, T fAlpha, long hX, long hY, int nXOff = 0, int nYOff = 0);
		long add_scalar(int n, T fAlpha, long hY, int nYOff = 0);
		long add(int n, long hA, long hB, long hY, T fAlpha);
		long add(int n, T* a, T* b, T* c);
		long add2(int n, long hA, long hB, long hY, T fAlphaA, T fAlphaB, int AOff = 0, int nBOff = 0, int nYOff = 0);

		long sub(int n, long hA, long hB, long hY, int nAOff = 0, int nBOff = 0, int nYOff = 0);
		long sub_and_dot(int n, int nN, int nLen, long hA, long hB, long hY, int nAOff = 0, int nBOff = 0, int nYOff = 0);
		long mul_scalar(int n, T fAlpha, long hY);
		long mul(int n, long hA, long hB, long hY, int nAOff = 0, int nBOff = 0, int nYOff = 0);
		long div(int n, long hA, long hB, long hY);
		long abs(int n, long hA, long hY);
		long exp(int n, long hA, long hY, int nAOff = 0, int nYOff = 0, T fBeta = 1);
		long log(int n, long hA, long hY, T fBeta = 1);
		long powx(int n, long hA, T fAlpha, long hY);
		long sign(int n, long hX, long hY, int nXOff = 0, int nYOff = 0);
		long sqrt(int n, long hX, long hY);
		long reciprocol(int n, long hX, long hY);
		long student(int n, long hX, long hY);
		long logistic1(int n, long hX, long hY);
		long logistic2(int n, long hX, long hY);
		long compare_signs(int n, long hA, long hB, long hY);
		long maxval(int n, long hA, T* pOut, int nAOff = 0);
		long minval(int n, long hA, T* pOut, int nAOff = 0);
		long minmaxval(int n, long hA, long hWork1, long hWork2, T* pMin, T* pMax, int nAOff = 0);
		long naninfval(int n, long hA, long hWork1, long hWork2, T* pNan, T* pInf, int nAOff = 0);
		long sumsq(int n, long hW, long hA, int nAOff, T* pOut);
		long sumsqdiff(int n, long hW, long hA, long hB, int nAOff, int nBOff, T* pOut);
		long sumsqdiff(int n, T* w, T* x, T* y, T* pOut);
		long width(int n, long hMean, long hMin, long hMax, T fAlpha, long hWidth);
		long contains_point(int n, long hMean, long hWidth, long hX, long hWork, T* pOut, int nXOff = 0);
		long denan(int n, long hX, T fReplacement);

		long channel_max(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY);
		long channel_sub(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY);
		long channel_sum(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY);
		long channel_div(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, int nMethod = 1);
		long channel_mul(int n, int nOutNum, int nChannels, int nInNum, long hX, long hY, int nMethod = 1);
		long channel_dot(int n, int nOutNum, int nChannels, int nInNum, long hX, long hA, long hY);

		long im2col(long hDataIm, int nDataImOffset, int nChannels, int nHeight, int nWidth, int nKernelH, int nKernelW, int nPadH, int nPadW, int nStrideH, int nStrideW, int nDilationH, int nDilationW, long hDataCol, int nDataColOffset);
		long col2im(long hDataCol, int nDataColOffset, int nChannels, int nHeight, int nWidth, int nKernelH, int nKernelW, int nPadH, int nPadW, int nStrideH, int nStrideW, int nDilationH, int nDilationW, long hDataIm, int nDataImOffset);
		long im2col_nd(long hDataIm, int nDataImOffset, int nNumSpatialAxes, int nImCount, int nChannelAxis, long hImShape, long hColShape, long hKernelShape, long hPad, long hStride, long hDilation, long hDataCol, int nDataColOffset);
		long col2im_nd(long hDataCol, int nDataColOffset, int nNumSpatialAxes, int nColCount, int nChannelAxis, long hImShape, long hColShape, long hKernelShape, long hPad, long hStride, long hDilation, long hDataIm, int nDataImOffset);

		long rng_setseed(long lSeed);
		long rng_uniform(int n, T fMin, T fMax, long hY);
		long rng_gaussian(int n, T fMu, T fSigma, long hY);
		long rng_bernoulli(int n, T fNonZeroProb, long hY);

		long batchreidx_fwd(int nCount, int nInnerDim, long hBottomData, long hPermutData, long hTopData);
		long batchreidx_bwd(int nCount, int nInnerDim, long hTopDiff, long hTopIdx, long hBegin, long hCounts, long hBottomDiff);

		long embed_fwd(int nCount, long hBottomData, long hWeight, int nM, int nN, int nK, long hTopData);
		long embed_bwd(int nCount, long hBottomData, long hTopDiff, int nM, int nN, int nK, long hWeightDiff);

		long pooling_fwd(int nMethod, int nCount, long hBottomData, int nNum, int nChannels, int h, int w, int hPooled, int wPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hTopData, long hMask, long hTopMask);
		long pooling_bwd(int nMethod, int nCount, long hTopDiff, int nNum, int nChannels, int h, int w, int hPooled, int wPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hBottomDiff, long hMask, long hTopMask);

		long unpooling_fwd(int nMethod, int nCount, long hBottomData, int nNum, int nChannels, int h, int w, int hUnPooled, int wUnPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hTopData, long hBottomMask);
		long unpooling_bwd(int nMethod, int nCount, long hTopDiff, int nNum, int nChannels, int h, int w, int hUnPooled, int wUnPooled, int hKernel, int wKernel, int hStride, int wStride, int hPad, int wPad, long hBottomDiff, long hBottomMask);

		long tanh_fwd(int nCount, long hBottomData, long hTopData);
		long tanh_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff);

		long sigmoid_fwd(int nCount, long hBottomData, long hTopData);
		long sigmoid_bwd(int nCount, long hTopDiff, long hTopData, long hBottomDiff);

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

		long softmaxloss_fwd(int nCount, long hProbData, long hLabels, long hLossData, int nOuterNum, int nDim, int nInnerNum, long hCounts, int nIgnoreLabel);
		long softmaxloss_bwd(int nCount, long hTopData, long hLabels, long hBottomDiff, int nOuterNum, int nDim, int nInnerNum, long hCounts, int nIgnoreLabel);

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

		long lrn_fillscale(int nCount, long hBottomData, int nNum, int nChannels, int nHeight, int nWidth, int nSize, T fA, T fB, long hScaleData);
		long lrn_computeoutput(int nCount, long hBottomData, long hScaleData, T fA, long hTopData);
		long lrn_computediff(int nCount, long hBottomData, long hTopData, long hScaleData, long hTopDiff, int nNum, int nChannels, int nHeight, int nWidth, int nSize, T fB, T fA, long hBottomDiff);

		long lstm_fwd(int t, int nN, int nH, long hWeight_h, long hWeight_i, long hClipData, int nClipOffset, long hTopData, int nTopOffset, long hCellData, int nCellOffset, long hPreGateData, int nPreGateOffset, long hGateData, int nGateOffset, long hHT1Data, int nHT1Offset, long hCT1Data, int nCT1Offset, long hHtoGateData);
		long lstm_bwd(int t, int nN, int nH, T fClip, long hWeight_h, long hClipData, int nClipOffset, long hTopDiff, int nTopOffset, long hCellData, long hCellDiff, int nCellOffset, long hPreGateDiff, int nPreGateOffset, long hGateData, long hGateDiff, int nGateOffset, long hCT1Data, int nCT1Offset, long hDHT1Diff, int nDHT1Offset, long hDCT1Diff, int nDCT1Offset, long hHtoHData);

		long lstm_unit_fwd(int nCount, int nHiddenDim, int nXCount, long hX, long hX_acts, long hC_prev, long hCont, long hC, long hH);
		long lstm_unit_bwd(int nCount, int nHiddenDim, int nXCount, long hC_prev, long hX_acts, long hC, long hH, long hCont, long hC_diff, long hH_diff, long hC_prev_diff, long hX_acts_diff, long hX_diff);

		long coeff_sum_fwd(int nCount, int nDim, int nNumOffset, T fCoeff, long hCoeffData, long hBottom, long hTop);
		long coeff_sum_bwd(int nCount, int nDim, int nNumOffset, T fCoeff, long hCoeffData, long hTopDiff, long hBottomDiff);

		long sigmoid_cross_entropy_fwd(int nCount, long hInput, long hTarget, long hLoss, bool bHasIgnoreLabel, int nIgnoreLabel, long hCount);
		long sigmoid_cross_entropy_ignore(int nCount, int nIgnoreLabel, long hTarget, long hData);

		long sgd_update(int nCount, long hNetParamDiff, long hHistoryData, T fMomentum, T fLearningRate);
		long nesterov_update(int nCount, long hNetParamDiff, long hHistoryData, T fMomentum, T fLearningRate);
		long adagrad_update(int nCount, long hNetParamDiff, long hHistoryData, T fDelta, T fLearningRate);
		long adadelta_update(int nCount, long hNetParamDiff, long hHistoryData1, long hHistoryData2, T fMomentum, T fDelta, T fLearningRate);
		long adam_update(int nCount, long hNetParamDiff, long hValM, long hValV, T fBeta1, T fBeta2, T fEpsHat, T fCorrectedLearningRate);
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
		long hamming_diff(int n, T fThreshold, long hA, long hB, long hY, int nOffA = 0, int nOffB = 0, int nOffY = 0);
};


//=============================================================================
//	Inline Methods
//=============================================================================

#endif // __DEVICE_CU__