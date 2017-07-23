//=============================================================================
//	main.mu
//
//	The kernel manages the interface to the DLL.
//=============================================================================

//=============================================================================
//	Includes
//=============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys\timeb.h>

// includes, project
#include "main.h"


//=============================================================================
//	Methods
//=============================================================================

template <class T>
long Kernel<T>::Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount)
{
	cudaGetLastError();

	switch (lfnIdx)
	{
		case CUDA_FN_SETDEVICE:
			return m_device.SetDevice(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETRANDOMSEED:
			return m_device.SetRandomSeed(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GETDEVICE:
			return m_device.GetDevice(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_RESETDEVICE:
			return m_device.ResetDevice(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SYNCHRONIZEDEVICE:
			return m_device.SynchronizeDevice(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GETDEVICEPROP:
			return m_device.GetDeviceProperty(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHECKMEMORYATTRIB:
			return m_device.CheckMemoryAttributes(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GETDEVICEMEMORY:
			return m_device.GetDeviceMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DEVICE_CANACCESSPEER:
			return m_device.CanAccessPeer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DEVICE_ENABLEPEERACCESS:
			return m_device.EnablePeerAccess(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DEVICE_DISABLEPEERACCESS:
			return m_device.DisablePeerAccess(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ALLOCMEM:
			return m_device.AllocMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREEMEM:
			return m_device.FreeMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GETMEM:		
			return m_device.GetMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETMEM:
			return m_device.SetMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETMEMAT:
			return m_device.SetMemoryAt(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ALLOCHOSTBUFFER:
			return m_device.AllocHostBuffer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREEHOSTBUFFER:
			return m_device.FreeHostBuffer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GETHOSTMEM:
			return m_device.GetHostMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETHOSTMEM:
			return m_device.SetHostMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_MEMORYPOINTER:
			return m_device.CreateMemoryPointer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_MEMORYPOINTER:
			return m_device.FreeMemoryPointer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_STREAM:
			return m_device.CreateStream(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_STREAM:
			return m_device.FreeStream(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SYNCHRONIZE_STREAM:
			return m_device.SynchronizeStream(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SYNCHRONIZE_THREAD:
			return m_device.SynchronizeThread(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_MEMTEST:
			return m_device.CreateMemoryTest(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_MEMTEST:
			return m_device.FreeMemoryTest(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_RUN_MEMTEST:
			return m_device.RunMemoryTest(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_NCCL:
			return m_device.CreateNCCL(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_NCCL:
			return m_device.FreeNCCL(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_NCCL_INIT_SINGLEPROCESS:
			return m_device.NcclInitSingleProcess(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_NCCL_INIT_MULTIPROCESS:
			return m_device.NcclInitMultiProcess(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_NCCL_BROADCAST:
			return m_device.NcclBroadcast(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_NCCL_ALLREDUCE:
			return m_device.NcclAllReduce(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_CUDNN:
			return m_device.CreateCuDNN(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_CUDNN:	
			return m_device.FreeCuDNN(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_TENSORDESC:
			return m_device.CreateTensorDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_TENSORDESC:
			return m_device.FreeTensorDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_TENSORDESC:
			return m_device.SetTensorDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_ADD_TENSOR:
			return m_device.AddTensor(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_FILTERDESC:
			return m_device.CreateFilterDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_FILTERDESC:
			return m_device.FreeFilterDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_FILTERDESC:	
			return m_device.SetFilterDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_CONVDESC:
			return m_device.CreateConvolutionDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_CONVDESC:	
			return m_device.FreeConvolutionDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_CONVDESC:	
			return m_device.SetConvolutionDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_GET_CONVINFO:
			return m_device.GetConvolutionInfo(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FWD_CONV:
			return m_device.ConvolutionForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_BWD_CONV_BIAS:
			return m_device.ConvolutionBackwardBias(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_BWD_CONV_FILTER:
			return m_device.ConvolutionBackwardFilter(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_BWD_CONV_DATA:
			return m_device.ConvolutionBackwardData(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_POOLDESC:
			return m_device.CreatePoolingDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_POOLDESC:	
			return m_device.FreePoolingDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_POOLDESC:	
			return m_device.SetPoolingDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_POOL_FWD:	
			return m_device.PoolingForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_POOL_BWD:	
			return m_device.PoolingBackward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_GET_DROPOUT_INFO:
			return m_device.GetDropoutInfo(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_DROPOUTDESC:
			return m_device.CreateDropoutDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_DROPOUTDESC:
			return m_device.FreeDropoutDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_DROPOUTDESC:
			return m_device.SetDropoutDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_DROPOUT_FWD:
			return m_device.DropoutForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_DROPOUT_BWD:
			return m_device.DropoutBackward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_LRNDESC:
			return m_device.CreateLRNDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_LRNDESC:	
			return m_device.FreeLRNDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_LRNDESC:	
			return m_device.SetLRNDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_TANH_FWD:	
			return m_device.TanhForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_TANH_BWD:	
			return m_device.TanhBackward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SIGMOID_FWD:	
			return m_device.SigmoidForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SIGMOID_BWD:	
			return m_device.SigmoidBackward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_RELU_FWD:	
			return m_device.ReLUForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_RELU_BWD:	
			return m_device.ReLUBackward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SOFTMAX_FWD:	
			return m_device.SoftmaxForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SOFTMAX_BWD:	
			return m_device.SoftmaxBackward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_LRN_CC_FWD:
			return m_device.LRNForwardCC(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_LRN_CC_BWD:
			return m_device.LRNBackwardCC(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_LCN_CC_FWD:
			return m_device.LCNForwardCC(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_LCN_CC_BWD:
			return m_device.LCNBackwardCC(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_PCA:
			return m_device.CreatePCA(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_PCA:
			return m_device.FreePCA(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_RUN_PCA:
			return m_device.RunPCA(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_TSNE_GAUSSIAN_PERPLEXITY:			
			return m_device.CreateTsneGaussianPerplexity(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_TSNE_GAUSSIAN_PERPLEXITY:
			return m_device.FreeTsneGaussianPerplexity(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FIND_TSNE_GAUSSIAN_PERPLEXITY:
			return m_device.FindTsneGaussianPerplexity(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_TSNE:			
			return m_device.CreateTsne(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_TSNE:
			return m_device.FreeTsne(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_COMPUTE_GRADIENT1:
			return m_device.ComputeTsneGradient(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_COMPUTE_ERROR1:
			return m_device.EvaluateTsneError(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SET:
			return m_device.cuda_set(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GET:
			return m_device.cuda_get(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY:
			return m_device.cuda_copy(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GEMM:
			return m_device.cuda_gemm(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GEMM2:
			return m_device.cuda_gemm2(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GEMV:
			return m_device.cuda_gemv(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_AXPY:
			return m_device.cuda_axpy(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_AXPBY:
			return m_device.cuda_axpby(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SCAL:
			return m_device.cuda_scal(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DOT:
			return m_device.cuda_dot(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ASUM:
			return m_device.cuda_asum(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SCALE:
			return m_device.cuda_scale(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ADD_SCALAR:
			return m_device.cuda_add_scalar(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ADD:
			return m_device.cuda_add(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ADD2:
			return m_device.cuda_add2(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SUB:
			return m_device.cuda_sub(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MUL:
			return m_device.cuda_mul(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SUB_AND_DOT:
			return m_device.cuda_sub_and_dot(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MUL_SCALAR:
			return m_device.cuda_mul_scalar(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DIV:
			return m_device.cuda_div(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ABS:
			return m_device.cuda_abs(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_EXP:
			return m_device.cuda_exp(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LOG:
			return m_device.cuda_log(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_POWX:
			return m_device.cuda_powx(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGN:
			return m_device.cuda_sign(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SQRT:
			return m_device.cuda_sqrt(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_RECIPROCOL:
			return m_device.cuda_reciprocol(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_STUDENT:
			return m_device.cuda_student(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LOGISTIC1:
			return m_device.cuda_logistic1(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LOGISTIC2:
			return m_device.cuda_logistic2(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COMPARE_SIGNS:
			return m_device.cuda_compare_signs(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DENAN:
			return m_device.cuda_denan(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MAXVAL:
			return m_device.cuda_maxval(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MINVAL:
			return m_device.cuda_minval(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MINMAXVAL:
			return m_device.cuda_minmaxval(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SUMSQ:
			return m_device.cuda_sumsq(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SUMSQDIFF:
			return m_device.cuda_sumsqdiff(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_WIDTH:
			return m_device.cuda_width(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CONTAINS_POINT:
			return m_device.cuda_contains_point(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_MAX:
			return m_device.cuda_channel_max(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_SUB:
			return m_device.cuda_channel_sub(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_SUM:
			return m_device.cuda_channel_sum(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_DIV:
			return m_device.cuda_channel_div(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_MUL:
			return m_device.cuda_channel_mul(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_DOT:
			return m_device.cuda_channel_dot(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_IM2COL:
			return m_device.cuda_im2col(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_IM2COL_ND:
			return m_device.cuda_im2col_nd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COL2IM:
			return m_device.cuda_col2im(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COL2IM_ND:
			return m_device.cuda_col2im_nd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_RNG_SETSEED:
			return m_device.cuda_rng_setseed(lCount, pfInput, plCount, ppfOutput);

		case CUDA_RNG_UNIFORM:
			return m_device.cuda_rng_uniform(lCount, pfInput, plCount, ppfOutput);

		case CUDA_RNG_GAUSSIAN:
			return m_device.cuda_rng_gaussian(lCount, pfInput, plCount, ppfOutput);

		case CUDA_RNG_BERNOULLI:
			return m_device.cuda_rng_bernoulli(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_BATCHREIDX_FWD:
			return m_device.cuda_batchreidx_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_BATCHREIDX_BWD:
			return m_device.cuda_batchreidx_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_EMBED_FWD:
			return m_device.cuda_embed_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_EMBED_BWD:
			return m_device.cuda_embed_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_POOL_FWD:
			return m_device.cuda_pooling_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_POOL_BWD:
			return m_device.cuda_pooling_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_UNPOOL_FWD:
			return m_device.cuda_unpooling_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_UNPOOL_BWD:
			return m_device.cuda_unpooling_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TANH_FWD:
			return m_device.cuda_tanh_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TANH_BWD:
			return m_device.cuda_tanh_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_FWD:
			return m_device.cuda_sigmoid_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_BWD:
			return m_device.cuda_sigmoid_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_RELU_FWD:
			return m_device.cuda_relu_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_RELU_BWD:
			return m_device.cuda_relu_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ELU_FWD:
			return m_device.cuda_elu_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ELU_BWD:
			return m_device.cuda_elu_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DROPOUT_FWD:
			return m_device.cuda_dropout_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DROPOUT_BWD:
			return m_device.cuda_dropout_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_BNLL_FWD:
			return m_device.cuda_bnll_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_BNLL_BWD:
			return m_device.cuda_bnll_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_PRELU_FWD:
			return m_device.cuda_prelu_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_PRELU_BWD:
			return m_device.cuda_prelu_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_PRELU_BWD_PARAM:
			return m_device.cuda_prelu_bwd_param(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SOFTMAXLOSS_FWD:
			return m_device.cuda_softmaxloss_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SOFTMAXLOSS_BWD:
			return m_device.cuda_softmaxloss_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MAX_FWD:
			return m_device.cuda_max_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MAX_BWD:
			return m_device.cuda_max_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CROP_FWD:
			return m_device.cuda_crop_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CROP_BWD:
			return m_device.cuda_crop_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CONCAT_FWD:
			return m_device.cuda_concat_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CONCAT_BWD:
			return m_device.cuda_concat_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SLICE_FWD:
			return m_device.cuda_slice_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SLICE_BWD:
			return m_device.cuda_slice_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TILE_FWD:
			return m_device.cuda_tile_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TILE_BWD:
			return m_device.cuda_tile_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_BIAS_FWD:
			return m_device.cuda_bias_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SCALE_FWD:
			return m_device.cuda_scale_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_THRESHOLD_FWD:
			return m_device.cuda_threshold_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CLL_BWD:
			return m_device.cuda_cll_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LRN_FILLSCALE:
			return m_device.cuda_lrn_fillscale(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LRN_COMPUTEOUTPUT:
			return m_device.cuda_lrn_computeoutput(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LRN_COMPUTEDIFF:
			return m_device.cuda_lrn_computediff(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LSTM_FWD:
			return m_device.cuda_lstm_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LSTM_BWD:
			return m_device.cuda_lstm_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LSTM_UNIT_FWD:
			return m_device.cuda_lstm_unit_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LSTM_UNIT_BWD:
			return m_device.cuda_lstm_unit_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COEFF_SUM_FWD:
			return m_device.cuda_coeff_sum_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COEFF_SUM_BWD:
			return m_device.cuda_coeff_sum_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_CROSS_ENTROPY_FWD:
			return m_device.cuda_sigmoid_cross_entropy_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_CROSS_ENTROPY_IGNORE:
			return m_device.cuda_sigmoid_cross_entropy_ignore(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SGD_UPDATE:
			return m_device.cuda_sgd_update(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_NESTEROV_UPDATE:
			return m_device.cuda_nesterov_update(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ADAGRAD_UPDATE:
			return m_device.cuda_adagrad_update(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ADADELTA_UPDATE:
			return m_device.cuda_adadelta_update(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ADAM_UPDATE:
			return m_device.cuda_adam_update(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_RMSPROP_UPDATE:
			return m_device.cuda_rmsprop_update(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COMBINE_DATA:
			return m_device.cuda_combine_data(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_SET_DIAGONAL:
			return m_device.cuda_mtx_set_diagonal(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_SET_DIAGONAL2:
			return m_device.cuda_mtx_set_diagonal2(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_ADD_VECTOR:
			return m_device.cuda_mtx_add_vector(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_TRANSPOSE_OP:
			return m_device.cuda_mtx_transpose_op(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_AGGREGATE_COLS:
			return m_device.cuda_mtx_aggregate_cols(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_AGGREGATE_ROWS:
			return m_device.cuda_mtx_aggregate_rows(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_TRANSPOSE:
			return m_device.cuda_mtx_transpose(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_MEANCENTER_BY_COL:
			return m_device.cuda_mtx_meancenter_by_column(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_EUCLIDEAN_DIST:
			return m_device.cuda_mtx_euclidean_dist(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_DOT:
			return m_device.cuda_mtx_dot(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_UPDATE:
			return m_device.cuda_tsne_update(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_UPDATE_GRAD:
			return m_device.cuda_tsne_update_grad(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_COMPUTE_EXACT_ERROR:
			return m_device.cuda_tsne_compute_exact_error(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_COMPUTE_SQUARED_EUCLIDEAN_DISTANCE:
			return m_device.cuda_tsne_compute_squared_euclidean_distance(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_COMPUTE_Q_MATRIX:
			return m_device.cuda_tsne_compute_q_matrix(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_COMPUTE_EXACT_GRADIENT:
			return m_device.cuda_tsne_compute_exact_gradient(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_SYMMETRIZE_MATRIX:
			return m_device.cuda_tsne_symmetrize_matrix(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TSNE_COMPUTE_KNN_BOUNDS:
			return m_device.cuda_tsne_compute_knn_bounds(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GUASSIAN_BLUR:
			return m_device.cuda_guassian_blur(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_HAMMING_DIFF:
			return m_device.cuda_hamming_diff(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CALC_BATCH_DIST:
			return m_device.cuda_calc_batch_dist(lCount, pfInput, plCount, ppfOutput);

		default:
			return ERROR_PARAM_OUT_OF_RANGE;
	}
}

template long Kernel<double>::Run(long lfnIdx, double* pfInput, long lCount, double** ppfOutput, long* plCount);
template long Kernel<float>::Run(long lfnIdx, float* pfInput, long lCount, float** ppfOutput, long* plCount);


template <class T>
long Kernel<T>::Query(long lfnIdx, LONG* pfInput, long lCount, LPTSTR* ppOutput)
{
	cudaGetLastError();

	switch (lfnIdx)
	{
		case CUDA_FN_GET_DEVICE_NAME:
			return m_device.GetDeviceName(lCount, pfInput, ppOutput);

		case CUDA_FN_GET_P2P_INFO:
			return m_device.GetDeviceP2PInfo(lCount, pfInput, ppOutput);

		case CUDA_FN_GET_DEVICE_INFO:
			return m_device.GetDeviceInfo(lCount, pfInput, ppOutput);

		default:
			return ERROR_PARAM_OUT_OF_RANGE;
	}
}

template long Kernel<double>::Query(long lfnIdx, LONG* pfInput, long lCount, LPTSTR* ppOutput);
template long Kernel<float>::Query(long lfnIdx, LONG* pfInput, long lCount, LPTSTR* ppOutput);


//end main.cu