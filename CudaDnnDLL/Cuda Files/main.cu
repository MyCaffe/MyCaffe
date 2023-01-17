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
#ifdef _TRACEAPI
#ifdef _TRACETOFILE
#include <iostream>
#include <fstream>
#endif // _TRACETOFILE
#endif // _TRACEAPI

// includes, project
#include "main.h"

#ifdef _DEBUG
#ifdef _TRACEAPI
static char s_msgbuf[256];

char* GetApiName(long lfnIdx);
#endif
#endif

//=============================================================================
//	Methods
//=============================================================================

template <class T>
long Kernel<T>::Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount)
{
	cudaGetLastError();

#ifdef _DEBUG
#ifdef _TRACEAPI
	snprintf(s_msgbuf, 256, "%s calling CudaDnnDLL FunctionID (%ld) %s\n", (sizeof(T) == sizeof(float)) ? "FLOAT" : "DOUBLE", lfnIdx, GetApiName(lfnIdx));
	OutputDebugStringA(s_msgbuf);
#ifdef _TRACETOFILE
	std::ofstream myfile;
	myfile.open("c:\\temp\\mycaffe_debug.txt", std::ofstream::out | std::ofstream::app);
	myfile << s_msgbuf;
	myfile.close();
#endif
#endif
#endif

	if (!m_device.IsInitialized())
		return ERROR_DEVICE_NOT_INITIALIZED;

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

		case CUDA_FN_GET_REQUIRED_COMPUTE:
			return m_device.GetRequiredCompute(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_ALLOCMEM_HALF:
			return m_device.AllocMemoryHalf(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREEMEM:
			return m_device.FreeMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GETMEM:		
			return m_device.GetMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETMEM:
			return m_device.SetMemory(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETMEMAT:
			return m_device.SetMemoryAt(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETPIXEL:
			return m_device.SetPixel(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY_DEVICE_TO_HOST:
			return m_device.CopyGpuToHostBuffer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY_HOST_TO_DEVICE:
			return m_device.CopyHostBufferToGpu(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ALLOCHOSTBUFFER:
			return m_device.AllocHostBuffer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREEHOSTBUFFER:
			return m_device.FreeHostBuffer(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GETHOSTBUFFERCAPACITY:
			return m_device.GetHostBufferCapacity(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_CREATE_IMAGEOP:
			return m_device.CreateImageOp(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_IMAGEOP:
			return m_device.FreeImageOp(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DISTORTIMAGE_IMAGEOP:
			return m_device.DistortImage(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_FREE_EXTENSION:
			return m_device.FreeExtension(lCount, pfInput, plCount, ppfOutput);

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

		case CUDNN_FN_SET_TENSORNDDESC:
			return m_device.SetTensorNdDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_ADD_TENSOR:
			return m_device.AddTensor(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_FILTERDESC:
			return m_device.CreateFilterDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_FILTERDESC:
			return m_device.FreeFilterDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_FILTERDESC:	
			return m_device.SetFilterDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_FILTERNDDESC:
			return m_device.SetFilterNdDesc(lCount, pfInput, plCount, ppfOutput);

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

		case CUDNN_FN_DERIVE_BN:
			return m_device.DeriveBatchNormDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_BN_FWD:
			return m_device.BatchNormForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_BN_BWD:
			return m_device.BatchNormBackward(lCount, pfInput, plCount, ppfOutput);

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

		case CUDNN_FN_ELU_FWD:
			return m_device.EluForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_ELU_BWD:
			return m_device.EluBackward(lCount, pfInput, plCount, ppfOutput);

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

		case CUDNN_FN_CREATE_RNN_DATA_DESC:
			return m_device.CreateRnnDataDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_RNN_DATA_DESC:
			return m_device.FreeRnnDataDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_RNN_DATA_DESC:
			return m_device.SetRnnDataDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_RNN_DATA_DESCEX:
			return m_device.CreateRnnDataDescEx(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_RNN_DATA_DESCEX:
			return m_device.FreeRnnDataDescEx(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_RNN_DATA_DESCEX:
			return m_device.SetRnnDataDescEx(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_CREATE_RNN_DESC:
			return m_device.CreateRnnDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_FREE_RNN_DESC:
			return m_device.FreeRnnDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_SET_RNN_DESC:
			return m_device.SetRnnDesc(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_GET_RNN_PARAMCOUNT:
			return m_device.GetRnnParamCount(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_GET_RNN_WORKSPACECOUNT:
			return m_device.GetRnnWorkspaceCount(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_GET_RNN_LINLAYERPARAMS:
			return m_device.GetRnnLinLayerParams(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_RNN_FWD:
			return m_device.RnnForward(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_RNN_BWD_DATA:
			return m_device.RnnBackwardData(lCount, pfInput, plCount, ppfOutput);

		case CUDNN_FN_RNN_BWD_WTS:
			return m_device.RnnBackwardWeights(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_COPY_SIM:
			return m_device.cuda_copy_sim(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY_FILL:
			return m_device.cuda_copy_fill(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SORT:
			return m_device.cuda_sort(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY_BATCH:
			return m_device.cuda_copy_batch(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY_SEQUENCE:
			return m_device.cuda_copy_sequence(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY_SEQUENCE2:
			return m_device.cuda_copy_sequence2(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COPY_EXPAND:
			return m_device.cuda_copy_expand(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GEMM:
			return m_device.cuda_gemm(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GEMM2:
			return m_device.cuda_gemm2(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GEMV:
			return m_device.cuda_gemv(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GEAM:
			return m_device.cuda_geam(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GER:
			return m_device.cuda_ger(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SET_BOUNDS:
			return m_device.cuda_set_bounds(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_AXPY:
			return m_device.cuda_axpy(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_AXPBY:
			return m_device.cuda_axpby(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MULBSX:
			return m_device.cuda_mulbsx(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DIVBSX:
			return m_device.cuda_divbsx(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SCAL:
			return m_device.cuda_scal(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DOT:
			return m_device.cuda_dot(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ASUM:
			return m_device.cuda_asum(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SCALE:
			return m_device.cuda_scale(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SCALE_TO_RANGE:
			return m_device.cuda_scale_to_range(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_ERF:
			return m_device.cuda_erf(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MASK:
			return m_device.cuda_mask(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MASK_BATCH:
			return m_device.cuda_mask_batch(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_INTERP2:
			return m_device.cuda_interp2(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_MINMAXVEC:
			return m_device.cuda_minmaxvec(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TRANSPOSE:
			return m_device.cuda_transpose(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SUMSQ:
			return m_device.cuda_sumsq(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SUMSQDIFF:
			return m_device.cuda_sumsqdiff(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SUM:
			return m_device.cuda_sum(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SQRT_SCALE:
			return m_device.cuda_sqrt_scale(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_WIDTH:
			return m_device.cuda_width(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CONTAINS_POINT:
			return m_device.cuda_contains_point(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_MIN:
			return m_device.cuda_channel_min(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_CHANNEL_MULV:
			return m_device.cuda_channel_mulv(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_SCALE:
			return m_device.cuda_channel_scale(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_DOT:
			return m_device.cuda_channel_dot(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_COMPARE:
			return m_device.cuda_channel_compare(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_FILL:
			return m_device.cuda_channel_fill(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_FILLFROM:
			return m_device.cuda_channel_fillfrom(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_COPY:
			return m_device.cuda_channel_copy(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CHANNEL_COPYALL:
			return m_device.cuda_channel_copyall(lCount, pfInput, plCount, ppfOutput);
			
		case CUDA_FN_CHANNEL_DUPLICATE:
			return m_device.cuda_channel_duplicate(lCount, pfInput, plCount, ppfOutput);
			
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

		case CUDA_FN_ACCURACY_FWD:
			return m_device.cuda_accuracy_fwd(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_CLIP_FWD:
			return m_device.cuda_clip_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CLIP_BWD:
			return m_device.cuda_clip_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TANH_FWD:
			return m_device.cuda_tanh_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_TANH_BWD:
			return m_device.cuda_tanh_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MEAN_ERROR_LOSS_BWD:
			return m_device.cuda_mean_error_loss_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MISH_FWD:
			return m_device.cuda_mish_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MISH_BWD:
			return m_device.cuda_mish_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GELU_FWD:
			return m_device.cuda_gelu_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GELU_BWD:
			return m_device.cuda_gelu_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_FWD:
			return m_device.cuda_sigmoid_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_BWD:
			return m_device.cuda_sigmoid_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SWISH_BWD:
			return m_device.cuda_swish_bwd(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_NLLLOSS_FWD:
			return m_device.cuda_nllloss_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_NLLLOSS_BWD:
			return m_device.cuda_nllloss_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SOFTMAXLOSS_FWD:
			return m_device.cuda_softmaxloss_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SOFTMAXLOSS_BWD:
			return m_device.cuda_softmaxloss_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MIN_FWD:
			return m_device.cuda_min_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MIN_BWD:
			return m_device.cuda_min_bwd(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_SMOOTHL1_FWD:
			return m_device.cuda_smoothl1_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SMOOTHL1_BWD:
			return m_device.cuda_smoothl1_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SERF_FWD:
			return m_device.cuda_serf_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SERF_BWD:
			return m_device.cuda_serf_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_PERMUTE:
			return m_device.cuda_permute(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GATHER_FWD:
			return m_device.cuda_gather_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_GATHER_BWD:
			return m_device.cuda_gather_bwd(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_MATH_FWD:
			return m_device.cuda_math_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MATH_BWD:
			return m_device.cuda_math_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COEFF_SUM_FWD:
			return m_device.cuda_coeff_sum_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COEFF_SUM_BWD:
			return m_device.cuda_coeff_sum_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COEFF_SUB_FWD:
			return m_device.cuda_coeff_sub_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_COEFF_SUB_BWD:
			return m_device.cuda_coeff_sub_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_CROSS_ENTROPY_FWD:
			return m_device.cuda_sigmoid_cross_entropy_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SIGMOID_CROSS_ENTROPY_BWD:
			return m_device.cuda_sigmoid_cross_entropy_bwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SOFTMAX_CROSS_ENTROPY_FWD:
			return m_device.cuda_softmax_cross_entropy_fwd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SOFTMAX_CROSS_ENTROPY_BWD:
			return m_device.cuda_softmax_cross_entropy_bwd(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_ADAMW_UPDATE:
			return m_device.cuda_adamw_update(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_MTX_MEAN:
			return m_device.cuda_mtx_mean(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_STDEV:
			return m_device.cuda_mtx_stdev(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_MTX_CORRELATION:
			return m_device.cuda_mtx_correlation(lCount, pfInput, plCount, ppfOutput);

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

		case CUDA_FN_CALC_DFT:
			return m_device.cuda_calc_dft(lCount, pfInput, plCount, ppfOutput);
	
		case CUDA_FN_HAMMING_DIFF:
			return m_device.cuda_hamming_diff(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CALC_BATCH_DIST:
			return m_device.cuda_calc_batch_dist(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_SSD:
			return m_device.CreateSsd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_SSD:
			return m_device.FreeSsd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SETUP_SSD:
			return m_device.SetupSsd(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SSD_FWD_MULTIBOXLOSS:
			return m_device.SsdMultiboxLossForward(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SSD_ENCODE_LOCPRED:
			return m_device.SsdEncodeLocPrediction(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_SSD_ENCODE_CONFPRED:
			return m_device.SsdEncodeConfPrediction(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_CREATE_LAYERNORM:
			return m_device.CreateLayerNorm(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_FREE_LAYERNORM:
			return m_device.FreeLayerNorm(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LAYERNORM_FWD:
			return m_device.LayerNormForward(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_LAYERNORM_BWD:
			return m_device.LayerNormBackward(lCount, pfInput, plCount, ppfOutput);

		case CUDA_FN_DEBUG:
			return 0;

		default:
			return ERROR_PARAM_OUT_OF_RANGE;
	}
}

template long Kernel<double>::Run(long lfnIdx, double* pfInput, long lCount, double** ppfOutput, long* plCount);
template long Kernel<float>::Run(long lfnIdx, float* pfInput, long lCount, float** ppfOutput, long* plCount);


template <class T>
long Kernel<T>::Run(long lfnIdx, T* pfInput, long lCount, LPTSTR pszInput, T** ppfOutput, long* plCount)
{
	cudaGetLastError();

#ifdef _DEBUG
#ifdef _TRACEAPI
	snprintf(s_msgbuf, 256, "%s calling CudaDnnDLL FunctionID (%ld) %s\n", (sizeof(T) == sizeof(float)) ? "FLOAT" : "DOUBLE", lfnIdx, GetApiName(lfnIdx));
	OutputDebugStringA(s_msgbuf);
#ifdef _TRACETOFILE
	std::ofstream myfile;
	myfile.open("c:\\temp\\mycaffe_debug.txt", std::ofstream::out | std::ofstream::app);
	myfile << s_msgbuf;
	myfile.close();
#endif
#endif
#endif

	//switch (lfnIdx)
	//{
	//	default:
			return ERROR_NOT_SUPPORTED;
	//}
}

template long Kernel<double>::Run(long lfnIdx, double* pfInput, long lCount, LPTSTR pszInput, double** ppfOutput, long* plCount);
template long Kernel<float>::Run(long lfnIdx, float* pfInput, long lCount, LPTSTR pszInput, float** ppfOutput, long* plCount);


template <class T>
long Kernel<T>::Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount, LPTSTR szErr, long lMaxErr)
{
	cudaGetLastError();

#ifdef _DEBUG
#ifdef _TRACEAPI
	snprintf(s_msgbuf, 256, "%s calling CudaDnnDLL FunctionID (%ld) %s\n", (sizeof(T) == sizeof(float)) ? "FLOAT" : "DOUBLE", lfnIdx, GetApiName(lfnIdx));
	OutputDebugStringA(s_msgbuf);
#ifdef _TRACETOFILE
	std::ofstream myfile;
	myfile.open("c:\\temp\\mycaffe_debug.txt", std::ofstream::out | std::ofstream::app);
	myfile << s_msgbuf;
	myfile.close();
#endif
#endif
#endif

	switch (lfnIdx)
	{
		case CUDA_FN_EXTENSION_RUN:
			return m_device.ExtensionRun(lCount, pfInput, plCount, ppfOutput, szErr, lMaxErr);

		default:
			return ERROR_PARAM_OUT_OF_RANGE;
	}
}

template long Kernel<double>::Run(long lfnIdx, double* pfInput, long lCount, double** ppfOutput, long* plCount, LPTSTR szErr, long lMaxErr);
template long Kernel<float>::Run(long lfnIdx, float* pfInput, long lCount, float** ppfOutput, long* plCount, LPTSTR szErr, long lMaxErr);


#ifdef _DEBUG
#ifdef _TRACEAPI
char* GetApiName(long lfnIdx)
{
	switch (lfnIdx)
	{
	case CUDA_FN_SETDEVICE:
		return "CUDA_FN_SETDEVICE";

	case CUDA_FN_SETRANDOMSEED:
		return "CUDA_FN_SETRANDOMSEED";

	case CUDA_FN_GETDEVICE:
		return "CUDA_FN_GETDEVICE";

	case CUDA_FN_RESETDEVICE:
		return "CUDA_FN_RESETDEVICE";

	case CUDA_FN_SYNCHRONIZEDEVICE:
		return "CUDA_FN_SYNCHRONIZEDEVICE";

	case CUDA_FN_GETDEVICEPROP:
		return "CUDA_FN_GETDEVICEPROP";

	case CUDA_FN_GET_REQUIRED_COMPUTE:
		return "CUDA_FN_GET_REQUIRED_COMPUTE";

	case CUDA_FN_CHECKMEMORYATTRIB:
		return "CUDA_FN_CHECKMEMORYATTRIB";

	case CUDA_FN_GETDEVICEMEMORY:
		return "CUDA_FN_GETDEVICEMEMORY";

	case CUDA_FN_DEVICE_CANACCESSPEER:
		return "CUDA_FN_DEVICE_CANACCESSPEER";

	case CUDA_FN_DEVICE_ENABLEPEERACCESS:
		return "CUDA_FN_DEVICE_ENABLEPEERACCESS";

	case CUDA_FN_DEVICE_DISABLEPEERACCESS:
		return "CUDA_FN_DEVICE_DISABLEPEERACCESS";

	case CUDA_FN_ALLOCMEM:
		return "CUDA_FN_ALLOCMEM";

	case CUDA_FN_FREEMEM:
		return "CUDA_FN_FREEMEM";

	case CUDA_FN_GETMEM:
		return "CUDA_FN_GETMEM";

	case CUDA_FN_SETMEM:
		return "CUDA_FN_SETMEM";

	case CUDA_FN_SETMEMAT:
		return "CUDA_FN_SETMEMAT";

	case CUDA_FN_SETPIXEL:
		return "CUDA_FN_SETPIXEL";

	case CUDA_FN_COPY_DEVICE_TO_HOST:
		return "CUDA_FN_COPY_DEVICE_TO_HOST";

	case CUDA_FN_COPY_HOST_TO_DEVICE:
		return "CUDA_FN_COPY_HOST_TO_DEVICE";

	case CUDA_FN_ALLOCHOSTBUFFER:
		return "CUDA_FN_ALLOCHOSTBUFFER";

	case CUDA_FN_FREEHOSTBUFFER:
		return "CUDA_FN_FREEHOSTBUFFER";

	case CUDA_FN_GETHOSTMEM:
		return "CUDA_FN_GETHOSTMEM";

	case CUDA_FN_SETHOSTMEM:
		return "CUDA_FN_SETHOSTMEM";

	case CUDA_FN_CREATE_MEMORYPOINTER:
		return "CUDA_FN_CREATE_MEMORYPOINTER";

	case CUDA_FN_FREE_MEMORYPOINTER:
		return "CUDA_FN_FREE_MEMORYPOINTER";

	case CUDA_FN_CREATE_STREAM:
		return "CUDA_FN_CREATE_STREAM";

	case CUDA_FN_FREE_STREAM:
		return "CUDA_FN_FREE_STREAM";

	case CUDA_FN_SYNCHRONIZE_STREAM:
		return "CUDA_FN_SYNCHRONIZE_STREAM";

	case CUDA_FN_SYNCHRONIZE_THREAD:
		return "CUDA_FN_SYNCHRONIZE_THREAD";

	case CUDA_FN_CREATE_MEMTEST:
		return "CUDA_FN_CREATE_MEMTEST";

	case CUDA_FN_FREE_MEMTEST:
		return "CUDA_FN_FREE_MEMTEST";

	case CUDA_FN_RUN_MEMTEST:
		return "CUDA_FN_RUN_MEMTEST";

	case CUDA_FN_CREATE_IMAGEOP:
		return "CUDA_FN_CREATE_IMAGEOP";

	case CUDA_FN_FREE_IMAGEOP:
		return "CUDA_FN_FREE_IMAGEOP";

	case CUDA_FN_DISTORTIMAGE_IMAGEOP:
		return "CUDA_FN_DISTORTIMAGE_IMAGEOP";

	case CUDA_FN_CREATE_NCCL:
		return "CUDA_FN_CREATE_NCCL";

	case CUDA_FN_FREE_NCCL:
		return "CUDA_FN_FREE_NCCL";

	case CUDA_FN_NCCL_INIT_SINGLEPROCESS:
		return "CUDA_FN_NCCL_INIT_SINGLEPROCESS";

	case CUDA_FN_NCCL_INIT_MULTIPROCESS:
		return "CUDA_FN_NCCL_INIT_MULTIPROCESS";

	case CUDA_FN_NCCL_BROADCAST:
		return "CUDA_FN_NCCL_BROADCAST";

	case CUDA_FN_NCCL_ALLREDUCE:
		return "CUDA_FN_NCCL_ALLREDUCE";

	case CUDA_FN_CREATE_EXTENSION:
		return "CUDA_FN_CREATE_EXTENSION";

	case CUDA_FN_FREE_EXTENSION:
		return "CUDA_FN_FREE_EXTENSION";

	case CUDA_FN_EXTENSION_RUN:
		return "CUDA_FN_EXTENSION_RUN";

	case CUDNN_FN_CREATE_CUDNN:
		return "CUDNN_FN_CREATE_CUDNN";

	case CUDNN_FN_FREE_CUDNN:
		return "CUDNN_FN_FREE_CUDNN";

	case CUDNN_FN_CREATE_TENSORDESC:
		return "CUDNN_FN_CREATE_TENSORDESC";

	case CUDNN_FN_FREE_TENSORDESC:
		return "CUDNN_FN_FREE_TENSORDESC";

	case CUDNN_FN_SET_TENSORDESC:
		return "CUDNN_FN_SET_TENSORDESC";

	case CUDNN_FN_SET_TENSORNDDESC:
		return "CUDNN_FN_SET_TENSORNDDESC";

	case CUDNN_FN_ADD_TENSOR:
		return "CUDNN_FN_ADD_TENSOR";

	case CUDNN_FN_CREATE_FILTERDESC:
		return "CUDNN_FN_CREATE_FILTERDESC";

	case CUDNN_FN_FREE_FILTERDESC:
		return "CUDNN_FN_FREE_FILTERDESC";

	case CUDNN_FN_SET_FILTERDESC:
		return "CUDNN_FN_SET_FILTERDESC";

	case CUDNN_FN_SET_FILTERNDDESC:
		return "CUDNN_FN_SET_FILTERNDDESC";

	case CUDNN_FN_CREATE_CONVDESC:
		return "CUDNN_FN_CREATE_CONVDESC";

	case CUDNN_FN_FREE_CONVDESC:
		return "CUDNN_FN_FREE_CONVDESC";

	case CUDNN_FN_SET_CONVDESC:
		return "CUDNN_FN_SET_CONVDESC";

	case CUDNN_FN_GET_CONVINFO:
		return "CUDNN_FN_GET_CONVINFO";

	case CUDNN_FN_FWD_CONV:
		return "CUDNN_FN_FWD_CONV";

	case CUDNN_FN_BWD_CONV_BIAS:
		return "CUDNN_FN_BWD_CONV_BIAS";

	case CUDNN_FN_BWD_CONV_FILTER:
		return "CUDNN_FN_BWD_CONV_FILTER";

	case CUDNN_FN_BWD_CONV_DATA:
		return "CUDNN_FN_BWD_CONV_DATA";

	case CUDNN_FN_CREATE_POOLDESC:
		return "CUDNN_FN_CREATE_POOLDESC";

	case CUDNN_FN_FREE_POOLDESC:
		return "CUDNN_FN_FREE_POOLDESC";

	case CUDNN_FN_SET_POOLDESC:
		return "CUDNN_FN_SET_POOLDESC";

	case CUDNN_FN_POOL_FWD:
		return "CUDNN_FN_POOL_FWD";

	case CUDNN_FN_POOL_BWD:
		return "CUDNN_FN_POOL_BWD";

	case CUDNN_FN_GET_DROPOUT_INFO:
		return "CUDNN_FN_GET_DROPOUT_INFO";

	case CUDNN_FN_CREATE_DROPOUTDESC:
		return "CUDNN_FN_CREATE_DROPOUTDESC";

	case CUDNN_FN_FREE_DROPOUTDESC:
		return "CUDNN_FN_FREE_DROPOUTDESC";

	case CUDNN_FN_SET_DROPOUTDESC:
		return "CUDNN_FN_SET_DROPOUTDESC";

	case CUDNN_FN_DROPOUT_FWD:
		return "CUDNN_FN_DROPOUT_FWD";

	case CUDNN_FN_DROPOUT_BWD:
		return "CUDNN_FN_DROPOUT_BWD";

	case CUDNN_FN_CREATE_LRNDESC:
		return "CUDNN_FN_CREATE_LRNDESC";

	case CUDNN_FN_FREE_LRNDESC:
		return "CUDNN_FN_FREE_LRNDESC";

	case CUDNN_FN_SET_LRNDESC:
		return "CUDNN_FN_SET_LRNDESC";

	case CUDNN_FN_TANH_FWD:
		return "CUDNN_FN_TANH_FWD";

	case CUDNN_FN_TANH_BWD:
		return "CUDNN_FN_TANH_BWD";

	case CUDNN_FN_SIGMOID_FWD:
		return "CUDNN_FN_SIGMOID_FWD";

	case CUDNN_FN_SIGMOID_BWD:
		return "CUDNN_FN_SIGMOID_BWD";

	case CUDNN_FN_RELU_FWD:
		return "CUDNN_FN_RELU_FWD";

	case CUDNN_FN_RELU_BWD:
		return "CUDNN_FN_RELU_BWD";

	case CUDNN_FN_SOFTMAX_FWD:
		return "CUDNN_FN_SOFTMAX_FWD";

	case CUDNN_FN_SOFTMAX_BWD:
		return "CUDNN_FN_SOFTMAX_BWD";

	case CUDNN_FN_CREATE_RNN_DATA_DESC:
		return "CUDNN_FN_CREATE_RNN_DATA_DESC";

	case CUDNN_FN_FREE_RNN_DATA_DESC:
		return "CUDNN_FN_FREE_RNN_DATA_DESC";

	case CUDNN_FN_SET_RNN_DATA_DESC:
		return "CUDNN_FN_SET_RNN_DATA_DESC";

	case CUDNN_FN_CREATE_RNN_DATA_DESCEX:
		return "CUDNN_FN_CREATE_RNN_DATA_DESCEX";

	case CUDNN_FN_FREE_RNN_DATA_DESCEX:
		return "CUDNN_FN_FREE_RNN_DATA_DESCEX";

	case CUDNN_FN_SET_RNN_DATA_DESCEX:
		return "CUDNN_FN_SET_RNN_DATA_DESCEX";

	case CUDNN_FN_CREATE_RNN_DESC:
		return "CUDNN_FN_CREATE_RNN_DESC";

	case CUDNN_FN_FREE_RNN_DESC:
		return "CUDNN_FN_FREE_RNN_DESC";

	case CUDNN_FN_SET_RNN_DESC:
		return "CUDNN_FN_SET_RNN_DESC";

	case CUDNN_FN_GET_RNN_PARAMCOUNT:
		return "CUDNN_FN_GET_RNN_PARAMCOUNT";

	case CUDNN_FN_GET_RNN_WORKSPACECOUNT:
		return "CUDNN_FN_GET_RNN_WORKSPACECOUNT";

	case CUDNN_FN_GET_RNN_LINLAYERPARAMS:
		return "CUDNN_FN_GET_RNN_LINLAYERPARAMS";

	case CUDNN_FN_RNN_FWD:
		return "CUDNN_FN_RNN_FWD";

	case CUDNN_FN_RNN_BWD_DATA:
		return "CUDNN_FN_RNN_BWD_DATA";

	case CUDNN_FN_RNN_BWD_WTS:
		return "CUDNN_FN_RNN_BWD_WTS";

	case CUDNN_FN_LRN_CC_FWD:
		return "CUDNN_FN_LRN_CC_FWD";

	case CUDNN_FN_LRN_CC_BWD:
		return "CUDNN_FN_LRN_CC_BWD";

	case CUDNN_FN_LCN_CC_FWD:
		return "CUDNN_FN_LCN_CC_FWD";

	case CUDNN_FN_LCN_CC_BWD:
		return "CUDNN_FN_LCN_CC_BWD";

	case CUDA_FN_CREATE_PCA:
		return "CUDA_FN_CREATE_PCA";

	case CUDA_FN_FREE_PCA:
		return "CUDA_FN_FREE_PCA";

	case CUDA_FN_RUN_PCA:
		return "CUDA_FN_RUN_PCA";

	case CUDA_FN_CREATE_TSNE_GAUSSIAN_PERPLEXITY:
		return "CUDA_FN_CREATE_TSNE_GAUSSIAN_PERPLEXITY";

	case CUDA_FN_FREE_TSNE_GAUSSIAN_PERPLEXITY:
		return "CUDA_FN_FREE_TSNE_GAUSSIAN_PERPLEXITY";

	case CUDA_FN_FIND_TSNE_GAUSSIAN_PERPLEXITY:
		return "CUDA_FN_FIND_TSNE_GAUSSIAN_PERPLEXITY";

	case CUDA_FN_CREATE_TSNE:
		return "CUDA_FN_CREATE_TSNE";

	case CUDA_FN_FREE_TSNE:
		return "CUDA_FN_FREE_TSNE";

	case CUDA_FN_TSNE_COMPUTE_GRADIENT1:
		return "CUDA_FN_TSNE_COMPUTE_GRADIENT1";

	case CUDA_FN_TSNE_COMPUTE_ERROR1:
		return "CUDA_FN_TSNE_COMPUTE_ERROR1";

	case CUDA_FN_SET:
		return "CUDA_FN_SET";

	case CUDA_FN_GET:
		return "CUDA_FN_GET";

	case CUDA_FN_COPY:
		return "CUDA_FN_COPY";

	case CUDA_FN_COPY_SIM:
		return "CUDA_FN_COPY_SIM";

	case CUDA_FN_COPY_FILL:
		return "CUDA_FN_COPY_FILL";

	case CUDA_FN_SORT:
		return "CUDA_FN_SORT";

	case CUDA_FN_COPY_BATCH:
		return "CUDA_FN_COPY_BATCH";

	case CUDA_FN_COPY_SEQUENCE:
		return "CUDA_FN_COPY_SEQUENCE";

	case CUDA_FN_COPY_SEQUENCE2:
		return "CUDA_FN_COPY_SEQUENCE2";

	case CUDA_FN_COPY_EXPAND:
		return "CUDA_FN_COPY_EXPAND";

	case CUDA_FN_GEMM:
		return "CUDA_FN_GEMM";

	case CUDA_FN_GEMM2:
		return "CUDA_FN_GEMM2";

	case CUDA_FN_GEMV:
		return "CUDA_FN_GEMV";

	case CUDA_FN_GEAM:
		return "CUDA_FN_GEAM";

	case CUDA_FN_GER:
		return "CUDA_FN_GER";

	case CUDA_FN_SET_BOUNDS:
		return "CUDA_FN_SET_BOUNDS";

	case CUDA_FN_AXPY:
		return "CUDA_FN_AXPY";

	case CUDA_FN_AXPBY:
		return "CUDA_FN_AXPBY";

	case CUDA_FN_MULBSX:
		return "CUDA_FN_MULBSX";

	case CUDA_FN_DIVBSX:
		return "CUDA_FN_DIVBSX";

	case CUDA_FN_SCAL:
		return "CUDA_FN_SCAL";

	case CUDA_FN_DOT:
		return "CUDA_FN_DOT";

	case CUDA_FN_ASUM:
		return "CUDA_FN_ASUM";

	case CUDA_FN_SCALE:
		return "CUDA_FN_SCALE";

	case CUDA_FN_SCALE_TO_RANGE:
		return "CUDA_FN_SCALE_TO_RANGE";

	case CUDA_FN_ERF:
		return "CUDA_FN_ERF";

	case CUDA_FN_MASK:
		return "CUDA_FN_MASK";

	case CUDA_FN_MASK_BATCH:
		return "CUDA_FN_MASK_BATCH";

	case CUDA_FN_INTERP2:
		return "CUDA_FN_INTERP2";

	case CUDA_FN_ADD_SCALAR:
		return "CUDA_FN_ADD_SCALAR";

	case CUDA_FN_ADD:
		return "CUDA_FN_ADD";

	case CUDA_FN_ADD2:
		return "CUDA_FN_ADD2";

	case CUDA_FN_SUB:
		return "CUDA_FN_SUB";

	case CUDA_FN_MUL:
		return "CUDA_FN_MUL";

	case CUDA_FN_SUB_AND_DOT:
		return "CUDA_FN_SUB_AND_DOT";

	case CUDA_FN_MUL_SCALAR:
		return "CUDA_FN_MUL_SCALAR";

	case CUDA_FN_DIV:
		return "CUDA_FN_DIV";

	case CUDA_FN_ABS:
		return "CUDA_FN_ABS";

	case CUDA_FN_EXP:
		return "CUDA_FN_EXP";

	case CUDA_FN_LOG:
		return "CUDA_FN_LOG";

	case CUDA_FN_POWX:
		return "CUDA_FN_POWX";

	case CUDA_FN_SIGN:
		return "CUDA_FN_SIGN";

	case CUDA_FN_SQRT:
		return "CUDA_FN_SQRT";

	case CUDA_FN_SQRT_SCALE:
		return "CUDA_FN_SQRT_SCALE";

	case CUDA_FN_RECIPROCOL:
		return "CUDA_FN_RECIPROCOL";

	case CUDA_FN_STUDENT:
		return "CUDA_FN_STUDENT";

	case CUDA_FN_LOGISTIC1:
		return "CUDA_FN_LOGISTIC1";

	case CUDA_FN_LOGISTIC2:
		return "CUDA_FN_LOGISTIC2";

	case CUDA_FN_COMPARE_SIGNS:
		return "CUDA_FN_COMPARE_SIGNS";

	case CUDA_FN_DENAN:
		return "CUDA_FN_DENAN";

	case CUDA_FN_MAXVAL:
		return "CUDA_FN_MAXVAL";

	case CUDA_FN_MINVAL:
		return "CUDA_FN_MINVAL";

	case CUDA_FN_MINMAXVAL:
		return "CUDA_FN_MINMAXVAL";

	case CUDA_FN_MINMAXVEC:
		return "CUDA_FN_MINMAXVEC";

	case CUDA_FN_TRANSPOSE:
		return "CUDA_FN_TRANSPOSE";

	case CUDA_FN_SUMSQ:
		return "CUDA_FN_SUMSQ";

	case CUDA_FN_SUMSQDIFF:
		return "CUDA_FN_SUMSQDIFF";

	case CUDA_FN_WIDTH:
		return "CUDA_FN_WIDTH";

	case CUDA_FN_CONTAINS_POINT:
		return "CUDA_FN_CONTAINS_POINT";

	case CUDA_FN_CHANNEL_MIN:
		return "CUDA_FN_CHANNEL_MIN";

	case CUDA_FN_CHANNEL_MAX:
		return "CUDA_FN_CHANNEL_MAX";

	case CUDA_FN_CHANNEL_SUB:
		return "CUDA_FN_CHANNEL_SUB";

	case CUDA_FN_CHANNEL_SUM:
		return "CUDA_FN_CHANNEL_SUM";

	case CUDA_FN_CHANNEL_DIV:
		return "CUDA_FN_CHANNEL_DIV";

	case CUDA_FN_CHANNEL_MUL:
		return "CUDA_FN_CHANNEL_MUL";

	case CUDA_FN_CHANNEL_MULV:
		return "CUDA_FN_CHANNEL_MULV";

	case CUDA_FN_CHANNEL_SCALE:
		return "CUDA_FN_CHANNEL_SCALE";

	case CUDA_FN_CHANNEL_DOT:
		return "CUDA_FN_CHANNEL_DOT";

	case CUDA_FN_CHANNEL_COMPARE:
		return "CUDA_FN_CHANNEL_COMPARE";

	case CUDA_FN_CHANNEL_FILL:
		return "CUDA_FN_CHANNEL_FILL";

	case CUDA_FN_CHANNEL_FILLFROM:
		return "CUDA_FN_CHANNEL_FILLFROM";

	case CUDA_FN_CHANNEL_COPY:
		return "CUDA_FN_CHANNEL_COPY";

	case CUDA_FN_CHANNEL_COPYALL:
		return "CUDA_FN_CHANNEL_COPYALL";

	case CUDA_FN_CHANNEL_DUPLICATE:
		return "CUDA_FN_CHANNEL_DUPLICATE";

	case CUDA_FN_IM2COL:
		return "CUDA_FN_IM2COL";

	case CUDA_FN_IM2COL_ND:
		return "CUDA_FN_IM2COL_ND";

	case CUDA_FN_COL2IM:
		return "CUDA_FN_COL2IM";

	case CUDA_FN_COL2IM_ND:
		return "CUDA_FN_COL2IM_ND";

	case CUDA_RNG_SETSEED:
		return "CUDA_RNG_SETSEED";

	case CUDA_RNG_UNIFORM:
		return "CUDA_RNG_UNIFORM";

	case CUDA_RNG_GAUSSIAN:
		return "CUDA_RNG_GAUSSIAN";

	case CUDA_RNG_BERNOULLI:
		return "CUDA_RNG_BERNOULLI";

	case CUDA_FN_ACCURACY_FWD:
		return "CUDA_FN_ACCURACY_FWD";

	case CUDA_FN_BATCHREIDX_FWD:
		return "CUDA_FN_BATCHREIDX_FWD";

	case CUDA_FN_BATCHREIDX_BWD:
		return "CUDA_FN_BATCHREIDX_BWD";

	case CUDA_FN_EMBED_FWD:
		return "CUDA_FN_EMBED_FWD";

	case CUDA_FN_EMBED_BWD:
		return "CUDA_FN_EMBED_BWD";

	case CUDA_FN_POOL_FWD:
		return "CUDA_FN_POOL_FWD";

	case CUDA_FN_POOL_BWD:
		return "CUDA_FN_POOL_BWD";

	case CUDA_FN_UNPOOL_FWD:
		return "CUDA_FN_UNPOOL_FWD";

	case CUDA_FN_UNPOOL_BWD:
		return "CUDA_FN_UNPOOL_BWD";

	case CUDA_FN_TANH_FWD:
		return "CUDA_FN_TANH_FWD";

	case CUDA_FN_TANH_BWD:
		return "CUDA_FN_TANH_BWD";

	case CUDA_FN_MEAN_ERROR_LOSS_BWD:
		return "CUDA_FN_MEAN_ERROR_LOSS_BWD";

	case CUDA_FN_MISH_FWD:
		return "CUDA_FN_MISH_FWD";

	case CUDA_FN_MISH_BWD:
		return "CUDA_FN_MISH_BWD";

	case CUDA_FN_GELU_FWD:
		return "CUDA_FN_GELU_FWD";

	case CUDA_FN_GELU_BWD:
		return "CUDA_FN_GELU_BWD";

	case CUDA_FN_SIGMOID_FWD:
		return "CUDA_FN_SIGMOID_FWD";

	case CUDA_FN_SIGMOID_BWD:
		return "CUDA_FN_SIGMOID_BWD";

	case CUDA_FN_SWISH_BWD:
		return "CUDA_FN_SWISH_BWD";

	case CUDA_FN_RELU_FWD:
		return "CUDA_FN_RELU_FWD";

	case CUDA_FN_RELU_BWD:
		return "CUDA_FN_RELU_BWD";

	case CUDA_FN_ELU_FWD:
		return "CUDA_FN_ELU_FWD";

	case CUDA_FN_ELU_BWD:
		return "CUDA_FN_ELU_BWD";

	case CUDA_FN_DROPOUT_FWD:
		return "CUDA_FN_DROPOUT_FWD";

	case CUDA_FN_DROPOUT_BWD:
		return "CUDA_FN_DROPOUT_BWD";

	case CUDA_FN_BNLL_FWD:
		return "CUDA_FN_BNLL_FWD";

	case CUDA_FN_BNLL_BWD:
		return "CUDA_FN_BNLL_BWD";

	case CUDA_FN_PRELU_FWD:
		return "CUDA_FN_PRELU_FWD";

	case CUDA_FN_PRELU_BWD:
		return "CUDA_FN_PRELU_BWD";

	case CUDA_FN_PRELU_BWD_PARAM:
		return "CUDA_FN_PRELU_BWD_PARAM";

	case CUDA_FN_NLLLOSS_FWD:
		return "CUDA_FN_NLLLOSS_FWD";

	case CUDA_FN_NLLLOSS_BWD:
		return "CUDA_FN_NLLLOSS_BWD";

	case CUDA_FN_SOFTMAXLOSS_FWD:
		return "CUDA_FN_SOFTMAXLOSS_FWD";

	case CUDA_FN_SOFTMAXLOSS_BWD:
		return "CUDA_FN_SOFTMAXLOSS_BWD";

	case CUDA_FN_MIN_FWD:
		return "CUDA_FN_MIN_FWD";

	case CUDA_FN_MIN_BWD:
		return "CUDA_FN_MIN_BWD";

	case CUDA_FN_MAX_FWD:
		return "CUDA_FN_MAX_FWD";

	case CUDA_FN_MAX_BWD:
		return "CUDA_FN_MAX_BWD";

	case CUDA_FN_CROP_FWD:
		return "CUDA_FN_CROP_FWD";

	case CUDA_FN_CROP_BWD:
		return "CUDA_FN_CROP_BWD";

	case CUDA_FN_CONCAT_FWD:
		return "CUDA_FN_CONCAT_FWD";

	case CUDA_FN_CONCAT_BWD:
		return "CUDA_FN_CONCAT_BWD";

	case CUDA_FN_SLICE_FWD:
		return "CUDA_FN_SLICE_FWD";

	case CUDA_FN_SLICE_BWD:
		return "CUDA_FN_SLICE_BWD";

	case CUDA_FN_TILE_FWD:
		return "CUDA_FN_TILE_FWD";

	case CUDA_FN_TILE_BWD:
		return "CUDA_FN_TILE_BWD";

	case CUDA_FN_BIAS_FWD:
		return "CUDA_FN_BIAS_FWD";

	case CUDA_FN_SCALE_FWD:
		return "CUDA_FN_SCALE_FWD";

	case CUDA_FN_THRESHOLD_FWD:
		return "CUDA_FN_THRESHOLD_FWD";

	case CUDA_FN_CLL_BWD:
		return "CUDA_FN_CLL_BWD";

	case CUDA_FN_SMOOTHL1_FWD:
		return "CUDA_FN_SMOOTHL1_FWD";

	case CUDA_FN_SMOOTHL1_BWD:
		return "CUDA_FN_SMOOTHL1_BWD";

	case CUDA_FN_SERF_FWD:
		return "CUDA_FN_SERF_FWD";

	case CUDA_FN_SERF_BWD:
		return "CUDA_FN_SERF_BWD";

	case CUDA_FN_PERMUTE:
		return "CUDA_FN_PERMUTE";

	case CUDA_FN_GATHER_FWD:
		return "CUDA_FN_GATHER_FWD";

	case CUDA_FN_GATHER_BWD:
		return "CUDA_FN_GATHER_BWD";

	case CUDA_FN_LRN_FILLSCALE:
		return "CUDA_FN_LRN_FILLSCALE";

	case CUDA_FN_LRN_COMPUTEOUTPUT:
		return "CUDA_FN_LRN_COMPUTEOUTPUT";

	case CUDA_FN_LRN_COMPUTEDIFF:
		return "CUDA_FN_LRN_COMPUTEDIFF";

	case CUDA_FN_LSTM_FWD:
		return "CUDA_FN_LSTM_FWD";

	case CUDA_FN_LSTM_BWD:
		return "CUDA_FN_LSTM_BWD";

	case CUDA_FN_LSTM_UNIT_FWD:
		return "CUDA_FN_LSTM_UNIT_FWD";

	case CUDA_FN_LSTM_UNIT_BWD:
		return "CUDA_FN_LSTM_UNIT_BWD";

	case CUDA_FN_MATH_FWD:
		return "CUDA_FN_MATH_FWD";

	case CUDA_FN_MATH_BWD:
		return "CUDA_FN_MATH_BWD";

	case CUDA_FN_COEFF_SUM_FWD:
		return "CUDA_FN_COEFF_SUM_FWD";

	case CUDA_FN_COEFF_SUM_BWD:
		return "CUDA_FN_COEFF_SUM_BWD";

	case CUDA_FN_COEFF_SUB_FWD:
		return "CUDA_FN_COEFF_SUB_FWD";

	case CUDA_FN_COEFF_SUB_BWD:
		return "CUDA_FN_COEFF_SUB_BWD";

	case CUDA_FN_SIGMOID_CROSS_ENTROPY_FWD:
		return "CUDA_FN_SIGMOID_CROSS_ENTROPY_FWD";

	case CUDA_FN_SIGMOID_CROSS_ENTROPY_BWD:
		return "CUDA_FN_SIGMOID_CROSS_ENTROPY_BWD";

	case CUDA_FN_SOFTMAX_CROSS_ENTROPY_FWD:
		return "CUDA_FN_SOFTMAX_CROSS_ENTROPY_FWD";

	case CUDA_FN_SOFTMAX_CROSS_ENTROPY_BWD:
		return "CUDA_FN_SOFTMAX_CROSS_ENTROPY_BWD";

	case CUDA_FN_SGD_UPDATE:
		return "CUDA_FN_SGD_UPDATE";

	case CUDA_FN_NESTEROV_UPDATE:
		return "CUDA_FN_NESTEROV_UPDATE";

	case CUDA_FN_ADAGRAD_UPDATE:
		return "CUDA_FN_ADAGRAD_UPDATE";

	case CUDA_FN_ADADELTA_UPDATE:
		return "CUDA_FN_ADADELTA_UPDATE";

	case CUDA_FN_ADAM_UPDATE:
		return "CUDA_FN_ADAM_UPDATE";

	case CUDA_FN_ADAMW_UPDATE:
		return "CUDA_FN_ADAMW_UPDATE";

	case CUDA_FN_RMSPROP_UPDATE:
		return "CUDA_FN_RMSPROP_UPDATE";

	case CUDA_FN_COMBINE_DATA:
		return "CUDA_FN_COMBINE_DATA";

	case CUDA_FN_MTX_SET_DIAGONAL:
		return "CUDA_FN_MTX_SET_DIAGONAL";

	case CUDA_FN_MTX_SET_DIAGONAL2:
		return "CUDA_FN_MTX_SET_DIAGONAL2";

	case CUDA_FN_MTX_ADD_VECTOR:
		return "CUDA_FN_MTX_ADD_VECTOR";

	case CUDA_FN_MTX_TRANSPOSE_OP:
		return "CUDA_FN_MTX_TRANSPOSE_OP";

	case CUDA_FN_MTX_AGGREGATE_COLS:
		return "CUDA_FN_MTX_AGGREGATE_COLS";

	case CUDA_FN_MTX_AGGREGATE_ROWS:
		return "CUDA_FN_MTX_AGGREGATE_ROWS";

	case CUDA_FN_MTX_TRANSPOSE:
		return "CUDA_FN_MTX_TRANSPOSE";

	case CUDA_FN_MTX_MEANCENTER_BY_COL:
		return "CUDA_FN_MTX_MEANCENTER_BY_COL";

	case CUDA_FN_MTX_EUCLIDEAN_DIST:
		return "CUDA_FN_MTX_EUCLIDEAN_DIST";

	case CUDA_FN_MTX_DOT:
		return "CUDA_FN_MTX_DOT";

	case CUDA_FN_MTX_MEAN:
		return "CUDA_FN_MTX_MEAN";

	case CUDA_FN_MTX_STDEV:
		return "CUDA_FN_MTX_STDEV";

	case CUDA_FN_MTX_CORRELATION:
		return "CUDA_FN_MTX_CORRELATION";

	case CUDA_FN_TSNE_UPDATE:
		return "CUDA_FN_TSNE_UPDATE";

	case CUDA_FN_TSNE_UPDATE_GRAD:
		return "CUDA_FN_TSNE_UPDATE_GRAD";

	case CUDA_FN_TSNE_COMPUTE_EXACT_ERROR:
		return "CUDA_FN_TSNE_COMPUTE_EXACT_ERROR";

	case CUDA_FN_TSNE_COMPUTE_SQUARED_EUCLIDEAN_DISTANCE:
		return "CUDA_FN_TSNE_COMPUTE_SQUARED_EUCLIDEAN_DISTANCE";

	case CUDA_FN_TSNE_COMPUTE_Q_MATRIX:
		return "CUDA_FN_TSNE_COMPUTE_Q_MATRIX";

	case CUDA_FN_TSNE_COMPUTE_EXACT_GRADIENT:
		return "CUDA_FN_TSNE_COMPUTE_EXACT_GRADIENT";

	case CUDA_FN_TSNE_SYMMETRIZE_MATRIX:
		return "CUDA_FN_TSNE_SYMMETRIZE_MATRIX";

	case CUDA_FN_TSNE_COMPUTE_KNN_BOUNDS:
		return "CUDA_FN_TSNE_COMPUTE_KNN_BOUNDS";

	case CUDA_FN_GUASSIAN_BLUR:
		return "CUDA_FN_GUASSIAN_BLUR";

	case CUDA_FN_HAMMING_DIFF:
		return "CUDA_FN_HAMMING_DIFF";

	case CUDA_FN_CALC_BATCH_DIST:
		return "CUDA_FN_CALC_BATCH_DIST";

	case CUDA_FN_CREATE_SSD:
		return "CUDA_FN_CREATE_SSD";

	case CUDA_FN_FREE_SSD:
		return "CUDA_FN_FREE_SSD";

	case CUDA_FN_SETUP_SSD:
		return "CUDA_FN_SETUP_SSD";

	case CUDA_FN_SSD_FWD_MULTIBOXLOSS:
		return "CUDA_FN_SSD_FWD_MULTIBOXLOSS";

	case CUDA_FN_SSD_ENCODE_LOCPRED:
		return "CUDA_FN_SSD_ENCODE_LOCPRED";

	case CUDA_FN_SSD_ENCODE_CONFPRED:
		return "CUDA_FN_SSD_ENCODE_CONFPRED";

	case CUDA_FN_CREATE_LAYERNORM:
		return "CUDA_FN_CREATE_LAYERNORM";

	case CUDA_FN_FREE_LAYERNORM:
		return "CUDA_FN_FREE_LAYERNORM";

	case CUDA_FN_LAYERNORM_FWD:
		return "CUDA_FN_LAYERNORM_FWD";

	case CUDA_FN_LAYERNORM_BWD:
		return "CUDA_FN_LAYERNORM_BWD";

	default:
		return "UNKNOWN";
	}
}
#endif
#endif

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