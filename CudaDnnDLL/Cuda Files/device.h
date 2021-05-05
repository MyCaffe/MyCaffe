//=============================================================================
//	FILE:	device.h
//
//	DESC:	This file implements the class used to manage the underlying
//			device.
//=============================================================================
#ifndef __DEVICE_CU__
#define __DEVICE_CU__

#include "util.h"
#include "memory.h"
#include "math.h"


//=============================================================================
//	Flags
//=============================================================================

const int DEVINIT_NONE    = 0x0000;
const int DEVINIT_CUBLAS  = 0x0001;
const int DEVINIT_CURAND  = 0x0002;
const int DEVINIT_SETSEED = 0x0004;
const int DEVINIT_RESETDEVICE = 0x0008;

const int DEVPROP_DEVICECOUNT			= 1;
const int DEVPROP_NAME					= 2;
const int DEVPROP_MULTIGPUBOARDGROUPID	= 3;

const int MAX_ARG = 4096 * 10;
const int MAX_DIM = 4096 * 10;

const long INITIAL_SET_MEM_BUFFER = 4096;


//-----------------------------------------------------------------------------
//	Device Class
//
//	The device class implements manages underying GPU device.
//-----------------------------------------------------------------------------
template <class T>
class Device
{
	protected:
		Memory<T> m_memory;
		Math<T> m_math;
		cublasHandle_t m_cublas;
		curandGenerator_t m_curand;
		long m_lSeed;
		int m_nDevice;
		HANDLE m_hEventSrc;
		int m_nMajor = 0;
		int m_nMinor = 0;
		long m_hSetMemHost = 0;
		CRITICAL_SECTION m_MemHostLock;

		long verifyInput(long lInput, T* pfInput, long lMin, long lMax, bool bExact = false);
		long verifyOutput(long* plOutput, T** ppfOutput);
		long setOutput(long hHandle, long* plOutput, T** ppfOutput);
		long setOutput(T fVal, long* plOutput, T** ppfOutput);

	public:
		Device();
		~Device();

		long GetDeviceName(int nDevice, LPTSTR* pszDevice);
		long GetDeviceP2PInfo(int nDevice, LPTSTR* pszDevice);
		long GetDeviceInfo(int nDevice, LPTSTR* pszDevice, bool bVerbose);

		long SetDevice(int nDevice, int nFlags = DEVINIT_CUBLAS | DEVINIT_CURAND | DEVINIT_SETSEED, long lSeed = 0);
		int GetDevice();
		long ResetDevice();
		long SynchronizeDevice();

		long GetPointer(HANDLE_TYPE ht, long hHandle, void** ppPtr)
		{
			return m_memory.GetPointer(ht, hHandle, ppPtr);
		}

		long GetMemory(long hHandle, MemoryItem** ppItem)
		{
			return m_memory.GetMemory(hHandle, ppItem);
		}

		HostBuffer<T>* GetHostBuffer(long hHandle)
		{
			return m_memory.GetHostBuffer(hHandle);
		}

		cudaStream_t GetStream(long hStream)
		{
			return m_memory.GetStream(hStream);
		}

		long GetDeviceName(long lInput, LONG* pfInput, LPTSTR* ppfOutput);
		long GetDeviceP2PInfo(long lInput, LONG* pfInput, LPTSTR* ppfOutput);
		long GetDeviceInfo(long lInput, LONG* pfInput, LPTSTR* ppfOutput);

		long SetDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetRandomSeed(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ResetDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SynchronizeDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetDeviceProperty(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetRequiredCompute(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long CheckMemoryAttributes(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetDeviceMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long CanAccessPeer(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long EnablePeerAccess(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long DisablePeerAccess(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long AllocMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long AllocMemoryHalf(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetMemoryAt(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetPixel(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetPixel(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CopyHostBufferToGpu(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long CopyGpuToHostBuffer(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long AllocHostBuffer(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeHostBuffer(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetHostBufferCapacity(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetHostMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetHostMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long FreeHost(T* pf)
		{
			if (!m_memory.IsHostBuffer(pf))
				return m_memory.FreeHost(pf);

			return 0;
		}

		long FreeHost(LPTSTR pf)
		{
			return m_memory.FreeHost(pf);
		}

		long AllocHost(size_t lCount, T** ppDst, T* pSrc, bool bSrcOnDevice = false, bool bHalf = false)
		{
			return m_memory.AllocHost(lCount, ppDst, pSrc, bSrcOnDevice, bHalf);
		}

		long CreateMemoryPointer(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeMemoryPointer(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateStream(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeStream(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SynchronizeStream(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SynchronizeThread(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateMemoryTest(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeMemoryTest(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long RunMemoryTest(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateImageOp(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeImageOp(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long DistortImage(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		ncclHandle<T>* GetNccl(long hNccl);
		long SetNccl(ncclHandle<T>* pNccl, long* plOutput, T** ppfOutput);

		long CreateNCCL(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeNCCL(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long NcclInitSingleProcess(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long NcclInitMultiProcess(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long NcclBroadcast(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long NcclAllReduce(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateExtensionFloat(HMODULE hParent, LONG lKernelIdx, long* plOutput, T** ppfOutput, LPTSTR pszInput);
		long CreateExtensionDouble(HMODULE hParent, LONG lKernelIdx, long* plOutput, T** ppfOutput, LPTSTR pszInput);
		long FreeExtension(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ExtensionRun(long lInput, T* pfInput, long* plOutput, T** ppfOutput, LPTSTR szErr, long lErrMax);

		long CreateCuDNN(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeCuDNN(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		cudnnHandle_t GetCuDNN(long h)
		{
			return m_memory.GetCuDNN(h);
		}

		long CreateTensorDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeTensorDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetTensorDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetTensorNdDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long AddTensor(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		cudnnTensorDescriptor_t GetTensorDesc(long h)
		{
			return m_memory.GetTensorDesc(h);
		}

		long CreateFilterDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeFilterDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetFilterDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetFilterNdDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		cudnnFilterDescriptor_t GetFilterDesc(long h)
		{
			return m_memory.GetFilterDesc(h);
		}

		long CreateConvolutionDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeConvolutionDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetConvolutionDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetConvolutionInfo(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ConvolutionForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ConvolutionBackwardBias(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ConvolutionBackwardFilter(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ConvolutionBackwardData(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		cudnnConvolutionDescriptor_t GetConvolutionDesc(long h)
		{
			return m_memory.GetConvolutionDesc(h);
		}

		long CreatePoolingDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreePoolingDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetPoolingDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long PoolingForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long PoolingBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long DeriveBatchNormDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long BatchNormForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long BatchNormBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateDropoutDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeDropoutDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetDropoutDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetDropoutInfo(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long DropoutForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long DropoutBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateLRNDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeLRNDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetLRNDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long TanhForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long TanhBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long EluForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long EluBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long SigmoidForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SigmoidBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long ReLUForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ReLUBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long SoftmaxForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SoftmaxBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long LRNForwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long LRNBackwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long LCNForwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long LCNBackwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateRnnDataDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeRnnDataDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetRnnDataDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long CreateRnnDataDescEx(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeRnnDataDescEx(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetRnnDataDescEx(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateRnnDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeRnnDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetRnnDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetRnnParamCount(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetRnnWorkspaceCount(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long GetRnnLinLayerParams(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long RnnForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long RnnBackwardData(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long RnnBackwardWeights(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreatePCA(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreePCA(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long RunPCA(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateTsneGaussianPerplexity(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeTsneGaussianPerplexity(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FindTsneGaussianPerplexity(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateTsne(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeTsne(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long ComputeTsneGradient(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long EvaluateTsneError(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long CreateSsd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long FreeSsd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SetupSsd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SsdMultiboxLossForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SsdEncodeLocPrediction(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long SsdEncodeConfPrediction(long lInput, T* pfInput, long* plOutput, T** ppfOutput);


		//---------------------------------------------------------------------------
		//	Math functions
		//---------------------------------------------------------------------------

		long cuda_set(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_get(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_copy(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_copy_sim(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_copy_fill(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sort(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_copy_batch(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_copy_sequence(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_copy_sequence2(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_copy_expand(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_gemm(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_gemm2(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_gemv(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_ger(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_axpy(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_axpby(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_scal(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_asum(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mulbsx(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_divbsx(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_scale(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_scale_to_range(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_add_scalar(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_add(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_add2(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sub(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mul_scalar(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sub_and_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mul(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_div(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_abs(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_exp(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_log(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_powx(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sign(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sqrt(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sqrt_scale(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_reciprocol(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_student(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_logistic1(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_logistic2(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_compare_signs(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_maxval(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_minval(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_minmaxval(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_minmaxvec(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_transpose(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sumsq(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sumsqdiff(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sum(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_width(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_contains_point(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_denan(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_set_bounds(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_channel_min(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_max(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_sub(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_sum(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_div(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_mul(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_scale(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_compare(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_channel_fill(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_im2col(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_im2col_nd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_col2im(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_col2im_nd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_rng_setseed(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_rng_uniform(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_rng_gaussian(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_rng_bernoulli(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_accuracy_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_batchreidx_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_batchreidx_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_embed_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_embed_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_pooling_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_pooling_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_unpooling_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_unpooling_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_clip_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_clip_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_math_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_math_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_tanh_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tanh_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_mae_loss_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_mish_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mish_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_sigmoid_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sigmoid_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_swish_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_relu_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_relu_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_elu_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_elu_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_dropout_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_dropout_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_bnll_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_bnll_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_prelu_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_prelu_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_prelu_bwd_param(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_softmaxloss_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_softmaxloss_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_min_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_min_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_max_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_max_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_crop_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_crop_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_concat_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_concat_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_slice_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_slice_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_tile_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tile_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_bias_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_scale_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_threshold_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_cll_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_smoothl1_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_smoothl1_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_permute(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_gather_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_gather_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_lrn_fillscale(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_lrn_computeoutput(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_lrn_computediff(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_lstm_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_lstm_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_lstm_unit_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_lstm_unit_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_coeff_sum_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_coeff_sum_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_coeff_sub_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_coeff_sub_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_sigmoid_cross_entropy_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_sigmoid_cross_entropy_ignore(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_sgd_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_nesterov_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_adagrad_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_adadelta_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_adam_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_rmsprop_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_combine_data(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_mtx_set_diagonal(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_set_diagonal2(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_add_vector(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_transpose_op(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_aggregate_cols(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_aggregate_rows(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_transpose(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_meancenter_by_column(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_euclidean_dist(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_dot(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_mean(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_stdev(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_mtx_correlation(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_tsne_update(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tsne_update_grad(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tsne_compute_exact_error(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tsne_compute_squared_euclidean_distance(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tsne_compute_q_matrix(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tsne_compute_exact_gradient(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tsne_symmetrize_matrix(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_tsne_compute_knn_bounds(long lInput, T* pfInput, long* plOutput, T** ppfOutput);

		long cuda_guassian_blur(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_hamming_diff(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
		long cuda_calc_batch_dist(long lInput, T* pfInput, long* plOutput, T** ppfOutput);
};


//=============================================================================
//	Inline Methods
//=============================================================================

template <class T>
inline long Device<T>::verifyInput(long lInput, T* pfInput, long lMin, long lMax, bool bExact)
{
	if (lInput < lMin || lInput > lMax)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lMin == 0 && lMax == 0)
		return 0;

	if (bExact && lInput != lMin && lInput != lMax)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (pfInput == NULL)
		return ERROR_PARAM_NULL;

	return 0;
}

template <class T>
inline long Device<T>::verifyOutput(long* plOutput, T** ppfOutput)
{
	if (plOutput == NULL)
		return ERROR_PARAM_NULL;

	if (ppfOutput == NULL)
		return ERROR_PARAM_NULL;

	return 0;
}

template <class T>
inline long Device<T>::setOutput(long hHandle, long* plOutput, T** ppfOutput)
{
	*plOutput = 1;
	(*ppfOutput)[0] = (T)hHandle;

	return 0;
}


template <class T>
inline long Device<T>::setOutput(T fVal, long* plOutput, T** ppfOutput)
{
	*plOutput = 1;
	(*ppfOutput)[0] = fVal;

	return 0;
}


//=============================================================================
//	Device Methods
//=============================================================================

template <class T>
inline Device<T>::Device() : m_memory(), m_math()
{
	m_hSetMemHost = 0;
	m_math.Connect(&m_memory);
	m_cublas = NULL;
	m_curand = NULL;
	m_lSeed = 0;
	m_nDevice = 0;
	m_hEventSrc = RegisterEventSource(NULL, L"MyCaffe");

	if (!InitializeCriticalSectionAndSpinCount(&m_MemHostLock, 0x00000400))
		return;
}

template <class T>
inline Device<T>::~Device()
{
	if (m_hSetMemHost != 0)
	{
		EnterCriticalSection(&m_MemHostLock);
		m_memory.FreeHostBuffer(m_hSetMemHost);
		m_hSetMemHost = 0;
		LeaveCriticalSection(&m_MemHostLock);
	}

	if (m_curand != NULL)
	{
		curandDestroyGenerator(m_curand);
		m_curand = NULL;
	}

	if (m_hEventSrc != NULL)
	{
		DeregisterEventSource(m_hEventSrc);
		m_hEventSrc = NULL;
	}

	if (m_cublas != NULL)
	{
		cublasDestroy(m_cublas);
		m_cublas = NULL;
	}

	DeleteCriticalSection(&m_MemHostLock);
}

template <class T>
inline int Device<T>::GetDevice()
{
	return m_nDevice;
}

template <class T>
inline long Device<T>::ResetDevice()
{
	return cudaDeviceReset();
}

template <class T>
inline long Device<T>::SynchronizeDevice()
{
	return cudaDeviceSynchronize();
}

template <class T>
inline long Device<T>::SetRandomSeed(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	
	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long lSeed = (long)pfInput[0];

	return m_math.rng_setseed(lSeed);
}

template <class T>
inline long Device<T>::GetDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	return setOutput((long)GetDevice(), plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::ResetDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	return ResetDevice();
}

template <class T>
inline long Device<T>::SynchronizeDevice(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	SynchronizeDevice();
	return 0;
}

template <class T>
inline long Device<T>::GetDeviceProperty(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nDeviceID = (int)pfInput[0];
	int nPropID = (int)pfInput[1];
	T fVal = 0;

	if (nPropID == DEVPROP_DEVICECOUNT)
	{
		int nCount = 0;

		if (lErr = cudaGetDeviceCount(&nCount))
			return lErr;

		fVal = (T)nCount;
	}
	else
	{
		cudaDeviceProp prop;

		if (lErr = cudaGetDeviceProperties(&prop, nDeviceID))
			return lErr;

		m_nMajor = prop.major;
		m_nMinor = prop.minor;

		switch (nPropID)
		{
			case DEVPROP_MULTIGPUBOARDGROUPID:
				fVal = (T)prop.multiGpuBoardGroupID;
				break;

			default:
				return ERROR_PARAM_OUT_OF_RANGE;
		}
	}

	return setOutput(fVal, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::CheckMemoryAttributes(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hSrc = (long)pfInput[0];
	int nSrcDeviceID = (int)pfInput[1];
	long hDst = (long)pfInput[2];
	int nDstDeviceID = (int)pfInput[3];
	bool bResult = false;

	if (lErr = m_memory.CheckMemoryAttributes(hSrc, nSrcDeviceID, hDst, nDstDeviceID, &bResult))
		return lErr;

	T fVal = (bResult) ? (T)1 : (T)0;

	return setOutput(fVal, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::GetDeviceMemory(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nDeviceID = (int)pfInput[0];
	T fTotal = 0;
	T fFree = 0;
	T fUsed = 0;
	bool bEstimate = false;

	if (lErr = m_memory.GetDeviceMemory(nDeviceID, &fTotal, &fFree, &fUsed, &bEstimate))
		return lErr;

	// ppfOutput has MAX_OUTPUT(16) pre-allocated items.
	T* pfOutput = *ppfOutput;

	pfOutput[0] = fTotal;
	pfOutput[1] = fFree;
	pfOutput[2] = fUsed;
	pfOutput[3] = (bEstimate) ? 1.0f : 0.0f;

	*ppfOutput = pfOutput;
	*plOutput = 4;

	return 0;
}

template <class T>
inline long Device<T>::GetRequiredCompute(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	long lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nMajor = 3;
	int nMinor = 5;

//#ifdef __SM__
//#if (__SM__ >= 530)
//	nMajor = 5;
//	nMinor = 3;
//#endif
//#endif

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = (T)nMajor;
	pfOutput[1] = (T)nMinor;

	*ppfOutput = pfOutput;
	*plOutput = 2;

	return 0;
}


template <class T>
inline long Device<T>::CreateMemoryPointer(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	long hData = (long)pfInput[0];
	long lOffset = (long)pfInput[1];
	long lCount = (long)pfInput[2];

	long hHandle = 0;
	
	if (lErr = m_memory.CreateMemoryPointer(hData, lOffset, lCount, &hHandle))
		return lErr;

	if (lErr = setOutput(hHandle, plOutput, ppfOutput))
		return lErr;

	return 0;
}

template <class T>
inline long Device<T>::FreeMemoryPointer(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeMemoryPointer(hHandle);
}


//=============================================================================
//	Memory Test Methods
//=============================================================================

template <class T>
inline long Device<T>::CreateMemoryTest(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	T fPctToAllocate = pfInput[0];

	size_t szTotalNumBlocks = 0;
	T fMemAllocated = 0;
	T fMemStartAddr = 0;
	T fMemBlockSize = 0;

	if (lErr = m_memory.CreateMemoryTest(fPctToAllocate, &hHandle, &szTotalNumBlocks, &fMemAllocated, &fMemStartAddr, &fMemBlockSize))
		return lErr;

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = (T)hHandle;
	pfOutput[1] = (T)szTotalNumBlocks;
	pfOutput[2] = fMemAllocated;
	pfOutput[3] = fMemStartAddr;
	pfOutput[4] = fMemBlockSize;

	*ppfOutput = pfOutput;
	*plOutput = 5;

	return 0;
}

template <class T>
inline long Device<T>::FreeMemoryTest(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeMemoryTest(hHandle);
}


//=============================================================================
//	ImageOp Methods
//=============================================================================

template <class T>
inline long Device<T>::CreateImageOp(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 10, 10))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nNum = (int)pfInput[0];
	T fBrightnessProb = pfInput[1];
	T fBrightnessDelta = pfInput[2];
	T fContrastProb = pfInput[3];
	T fContrastLower = pfInput[4];
	T fContrastUpper = pfInput[5];
	T fSaturationProb = pfInput[6];
	T fSaturationLower = pfInput[7];
	T fSaturationUpper = pfInput[8];
	long lRandomSeed = (long)pfInput[9];

	if (lErr = m_memory.CreateImageOp(nNum, fBrightnessProb, fBrightnessDelta, fContrastProb, fContrastLower, fContrastUpper, fSaturationProb, fSaturationLower, fSaturationUpper, lRandomSeed, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeImageOp(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeImageOp(hHandle);
}


//=============================================================================
//	Cuda Methods
//=============================================================================

template <class T>
inline long Device<T>::CreateStream(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;

	bool bNonBlocking = (pfInput[0] == 1.0) ? true : false;
	int nIndex = (int)pfInput[1];

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateStream(&hHandle, bNonBlocking, nIndex))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeStream(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeStream(hHandle);
}

template <class T>
inline long Device<T>::SynchronizeStream(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.SynchronizeStream(hHandle);
}

template <class T>
inline long Device<T>::SynchronizeThread(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	return m_memory.SynchronizeThread();
}


//=============================================================================
//	CuDnn Methods
//=============================================================================

template <class T>
inline long Device<T>::CreateCuDNN(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;
	long hStream = 0;

	if (lErr = verifyInput(lInput, pfInput, 0, 1))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lInput > 0)
		hStream = (long)pfInput[0];

	if (lErr = m_memory.CreateCuDNN(hStream, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeCuDNN(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeCuDNN(hHandle);
}

template <class T>
inline long Device<T>::CreateTensorDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateTensorDesc(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeTensorDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeTensorDesc(hHandle);
}


template <class T>
inline long Device<T>::AddTensor(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hSrcDesc = (long)pfInput[2];
	long hSrc = (long)pfInput[3];
	int nSrcOffset = (int)pfInput[4];
	T fBeta = pfInput[5];
	long hDstDesc = (long)pfInput[6];
	long hDst = (long)pfInput[7];
	int nDstOffset = (int)pfInput[8];

	return m_memory.AddTensor(hHandle, fAlpha, hSrcDesc, hSrc, nSrcOffset, fBeta, hDstDesc, hDst, nDstOffset);
}


template <class T>
inline long Device<T>::CreateFilterDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateFilterDesc(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeFilterDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeFilterDesc(hHandle);
}

template <class T>
inline long Device<T>::SetFilterDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	long hHandle = (long)pfInput[0];
	bool bHalf = (bool)(pfInput[1] == 1) ? true : false;
	int n = (int)pfInput[2];
	int c = (int)pfInput[3];
	int h = (int)pfInput[4];
	int w = (int)pfInput[5];

	return m_memory.SetFilterDesc(hHandle, n, c, h, w, bHalf);
}

template <class T>
inline long Device<T>::CreateConvolutionDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateConvolutionDesc(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeConvolutionDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeConvolutionDesc(hHandle);
}

template <class T>
inline long Device<T>::SetConvolutionDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	long hHandle = (long)pfInput[0];
	bool bHalf = (bool)(pfInput[1] == 1) ? true : false;
	int hPad = (int)pfInput[2];
	int wPad = (int)pfInput[3];
	int hStride = (int)pfInput[4];
	int wStride = (int)pfInput[5];
	int hDilation = (int)pfInput[6];
	int wDilation = (int)pfInput[7];
	bool bUseTensorCores = (bool)(pfInput[8] == 1) ? true : false;

	if (m_nMajor == 0 && m_nMinor == 0)
	{
		cudaDeviceProp prop;
		if (lErr = cudaGetDeviceProperties(&prop, m_nDevice))
			return lErr;

		m_nMajor = prop.major;
		m_nMinor = prop.minor;
	}

	// FULL HALF mode only supported on compute mode > 5.3
	if (m_nMajor < 5 || (m_nMajor == 5 && m_nMinor < 3))
		bHalf = false;

	return m_memory.SetConvolutionDesc(hHandle, hPad, wPad, hStride, wStride, hDilation, wDilation, bHalf, bUseTensorCores);
}

template <class T>
inline long Device<T>::GetConvolutionInfo(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 8))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hBottomDesc = (long)pfInput[1];
	long hFilter = (long)pfInput[2];
	long hConvDesc = (long)pfInput[3];
	long hTopDesc = (long)pfInput[4];
	size_t lWsLimitInBytes = (size_t)pfInput[5];
	bool bUseTensorCores = (bool)(pfInput[6] == 1) ? true : false;
	int nPreferredFwdAlgo = -1;
	long algoFwd = 0;
	size_t lWsSizeFwd = 0;
	long algoBwdFilter = 0;
	size_t lWsSizeBwdFilter = 0;
	long algoBwdData = 0;
	size_t lWsSizeBwdData = 0;

	if (lInput >= 8)
		nPreferredFwdAlgo = (int)pfInput[7];

	if (lErr = m_memory.GetConvolutionInfo(hHandle, hBottomDesc, hFilter, hConvDesc, hTopDesc, lWsLimitInBytes, bUseTensorCores, &algoFwd, &lWsSizeFwd, &algoBwdFilter, &lWsSizeBwdFilter, &algoBwdData, &lWsSizeBwdData, nPreferredFwdAlgo))
		return lErr;

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pOutput = *ppfOutput;

	pOutput[0] = (T)algoFwd;
	pOutput[1] = (T)lWsSizeFwd;
	pOutput[2] = (T)algoBwdFilter;
	pOutput[3] = (T)lWsSizeBwdFilter;
	pOutput[4] = (T)algoBwdData;
	pOutput[5] = (T)lWsSizeBwdData;

	*plOutput = 6;
	*ppfOutput = pOutput;

	return 0;
}

template <class T>
inline long Device<T>::ConvolutionForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 17, 18))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	int nBottomOffset = (int)pfInput[4];
	long hFilterDesc = (long)pfInput[5];
	long hWeight = (long)pfInput[6];
	int nWeightOffset = (int)pfInput[7];
	long hConvDesc = (long)pfInput[8];
	long algo = (long)pfInput[9];
	long hWorkspace = (long)pfInput[10];
	int nWorkspaceOffset = (int)pfInput[11];
	size_t lWorkspaceSize = (size_t)pfInput[12];
	T fBeta = pfInput[13];
	long hTopDesc = (long)pfInput[14];
	long hTopData = (long)pfInput[15];
	int nTopOffset = (int)pfInput[16];
	bool bSyncStream = true;

	if (lInput > 17)
		bSyncStream = (pfInput[17] == 0) ? false : true;

	return m_memory.ConvolutionForward(hHandle, fAlpha, hBottomDesc, hBottomData, nBottomOffset, hFilterDesc, hWeight, nWeightOffset, hConvDesc, algo, hWorkspace, nWorkspaceOffset, lWorkspaceSize, fBeta, hTopDesc, hTopData, nTopOffset, bSyncStream);
}

template <class T>
inline long Device<T>::ConvolutionBackwardBias(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 10))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hTopDesc = (long)pfInput[2];
	long hTopDiff = (long)pfInput[3];
	int nTopOffset = (int)pfInput[4];
	T fBeta = pfInput[5];
	long hBiasDesc = (long)pfInput[6];
	long hBiasDiff = (long)pfInput[7];
	int nBiasOffset = (int)pfInput[8];
	bool bSyncStream = true;

	if (lInput > 9)
		bSyncStream = (pfInput[9] == 0) ? false : true;

	return m_memory.ConvolutionBackwardBias(hHandle, fAlpha, hTopDesc, hTopDiff, nTopOffset, fBeta, hBiasDesc, hBiasDiff, nBiasOffset, bSyncStream);
}

template <class T>
inline long Device<T>::ConvolutionBackwardFilter(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 17, 18))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	int nBottomOffset = (int)pfInput[4];
	long hTopDesc = (long)pfInput[5];
	long hTopDiff = (long)pfInput[6];
	int nTopOffset = (int)pfInput[7];
	long hConvDesc = (long)pfInput[8];
	long algo = (long)pfInput[9];
	long hWorkspace = (long)pfInput[10];
	int nWorkspaceOffset = (int)pfInput[11];
	size_t lWorkspaceSize = (size_t)pfInput[12];
	T fBeta = pfInput[13];
	long hFilterDesc = (long)pfInput[14];
	long hWeightDiff = (long)pfInput[15];
	int nWeightOffset = (int)pfInput[16];
	bool bSyncStream = true;

	if (lInput > 17)
		bSyncStream = (pfInput[17] == 0) ? false : true;

	return m_memory.ConvolutionBackwardFilter(hHandle, fAlpha, hBottomDesc, hBottomData, nBottomOffset, hTopDesc, hTopDiff, nTopOffset, hConvDesc, algo, hWorkspace, nWorkspaceOffset, lWorkspaceSize, fBeta, hFilterDesc, hWeightDiff, nWeightOffset, bSyncStream);
}

template <class T>
inline long Device<T>::ConvolutionBackwardData(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 17, 18))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hFilterDesc = (long)pfInput[2];
	long hWeight = (long)pfInput[3];
	int nWeightOffset = (int)pfInput[4];
	long hTopDesc = (long)pfInput[5];
	long hTopDiff = (long)pfInput[6];
	int nTopOffset = (int)pfInput[7];
	long hConvDesc = (long)pfInput[8];
	long algo = (long)pfInput[9];
	long hWorkspace = (long)pfInput[10];
	int nWorkspaceOffset = (int)pfInput[11];
	size_t lWorkspaceSize = (size_t)pfInput[12];
	T fBeta = pfInput[13];
	long hBottomDesc = (long)pfInput[14];
	long hBottomDiff = (long)pfInput[15];
	int nBottomOffset = (int)pfInput[16];
	bool bSyncStream = true;

	if (lInput > 17)
		bSyncStream = (pfInput[17] == 0) ? false : true;

	return m_memory.ConvolutionBackwardData(hHandle, fAlpha, hFilterDesc, hWeight, nWeightOffset, hTopDesc, hTopDiff, nTopOffset, hConvDesc, algo, hWorkspace, nWorkspaceOffset, lWorkspaceSize, fBeta, hBottomDesc, hBottomDiff, nBottomOffset, bSyncStream);
}


template <class T>
inline long Device<T>::CreatePoolingDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreatePoolingDesc(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreePoolingDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreePoolingDesc(hHandle);
}

template <class T>
inline long Device<T>::SetPoolingDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	long hHandle = (long)pfInput[0];
	PoolingMethod nMethod = (PoolingMethod)(int)pfInput[1];
	int h = (int)pfInput[2];
	int w = (int)pfInput[3];
	int hPad = (int)pfInput[4];
	int wPad = (int)pfInput[5];
	int hStride = (int)pfInput[6];
	int wStride = (int)pfInput[7];

	return m_memory.SetPoolingDesc(hHandle, nMethod, h, w, hPad, wPad, hStride, wStride);
}

template <class T>
inline long Device<T>::PoolingForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hPoolingDesc = (long)pfInput[1];
	T fAlpha = pfInput[2];
	long hBottomDesc = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	T fBeta = pfInput[5];
	long hTopDesc = (long)pfInput[6];
	long hTopData = (long)pfInput[7];

	return m_memory.PoolingForward(hHandle, hPoolingDesc, fAlpha, hBottomDesc, hBottomData, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::PoolingBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 12, 12))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hPoolingDesc = (long)pfInput[1];
	T fAlpha = pfInput[2];
	long hTopDataDesc = (long)pfInput[3];
	long hTopData = (long)pfInput[4];
	long hTopDiffDesc = (long)pfInput[5];
	long hTopDiff = (long)pfInput[6];
	long hBottomDataDesc = (long)pfInput[7];
	long hBottomData = (long)pfInput[8];
	T fBeta = pfInput[9];
	long hBottomDiffDesc = (long)pfInput[10];
	long hBottomDiff = (long)pfInput[11];

	return m_memory.PoolingBackward(hHandle, hPoolingDesc, fAlpha, hTopDataDesc, hTopData, hTopDiffDesc, hTopDiff, hBottomDataDesc, hBottomData, fBeta, hBottomDiffDesc, hBottomDiff);
}

template <class T>
inline long Device<T>::DeriveBatchNormDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	long hFwdScaleBiasMeanVarDesc = (long)pfInput[0];
	long hFwdBottomDesc = (long)pfInput[1];
	long hBwdScaleBiasMeanVarDesc = (long)pfInput[2];
	long hBwdBottomDesc = (long)pfInput[3];
	int mode = (int)pfInput[4];

	return m_memory.DeriveBatchNormDesc(hFwdScaleBiasMeanVarDesc, hFwdBottomDesc, hBwdScaleBiasMeanVarDesc, hBwdBottomDesc, mode);
}

template <class T>
inline long Device<T>::BatchNormForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 18, 18))
		return lErr;

	long hHandle = (long)pfInput[0];
	int mode = (int)pfInput[1];
	T fAlpha = pfInput[2];
	T fBeta = pfInput[3];
	long hFwdBottomDesc = (long)pfInput[4];
	long hBottomData = (long)pfInput[5];
	long hFwdTopDesc = (long)pfInput[6];
	long hTopData = (long)pfInput[7];
	long hFwdScaleBiasMeanVarDesc = (long)pfInput[8];
	long hScaleData = (long)pfInput[9];
	long hBiasData = (long)pfInput[10];
	T fFactor = pfInput[11];
	long hGlobalMean = (long)pfInput[12];
	long hGlobalVar = (long)pfInput[13];
	T fEps = pfInput[14];
	long hSaveMean = (long)pfInput[15];
	long hSaveVar = (long)pfInput[16];
	bool bTraining = (pfInput[17] == 0) ? false : true;

	return m_memory.BatchNormForward(hHandle, mode, fAlpha, fBeta, hFwdBottomDesc, hBottomData, hFwdTopDesc, hTopData, hFwdScaleBiasMeanVarDesc, hScaleData, hBiasData, fFactor, hGlobalMean, hGlobalVar, fEps, hSaveMean, hSaveVar, bTraining);
}

template <class T>
inline long Device<T>::BatchNormBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 19, 19))
		return lErr;

	long hHandle = (long)pfInput[0];
	int mode = (int)pfInput[1];
	T fAlphaDiff = pfInput[2];
	T fBetaDiff = pfInput[3];
	T fAlphaParamDiff = pfInput[4];
	T fBetaParamDiff = pfInput[5];
	long hBtmBottomDesc = (long)pfInput[6];
	long hBottomData = (long)pfInput[7];
	long hTopDiffDesc = (long)pfInput[8];
	long hTopDiff = (long)pfInput[9];
	long hBottomDiffDesc = (long)pfInput[10];
	long hBottomDiff = (long)pfInput[11];
	long hBwdScaleBiasMeanVarDesc = (long)pfInput[12];
	long hScaleData = (long)pfInput[13];
	long hScaleDiff = (long)pfInput[14];
	long hBiasDiff = (long)pfInput[15];
	T fEps = pfInput[16];
	long hSaveMean = (long)pfInput[17];
	long hSaveVar = (long)pfInput[18];

	return m_memory.BatchNormBackward(hHandle, mode, fAlphaDiff, fBetaDiff, fAlphaParamDiff, fBetaParamDiff, hBtmBottomDesc, hBottomData, hTopDiffDesc, hTopDiff, hBottomDiffDesc, hBottomDiff, hBwdScaleBiasMeanVarDesc, hScaleData, hScaleDiff, hBiasDiff, fEps, hSaveMean, hSaveVar);
}


template <class T>
inline long Device<T>::CreateRnnDataDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateRnnDataDesc1(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeRnnDataDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeRnnDataDesc1(hHandle);
}

template <class T>
inline long Device<T>::SetRnnDataDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 8))
		return lErr;

	long hRnnDataDesc = (long)pfInput[0];
	int layout = (int)pfInput[1];
	int nMaxSeqLen = (int)pfInput[2];
	int nBatchSize = (int)pfInput[3];
	long nInputSize = (long)pfInput[4];
	bool bBidirectional = false;
	int* rgSeqLen = NULL;
	int nIdx1 = 6;
	int nIdx2 = 8;

	if (lInput > 5)
		bBidirectional = (pfInput[5] == 0) ? false : true;

	if (lInput != nIdx1 && lInput < nIdx2)
		return ERROR_PARAM_OUT_OF_RANGE;

	if (lInput > nIdx1)
	{
		int nCount = nIdx2 - nIdx1;
		rgSeqLen = (int*)malloc(sizeof(int) * nCount);
		if (rgSeqLen == NULL)
			return ERROR_OUTOFMEMORY;

		for (int i = 0; i < nCount; i++)
		{
			rgSeqLen[i] = (int)pfInput[i];
		}
	}

	lErr = m_memory.SetRnnDataDesc1(hRnnDataDesc, (RnnDataLayout)layout, nMaxSeqLen, nBatchSize, nInputSize, bBidirectional, rgSeqLen);

	if (rgSeqLen != NULL)
		free(rgSeqLen);

	return lErr;
}


template <class T>
inline long Device<T>::CreateRnnDataDescEx(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateRnnDataDesc2(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeRnnDataDescEx(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeRnnDataDesc2(hHandle);
}

template <class T>
inline long Device<T>::SetRnnDataDescEx(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 7))
		return lErr;

	long hRnnDataDesc = (long)pfInput[0];
	int layout = (int)pfInput[1];
	int nMaxSeqLen = (int)pfInput[2];
	int nBatchSize = (int)pfInput[3];
	long nVectorSize = (long)pfInput[4];
	int nIdx1 = 5;
	int nIdx2 = 7;

	if (lInput != nIdx1 && lInput < nIdx2)
		return ERROR_PARAM_OUT_OF_RANGE;

	int* rgSeqLen = NULL;
	if (lInput > nIdx1)
	{
		int nCount = nIdx2 - nIdx2;
		rgSeqLen = (int*)malloc(sizeof(int) * nCount);
		if (rgSeqLen == NULL)
			return ERROR_OUTOFMEMORY;

		for (int i = 0; i < nCount; i++)
		{
			rgSeqLen[i] = (int)pfInput[nIdx1];
			nIdx1++;
		}
	}

	lErr = m_memory.SetRnnDataDesc2(hRnnDataDesc, (RnnDataLayout)layout, nMaxSeqLen, nBatchSize, nVectorSize, rgSeqLen);

	free(rgSeqLen);

	return lErr;
}

template <class T>
inline long Device<T>::CreateRnnDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateRnnDesc(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeRnnDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeRnnDesc(hHandle);
}

template <class T>
inline long Device<T>::SetRnnDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 8))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hRnnDesc = (long)pfInput[1];
	int nHiddenCount = (int)pfInput[2];
	int nNumLayers = (int)pfInput[3];
	long hDropoutDesc = (long)pfInput[4];
	int mode = (int)pfInput[5];
	bool bUseTensorCores = (bool)(pfInput[6] == 1) ? true : false;
	int dir = 0;

	if (lInput > 7)
		dir = (int)pfInput[7];

	return m_memory.SetRnnDesc(hHandle, hRnnDesc, nHiddenCount, nNumLayers, hDropoutDesc, (RnnMode)mode, bUseTensorCores, (RnnDirection)dir);
}

template <class T>
inline long Device<T>::GetRnnParamCount(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 4))
		return lErr;

	int nCount;
	long hHandle = (long)pfInput[0];
	long hRnnDesc = (long)pfInput[1];
	long hXDesc = (long)pfInput[2];

	bool bUseExtendedVersion = false;	// When true, requires that hXDesc and hYDesc be a handle to cudnnRnnDataDesc_t (created with CreateRnnDataDescEx)
	if (lInput > 3)
		bUseExtendedVersion = (pfInput[3] != 0) ? true : false;

	if (bUseExtendedVersion)
	{
		if (lErr = m_memory.GetRnnParamCountEx(hHandle, hRnnDesc, hXDesc, &nCount))
			return lErr;
	}
	else
	{
		if (lErr = m_memory.GetRnnParamCount(hHandle, hRnnDesc, hXDesc, &nCount))
			return lErr;
	}

	setOutput((T)nCount, plOutput, ppfOutput);

	return 0;
}

template <class T>
inline long Device<T>::GetRnnWorkspaceCount(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, MAX_ARG))
		return lErr;

	size_t nWsCount = 0;
	size_t nResCount = 0;
	long hHandle = (long)pfInput[0];
	long hRnnDesc = (long)pfInput[1];
	bool bUseExtendedVersion = (pfInput[2] != 0) ? true : false;
	long hXDesc = (long)pfInput[3];

	if (bUseExtendedVersion)
	{
		if (lErr = m_memory.GetRnnWorkspaceCountEx(hHandle, hRnnDesc, hXDesc, &nWsCount, &nResCount))
			return lErr;
	}
	else
	{
		if (lErr = m_memory.GetRnnWorkspaceCount(hHandle, hRnnDesc, hXDesc, &nWsCount, &nResCount))
			return lErr;
	}

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pOutput = *ppfOutput;

	pOutput[0] = (T)nWsCount;
	pOutput[1] = (T)nResCount;

	*plOutput = 2;
	*ppfOutput = pOutput;

	return lErr;
}

template <class T>
inline long Device<T>::GetRnnLinLayerParams(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 8))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hRnnDesc = (long)pfInput[1];
	int nLayer = (int)pfInput[2];
	long hXDesc = (long)pfInput[3];
	long hWtDesc = (long)pfInput[4];
	long hWtData = (long)pfInput[5];
	int nLinLayer = (int)pfInput[6];

	bool bUseExtendedVersion = false;	// When true, requires that hXDesc and hYDesc be a handle to cudnnRnnDataDesc_t (created with CreateRnnDataDescEx)
	if (lInput > 7)
		bUseExtendedVersion = (pfInput[7] != 0) ? true : false;

	int nWtCount = 0;
	long hWt = 0;
	int nBiasCount = 0;
	long hBias = 0;

	if (bUseExtendedVersion)
	{
		if (lErr = m_memory.GetRnnLinLayerParamsEx(hHandle, hRnnDesc, nLayer, hXDesc, hWtDesc, hWtData, nLinLayer, &nWtCount, &hWt, &nBiasCount, &hBias))
			return lErr;
	}
	else
	{
		if (lErr = m_memory.GetRnnLinLayerParams(hHandle, hRnnDesc, nLayer, hXDesc, hWtDesc, hWtData, nLinLayer, &nWtCount, &hWt, &nBiasCount, &hBias))
			return lErr;
	}

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pOutput = *ppfOutput;

	pOutput[0] = (T)nWtCount;
	pOutput[1] = (T)hWt;
	pOutput[2] = (T)nBiasCount;
	pOutput[3] = (T)hBias;

	*plOutput = 4;
	*ppfOutput = pOutput;

	return 0;
}

template <class T>
inline long Device<T>::RnnForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 21, 22))
		return lErr;
	
	long hHandle = (long)pfInput[0];
	long hRnnDesc = (long)pfInput[1];
	long hXDesc = (long)pfInput[2];

	int nIdx = 3;
	long hXData = (long)pfInput[nIdx];
	nIdx++;
	long hHxDesc = (long)pfInput[nIdx];
	nIdx++;
	long hHxData = (long)pfInput[nIdx];
	nIdx++;
	long hCxDesc = (long)pfInput[nIdx];
	nIdx++;
	long hCxData = (long)pfInput[nIdx];
	nIdx++;
	long hWtDesc = (long)pfInput[nIdx];
	nIdx++;
	long hWtData = (long)pfInput[nIdx];
	nIdx++;
	long hYDesc = (long)pfInput[nIdx];
	nIdx++;
	long hYData = (long)pfInput[nIdx];
	nIdx++;
	long hHyDesc = (long)pfInput[nIdx];
	nIdx++;
	long hHyData = (long)pfInput[nIdx];
	nIdx++;
	long hCyDesc = (long)pfInput[nIdx];
	nIdx++;
	long hCyData = (long)pfInput[nIdx];
	nIdx++;
	long hWorkspace = (long)pfInput[nIdx];
	nIdx++;
	size_t nWsCount = (size_t)pfInput[nIdx];
	nIdx++;
	long hReserved = (long)pfInput[nIdx];
	nIdx++;
	size_t nResCount = (size_t)pfInput[nIdx];
	nIdx++;
	bool bTraining = (pfInput[nIdx] == 0) ? false : true;
	nIdx++;

	bool bUseExtendedVersion = false;	// When true, requires that hXDesc and hYDesc be a handle to cudnnRnnDataDesc_t (created with CreateRnnDataDescEx)
	if (nIdx < lInput)
		bUseExtendedVersion = (pfInput[nIdx] != 0) ? true : false;

	if (bUseExtendedVersion)
		return m_memory.RnnForwardEx(hHandle, hRnnDesc, hXDesc, hXData, hHxDesc, hHxData, hCxDesc, hCxData, hWtDesc, hWtData, hYDesc, hYData, hHyDesc, hHyData, hCyDesc, hCyData, hWorkspace, nWsCount, hReserved, nResCount, bTraining);

	return m_memory.RnnForward(hHandle, hRnnDesc, hXDesc, hXData, hHxDesc, hHxData, hCxDesc, hCxData, hWtDesc, hWtData, hYDesc, hYData, hHyDesc, hHyData, hCyDesc, hCyData, hWorkspace, nWsCount, hReserved, nResCount, bTraining);
}

template <class T>
inline long Device<T>::RnnBackwardData(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 25, 26))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hRnnDesc = (long)pfInput[1];
	long hYDesc = (int)pfInput[2];

	int nIdx = 3;
	long hYData = (long)pfInput[nIdx];
	nIdx++;
	long hYDiff = (long)pfInput[nIdx];
	nIdx++;

	long hHyDesc = (long)pfInput[nIdx];
	nIdx++;
	long hHyDiff = (long)pfInput[nIdx];
	nIdx++;
	long hCyDesc = (long)pfInput[nIdx];
	nIdx++;
	long hCyDiff = (long)pfInput[nIdx];
	nIdx++;

	long hWtDesc = (long)pfInput[nIdx];
	nIdx++;
	long hWtData = (long)pfInput[nIdx];
	nIdx++;

	long hHxDesc = (long)pfInput[nIdx];
	nIdx++;
	long hHxData = (long)pfInput[nIdx];
	nIdx++;
	long hCxDesc = (long)pfInput[nIdx];
	nIdx++;
	long hCxData = (long)pfInput[nIdx];
	nIdx++;
	long hXDesc = (long)pfInput[nIdx];
	nIdx++;
	long hXDiff = (long)pfInput[nIdx];
	nIdx++;

	long hdHxDesc = (long)pfInput[nIdx];
	nIdx++;
	long hHxDiff = (long)pfInput[nIdx];
	nIdx++;
	long hdCxDesc = (long)pfInput[nIdx];
	nIdx++;
	long hCxDiff = (long)pfInput[nIdx];
	nIdx++;

	long hWorkspace = (long)pfInput[nIdx];
	nIdx++;
	size_t nWsCount = (size_t)pfInput[nIdx];
	nIdx++;
	long hReserved = (long)pfInput[nIdx];
	nIdx++;
	size_t nResCount = (size_t)pfInput[nIdx];
	nIdx++;

	bool bUseExtendedVersion = false;	// When true, requires that hXDesc and hYDesc be a handle to cudnnRnnDataDesc_t (created with CreateRnnDataDescEx)
	if (nIdx < lInput)
		bUseExtendedVersion = (pfInput[nIdx] != 0) ? true : false;

	if (bUseExtendedVersion)
		return m_memory.RnnBackwardDataEx(hHandle, hRnnDesc, hYDesc, hYData, hYDiff, hHyDesc, hHyDiff, hCyDesc, hCyDiff, hWtDesc, hWtData, hHxDesc, hHxData, hCxDesc, hCxData, hXDesc, hXDiff, hdHxDesc, hHxDiff, hdCxDesc, hCxDiff, hWorkspace, nWsCount, hReserved, nResCount);

	return m_memory.RnnBackwardData(hHandle, hRnnDesc, hYDesc, hYData, hYDiff, hHyDesc, hHyDiff, hCyDesc, hCyDiff, hWtDesc, hWtData, hHxDesc, hHxData, hCxDesc, hCxData, hXDesc, hXDiff, hdHxDesc, hHxDiff, hdCxDesc, hCxDiff, hWorkspace, nWsCount, hReserved, nResCount);
}

template <class T>
inline long Device<T>::RnnBackwardWeights(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 14, 15))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hRnnDesc = (long)pfInput[1];
	long hXDesc = (long)pfInput[2];

	int nIdx = 3;
	long hXData = (long)pfInput[nIdx];
	nIdx++;

	long hHxDesc = (long)pfInput[nIdx];
	nIdx++;
	long hHxData = (long)pfInput[nIdx];
	nIdx++;

	long hYDesc = (long)pfInput[nIdx];
	nIdx++;
	long hYData = (long)pfInput[nIdx];
	nIdx++;

	long hWorkspace = (long)pfInput[nIdx];
	nIdx++;
	size_t nWsCount = (size_t)pfInput[nIdx];
	nIdx++;

	long hWtDesc = (long)pfInput[nIdx];
	nIdx++;
	long hWtDiff = (long)pfInput[nIdx];
	nIdx++;

	long hReserved = (long)pfInput[nIdx];
	nIdx++;
	size_t nResCount = (size_t)pfInput[nIdx];
	nIdx++;

	bool bUseExtendedVersion = false;	// When true, requires that hXDesc and hYDesc be a handle to cudnnRnnDataDesc_t (created with CreateRnnDataDescEx)
	if (nIdx < lInput)
		bUseExtendedVersion = (pfInput[nIdx] != 0) ? true : false;

	if (bUseExtendedVersion)
		return m_memory.RnnBackwardWeightsEx(hHandle, hRnnDesc, hXDesc, hXData, hHxDesc, hHxData, hYDesc, hYData, hWorkspace, nWsCount, hWtDesc, hWtDiff, hReserved, nResCount);

	return m_memory.RnnBackwardWeights(hHandle, hRnnDesc, hXDesc, hXData, hHxDesc, hHxData, hYDesc, hYData, hWorkspace, nWsCount, hWtDesc, hWtDiff, hReserved, nResCount);
}

template <class T>
inline long Device<T>::cuda_accuracy_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 11, 12))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBtmData = (long)pfInput[1];
	long hBtmLabel = (long)pfInput[2];
	long hAccData = (long)pfInput[3];
	int nOuterNum = (int)pfInput[4];
	int nDim = (int)pfInput[5];
	int nInnerNum = (int)pfInput[6];
	int nNumLabels = (int)pfInput[7];
	int nTopK = (int)pfInput[8];
	long hCounts = (long)pfInput[9];
	bool bPerClass = (pfInput[10] == 0) ? false : true;
	int nIgnoreLabel = 0;
	bool bIgnoreLabel = false;

	if (lInput > 11)
	{
		nIgnoreLabel = (int)pfInput[11];
		bIgnoreLabel = true;
	}

	return m_math.accuracy_fwd(nCount, hBtmData, hBtmLabel, hAccData, nOuterNum, nDim, nInnerNum, nNumLabels, nTopK, hCounts, bPerClass, bIgnoreLabel, nIgnoreLabel);
}


template <class T>
inline long Device<T>::cuda_batchreidx_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int nCount = (int)pfInput[0];
	int nInnerDim = (int)pfInput[1];
	long hBottomData = (long)pfInput[2];
	long hPermutData = (long)pfInput[3];
	long hTopData = (long)pfInput[4];

	return m_math.batchreidx_fwd(nCount, nInnerDim, hBottomData, hPermutData, hTopData);
}


template <class T>
inline long Device<T>::cuda_batchreidx_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nInnerDim = (int)pfInput[1];
	long hTopDiff = (long)pfInput[2];
	long hTopIdx = (long)pfInput[3];
	long hBegins = (long)pfInput[4];
	long hCounts = (long)pfInput[5];
	long hBottomDiff = (long)pfInput[6];

	return m_math.batchreidx_bwd(nCount, nInnerDim, hTopDiff, hTopIdx, hBegins, hCounts, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_embed_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hWeight = (int)pfInput[2];
	int nM = (int)pfInput[3];
	int nN = (int)pfInput[4];
	int nK = (int)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_math.embed_fwd(nCount, hBottomData, hWeight, nM, nN, nK, hTopData);
}


template <class T>
inline long Device<T>::cuda_embed_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopDiff = (int)pfInput[2];
	int nM = (int)pfInput[3];
	int nN = (int)pfInput[4];
	int nK = (int)pfInput[5];
	long hWeightDiff = (long)pfInput[6];

	return m_math.embed_bwd(nCount, hBottomData, hTopDiff, nM, nN, nK, hWeightDiff);
}


template <class T>
inline long Device<T>::cuda_pooling_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 18, 18))
		return lErr;

	int nMethod = (int)pfInput[0];
	int nCount = (int)pfInput[1];
	long hBottomData = (long)pfInput[2];
	int nNum = (int)pfInput[3];
	int nChannels = (int)pfInput[4];
	int h = (int)pfInput[5];
	int w = (int)pfInput[6];
	int hPooled = (int)pfInput[7];
	int wPooled = (int)pfInput[8];
	int hKernel = (int)pfInput[9];
	int wKernel = (int)pfInput[10];
	int hStride = (int)pfInput[11];
	int wStride = (int)pfInput[12];
	int hPad = (int)pfInput[13];
	int wPad = (int)pfInput[14];
	long hTopData = (long)pfInput[15];
	long hMask = (long)pfInput[16];
	long hTopMask = (long)pfInput[17];

	return m_math.pooling_fwd(nMethod, nCount, hBottomData, nNum, nChannels, h, w, hPooled, wPooled, hKernel, wKernel, hStride, wStride, hPad, wPad, hTopData, hMask, hTopMask);
}

template <class T>
inline long Device<T>::cuda_pooling_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 18, 18))
		return lErr;

	int nMethod = (int)pfInput[0];
	int nCount = (int)pfInput[1];
	long hTopDiff = (long)pfInput[2];
	int nNum = (int)pfInput[3];
	int nChannels = (int)pfInput[4];
	int h = (int)pfInput[5];
	int w = (int)pfInput[6];
	int hPooled = (int)pfInput[7];
	int wPooled = (int)pfInput[8];
	int hKernel = (int)pfInput[9];
	int wKernel = (int)pfInput[10];
	int hStride = (int)pfInput[11];
	int wStride = (int)pfInput[12];
	int hPad = (int)pfInput[13];
	int wPad = (int)pfInput[14];
	long hBottomDiff = (long)pfInput[15];
	long hMask = (long)pfInput[16];
	long hTopMask = (long)pfInput[17];

	return m_math.pooling_bwd(nMethod, nCount, hTopDiff, nNum, nChannels, h, w, hPooled, wPooled, hKernel, wKernel, hStride, wStride, hPad, wPad, hBottomDiff, hMask, hTopMask);
}


template <class T>
inline long Device<T>::cuda_unpooling_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 17, 17))
		return lErr;

	int nMethod = (int)pfInput[0];
	int nCount = (int)pfInput[1];
	long hBottomData = (long)pfInput[2];
	int nNum = (int)pfInput[3];
	int nChannels = (int)pfInput[4];
	int h = (int)pfInput[5];
	int w = (int)pfInput[6];
	int hUnPooled = (int)pfInput[7];
	int wUnPooled = (int)pfInput[8];
	int hKernel = (int)pfInput[9];
	int wKernel = (int)pfInput[10];
	int hStride = (int)pfInput[11];
	int wStride = (int)pfInput[12];
	int hPad = (int)pfInput[13];
	int wPad = (int)pfInput[14];
	long hTopData = (long)pfInput[15];
	long hBottomMask = (long)pfInput[16];

	return m_math.unpooling_fwd(nMethod, nCount, hBottomData, nNum, nChannels, h, w, hUnPooled, wUnPooled, hKernel, wKernel, hStride, wStride, hPad, wPad, hTopData, hBottomMask);
}

template <class T>
inline long Device<T>::cuda_unpooling_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 17, 17))
		return lErr;

	int nMethod = (int)pfInput[0];
	int nCount = (int)pfInput[1];
	long hTopDiff = (long)pfInput[2];
	int nNum = (int)pfInput[3];
	int nChannels = (int)pfInput[4];
	int h = (int)pfInput[5];
	int w = (int)pfInput[6];
	int hUnPooled = (int)pfInput[7];
	int wUnPooled = (int)pfInput[8];
	int hKernel = (int)pfInput[9];
	int wKernel = (int)pfInput[10];
	int hStride = (int)pfInput[11];
	int wStride = (int)pfInput[12];
	int hPad = (int)pfInput[13];
	int wPad = (int)pfInput[14];
	long hBottomDiff = (long)pfInput[15];
	long hBottomMask = (long)pfInput[16];

	return m_math.unpooling_bwd(nMethod, nCount, hTopDiff, nNum, nChannels, h, w, hUnPooled, wUnPooled, hKernel, wKernel, hStride, wStride, hPad, wPad, hBottomDiff, hBottomMask);
}


template <class T>
inline long Device<T>::CreateDropoutDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateDropoutDesc(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}


template <class T>
inline long Device<T>::FreeDropoutDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeDropoutDesc(hHandle);
}

template <class T>
inline long Device<T>::SetDropoutDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hDropoutDesc = (long)pfInput[1];
	T fDropout = pfInput[2];
	long hStates = (long)pfInput[3];
	long lSeed = (long)pfInput[4];

	return m_memory.SetDropoutDesc(hHandle, hDropoutDesc, fDropout, hStates, lSeed);
}

template <class T>
inline long Device<T>::DropoutForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hDropoutDesc = (long)pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottom = (long)pfInput[3];
	long hTopDesc = (long)pfInput[4];
	long hTop = (long)pfInput[5];
	long hReserved = (long)pfInput[6];

	return m_memory.DropoutForward(hHandle, hDropoutDesc, hBottomDesc, hBottom, hTopDesc, hTop, hReserved);
}

template <class T>
inline long Device<T>::DropoutBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hDropoutDesc = (long)pfInput[1];
	long hTopDesc = (long)pfInput[2];
	long hTop = (long)pfInput[3];
	long hBottomDesc = (long)pfInput[4];
	long hBottom = (long)pfInput[5];
	long hReserved = (long)pfInput[6];

	return m_memory.DropoutBackward(hHandle, hDropoutDesc, hTopDesc, hTop, hBottomDesc, hBottom, hReserved);
}


template <class T>
inline long Device<T>::CreateLRNDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateLRNDesc(&hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeLRNDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeLRNDesc(hHandle);
}

template <class T>
inline long Device<T>::SetLRNDesc(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	long hHandle = (long)pfInput[0];
	unsigned int nSize = (unsigned int)pfInput[1];
	T fAlpha = pfInput[2];
	T fBeta = pfInput[3];
	T fK = pfInput[4];

	return m_memory.SetLRNDesc(hHandle, nSize, fAlpha, fBeta, fK);
}


template <class T>
inline long Device<T>::TanhForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	T fBeta = pfInput[4];
	long hTopDesc = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_memory.TanhForward(hHandle, fAlpha, hBottomDesc, hBottomData, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::TanhBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 11, 11))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hTopDataDesc = (long)pfInput[2];
	long hTopData = (long)pfInput[3];
	long hTopDiffDesc = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	long hBottomDataDesc = (long)pfInput[6];
	long hBottomData = (long)pfInput[7];
	T fBeta = pfInput[8];
	long hBottomDiffDesc = (long)pfInput[9];
	long hBottomDiff = (long)pfInput[10];

	return m_memory.TanhBackward(hHandle, fAlpha, hTopDataDesc, hTopData, hTopDiffDesc, hTopDiff, hBottomDataDesc, hBottomData, fBeta, hBottomDiffDesc, hBottomDiff);
}


template <class T>
inline long Device<T>::EluForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	T fBeta = pfInput[4];
	long hTopDesc = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_memory.EluForward(hHandle, fAlpha, hBottomDesc, hBottomData, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::EluBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 11, 11))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hTopDataDesc = (long)pfInput[2];
	long hTopData = (long)pfInput[3];
	long hTopDiffDesc = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	long hBottomDataDesc = (long)pfInput[6];
	long hBottomData = (long)pfInput[7];
	T fBeta = pfInput[8];
	long hBottomDiffDesc = (long)pfInput[9];
	long hBottomDiff = (long)pfInput[10];

	return m_memory.EluBackward(hHandle, fAlpha, hTopDataDesc, hTopData, hTopDiffDesc, hTopDiff, hBottomDataDesc, hBottomData, fBeta, hBottomDiffDesc, hBottomDiff);
}

template <class T>
inline long Device<T>::cuda_clip_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	T fMin = pfInput[3];
	T fMax = pfInput[4];

	return m_math.clip_fwd(nCount, hBottomData, hTopData, fMin, fMax);
}

template <class T>
inline long Device<T>::cuda_clip_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hBottomData = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];
	T fMin = pfInput[4];
	T fMax = pfInput[5];

	return m_math.clip_bwd(nCount, hTopDiff, hBottomData, hBottomDiff, fMin, fMax);
}

template <class T>
inline long Device<T>::cuda_math_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	int nFunction = (int)pfInput[3];

	return m_math.math_fwd(nCount, hBottomData, hTopData, nFunction);
}

template <class T>
inline long Device<T>::cuda_math_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	int nFunction = (int)pfInput[5];

	return m_math.math_bwd(nCount, hTopDiff, hTopData, hBottomDiff, hBottomData, nFunction);
}

template <class T>
inline long Device<T>::cuda_mae_loss_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hPredicted = (long)pfInput[1];
	long hTarget = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];

	return m_math.mae_loss_bwd(nCount, hPredicted, hTarget, hBottomDiff);
}

template <class T>
inline long Device<T>::cuda_mish_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	T fThreshold = pfInput[3];

	return m_math.mish_fwd(nCount, hBottomData, hTopData, fThreshold);
}

template <class T>
inline long Device<T>::cuda_mish_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	T fThreshold = pfInput[5];
	int nMethod = 1;

	if (lInput > 6)
		nMethod = (int)pfInput[6];

	return m_math.mish_bwd(nCount, hTopDiff, hTopData, hBottomDiff, hBottomData, fThreshold, nMethod);
}

template <class T>
inline long Device<T>::cuda_tanh_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];

	return m_math.tanh_fwd(nCount, hBottomData, hTopData);
}

template <class T>
inline long Device<T>::cuda_tanh_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];

	return m_math.tanh_bwd(nCount, hTopDiff, hTopData, hBottomDiff);
}


template <class T>
inline long Device<T>::SigmoidForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	T fBeta = pfInput[4];
	long hTopDesc = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_memory.SigmoidForward(hHandle, fAlpha, hBottomDesc, hBottomData, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::SigmoidBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 11, 11))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hTopDataDesc = (long)pfInput[2];
	long hTopData = (long)pfInput[3];
	long hTopDiffDesc = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	long hBottomDataDesc = (long)pfInput[6];
	long hBottomData = (long)pfInput[7];
	T fBeta = pfInput[8];
	long hBottomDiffDesc = (long)pfInput[9];
	long hBottomDiff = (long)pfInput[10];

	return m_memory.SigmoidBackward(hHandle, fAlpha, hTopDataDesc, hTopData, hTopDiffDesc, hTopDiff, hBottomDataDesc, hBottomData, fBeta, hBottomDiffDesc, hBottomDiff);
}

template <class T>
inline long Device<T>::CreatePCA(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 7, 9))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nMaxIterations = (int)pfInput[0];
	int nM = (int)pfInput[1];
	int nN = (int)pfInput[2];
	int nK = (int)pfInput[3];
	long hData = (long)pfInput[4];
	long hScoresResult = (long)pfInput[5];
	long hLoadsResult = (long)pfInput[6];
	long hResiduals = 0;
	long hEigenvalues = 0;

	if (lInput > 7)
		hResiduals = (long)pfInput[7];

	if (lInput > 8)
		hEigenvalues = (long)pfInput[8];

	if (lErr = m_memory.CreatePCA(nMaxIterations, nM, nN, nK, hData, hScoresResult, hLoadsResult, hResiduals, hEigenvalues, &m_math, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreePCA(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreePCA(hHandle);
}

template <class T>
inline long Device<T>::RunPCA(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 2))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)pfInput[0];
	int nSteps = 1;
	bool bDone = FALSE;
	int nCurrentIteration = 0;
	int nCurrentK = 0;

	if (lInput > 1)
		nSteps = (int)pfInput[1];

	if (lErr = m_memory.RunPCA(hHandle, nSteps, &bDone, &nCurrentIteration, &nCurrentK))
		return lErr;

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = (bDone) ? T(0) : T(1);
	pfOutput[1] = T(nCurrentIteration);
	pfOutput[2] = T(nCurrentK);

	*plOutput = 3;
	*ppfOutput = pfOutput;

	return 0;
}


template <class T>
inline long Device<T>::CreateTsneGaussianPerplexity(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	unsigned int nN = (unsigned int)pfInput[0];
	unsigned int nD = (unsigned int)pfInput[1];
	unsigned int nK = (unsigned int)pfInput[2];
	long hX = (long)pfInput[3];			// on gpu
	long hCurP = (long)pfInput[4];		// on gpu
	long hValP = (long)pfInput[5];		// on gpu
	long hRowP = (long)pfInput[6];		// on host
	long hColP = (long)pfInput[7];		// on host
	T fPerplexity = pfInput[8];

	if (lErr = m_memory.CreateTsneGaussianPerplexity(nN, nD, nK, hX, hCurP, hValP, hRowP, hColP, fPerplexity, &m_math, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeTsneGaussianPerplexity(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeTsneGaussianPerplexity(hHandle);
}

template <class T>
inline long Device<T>::FindTsneGaussianPerplexity(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)pfInput[0];
	bool bDone = FALSE;
	int nCurrentIteration = 0;
	int nMaxIteration = 0;

	if (lErr = m_memory.FindTsneGaussianPerplexity(hHandle, &bDone, &nCurrentIteration, &nMaxIteration))
		return lErr;

	// ppfOutput has up to MAX_OUTPUT(16) pre-allocated items
	T* pfOutput = *ppfOutput;

	pfOutput[0] = (bDone) ? T(0) : T(1);
	pfOutput[1] = T(nCurrentIteration);
	pfOutput[2] = T(nMaxIteration);

	*plOutput = 3;
	*ppfOutput = pfOutput;

	return 0;
}


template <class T>
inline long Device<T>::CreateTsne(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	unsigned int nN = (unsigned int)pfInput[0];
	unsigned int nD = (unsigned int)pfInput[1];
	long hY = (long)pfInput[2];
	long hValP = (long)pfInput[3];
	long hRowP = (long)pfInput[4];
	long hColP = (long)pfInput[5];
	long hdC = (long)pfInput[6];
	T fTheta = pfInput[7];

	if (lErr = m_memory.CreateTsne(nN, nD, hY, hValP, hRowP, hColP, hdC, fTheta, &m_math, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeTsne(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeTsne(hHandle);
}


template <class T>
inline long Device<T>::ComputeTsneGradient(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;

	long hHandle = (long)pfInput[0];
	bool bValPUpdated = (pfInput[1] == 1) ? true : false;

	return m_memory.ComputeTsneGradient(hHandle, bValPUpdated);
}


template <class T>
inline long Device<T>::EvaluateTsneError(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fErr;

	if (lErr = m_memory.EvaluateTsneError(hHandle, &fErr))
		return lErr;

	return setOutput(fErr, plOutput, ppfOutput);
}


template <class T>
inline ncclHandle<T>* Device<T>::GetNccl(long hNccl)
{
	return m_memory.GetNCCL(hNccl);
}

template <class T>
inline long Device<T>::SetNccl(ncclHandle<T>* pNccl, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	pNccl->Update(&m_memory, &m_math);

	if (lErr = m_memory.SetNCCL(pNccl, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}


template <class T>
inline long Device<T>::CreateNCCL(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nGpuId = (int)pfInput[0];
	int nCount = (int)pfInput[1];
	int nRank = (int)pfInput[2];
	int nGuidCount = (int)pfInput[3];

	if (nGuidCount != 5)
		return ERROR_PARAM_OUT_OF_RANGE;

	unsigned long g1 = (unsigned long)pfInput[4];
	unsigned long g2 = (unsigned long)pfInput[5];
	unsigned long g3 = (unsigned long)pfInput[6];
	unsigned long g4 = (unsigned long)pfInput[7];
	unsigned long g5 = (unsigned long)pfInput[8];

	char szGuid[128];
	snprintf(szGuid, 128, "nccl-%08x-%04x-%04x-%04x-%012x", g1, g2, g3, g4, g5);

	if (lErr = m_memory.CreateNCCL(nGpuId, nCount, nRank, szGuid, &m_math, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeNCCL(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hNccl = (long)pfInput[0];

	return m_memory.FreeNCCL(hNccl);
}

template <class T>
inline long Device<T>::NcclInitSingleProcess(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, INT_MAX))
		return lErr;

	long lBufferCount = (long)pfInput[0];
	int nCount = (int)pfInput[1];
	if (nCount != lInput - 2)
		return ERROR_PARAM_OUT_OF_RANGE;

	long* rgHandles = new long[nCount];
	if (rgHandles == NULL)
		return ERROR_MEMORY_OUT;

	for (int i = 0; i < nCount; i++)
	{
		rgHandles[i] = (long)pfInput[i + 2];
	}

	lErr = m_memory.NcclInitSingleProcess(lBufferCount, rgHandles, nCount);
	delete rgHandles;

	return lErr;
}

template <class T>
inline long Device<T>::NcclInitMultiProcess(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;

	long lBufferCount = (long)pfInput[0];
	long hNccl = (long)pfInput[1];

	return m_memory.NcclInitMultiProcess(lBufferCount, hNccl);
}

template <class T>
inline long Device<T>::NcclBroadcast(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	long hNccl = (long)pfInput[0];
	long hStream = (long)pfInput[1];
	long hX = (long)pfInput[2];
	int nCount = (int)pfInput[3];

	return m_memory.NcclBroadcast(hNccl, hStream, hX, nCount);
}

template <class T>
inline long Device<T>::NcclAllReduce(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	long hNccl = (long)pfInput[0];
	long hStream = (long)pfInput[1];
	long hX = (long)pfInput[2];
	int nCount = (int)pfInput[3];
	NCCL_OP op = (NCCL_OP)(int)pfInput[4];
	T fScale = pfInput[5];

	return m_memory.NcclAllReduce(hNccl, hStream, hX, nCount, op, fScale);
}


template <class T>
inline long Device<T>::CreateSsd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyInput(lInput, pfInput, 22, 25))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	int nGpuID = (int)pfInput[0];
	int nNumClasses = (int)pfInput[1];
	bool bShareLocation = (pfInput[2] == 0) ? false : true;
	int nLocClasses = (int)pfInput[3];
	int nBackgroundLabelId = (int)pfInput[4];
	bool bUseDifficultGt = (pfInput[5] == 0) ? false : true;
	SsdMiningType miningType = (SsdMiningType)(int)pfInput[6];
	SsdMatchingType matchingType = (SsdMatchingType)(int)pfInput[7];
	T fOverlapThreshold = pfInput[8];
	bool bUsePriorForMatching = (pfInput[9] == 0) ? false : true;
	SsdCodeType codeType = (SsdCodeType)(int)pfInput[10];
	bool bEncodeVariantInTgt = (pfInput[11] == 0) ? false : true;
	bool bBpInside = (pfInput[12] == 0) ? false : true;
	bool bIgnoreCrossBoundaryBbox = (pfInput[13] == 0) ? false : true;
	bool bUsePriorForNms = (pfInput[14] == 0) ? false : true;
	SsdConfLossType confLossType = (SsdConfLossType)(int)pfInput[15];
	SsdLocLossType locLossType = (SsdLocLossType)(int)pfInput[16];
	T fNegPosRatio = pfInput[17];
	T fNegOverlap = pfInput[18];
	int nSampleSize = (int)pfInput[19];
	bool bMapObjectToAgnostic = (pfInput[20] == 0) ? false : true;
	bool bNmsActive = (pfInput[21] == 0) ? false : true;

	T fNmsThreshold = T(0.0);
	if (lInput > 22)
		fNmsThreshold = pfInput[22];

	int nTopK = -1;
	if (lInput > 23)
		nTopK = (int)pfInput[23];

	T fEta = 1.0;
	if (lInput > 24)
		fEta = pfInput[23];

	if (lErr = m_memory.CreateSSD(nGpuID, nNumClasses, bShareLocation, nLocClasses, nBackgroundLabelId, bUseDifficultGt, miningType, matchingType, fOverlapThreshold, bUsePriorForMatching, codeType, bEncodeVariantInTgt, bBpInside, bIgnoreCrossBoundaryBbox, bUsePriorForNms, confLossType, locLossType, fNegPosRatio, fNegOverlap, nSampleSize, bMapObjectToAgnostic, bNmsActive, fNmsThreshold, nTopK, fEta, &m_math, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeSsd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hHandle = (long)pfInput[0];

	return m_memory.FreeSSD(hHandle);
}

template <class T>
inline long Device<T>::SetupSsd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	long hSsd = (long)pfInput[0];
	int nNum = (int)pfInput[1];
	int nNumPriors = (int)pfInput[2];
	int nNumGt = (int)pfInput[3];

	return m_memory.SetupSSD(hSsd, nNum, nNumPriors, nNumGt);
}

template <class T>
inline long Device<T>::SsdMultiboxLossForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	long hSsd = (long)pfInput[0];
	int nLocDataCount = (int)pfInput[1];
	long hLocData = (long)pfInput[2];
	int nConfDataCount = (int)pfInput[3];
	long hConfData = (long)pfInput[4];
	int nPriorDataCount = (int)pfInput[5];
	long hPriorData = (long)pfInput[6];
	int nGtDataCount = (int)pfInput[7];
	long hGtData = (long)pfInput[8];

	int nNumMatches;
	int nNumNegs;

	if (lErr = m_memory.SsdMultiboxLossForward(hSsd, nLocDataCount, hLocData, nConfDataCount, hConfData, nPriorDataCount, hPriorData, nGtDataCount, hGtData, &nNumMatches, &nNumNegs))
		return lErr;

	vector<map<int, vector<int>>> all_match_indices;
	if (lErr = m_memory.SsdGetAllMatchIndices(hSsd, &all_match_indices))
		return lErr;

	vector<vector<int>> all_neg_indices;
	if (lErr = m_memory.SsdGetAllNegIndices(hSsd, &all_neg_indices))
		return lErr;

	//-------------------------------------------
	//	Get the return values in the following
	//	ordering:
	//
	// [0] num matches.
	// [1] num negs.
	// [2] list of maps count
	// [2] first map - count.
	// [3] first map, first item, - label
	// [4] first map, first item, - idx count
	// [5] first map, first item, - idx 1
	//     
	// [n + 0] list of negidx list count.
	// [n + 1] first negidx set - count
	// [n + 2] first negidx set - idx 1
	//                              :
	vector<int> retval;
	retval.push_back(nNumMatches);
	retval.push_back(nNumNegs);

	// Match Indexes
	retval.push_back((int)all_match_indices.size());
	for (int i = 0; i < all_match_indices.size(); i++)
	{
		int nCountIdx = (int)retval.size();
		int nMapCount = 0;
		retval.push_back(0);

		for (map<int, vector<int>>::const_iterator it = all_match_indices[i].begin(); it != all_match_indices[i].end(); it++)
		{
			const int nLabel = it->first;
			const vector<int>& match_index = it->second;

			retval.push_back(nLabel);
			int nCount = (int)match_index.size();
			retval.push_back(nCount);

			for (int j = 0; j < nCount; j++)
			{
				retval.push_back(match_index[j]);
			}

			nMapCount++;
		}

		retval[nCountIdx] = nMapCount;
	}

	// Neg Indexes
	retval.push_back((int)all_neg_indices.size());
	for (int i = 0; i < all_neg_indices.size(); i++)
	{
		int nCount = (int)all_neg_indices[i].size();
		retval.push_back(nCount);

		for (int j = 0; j < nCount; j++)
		{
			retval.push_back(all_neg_indices[i][j]);
		}
	}

	// Allocate the return mem and copy the values.
	T* pfOutput = NULL;
	if (lErr = m_memory.AllocHost(retval.size(), &pfOutput, NULL, false, false, false))
		return lErr;

	for (int i = 0; i < retval.size(); i++)
	{
		pfOutput[i] = T(retval[i]);
	}

	*plOutput = (long)retval.size();
	*ppfOutput = pfOutput;

	return 0;
}

template <class T>
inline long Device<T>::SsdEncodeLocPrediction(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	long hSsd = (long)pfInput[0];
	int nLocPredCount = (int)pfInput[1];
	long hLocPred = (long)pfInput[2];
	int nLocGtCount = (int)pfInput[3];
	long hLocGt = (long)pfInput[4];

	return m_memory.SsdEncodeLocPrediction(hSsd, nLocPredCount, hLocPred, nLocGtCount, hLocGt);
}

template <class T>
inline long Device<T>::SsdEncodeConfPrediction(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	long hSsd = (long)pfInput[0];
	int nConfPredCount = (int)pfInput[1];
	long hConfPred = (long)pfInput[2];
	int nConfGtCount = (int)pfInput[3];
	long hConfGt = (long)pfInput[4];

	return m_memory.SsdEncodeConfPrediction(hSsd, nConfPredCount, hConfPred, nConfGtCount, hConfGt);
}


template <class T>
inline long Device<T>::CreateExtensionFloat(HMODULE hParent, LONG lKernelIdx, long* plOutput, T** ppfOutput, LPTSTR pszInput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateExtensionFloat(hParent, lKernelIdx, pszInput, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::CreateExtensionDouble(HMODULE hParent, LONG lKernelIdx, long* plOutput, T** ppfOutput, LPTSTR pszInput)
{
	LONG lErr;
	long hHandle = 0;

	if (lErr = verifyOutput(plOutput, ppfOutput))
		return lErr;

	if (lErr = m_memory.CreateExtensionDouble(hParent, lKernelIdx, pszInput, &hHandle))
		return lErr;

	return setOutput(hHandle, plOutput, ppfOutput);
}

template <class T>
inline long Device<T>::FreeExtension(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 1, 1))
		return lErr;

	long hExtension = (long)pfInput[0];

	return m_memory.FreeExtension(hExtension);
}


template <class T>
inline long Device<T>::ExtensionRun(long lInput, T* pfInput, long* plOutput, T** ppfOutput, LPTSTR szErr, long lErrMax)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, INT_MAX))
		return lErr;

	long hExtension = (long)pfInput[0];
	long lfnIdx = (long)pfInput[1];

	lInput -= 2;
	
	if (lInput == 0)
		pfInput = NULL;
	else
		pfInput = &pfInput[2];

	return m_memory.ExtensionRun(hExtension, lfnIdx, pfInput, lInput, ppfOutput, plOutput, szErr, lErrMax);
}

template <class T>
inline long Device<T>::cuda_sigmoid_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];

	return m_math.sigmoid_fwd(nCount, hBottomData, hTopData);
}

template <class T>
inline long Device<T>::cuda_sigmoid_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];

	return m_math.sigmoid_bwd(nCount, hTopDiff, hTopData, hBottomDiff);
}

template <class T>
inline long Device<T>::cuda_swish_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hSigmoidOutputData = (long)pfInput[3];
	long hBottomDiff = (long)pfInput[4];
	T fBeta = pfInput[5];

	return m_math.swish_bwd(nCount, hTopDiff, hTopData, hSigmoidOutputData, hBottomDiff, fBeta);
}


template <class T>
inline long Device<T>::ReLUForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	T fBeta = pfInput[4];
	long hTopDesc = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_memory.ReLUForward(hHandle, fAlpha, hBottomDesc, hBottomData, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::ReLUBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 11, 11))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hTopDataDesc = (long)pfInput[2];
	long hTopData = (long)pfInput[3];
	long hTopDiffDesc = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	long hBottomDataDesc = (long)pfInput[6];
	long hBottomData = (long)pfInput[7];
	T fBeta = pfInput[8];
	long hBottomDiffDesc = (long)pfInput[9];
	long hBottomDiff = (long)pfInput[10];

	return m_memory.ReLUBackward(hHandle, fAlpha, hTopDataDesc, hTopData, hTopDiffDesc, hTopDiff, hBottomDataDesc, hBottomData, fBeta, hBottomDiffDesc, hBottomDiff);
}


template <class T>
inline long Device<T>::SoftmaxForward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hBottomDesc = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	T fBeta = pfInput[4];
	long hTopDesc = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_memory.SoftmaxForward(hHandle, fAlpha, hBottomDesc, hBottomData, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::SoftmaxBackward(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	long hHandle = (long)pfInput[0];
	T fAlpha = pfInput[1];
	long hTopDataDesc = (long)pfInput[2];
	long hTopData = (long)pfInput[3];
	long hTopDiffDesc = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	T fBeta = pfInput[6];
	long hBottomDiffDesc = (long)pfInput[7];
	long hBottomDiff = (long)pfInput[8];

	return m_memory.SoftmaxBackward(hHandle, fAlpha, hTopDataDesc, hTopData, hTopDiffDesc, hTopDiff, fBeta, hBottomDiffDesc, hBottomDiff);
}


template <class T>
inline long Device<T>::LRNForwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hNormDesc = (long)pfInput[1];
	T fAlpha = pfInput[2];
	long hBottomDesc = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	T fBeta = pfInput[5];
	long hTopDesc = (long)pfInput[6];
	long hTopData = (long)pfInput[7];

	return m_memory.LRNForwardCC(hHandle, hNormDesc, fAlpha, hBottomDesc, hBottomData, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::LRNBackwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 12, 12))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hNormDesc = (long)pfInput[1];
	T fAlpha = pfInput[2];
	long hTopDataDesc = (long)pfInput[3];
	long hTopData = (long)pfInput[4];
	long hTopDiffDesc = (long)pfInput[5];
	long hTopDiff = (long)pfInput[6];
	long hBottomDataDesc = (long)pfInput[7];
	long hBottomData = (long)pfInput[8];
	T fBeta = pfInput[9];
	long hBottomDiffDesc = (long)pfInput[10];
	long hBottomDiff = (long)pfInput[11];

	return m_memory.LRNBackwardCC(hHandle, hNormDesc, fAlpha, hTopDataDesc, hTopData, hTopDiffDesc, hTopDiff, hBottomDataDesc, hBottomData, fBeta, hBottomDiffDesc, hBottomDiff);
}

template <class T>
inline long Device<T>::LCNForwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 10, 10))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hNormDesc = (long)pfInput[1];
	T fAlpha = pfInput[2];
	long hBottomDesc = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	long hTemp1 = (long)pfInput[5];
	long hTemp2 = (long)pfInput[6];
	T fBeta = pfInput[7];
	long hTopDesc = (long)pfInput[8];
	long hTopData = (long)pfInput[9];

	return m_memory.LCNForwardCC(hHandle, hNormDesc, fAlpha, hBottomDesc, hBottomData, hTemp1, hTemp2, fBeta, hTopDesc, hTopData);
}

template <class T>
inline long Device<T>::LCNBackwardCC(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 11, 11))
		return lErr;

	long hHandle = (long)pfInput[0];
	long hNormDesc = (long)pfInput[1];
	T fAlpha = pfInput[2];
	long hBottomDataDesc = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	long hTemp1 = (long)pfInput[6];
	long hTemp2 = (long)pfInput[7];
	T fBeta = pfInput[8];
	long hBottomDiffDesc = (long)pfInput[9];
	long hBottomDiff = (long)pfInput[10];

	return m_memory.LCNBackwardCC(hHandle, hNormDesc, fAlpha, hBottomDataDesc, hBottomData, hTopDiff, hTemp1, hTemp2, fBeta, hBottomDiffDesc, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_relu_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	T fNegativeSlope = pfInput[3];

	return m_math.relu_fwd(nCount, hBottomData, hTopData, fNegativeSlope);
}

template <class T>
inline long Device<T>::cuda_relu_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];
	T fNegativeSlope = pfInput[4];

	return m_math.relu_bwd(nCount, hTopDiff, hTopData, hBottomDiff, fNegativeSlope);
}


template <class T>
inline long Device<T>::cuda_elu_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	T fAlpha = pfInput[3];

	return m_math.elu_fwd(nCount, hBottomData, hTopData, fAlpha);
}

template <class T>
inline long Device<T>::cuda_elu_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hBottomData = (long)pfInput[3];
	long hBottomDiff = (long)pfInput[4];
	T fAlpha = pfInput[5];

	return m_math.elu_bwd(nCount, hTopDiff, hTopData, hBottomData, hBottomDiff, fAlpha);
}


template <class T>
inline long Device<T>::cuda_dropout_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hMask = (long)pfInput[2];
	unsigned int uiThreshold = (unsigned int)pfInput[3];
	T fScale = pfInput[4];
	long hTopData = (long)pfInput[5];

	return m_math.dropout_fwd(nCount, hBottomData, hMask, uiThreshold, fScale, hTopData);
}

template <class T>
inline long Device<T>::cuda_dropout_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hMask = (long)pfInput[2];
	unsigned int uiThreshold = (unsigned int)pfInput[3];
	T fScale = pfInput[4];
	long hBottomDiff = (long)pfInput[5];

	return m_math.dropout_bwd(nCount, hTopDiff, hMask, uiThreshold, fScale, hBottomDiff);
}

template <class T>
inline long Device<T>::cuda_bnll_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];

	return m_math.bnll_fwd(nCount, hBottomData, hTopData);
}

template <class T>
inline long Device<T>::cuda_bnll_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	long hBottomData = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];

	return m_math.bnll_bwd(nCount, hTopDiff, hBottomData, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_prelu_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nChannels = (int)pfInput[1];
	int nDim = (int)pfInput[2];
	long hBottomData = (long)pfInput[3];
	long hTopData = (long)pfInput[4];
	long hSlopeData = (long)pfInput[5];
	int nDivFactor = (int)pfInput[6];

	return m_math.prelu_fwd(nCount, nChannels, nDim, hBottomData, hTopData, hSlopeData, nDivFactor);
}

template <class T>
inline long Device<T>::cuda_prelu_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	int nChannels = (int)pfInput[1];
	int nDim = (int)pfInput[2];
	long hTopDiff = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	long hBottomDiff = (long)pfInput[5];
	long hSlopeData = (long)pfInput[6];
	int nDivFactor = (int)pfInput[7];

	return m_math.prelu_bwd(nCount, nChannels, nDim, hTopDiff, hBottomData, hBottomDiff, hSlopeData, nDivFactor);
}

template <class T>
inline long Device<T>::cuda_prelu_bwd_param(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCDim = (int)pfInput[0];
	int nNum = (int)pfInput[1];
	int nTopOffset = (int)pfInput[2];
	long hTopDiff = (long)pfInput[3];
	long hBottomData = (long)pfInput[4];
	long hBackBuffDiff = (long)pfInput[5];

	return m_math.prelu_bwd_param(nCDim, nNum, nTopOffset, hTopDiff, hBottomData, hBackBuffDiff);
}


template <class T>
inline long Device<T>::cuda_softmaxloss_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 9))
		return lErr;

	int nCount = (int)pfInput[0];
	long hProbData = (long)pfInput[1];
	long hLabels = (long)pfInput[2];
	long hLossData = (long)pfInput[3];
	int nOuterNum = (int)pfInput[4];
	int nDim = (int)pfInput[5];
	int nInnerNum = (int)pfInput[6];
	long hCounts = (long)pfInput[7];
	int nIgnoreLabel = -1;

	if (lInput > 8)
		nIgnoreLabel = (int)pfInput[8];

	return m_math.softmaxloss_fwd(nCount, hProbData, hLabels, hLossData, nOuterNum, nDim, nInnerNum, hCounts, nIgnoreLabel);
}

template <class T>
inline long Device<T>::cuda_softmaxloss_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 9))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopData = (long)pfInput[1];
	long hLabels = (long)pfInput[2];
	long hBottomDiff = (long)pfInput[3];
	int nOuterNum = (int)pfInput[4];
	int nDim = (int)pfInput[5];
	int nInnerNum = (int)pfInput[6];
	long hCounts = (long)pfInput[7];
	int nIgnoreLabel = -1;

	if (lInput > 8)
		nIgnoreLabel = (int)pfInput[8];

	return m_math.softmaxloss_bwd(nCount, hTopData, hLabels, hBottomDiff, nOuterNum, nDim, nInnerNum, hCounts, nIgnoreLabel);
}

template <class T>
inline long Device<T>::cuda_min_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	int nIdx = (int)pfInput[3];
	long hY = (long)pfInput[4];
	long hMask = (long)pfInput[5];

	return m_math.min_fwd(nCount, hA, hB, nIdx, hY, hMask);
}

template <class T>
inline long Device<T>::cuda_min_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	int nIdx = (int)pfInput[2];
	long hMask = (long)pfInput[3];
	long hY = (long)pfInput[4];

	return m_math.min_bwd(nCount, hX, nIdx, hMask, hY);
}

template <class T>
inline long Device<T>::cuda_max_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hA = (long)pfInput[1];
	long hB = (long)pfInput[2];
	int nIdx = (int)pfInput[3];
	long hY = (long)pfInput[4];
	long hMask = (long)pfInput[5];

	return m_math.max_fwd(nCount, hA, hB, nIdx, hY, hMask);
}

template <class T>
inline long Device<T>::cuda_max_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	int nIdx = (int)pfInput[2];
	long hMask = (long)pfInput[3];
	long hY = (long)pfInput[4];

	return m_math.max_bwd(nCount, hX, nIdx, hMask, hY);
}


template <class T>
inline long Device<T>::cuda_crop_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nNumAxes = (int)pfInput[1];
	int hSrcStrides = (long)pfInput[2];
	int hDstStrides = (long)pfInput[3];
	int hOffsets = (long)pfInput[4];
	long hBottomData = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_math.crop_fwd(nCount, nNumAxes, hSrcStrides, hDstStrides, hOffsets, hBottomData, hTopData);
}


template <class T>
inline long Device<T>::cuda_crop_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nNumAxes = (int)pfInput[1];
	int hSrcStrides = (long)pfInput[2];
	int hDstStrides = (long)pfInput[3];
	int hOffsets = (long)pfInput[4];
	long hBottomDiff = (long)pfInput[5];
	long hTopDiff = (long)pfInput[6];

	return m_math.crop_bwd(nCount, nNumAxes, hSrcStrides, hDstStrides, hOffsets, hBottomDiff, hTopDiff);
}


template <class T>
inline long Device<T>::cuda_concat_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	int nNumConcats = (int)pfInput[2];
	int nConcatInputSize = (int)pfInput[3];
	int nTopConcatAxis = (int)pfInput[4];
	int nBottomConcatAxis = (int)pfInput[5];
	int nOffsetConcatAxis = (int)pfInput[6];
	long hTopData = (long)pfInput[7];

	return m_math.concat_fwd(nCount, hBottomData, nNumConcats, nConcatInputSize, nTopConcatAxis, nBottomConcatAxis, nOffsetConcatAxis, hTopData);
}


template <class T>
inline long Device<T>::cuda_concat_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	int nNumConcats = (int)pfInput[2];
	int nConcatInputSize = (int)pfInput[3];
	int nTopConcatAxis = (int)pfInput[4];
	int nBottomConcatAxis = (int)pfInput[5];
	int nOffsetConcatAxis = (int)pfInput[6];
	long hBottomDiff = (long)pfInput[7];

	return m_math.concat_bwd(nCount, hTopDiff, nNumConcats, nConcatInputSize, nTopConcatAxis, nBottomConcatAxis, nOffsetConcatAxis, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_slice_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	int nNumSlices = (int)pfInput[2];
	int nSliceInputSize = (int)pfInput[3];
	int nBottomSliceAxis = (int)pfInput[4];
	int nTopSliceAxis = (int)pfInput[5];
	int nOffsetSliceAxis = (int)pfInput[6];
	long hTopData = (long)pfInput[7];

	return m_math.slice_fwd(nCount, hBottomData, nNumSlices, nSliceInputSize, nBottomSliceAxis, nTopSliceAxis, nOffsetSliceAxis, hTopData);
}


template <class T>
inline long Device<T>::cuda_slice_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	int nNumSlices = (int)pfInput[2];
	int nSliceInputSize = (int)pfInput[3];
	int nBottomSliceAxis = (int)pfInput[4];
	int nTopSliceAxis = (int)pfInput[5];
	int nOffsetSliceAxis = (int)pfInput[6];
	long hBottomDiff = (long)pfInput[7];

	return m_math.slice_bwd(nCount, hTopDiff, nNumSlices, nSliceInputSize, nBottomSliceAxis, nTopSliceAxis, nOffsetSliceAxis, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_tile_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	int nInnerDim = (int)pfInput[2];
	int nTiles = (int)pfInput[3];
	int nBottomTileAxis = (int)pfInput[4];
	long hTopData = (long)pfInput[5];

	return m_math.tile_fwd(nCount, hBottomData, nInnerDim, nTiles, nBottomTileAxis, hTopData);
}


template <class T>
inline long Device<T>::cuda_tile_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hTopDiff = (long)pfInput[1];
	int nTileSize = (int)pfInput[2];
	int nTiles = (int)pfInput[3];
	int nBottomTileAxis = (int)pfInput[4];
	long hBottomDiff = (long)pfInput[5];

	return m_math.tile_bwd(nCount, hTopDiff, nTileSize, nTiles, nBottomTileAxis, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_bias_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hBiasData = (long)pfInput[2];
	int nBiasDim = (int)pfInput[3];
	int nInnerDim = (int)pfInput[4];
	long hTopData = (long)pfInput[5];

	return m_math.bias_fwd(nCount, hBottomData, hBiasData, nBiasDim, nInnerDim, hTopData);
}


template <class T>
inline long Device<T>::cuda_scale_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hScaleData = (long)pfInput[2];
	int nScaleDim = (int)pfInput[3];
	int nInnerDim = (int)pfInput[4];
	long hY = (long)pfInput[5];
	long hBiasData = 0;

	if (lInput > 6)
		hBiasData = (long)pfInput[6];

	return m_math.scale_fwd(nCount, hX, hScaleData, nScaleDim, nInnerDim, hY, hBiasData);
}


template <class T>
inline long Device<T>::cuda_threshold_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	T fThreshold = pfInput[1];
	long hX = (long)pfInput[2];
	long hY = (long)pfInput[3];

	return m_math.threshold_fwd(nCount, fThreshold, hX, hY);
}



template <class T>
inline long Device<T>::cuda_cll_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	int nCount = (int)pfInput[0];
	int nChannels = (int)pfInput[1];
	T fMargin = pfInput[2];
	bool bLegacyVersion = (pfInput[3] != 0) ? true : false;
	T fAlpha = pfInput[4];
	long hY = (long)pfInput[5];
	long hDiff = (long)pfInput[6];
	long hDistSq = (long)pfInput[7];
	long hBottomDiff = (long)pfInput[8];

	return m_math.cll_bwd(nCount, nChannels, fMargin, bLegacyVersion, fAlpha, hY, hDiff, hDistSq, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_smoothl1_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.smoothl1_fwd(nCount, hX, hY);
}


template <class T>
inline long Device<T>::cuda_smoothl1_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 3))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];

	return m_math.smoothl1_bwd(nCount, hX, hY);
}


template <class T>
inline long Device<T>::cuda_permute(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 8, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	bool bFwd = (pfInput[2] == 0) ? false : true;
	long hPermuteOrder = (long)pfInput[3];
	long hOldSteps = (long)pfInput[4];
	long hNewSteps = (long)pfInput[5];
	int nNumAxes = (int)pfInput[6];
	long hY = (long)pfInput[7];

	return m_math.permute(nCount, hX, bFwd, hPermuteOrder, hOldSteps, hNewSteps, nNumAxes, hY);
}


template <class T>
inline long Device<T>::cuda_gather_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];
	int nAxis = (int)pfInput[3];
	int nDim = (int)pfInput[4];
	int nDimAtAxis = (int)pfInput[5];
	int nM = (int)pfInput[6];
	int nN = (int)pfInput[7];
	long hIdx = (long)pfInput[8];

	return m_math.gather_fwd(nCount, hX, hY, nAxis, nDim, nDimAtAxis, nM, nN, hIdx);
}

template <class T>
inline long Device<T>::cuda_gather_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	int nCount = (int)pfInput[0];
	long hX = (long)pfInput[1];
	long hY = (long)pfInput[2];
	int nAxis = (int)pfInput[3];
	int nDim = (int)pfInput[4];
	int nDimAtAxis = (int)pfInput[5];
	int nM = (int)pfInput[6];
	int nN = (int)pfInput[7];
	long hIdx = (long)pfInput[8];

	return m_math.gather_bwd(nCount, hX, hY, nAxis, nDim, nDimAtAxis, nM, nN, hIdx);
}

template <class T>
inline long Device<T>::cuda_lrn_fillscale(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 10, 10))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	int nNum = (int)pfInput[2];
	int nChannels = (int)pfInput[3];
	int nHeight = (int)pfInput[4];
	int nWidth = (int)pfInput[5];
	int nSize = (int)pfInput[6];
	T fA = pfInput[7];
	T fB = pfInput[8];
	long hScaleData = (long)pfInput[9];

	return m_math.lrn_fillscale(nCount, hBottomData, nNum, nChannels, nHeight, nWidth, nSize, fA, fB, hScaleData);
}


template <class T>
inline long Device<T>::cuda_lrn_computeoutput(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hScaleData = (long)pfInput[2];
	T fA = pfInput[3];
	long hTopData = (long)pfInput[4];

	return m_math.lrn_computeoutput(nCount, hBottomData, hScaleData, fA, hTopData);
}


template <class T>
inline long Device<T>::cuda_lrn_computediff(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 13, 13))
		return lErr;

	int nCount = (int)pfInput[0];
	long hBottomData = (long)pfInput[1];
	long hTopData = (long)pfInput[2];
	long hScaleData = (long)pfInput[3];
	long hTopDiff = (long)pfInput[4];
	int nNum = (int)pfInput[5];
	int nChannels = (int)pfInput[6];
	int nHeight = (int)pfInput[7];
	int nWidth = (int)pfInput[8];
	int nSize = (int)pfInput[9];
	T fB = pfInput[10];
	T fA = pfInput[11];
	long hBottomDiff = (long)pfInput[12];

	return m_math.lrn_computediff(nCount, hBottomData, hTopData, hScaleData, hTopDiff, nNum, nChannels, nHeight, nWidth, nSize, fB, fA, hBottomDiff);
}


template <class T>
inline long Device<T>::cuda_lstm_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 21, 24))
		return lErr;

	int t = (int)pfInput[0];
	int nN = (int)pfInput[1];
	int nH = (int)pfInput[2];
	int nI = (int)pfInput[3];
	long hWeight_h = (long)pfInput[4];
	long hWeight_i = (long)pfInput[5];
	long hClipData = (long)pfInput[6];
	int nClipOffset = (int)pfInput[7];
	long hTopData = (long)pfInput[8];
	int nTopOffset = (int)pfInput[9];
	long hCellData = (long)pfInput[10];
	int nCellOffset = (int)pfInput[11];
	long hPreGateData = (long)pfInput[12];
	int nPreGateOffset = (int)pfInput[13];
	long hGateData = (long)pfInput[14];
	int nGateOffset = (int)pfInput[15];
	long hHT1Data = (long)pfInput[16];
	int nHT1Offset = (int)pfInput[17];
	long hCT1Data = (long)pfInput[18];
	int nCT1Offset = (int)pfInput[19];
	long hHtoGateData = (long)pfInput[20];
	long hContext = 0;
	long hWeight_c = 0;
	long hCtoGateData = 0;

	if (lInput > 21)
		hContext = (long)pfInput[21];

	if (lInput > 22)
		hWeight_c = (long)pfInput[22];

	if (lInput > 23)
		hCtoGateData = (long)pfInput[23];

	return m_math.lstm_fwd(t, nN, nH, nI, hWeight_h, hWeight_i, hClipData, nClipOffset, hTopData, nTopOffset, hCellData, nCellOffset, hPreGateData, nPreGateOffset, hGateData, nGateOffset, hHT1Data, nHT1Offset, hCT1Data, nCT1Offset, hHtoGateData, hContext, hWeight_c, hCtoGateData);
}


template <class T>
inline long Device<T>::cuda_lstm_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 25, 27))
		return lErr;

	int t = (int)pfInput[0];
	int nN = (int)pfInput[1];
	int nH = (int)pfInput[2];
	int nI = (int)pfInput[3];
	T fClip = pfInput[4];
	long hWeight_h = (long)pfInput[5];
	long hClipData = (long)pfInput[6];
	int nClipOffset = (int)pfInput[7];
	long hTopDiff = (long)pfInput[8];
	int nTopOffset = (int)pfInput[9];
	long hCellData = (long)pfInput[10];
	long hCellDiff = (long)pfInput[11];
	int nCellOffset = (int)pfInput[12];
	long hPreGateDiff = (long)pfInput[13];
	int nPreGateOffset = (int)pfInput[14];
	long hGateData = (long)pfInput[15];
	long hGateDiff = (long)pfInput[16];
	int nGateOffset = (int)pfInput[17];
	long hCT1Data = (long)pfInput[18];
	int nCT1Offset = (int)pfInput[19];
	long hDHT1Diff = (long)pfInput[20];
	int nDHT1Offset = (int)pfInput[21];
	long hDCT1Diff = (long)pfInput[22];
	int nDCT1Offset = (int)pfInput[23];
	long hHtoHData = (long)pfInput[24];
	long hContextDiff = 0;
	long hWeight_c = 0;

	if (lInput > 25)
		hContextDiff = (long)pfInput[25];

	if (lInput > 26)
		hWeight_c = (long)pfInput[26];

	return m_math.lstm_bwd(t, nN, nH, nI, fClip, hWeight_h, hClipData, nClipOffset, hTopDiff, nTopOffset, hCellData, hCellDiff, nCellOffset, hPreGateDiff, nPreGateOffset, hGateData, hGateDiff, nGateOffset, hCT1Data, nCT1Offset, hDHT1Diff, nDHT1Offset, hDCT1Diff, nDCT1Offset, hHtoHData, hContextDiff, hWeight_c);
}


template <class T>
inline long Device<T>::cuda_lstm_unit_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 9, 9))
		return lErr;

	int nCount = (int)pfInput[0];
	int nHiddenDim = (int)pfInput[1];
	int nXCount = (int)pfInput[2];
	long hX = (long)pfInput[3];
	long hX_acts = (long)pfInput[4];
	long hC_prev = (long)pfInput[5];
	long hCont = (long)pfInput[6];
	long hC = (long)pfInput[7];
	long hH = (long)pfInput[8];

	return m_math.lstm_unit_fwd(nCount, nHiddenDim, nXCount, hX, hX_acts, hC_prev, hCont, hC, hH);
}


template <class T>
inline long Device<T>::cuda_lstm_unit_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 13, 13))
		return lErr;

	int nCount = (int)pfInput[0];
	int nHiddenDim = (int)pfInput[1];
	int nXCount = (int)pfInput[2];
	long hC_prev = (long)pfInput[3];
	long hX_acts = (long)pfInput[4];
	long hC = (long)pfInput[5];
	long hH = (long)pfInput[6];
	long hCont = (long)pfInput[7];
	long hC_diff = (long)pfInput[8];
	long hH_diff = (long)pfInput[9];
	long hC_prev_diff = (long)pfInput[10];
	long hX_acts_diff = (long)pfInput[11];
	long hX_diff = (long)pfInput[12];

	return m_math.lstm_unit_bwd(nCount, nHiddenDim, nXCount, hC_prev, hX_acts, hC, hH, hCont, hC_diff, hH_diff, hC_prev_diff, hX_acts_diff, hX_diff);
}


template <class T>
inline long Device<T>::cuda_coeff_sum_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nDim = (int)pfInput[1];
	int nNumOffset = (int)pfInput[2];
	T fCoeff = pfInput[3];
	long hCoeffData = (long)pfInput[4];
	long hBottomData = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_math.coeff_sum_fwd(nCount, nDim, nNumOffset, fCoeff, hCoeffData, hBottomData, hTopData);
}

template <class T>
inline long Device<T>::cuda_coeff_sum_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nDim = (int)pfInput[1];
	int nNumOffset = (int)pfInput[2];
	T fCoeff = pfInput[3];
	long hCoeffData = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	long hBottomDiff = (long)pfInput[6];

	return m_math.coeff_sum_bwd(nCount, nDim, nNumOffset, fCoeff, hCoeffData, hTopDiff, hBottomDiff);
}

template <class T>
inline long Device<T>::cuda_coeff_sub_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nDim = (int)pfInput[1];
	int nNumOffset = (int)pfInput[2];
	T fCoeff = pfInput[3];
	long hCoeffData = (long)pfInput[4];
	long hBottomData = (long)pfInput[5];
	long hTopData = (long)pfInput[6];

	return m_math.coeff_sub_fwd(nCount, nDim, nNumOffset, fCoeff, hCoeffData, hBottomData, hTopData);
}

template <class T>
inline long Device<T>::cuda_coeff_sub_bwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	int nDim = (int)pfInput[1];
	int nNumOffset = (int)pfInput[2];
	T fCoeff = pfInput[3];
	long hCoeffData = (long)pfInput[4];
	long hTopDiff = (long)pfInput[5];
	long hBottomDiff = (long)pfInput[6];

	return m_math.coeff_sub_bwd(nCount, nDim, nNumOffset, fCoeff, hCoeffData, hTopDiff, hBottomDiff);
}

template <class T>
inline long Device<T>::cuda_sigmoid_cross_entropy_fwd(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 7))
		return lErr;

	int nCount = (int)pfInput[0];
	long hInput = (long)pfInput[1];
	long hTarget = (long)pfInput[2];
	long hLoss = (long)pfInput[3];
	bool bHasIgnoreLabel = (pfInput[4] == 1) ? true : false;
	int nIgnoreLabel = (int)pfInput[5];
	long hCount = (long)pfInput[6];

	return m_math.sigmoid_cross_entropy_fwd(nCount, hInput, hTarget, hLoss, bHasIgnoreLabel, nIgnoreLabel, hCount);
}

template <class T>
inline long Device<T>::cuda_sigmoid_cross_entropy_ignore(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 4, 4))
		return lErr;

	int nCount = (int)pfInput[0];
	int nIgnoreLabel = (int)pfInput[1];
	long hTarget = (long)pfInput[2];
	long hData = (long)pfInput[3];

	return m_math.sigmoid_cross_entropy_ignore(nCount, nIgnoreLabel, hTarget, hData);
}


//=============================================================================
//	Math Methods
//=============================================================================

template <class T>
inline long Device<T>::cuda_set(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 5))
		return lErr;

	int nCount = (int)pfInput[0];
	long hHandle = (long)pfInput[1];
	T fVal = pfInput[2];
	int nIdx = -1;
	int nXOff = 0;

	if (lInput > 3)
		nIdx = (int)pfInput[3];

	if (lInput > 4)
		nXOff = (int)pfInput[4];

	return m_math.set(nCount, hHandle, fVal, nIdx, nXOff);
}

template <class T>
inline long Device<T>::cuda_copy(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 3, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	long hSrc = (long)pfInput[1];
	long hDst = (long)pfInput[2];
	int nSrcOffset = 0;
	int nDstOffset = 0;
	long hAsyncStream = -1;
	int nSrcHalfSizeOverride = -1;
	int nDstHalfSizeOverride = -1;

	if (lInput > 3)
		nSrcOffset = (int)pfInput[3];

	if (lInput > 4)
		nDstOffset = (int)pfInput[4];

	if (lInput > 5)
		hAsyncStream = (long)pfInput[5];

	if (lInput > 6)
		nSrcHalfSizeOverride = (int)pfInput[6];

	if (lInput > 7)
		nDstHalfSizeOverride = (int)pfInput[7];

	return m_math.copy(nCount, hSrc, hDst, nSrcOffset, nDstOffset, hAsyncStream, nSrcHalfSizeOverride, nDstHalfSizeOverride);
}

template <class T>
inline long Device<T>::cuda_copy_sim(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 7, 8))
		return lErr;

	int nCount = (int)pfInput[0];
	int nNum = (int)pfInput[1];
	int nDim = (int)pfInput[2];
	long hSrc1 = (long)pfInput[3];
	long hSrc2 = (long)pfInput[4];
	long hDst = (long)pfInput[5];
	long hSim = (long)pfInput[6];
	bool bInvert = false;

	if (lInput > 7)
		bInvert = (pfInput[7] == 0) ? false : true;

	return m_math.copy_sim(nCount, nNum, nDim, hSrc1, hSrc2, hDst, hSim, bInvert);
}


template <class T>
inline long Device<T>::cuda_copy_fill(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 6, 6))
		return lErr;

	int n = (int)pfInput[0];
	int nDim = (int)pfInput[1];
	long hSrc = (long)pfInput[2];
	int nSrcOff = (int)pfInput[3];
	int nCount = (int)pfInput[4];
	long hDst = (long)pfInput[5];

	return m_math.copy_fill(n, nDim, hSrc, nSrcOff, nCount, hDst);
}


template <class T>
inline long Device<T>::cuda_sort(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 2, 2))
		return lErr;

	int nCount = (int)pfInput[0];
	long hY = (long)pfInput[1];

	return m_math.sort(nCount, hY);
}

template <class T>
inline long Device<T>::cuda_copy_batch(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 13, 13))
		return lErr;

	int n = (int)pfInput[0];
	int nNum = (int)pfInput[1];
	int nDim = (int)pfInput[2];
	long hSrcData = (long)pfInput[3];
	long hSrcLbl = (long)pfInput[4];
	int nDstCount = (int)pfInput[5];
	long hDstCache = (long)pfInput[6];
	long hWorkDevData = (long)pfInput[7];
	int nLabelStart = (int)pfInput[8];
	int nLabelCount = (int)pfInput[9];
	int nCacheSize = (int)pfInput[10];
	long hCacheHostCursors = (long)pfInput[11];
	long hWorkHostData = (long)pfInput[12];

	return m_math.copy_batch(n, nNum, nDim, hSrcData, hSrcLbl, nDstCount, hDstCache, hWorkDevData, nLabelStart, nLabelCount, nCacheSize, hCacheHostCursors, hWorkHostData);
}

template <class T>
inline long Device<T>::cuda_copy_sequence(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 15 + (2 * 2), 15 + (12 * 2)))
		return lErr;

	int nK = (int)pfInput[0];
	int nNum = (int)pfInput[1];
	int nDim = (int)pfInput[2];
	long hSrcData = (long)pfInput[3];
	long hSrcLabel = (long)pfInput[4];
	int nSrcCacheCount = (int)pfInput[5];
	long hSrcCache = (long)pfInput[6];
	int nLabelStart = (int)pfInput[7];
	int nLabelCount = (int)pfInput[8];
	int nCacheSize = (int)pfInput[9];
	long hCacheHostCursors = (long)pfInput[10];
	bool bOutputLabels = (pfInput[11] == 1) ? true : false;
	long hWorkHostData = (long)pfInput[12];
	bool bCombinePositiveAndNegative = (pfInput[13] == 1) ? true : false;
	int nRandomSeed = (int)pfInput[14];

	int nTopCount = 2 + nK;
	if (bOutputLabels)
		nTopCount++;

	long* rghTop = (long*)malloc(sizeof(long) * nTopCount);
	if (rghTop == NULL)
		return ERROR_MEMORY_OUT;

	int* rgnTopCounts = (int*)malloc(sizeof(int) * nTopCount);
	if (rghTop == NULL)
	{
		free(rghTop);
		return ERROR_MEMORY_OUT;
	}

	int nIdx = 15;
	for (int i = 0; i < nTopCount; i++)
	{
		rghTop[i] = (long)pfInput[nIdx];
		nIdx++;
	}

	for (int i = 0; i < nTopCount; i++)
	{
		rgnTopCounts[i] = (int)pfInput[nIdx];
		nIdx++;
	}

	lErr = m_math.copy_sequence(nK, nNum, nDim, hSrcData, hSrcLabel, nSrcCacheCount, hSrcCache, nLabelStart, nLabelCount, nCacheSize, hCacheHostCursors, bOutputLabels, nTopCount, rghTop, rgnTopCounts, hWorkHostData, bCombinePositiveAndNegative, nRandomSeed);

	free(rghTop);
	free(rgnTopCounts);

	return lErr;
}

template <class T>
inline long Device<T>::cuda_copy_sequence2(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 11, 14))
		return lErr;

	int n = (int)pfInput[0];
	long hSrc = (long)pfInput[1];
	int nSrcStep = (int)pfInput[2];
	int nSrcStartIdx = (int)pfInput[3];
	int nCopyCount = (int)pfInput[4];
	int nCopyDim = (int)pfInput[5];
	long hDst = (long)pfInput[6];
	int nDstStep = (int)pfInput[7];
	int nDstStartIdx = (int)pfInput[8];
	int nSrcSpatialDim = (int)pfInput[9];
	int nDstSpatialDim = (int)pfInput[10];
	int nSrcSpatialDimStartIdx = 0;
	int nDstSpatialDimStartIdx = 0;
	int nSpatialDimCount = -1;

	if (lInput > 11)
	{
		nSrcSpatialDimStartIdx = (int)pfInput[11];
		if (nSrcSpatialDimStartIdx < 0)
			nSrcSpatialDimStartIdx += nSrcSpatialDim;
	}

	if (lInput > 12)
	{
		nDstSpatialDimStartIdx = (int)pfInput[12];
		if (nDstSpatialDimStartIdx < 0)
			nDstSpatialDimStartIdx += nDstSpatialDim;
	}

	if (lInput > 13)
		nSpatialDimCount = (int)pfInput[13];

	return m_math.copy_sequence(n, hSrc, nSrcStep, nSrcStartIdx, nCopyCount, nCopyDim, hDst, nDstStep, nDstStartIdx, nSrcSpatialDim, nDstSpatialDim, nSrcSpatialDimStartIdx, nDstSpatialDimStartIdx, nSpatialDimCount);
}

template <class T>
inline long Device<T>::cuda_copy_expand(long lInput, T* pfInput, long* plOutput, T** ppfOutput)
{
	LONG lErr;

	if (lErr = verifyInput(lInput, pfInput, 5, 5))
		return lErr;

	int n = (int)pfInput[0];
	int nNum = (int)pfInput[1];
	int nDim = (int)pfInput[2];
	long hX = (long)pfInput[3];
	long hA = (long)pfInput[4];

	return m_math.copy_expand(n, nNum, nDim, hX, hA);
}

#endif // __DEVICE_CU__