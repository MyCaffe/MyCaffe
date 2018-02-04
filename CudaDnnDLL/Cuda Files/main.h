//=============================================================================
//	main.h
//
//	The kernel manages the interface to the DLL.
//=============================================================================
#ifndef __MAIN_H_
#define __MAIN_H_

//=============================================================================
//	Includes
//=============================================================================
#include "util.h"
#include "device.h"


//=============================================================================
//	Defines
//=============================================================================

const int CUDA_DLL_KERNEL_COPY_NCCL = -10;
const int CUDA_DLL_CREATEKERNEL		= -9;
const int CUDA_DLL_DESTROYKERNEL	= -8;
const int CUDA_DLL_SETKERNEL		= -7;
const int CUDA_DLL_KERNEL_ADD		= -5;
const int CUDA_DLL_KERNEL_MEMCPY	= -4;

const int DEVICE_PROP_NAME					= 1;
const int DEVICE_PROP_MULTIGPUBOARDGROUPID  = 2;

const int CUDA_DLL_FREEMEM			= -1;
const int CUDA_DLL_INITIALIZE		= -2;
const int CUDA_DLL_CLEANUP			= -3;

const int CUDA_FN_SETDEVICE			= 1;
const int CUDA_FN_SETRANDOMSEED		= 2;
const int CUDA_FN_GETDEVICE			= 3;
const int CUDA_FN_RESETDEVICE		= 4;
const int CUDA_FN_SYNCHRONIZEDEVICE	= 5;
const int CUDA_FN_GETDEVICEPROP		= 6;
const int CUDA_FN_CHECKMEMORYATTRIB = 7;
const int CUDA_FN_GETDEVICEMEMORY   = 8;

const int CUDA_FN_DEVICE_CANACCESSPEER = 10;
const int CUDA_FN_DEVICE_ENABLEPEERACCESS = 11;
const int CUDA_FN_DEVICE_DISABLEPEERACCESS = 12;

const int CUDA_FN_CREATE_MEMORYPOINTER = 18;
const int CUDA_FN_FREE_MEMORYPOINTER = 19;

const int CUDA_FN_ALLOCMEM			= 20;
const int CUDA_FN_FREEMEM			= 21;
const int CUDA_FN_GETMEM			= 22;
const int CUDA_FN_SETMEM			= 23;
const int CUDA_FN_SETMEMAT			= 24;

const int CUDA_FN_ALLOCHOSTBUFFER	= 25;
const int CUDA_FN_FREEHOSTBUFFER	= 26;
const int CUDA_FN_GETHOSTMEM        = 27;
const int CUDA_FN_SETHOSTMEM		= 28;

const int CUDA_FN_CREATE_STREAM		= 30;
const int CUDA_FN_FREE_STREAM		= 31;
const int CUDA_FN_SYNCHRONIZE_STREAM = 32;
const int CUDA_FN_SYNCHRONIZE_THREAD = 33;

const int CUDA_FN_CREATE_MEMTEST    = 34;
const int CUDA_FN_FREE_MEMTEST      = 35;
const int CUDA_FN_RUN_MEMTEST       = 36;

const int CUDA_FN_CREATE_NCCL		= 40;
const int CUDA_FN_FREE_NCCL			= 41;
const int CUDA_FN_NCCL_INIT_SINGLEPROCESS = 42;
const int CUDA_FN_NCCL_INIT_MULTIPROCESS = 43;
const int CUDA_FN_NCCL_BROADCAST	= 44;
const int CUDA_FN_NCCL_ALLREDUCE	= 45;

const int CUDNN_FN_CREATE_CUDNN		= 47;
const int CUDNN_FN_FREE_CUDNN		= 48;

const int CUDNN_FN_CREATE_TENSORDESC = 50;
const int CUDNN_FN_FREE_TENSORDESC	 = 51;
const int CUDNN_FN_SET_TENSORDESC	 = 52;
const int CUDNN_FN_ADD_TENSOR        = 53;

const int CUDNN_FN_CREATE_FILTERDESC = 60;
const int CUDNN_FN_FREE_FILTERDESC	 = 61; 
const int CUDNN_FN_SET_FILTERDESC	 = 62;

const int CUDNN_FN_CREATE_CONVDESC	= 70;
const int CUDNN_FN_FREE_CONVDESC	= 71;
const int CUDNN_FN_SET_CONVDESC		= 72;
const int CUDNN_FN_GET_CONVINFO		= 73;
const int CUDNN_FN_FWD_CONV			= 74;
const int CUDNN_FN_BWD_CONV_BIAS	= 75;
const int CUDNN_FN_BWD_CONV_FILTER  = 76;
const int CUDNN_FN_BWD_CONV_DATA    = 77;

const int CUDNN_FN_CREATE_POOLDESC	= 80;
const int CUDNN_FN_FREE_POOLDESC	= 81;
const int CUDNN_FN_SET_POOLDESC		= 82;
const int CUDNN_FN_POOL_FWD			= 83;
const int CUDNN_FN_POOL_BWD			= 84;

const int CUDNN_FN_CREATE_LRNDESC	= 90;
const int CUDNN_FN_FREE_LRNDESC		= 91;
const int CUDNN_FN_SET_LRNDESC		= 92;

const int CUDNN_FN_GET_DROPOUT_INFO = 94;
const int CUDNN_FN_CREATE_DROPOUTDESC = 95;
const int CUDNN_FN_FREE_DROPOUTDESC = 96;
const int CUDNN_FN_SET_DROPOUTDESC	= 97;
const int CUDNN_FN_DROPOUT_FWD		= 98;
const int CUDNN_FN_DROPOUT_BWD		= 99;

const int CUDNN_FN_TANH_FWD			= 100;
const int CUDNN_FN_TANH_BWD			= 101;

const int CUDNN_FN_SIGMOID_FWD		= 104;
const int CUDNN_FN_SIGMOID_BWD		= 105;

const int CUDNN_FN_RELU_FWD			= 108;
const int CUDNN_FN_RELU_BWD			= 109;

const int CUDNN_FN_SOFTMAX_FWD		= 111;
const int CUDNN_FN_SOFTMAX_BWD		= 112;

const int CUDNN_FN_LRN_CC_FWD		= 120;
const int CUDNN_FN_LRN_CC_BWD		= 121;
const int CUDNN_FN_LCN_CC_FWD		= 122;
const int CUDNN_FN_LCN_CC_BWD		= 123;

const int CUDA_FN_SET				= 200;
const int CUDA_FN_GET				= 201;
const int CUDA_FN_COPY				= 202;

const int CUDA_FN_GEMM2				= 219;
const int CUDA_FN_GEMM				= 220;
const int CUDA_FN_GEMV				= 221;
const int CUDA_FN_AXPY				= 222;
const int CUDA_FN_AXPBY				= 223;
const int CUDA_FN_SCAL				= 224;
const int CUDA_FN_DOT				= 225;
const int CUDA_FN_ASUM				= 226;
const int CUDA_FN_SCALE				= 227;
const int CUDA_FN_ADD_SCALAR		= 228;
const int CUDA_FN_ADD				= 229;
const int CUDA_FN_SUB				= 230;
const int CUDA_FN_MUL				= 231;
const int CUDA_FN_MUL_SCALAR		= 232;
const int CUDA_FN_DIV				= 233;
const int CUDA_FN_ABS				= 234;
const int CUDA_FN_EXP				= 235;
const int CUDA_FN_LOG				= 236;
const int CUDA_FN_POWX				= 237;
const int CUDA_FN_SIGN				= 238;
const int CUDA_FN_SQRT				= 239;
const int CUDA_FN_RECIPROCOL		= 240;
const int CUDA_FN_STUDENT			= 241;
const int CUDA_FN_LOGISTIC1			= 242;
const int CUDA_FN_LOGISTIC2			= 243;
const int CUDA_FN_ADD2				= 244;
const int CUDA_FN_COMPARE_SIGNS		= 245;
const int CUDA_FN_MAXVAL			= 246;
const int CUDA_FN_MINVAL			= 247;
const int CUDA_FN_SUMSQ				= 248;
const int CUDA_FN_SUMSQDIFF			= 249;
const int CUDA_FN_WIDTH				= 250;
const int CUDA_FN_CONTAINS_POINT	= 251;
const int CUDA_FN_DENAN				= 252;
const int CUDA_FN_SUB_AND_DOT		= 253;
const int CUDA_FN_MINMAXVAL			= 254;

const int CUDA_FN_IM2COL			= 280;
const int CUDA_FN_IM2COL_ND			= 281;
const int CUDA_FN_COL2IM			= 282;
const int CUDA_FN_COL2IM_ND			= 283;

const int CUDA_FN_ACCURACY_FWD      = 286;

const int CUDA_FN_CHANNEL_MAX       = 290;
const int CUDA_FN_CHANNEL_SUB       = 291;
const int CUDA_FN_CHANNEL_SUM       = 292;
const int CUDA_FN_CHANNEL_DIV       = 293;
const int CUDA_FN_CHANNEL_DOT       = 294;
const int CUDA_FN_CHANNEL_MUL		= 295;

const int CUDA_RNG_SETSEED			= 349;
const int CUDA_RNG_UNIFORM			= 350;
const int CUDA_RNG_GAUSSIAN			= 351;
const int CUDA_RNG_BERNOULLI		= 352;

const int CUDA_FN_BATCHREIDX_FWD	= 386;
const int CUDA_FN_BATCHREIDX_BWD	= 387;

const int CUDA_FN_EMBED_FWD			= 390;
const int CUDA_FN_EMBED_BWD			= 391;

const int CUDA_FN_POOL_FWD			= 400;
const int CUDA_FN_POOL_BWD			= 401;

const int CUDA_FN_UNPOOL_FWD		= 410;
const int CUDA_FN_UNPOOL_BWD		= 411;

const int CUDA_FN_TANH_FWD			= 420;
const int CUDA_FN_TANH_BWD			= 421;

const int CUDA_FN_SIGMOID_FWD		= 424;
const int CUDA_FN_SIGMOID_BWD		= 425;

const int CUDA_FN_RELU_FWD			= 428;
const int CUDA_FN_RELU_BWD			= 429;

const int CUDA_FN_ELU_FWD			= 430;
const int CUDA_FN_ELU_BWD			= 431;

const int CUDA_FN_DROPOUT_FWD		= 432;
const int CUDA_FN_DROPOUT_BWD		= 433;

const int CUDA_FN_BNLL_FWD			= 435;
const int CUDA_FN_BNLL_BWD			= 436;

const int CUDA_FN_PRELU_FWD			= 438;
const int CUDA_FN_PRELU_BWD			= 439;
const int CUDA_FN_PRELU_BWD_PARAM	= 440;

const int CUDA_FN_SOFTMAXLOSS_FWD	= 444;
const int CUDA_FN_SOFTMAXLOSS_BWD	= 445;

const int CUDA_FN_MAX_FWD			= 448;
const int CUDA_FN_MAX_BWD			= 449;

const int CUDA_FN_CROP_FWD  		= 450;
const int CUDA_FN_CROP_BWD          = 451;

const int CUDA_FN_CONCAT_FWD		= 452;
const int CUDA_FN_CONCAT_BWD		= 453;

const int CUDA_FN_SLICE_FWD			= 455;
const int CUDA_FN_SLICE_BWD			= 456;

const int CUDA_FN_TILE_FWD			= 457;
const int CUDA_FN_TILE_BWD			= 458;

const int CUDA_FN_BIAS_FWD			= 460;

const int CUDA_FN_SCALE_FWD			= 461;

const int CUDA_FN_THRESHOLD_FWD		= 462;

const int CUDA_FN_CLL_BWD			= 463;

const int CUDA_FN_LRN_FILLSCALE		= 465;
const int CUDA_FN_LRN_COMPUTEOUTPUT = 466;
const int CUDA_FN_LRN_COMPUTEDIFF   = 467;

const int CUDA_FN_LSTM_FWD			= 480;
const int CUDA_FN_LSTM_BWD			= 481;
const int CUDA_FN_LSTM_UNIT_FWD		= 482;
const int CUDA_FN_LSTM_UNIT_BWD		= 483;

const int CUDA_FN_COEFF_SUM_FWD		= 490;
const int CUDA_FN_COEFF_SUM_BWD		= 491;

const int CUDA_FN_SIGMOID_CROSS_ENTROPY_FWD = 496;
const int CUDA_FN_SIGMOID_CROSS_ENTROPY_IGNORE = 497;

const int CUDA_FN_SGD_UPDATE		= 500;
const int CUDA_FN_NESTEROV_UPDATE	= 501;
const int CUDA_FN_ADAGRAD_UPDATE	= 502;
const int CUDA_FN_ADADELTA_UPDATE	= 503;
const int CUDA_FN_ADAM_UPDATE		= 504;
const int CUDA_FN_RMSPROP_UPDATE	= 505;

const int CUDA_FN_COMBINE_DATA		= 550;

const int CUDA_FN_MTX_SET_DIAGONAL  = 700;
const int CUDA_FN_MTX_SET_DIAGONAL2 = 701;
const int CUDA_FN_MTX_ADD_VECTOR	= 702;
const int CUDA_FN_MTX_TRANSPOSE_OP  = 703;
const int CUDA_FN_MTX_AGGREGATE_COLS = 704;
const int CUDA_FN_MTX_AGGREGATE_ROWS = 705;
const int CUDA_FN_MTX_TRANSPOSE		= 706;
const int CUDA_FN_MTX_MEANCENTER_BY_COL	= 707;
const int CUDA_FN_MTX_EUCLIDEAN_DIST = 709;
const int CUDA_FN_MTX_DOT			= 710;

const int CUDA_FN_CREATE_PCA		= 800;
const int CUDA_FN_RUN_PCA			= 801;
const int CUDA_FN_FREE_PCA			= 802;

const int CUDA_FN_TSNE_UPDATE				= 850;
const int CUDA_FN_TSNE_UPDATE_GRAD			= 851;
const int CUDA_FN_TSNE_COMPUTE_EXACT_ERROR  = 852;
const int CUDA_FN_TSNE_COMPUTE_SQUARED_EUCLIDEAN_DISTANCE = 854;
const int CUDA_FN_TSNE_COMPUTE_Q_MATRIX		= 855;
const int CUDA_FN_TSNE_COMPUTE_EXACT_GRADIENT = 856;
const int CUDA_FN_TSNE_SYMMETRIZE_MATRIX    = 858;
const int CUDA_FN_TSNE_COMPUTE_KNN_BOUNDS   = 859;

const int CUDA_FN_CREATE_TSNE_GAUSSIAN_PERPLEXITY = 870;
const int CUDA_FN_FREE_TSNE_GAUSSIAN_PERPLEXITY = 871;
const int CUDA_FN_FIND_TSNE_GAUSSIAN_PERPLEXITY = 872;

const int CUDA_FN_CREATE_TSNE				= 875;
const int CUDA_FN_FREE_TSNE					= 876;
const int CUDA_FN_TSNE_COMPUTE_GRADIENT1	= 877;
const int CUDA_FN_TSNE_COMPUTE_ERROR1		= 878;

const int CUDA_FN_GUASSIAN_BLUR     = 900;
const int CUDA_FN_HAMMING_DIFF      = 901;
const int CUDA_FN_CALC_BATCH_DIST   = 902;

const int CUDA_FN_GET_DEVICE_NAME	= 1000;
const int CUDA_FN_GET_P2P_INFO		= 1001;
const int CUDA_FN_GET_DEVICE_INFO   = 1002;


//=============================================================================
//	Kernel Classses
//=============================================================================

template <class T>
class Kernel
{
	Device<T> m_device;

public:
	Kernel() : m_device()
	{
	}

	~Kernel()
	{
		CleanUp();
	}

	long Initialize(T* pfInput, long lCount)
	{
		return Run(CUDA_FN_SETDEVICE, pfInput, lCount, NULL, NULL);
	}

	void CleanUp()
	{
	}

	int GetDevice()
	{
		return m_device.GetDevice();
	}

	HostBuffer<T>* GetHostBuffer(long hHandle)
	{
		return m_device.GetHostBuffer(hHandle);
	}

	cudaStream_t GetStream(long hStream)
	{
		return m_device.GetStream(hStream);
	}

	long GetMemory(long hHandle, MemoryItem** ppItem)
	{
		return m_device.GetMemory(hHandle, ppItem);
	}

	long AllocHost(long lCount, T** ppfOutput, T* pSrc, bool bSrcOnDevice = false)
	{
		return m_device.AllocHost(lCount, ppfOutput, pSrc, bSrcOnDevice);
	}

	long FreeHost(T* pfInput)
	{
		return m_device.FreeHost(pfInput);
	}

	long FreeHost(LPTSTR pfInput)
	{
		return m_device.FreeHost(pfInput);
	}

	ncclHandle<T>* GetNCCL(long hNccl)
	{
		return m_device.GetNccl(hNccl);
	}

	long SetNCCL(ncclHandle<T>* pNccl, T** ppfOutput, long* plCount)
	{
		return m_device.SetNccl(pNccl, plCount, ppfOutput);
	}

	long Run(long lfnIdx, T* pfInput, long lCount, T** ppfOutput, long* plCount);

	long Query(long lfnIdx, LONG* pfInput, long lCount, LPTSTR* ppfOutput);
};


#endif // #ifndef __MAIN_H_
