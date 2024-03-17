//=============================================================================
//	FILE:	Util.h
//	DESC:	This file contains helpful macros used when building CUDA files.
//=============================================================================

#ifndef __UTIL_H_
#define __UTIL_H_

#include <math.h>
#include <tchar.h>
#include <windows.h>
#include <stdio.h>
//#include <atlbase.h>
#include <atlconv.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cudnn.h>

enum FILLER_TYPE
{
    FT_CONSTANT = 0,
    FT_XAVIER = 1,
    FT_GAUSSIAN = 2
};

//-----------------------------------------------------------------------------
//	Set the version of CUDA
//-----------------------------------------------------------------------------

#define CUDNN_4 1	// depreciated
#define CUDNN_5 1	// cuda 5.1
#define CUDNN_6 1


//=============================================================================
//	Constants
//=============================================================================

const int CAFFE_CUDA_NUM_THREADS	= 512;

//=============================================================================
//	Local Classes
//=============================================================================

class Lock
{
    CRITICAL_SECTION* m_plock;

public:
    Lock(CRITICAL_SECTION* plock)
    {
        m_plock = NULL;

        if (plock->LockCount <= -1)
        {
            m_plock = plock;
            EnterCriticalSection(m_plock);
        }
    }

    ~Lock()
    {
        if (m_plock != NULL && m_plock->LockCount < -1)
            LeaveCriticalSection(m_plock);
        m_plock = NULL;
    }
};


//-----------------------------------------------------------------------------
//	Funtion Overrides
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define Stringize( L )     #L 
#define MakeString( M, L ) M(L)
#define $Line MakeString( Stringize, __LINE__ )
#define Reminder __FILE__ "(" $Line ") : Reminder: "

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
    cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);                                            \

    //! Check for CUDA error
#ifdef _DEBUG
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    err = cudaThreadSynchronize();                                           \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#else
#  define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }
#endif

//-----------------------------------------------------------------------------
//	Error Codes
//-----------------------------------------------------------------------------

const int ERROR_BASE							= 90000;
const int ERROR_DLL_NOT_INIT = ERROR_BASE - 1;
const int ERROR_PARAM							= ERROR_BASE;
const int ERROR_PARAM_OUT_OF_RANGE				= ERROR_PARAM + 1;
const int ERROR_PARAM_NULL						= ERROR_PARAM + 2;
const int ERROR_FILE_WRITE						= ERROR_PARAM + 3;
const int ERROR_FILE_READ						= ERROR_PARAM + 4;
const int ERROR_FILE_OPEN						= ERROR_PARAM + 5;
const int ERROR_MEMORY_EXPECTED_DEVICE			= ERROR_PARAM + 6;
const int ERROR_MEMORY_OUT						= ERROR_PARAM + 7;
const int ERROR_MEMORY_NULL						= ERROR_PARAM + 8;
const int ERROR_NOT_IMPLEMENTED					= ERROR_PARAM + 9;
const int ERROR_MEMORY_RANGE_EXCEEDED			= ERROR_PARAM + 10;
const int ERROR_MEMORY_MIXED_HALF_TYPES			= ERROR_PARAM + 11;
const int ERROR_MEMORY_HALF_TYPE_NOT_SUPPORTED  = ERROR_PARAM + 12;
const int ERROR_MEMORY_NOT_FOUND				= ERROR_PARAM + 14;
const int ERROR_BATCH_TOO_SMALL					= ERROR_PARAM + 15;
const int ERROR_MEMORY_TOO_SMALL                = ERROR_PARAM + 16;
const int ERROR_DEVICE_NOT_INITIALIZED          = ERROR_PARAM + 17;

const int ERROR_CUBLAS_NULL						= ERROR_PARAM + 18;

const int ERROR_MATRIX							= ERROR_PARAM + 20;
const int ERROR_MATRIX_DIMENSIONS_DONT_MATCH	= ERROR_MATRIX + 1;
const int ERROR_MATRIX_DIMENSIONS_EXCEED_THREADS = ERROR_MATRIX + 2;
const int ERROR_MATRIX_NOT_SQUARE				= ERROR_MATRIX + 3;

const int ERROR_VECTOR							= ERROR_MATRIX + 20;
const int ERROR_VECTOR_DIMENSIONS_DONT_MATCH    = ERROR_VECTOR + 1;

const int ERROR_NN								= ERROR_VECTOR + 20;
const int ERROR_NN_LAYER_COUNTS_DONT_MATCH      = ERROR_NN + 1;

const int ERROR_CUDA							= ERROR_NN + 20;
const int ERROR_CUDA_NOTSUPPORED_ON_DISPLAYGPU  = ERROR_CUDA + 1;
const int ERROR_CUDA_MISSING_NCCL64DLL			= ERROR_CUDA + 2;
const int ERROR_CUDA_KERNEL_NOT_IMPLEMENTED     = ERROR_CUDA + 3;

const int ERROR_TSNE							= ERROR_NN + 40;
const int ERROR_TSNE_NO_DISTANCES_FOUND			= ERROR_TSNE + 1;

const int ERROR_SSD								= ERROR_TSNE + 20;
const int ERROR_SSD_NOT_INITIALIZED				= ERROR_SSD + 1;
const int ERROR_SSD_INVALID_CODE_TYPE			= ERROR_SSD + 2;
const int ERROR_SSD_INVALID_BBOX_DIMENSION		= ERROR_SSD + 3;
const int ERROR_SSD_INVALID_NUM_CLASSES			= ERROR_SSD + 4;
const int ERROR_SSD_INVALID_LOC_LOSS_TYPE		= ERROR_SSD + 5;
const int ERROR_SSD_INVALID_CONF_LOSS_TYPE		= ERROR_SSD + 6;
const int ERROR_SSD_INVALID_PRIOR_VARIANCE_COUNT = ERROR_SSD + 9;
const int ERROR_SSD_INVALID_NUMLOCCLASSES_FOR_SHARED = ERROR_SSD + 10;
const int ERROR_SSD_INVALID_LOCCOUNT_GTCOUNT	= ERROR_SSD + 11;
const int ERROR_SSD_INVALID_LOC_LOSS_MATCH_COUNT = ERROR_SSD + 12;

const int ERROR_SSD_HOST_TYPE_NOT_SUPPORTED		= ERROR_SSD + 13;
const int ERROR_SSD_BAD_MATCH					= ERROR_SSD + 14;
const int ERROR_SSD_GT_LABEL_OUT_OF_RANGE		= ERROR_SSD + 15;
const int ERROR_SSD_BACKGROUND_LABEL_OUT_OF_RANGE = ERROR_SSD + 16;
const int ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_INCORRECT = ERROR_SSD + 17;
const int ERROR_SSD_COMPUTE_CONF_LOSS_GT_MISSING_ITEM = ERROR_SSD + 18;
const int ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_OUT_OF_RANGE = ERROR_SSD + 19;
const int ERROR_SSD_COMPUTE_CONF_LOSS_INVALID_LABEL = ERROR_SSD + 20;
const int ERROR_SSD_NOT_SUPPORTED_IN_HALF_BBOX  = ERROR_SSD + 21;
const int ERROR_SSD_LOC_PRED_LABEL_NOT_FOUND    = ERROR_SSD + 22;
const int ERROR_SSD_SAMPLE_SIZE_TOO_SMALL		= ERROR_SSD + 23;
const int ERROR_SSD_BACKGROUND_LABEL_IN_DATASET = ERROR_SSD + 24;
const int ERROR_SSD_MINEHARDEXAMPLES_NO_MATCHES = ERROR_SSD + 25;

const int ERROR_LAYERNORM                       = ERROR_SSD + 40;     
const int ERROR_LAYERNORM_NOT_INITIALIZED       = ERROR_LAYERNORM + 1;

const int ERROR_ROPE                            = ERROR_LAYERNORM + 40;
const int ERROR_ROPE_NOT_INITIALIZED            = ERROR_ROPE + 1;

const int ERROR_RNN8							 = ERROR_ROPE + 40;
const int ERROR_RNN8_NOT_INITIALIZED			 = ERROR_RNN8 + 1;
const int ERROR_RNN8_INCOMPATIBLE_CUDNN_VER		 = ERROR_RNN8 + 2;
const int ERROR_RNN8_INVALID_FILLER			     = ERROR_RNN8 + 3;

const int ERROR_ATTN = ERROR_RNN8 + 40;
const int ERROR_ATTN_NOT_INITIALIZED             = ERROR_ATTN + 1;
const int ERROR_ATTN_INCOMPATIBLE_CUDNN_VER      = ERROR_ATTN + 2;
const int ERROR_ATTN_MISSING_DESCRIPTOR		     = ERROR_ATTN + 3;
const int ERROR_ATTN_INCOMPATIBLE_BLOB_SIZE      = ERROR_ATTN + 4;
const int ERROR_ATTN_INVALID_BLOCK_SIZE          = ERROR_ATTN + 5;

const int ERROR_BLOBLOADER                       = ERROR_ATTN + 40;
const int ERROR_BLOBLOADER_NOT_INITIALIZED       = ERROR_BLOBLOADER + 1;

const int ERROR_FUSEDCOMP                        = ERROR_BLOBLOADER + 20;
const int ERROR_FUSEDCOMP_NOT_INITIALIZED        = ERROR_FUSEDCOMP + 1;

const int ERROR_CPD                              = ERROR_BLOBLOADER + 40;
const int ERROR_CPD_NOT_INITIALIZED              = ERROR_CPD + 1;

const int ERROR_CUDNN_OFFSET					= 0x10000;
const int ERROR_CUBLAS_OFFSET					= 0x20000;
const int ERROR_CUDNNFE_OFFSET					= 0x40000;

//-----------------------------------------------------------------------------
// Cudnn Front End Errors
//-----------------------------------------------------------------------------

const int CUDNNFE_ATTRIBUTE_NOT_SET             = 1;
const int CUDNNFE_SHAPE_DEDUCTION_FAILED        = 2;
const int CUDNNFE_INVALID_TENSOR_NAME           = 3;
const int CUDNNFE_INVALID_VARIANT_PACK          = 4;
const int CUDNNFE_GRAPH_NOT_SUPPORTED           = 5;
const int CUDNNFE_GRAPH_EXECUTION_PLAN_CREATION_FAILED = 6;
const int CUDNNFE_GRAPH_EXECUTION_FAILED        = 7;
const int CUDNNFE_HEURISTIC_QUERY_FAILED        = 8;
const int CUDNNFE_UNSUPPORTED_GRAPH_FORMAT      = 9;
const int CUDNNFE_CUDA_API_FAILED               = 10;
const int CUDNNFE_CUDNN_BACKEND_API_FAILED      = 11;
const int CUDNNFE_INVALID_CUDA_DEVICE           = 12;
const int CUDNNFE_HANDLE_ERROR                  = 13;


//-----------------------------------------------------------------------------
//	Helper Functions
//-----------------------------------------------------------------------------

bool GetCudaErrorString(long lKernel, long lErr, char* szErr, long lErrMax);
bool GetErrorString(long lKernel, long lErr, char* szErr, long lErrMax);

inline int CAFFE_GET_BLOCKS(const int n)
{
	return (n + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

#endif // __UTIL_H_
