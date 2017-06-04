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


//-----------------------------------------------------------------------------
//	Funtion Overrides
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

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

const int ERROR_CUBLAS_NULL						= ERROR_PARAM + 10;

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


//-----------------------------------------------------------------------------
//	Helper Types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//	Helper Functions
//-----------------------------------------------------------------------------

bool GetCudaErrorString(long lErr, char* szErr, long lErrMax);
bool GetErrorString(long lErr, char* szErr, long lErrMax);

inline int CAFFE_GET_BLOCKS(const int n)
{
	return (n + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

#endif // __UTIL_H_
