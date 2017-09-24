//=============================================================================
//	FILE:	util.cu
//
//	DESC:	This file implements the utility functions.
//=============================================================================

#include "util.h"

//=============================================================================
//	Constants
//=============================================================================

//=============================================================================
//	Public Methods
//=============================================================================

inline double cint(double x)
{
	double dfInt = 0;

	if (modf(x, &dfInt) >= 0.5)
		return (x >= 0) ? ceil(x) : floor(x);
	else
		return (x < 0) ? ceil(x) : floor(x);
}

inline double round2(double r, int places)
{
	double off = pow(10.0, places);
	return cint(r*off)/off;
}

inline double roundex(double r)
{
	char sz[256];

	sprintf(sz, "%0.7lf", r);
	return atof(sz);
}

bool GetErrorString(long lErr, char* szErr, long lMaxErr)
{
	if (GetCudaErrorString(lErr, szErr, lMaxErr))
		return true;

	switch (lErr)
	{
		case ERROR_PARAM:
			_snprintf(szErr, lMaxErr, "GENERAL: Parameter error (%ld)", lErr);
			return true;

		case ERROR_PARAM_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "GENERAL: Parameter out of range (%ld)", lErr);
			return true;

		case ERROR_PARAM_NULL:
			_snprintf(szErr, lMaxErr, "GENERAL: Parameter is NULL (%ld)", lErr);
			return true;

		case ERROR_FILE_WRITE:
			_snprintf(szErr, lMaxErr, "GENERAL: Failure when writing to file (%ld)", lErr);
			return true;

		case ERROR_FILE_READ:
			_snprintf(szErr, lMaxErr, "GENERAL: Failure when reading from file (%ld)", lErr);
			return true;

		case ERROR_FILE_OPEN:
			_snprintf(szErr, lMaxErr, "GENERAL: Failure when opening a file (%ld)", lErr);
			return true;

		case ERROR_MATRIX:
			_snprintf(szErr, lMaxErr, "MATRIX: general matrix error (%ld)", lErr);
			return true;

		case ERROR_MEMORY_EXPECTED_DEVICE:
			_snprintf(szErr, lMaxErr, "GENERAL: Expected device memory but received host memory (%ld)", lErr);
			return true;

		case ERROR_MEMORY_OUT:
			_snprintf(szErr, lMaxErr, "GENERAL: Out of memory (%ld)", lErr);
			return true;

		case ERROR_MATRIX_DIMENSIONS_DONT_MATCH:
			_snprintf(szErr, lMaxErr, "MATRIX: matrix dimensions do not match (%ld)", lErr);
			return true;

		case ERROR_MATRIX_DIMENSIONS_EXCEED_THREADS:
			_snprintf(szErr, lMaxErr, "MATRIX: matrix dimensions exceed number of threads (%ld)", lErr);
			return true;

		case ERROR_MATRIX_NOT_SQUARE:
			_snprintf(szErr, lMaxErr, "MATRIX: the current operation is only supported on square matrices (%ld)", lErr);
			return true;

		case ERROR_VECTOR:
			_snprintf(szErr, lMaxErr, "VECTOR: general vector error (%ld)", lErr);
			return true;

		case ERROR_VECTOR_DIMENSIONS_DONT_MATCH:
			_snprintf(szErr, lMaxErr, "VECTOR: vector dimensions do not match (%ld)", lErr);
			return true;

		case ERROR_NN:
			_snprintf(szErr, lMaxErr, "NN: general neural net error (%ld)", lErr);
			return true;

		case ERROR_NN_LAYER_COUNTS_DONT_MATCH:
			_snprintf(szErr, lMaxErr, "NN: layer counts do not match (%ld)", lErr);
			return true;

		case ERROR_CUBLAS_NULL:
			_snprintf(szErr, lMaxErr, "NN: The cublas handle is NULL! (%ld)", lErr);
			return true;

		case ERROR_CUDA_NOTSUPPORED_ON_DISPLAYGPU:
			_snprintf(szErr, lMaxErr, "CUDA: The function you are attempting to run is not supported on the display GPU (only supported on headless gpus)! (%ld)", lErr);
			return true;

		case ERROR_CUDA_MISSING_NCCL64DLL:
			_snprintf(szErr, lMaxErr, "CUDA: The 'nccl64' DLL is missing from the executable directory!  For example when using version 134, the file 'nccl64_134.dll' should be in the same directory as the executable. (%ld)", lErr);
			return true;
	}

	return false;
}

bool GetCudaErrorString(long lErr, char* szErr, long lMaxErr)
{
	if (lErr == 0)
		return false;

	switch (lErr)
	{
		case cudaErrorMissingConfiguration:
			_snprintf(szErr, lMaxErr, "CUDA: Missing configuration error (%ld)", lErr);
			return true;
			
		case cudaErrorMemoryAllocation:
			_snprintf(szErr, lMaxErr, "CUDA: Memory allocation error (%ld)", lErr);
			return true;
			
		case cudaErrorInitializationError:
			_snprintf(szErr, lMaxErr, "CUDA: Initialization error (%ld)", lErr);
			return true;
			
		case cudaErrorLaunchFailure:
			_snprintf(szErr, lMaxErr, "CUDA: Launch failure (%ld)", lErr);
			return true;
			
		case cudaErrorPriorLaunchFailure:
			_snprintf(szErr, lMaxErr, "CUDA: Prior launch failure (%ld)", lErr);
			return true;
			
		case cudaErrorLaunchTimeout:
			_snprintf(szErr, lMaxErr, "CUDA: Prior launch failure (%ld)", lErr);
			return true;
			
		case cudaErrorLaunchOutOfResources:
			_snprintf(szErr, lMaxErr, "CUDA: Launch out of resources error (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidDeviceFunction:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid device function (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidConfiguration:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid configuration (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidDevice:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid device (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidValue:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid value (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidPitchValue:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid pitch value (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidSymbol:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid symbol (%ld)", lErr);
			return true;
			
		case cudaErrorMapBufferObjectFailed:
			_snprintf(szErr, lMaxErr, "CUDA: Map buffer object failed (%ld)", lErr);
			return true;
			
		case cudaErrorUnmapBufferObjectFailed:
			_snprintf(szErr, lMaxErr, "CUDA: Unmap buffer object failed (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidHostPointer:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid host pointer (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidDevicePointer:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid device pointer (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidTexture:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid texture (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidTextureBinding:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid texture binding (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidChannelDescriptor:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid channel descriptor (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidMemcpyDirection:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid memcpy direction (%ld)", lErr);
			return true;
			
		case cudaErrorAddressOfConstant:
			_snprintf(szErr, lMaxErr, "CUDA: Address of constant error (%ld)", lErr);
			return true;
			
		case cudaErrorTextureFetchFailed:
			_snprintf(szErr, lMaxErr, "CUDA: Texture fetch failed (%ld)", lErr);
			return true;
			
		case cudaErrorTextureNotBound:
			_snprintf(szErr, lMaxErr, "CUDA: Texture not bound error (%ld)", lErr);
			return true;
			
		case cudaErrorSynchronizationError:
			_snprintf(szErr, lMaxErr, "CUDA: Synchronization error (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidFilterSetting:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid filter setting (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidNormSetting:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid norm setting (%ld)", lErr);
			return true;
			
		case cudaErrorMixedDeviceExecution:
			_snprintf(szErr, lMaxErr, "CUDA: Mixed device execution (%ld)", lErr);
			return true;
			
		case cudaErrorCudartUnloading:
			_snprintf(szErr, lMaxErr, "CUDA: cuda runtime unloading (%ld)", lErr);
			return true;
			
		case cudaErrorUnknown:
			_snprintf(szErr, lMaxErr, "CUDA: Unknown error condition (%ld)", lErr);
			return true;
			
		case cudaErrorNotYetImplemented:
			_snprintf(szErr, lMaxErr, "CUDA: Function not yet implemented (%ld)", lErr);
			return true;
			
		case cudaErrorMemoryValueTooLarge:
			_snprintf(szErr, lMaxErr, "CUDA: Memory value too large (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidResourceHandle:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid resource handle (%ld)", lErr);
			return true;
			
		case cudaErrorNotReady:
			_snprintf(szErr, lMaxErr, "CUDA: Not ready error (%ld)", lErr);
			return true;
			
		case cudaErrorInsufficientDriver:
			_snprintf(szErr, lMaxErr, "CUDA: cuda runtime is newer than driver (%ld)", lErr);
			return true;
			
		case cudaErrorSetOnActiveProcess:
			_snprintf(szErr, lMaxErr, "CUDA: Set on active process error (%ld)", lErr);
			return true;
			
		case cudaErrorNoDevice:
			_snprintf(szErr, lMaxErr, "CUDA: No available CUDA device (%ld)", lErr);
			return true;
			
		case cudaErrorECCUncorrectable:
			_snprintf(szErr, lMaxErr, "CUDA: Uncorrectable ECC error detected (%ld)", lErr);
			return true;
			
		case cudaErrorStartupFailure:
			_snprintf(szErr, lMaxErr, "CUDA: Startup failure (%ld)", lErr);
			return true;
	}

	return false;
}


//=============================================================================
//	Device Functions
//=============================================================================

//end util.cu