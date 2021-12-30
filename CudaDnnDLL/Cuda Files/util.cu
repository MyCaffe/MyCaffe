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

bool GetErrorString(long lKernel, long lErr, char* szErr, long lMaxErr)
{
	if (GetCudaErrorString(lKernel, lErr, szErr, lMaxErr))
		return true;

	switch (lErr)
	{
		case ERROR_PARAM:
			_snprintf(szErr, lMaxErr, "GENERAL: Parameter error (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_PARAM_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "GENERAL: Parameter out of range (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_PARAM_NULL:
			_snprintf(szErr, lMaxErr, "GENERAL: Parameter is NULL (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_FILE_WRITE:
			_snprintf(szErr, lMaxErr, "GENERAL: Failure when writing to file (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_FILE_READ:
			_snprintf(szErr, lMaxErr, "GENERAL: Failure when reading from file (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_FILE_OPEN:
			_snprintf(szErr, lMaxErr, "GENERAL: Failure when opening a file (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MATRIX:
			_snprintf(szErr, lMaxErr, "MATRIX: general matrix error (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MEMORY_EXPECTED_DEVICE:
			_snprintf(szErr, lMaxErr, "MEMORY: Expected device memory but received host memory (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MEMORY_RANGE_EXCEEDED:
			_snprintf(szErr, lMaxErr, "MEMORY: Exceeded the maximum amount of memory size available as a chunk (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MEMORY_MIXED_HALF_TYPES:
			_snprintf(szErr, lMaxErr, "MEMORY: You are using a mix of half types and non-half types.  All types for this function must be of the same type (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MEMORY_HALF_TYPE_NOT_SUPPORTED:
			_snprintf(szErr, lMaxErr, "MEMORY: The GPU that you are using has limited half-type support.  Full half-type support is only available on Maxwell gpu's with compute 5.3 and above (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MEMORY_OUT:
			_snprintf(szErr, lMaxErr, "MEMORY: Out of memory (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MEMORY_NOT_FOUND:
			_snprintf(szErr, lMaxErr, "MEMORY: Memory was not found and therefore could not be freed. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_BATCH_TOO_SMALL:
			_snprintf(szErr, lMaxErr, "DATA: The batch size used is too small - not enough label variety for sequencing. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MEMORY_TOO_SMALL:
			_snprintf(szErr, lMaxErr, "MEMORY: Memory size allocated is too small - must allocate more memory for this operation. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MATRIX_DIMENSIONS_DONT_MATCH:
			_snprintf(szErr, lMaxErr, "MATRIX: matrix dimensions do not match (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MATRIX_DIMENSIONS_EXCEED_THREADS:
			_snprintf(szErr, lMaxErr, "MATRIX: matrix dimensions exceed number of threads (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_MATRIX_NOT_SQUARE:
			_snprintf(szErr, lMaxErr, "MATRIX: the current operation is only supported on square matrices (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_VECTOR:
			_snprintf(szErr, lMaxErr, "VECTOR: general vector error (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_VECTOR_DIMENSIONS_DONT_MATCH:
			_snprintf(szErr, lMaxErr, "VECTOR: vector dimensions do not match (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_NN:
			_snprintf(szErr, lMaxErr, "NN: general neural net error (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_NN_LAYER_COUNTS_DONT_MATCH:
			_snprintf(szErr, lMaxErr, "NN: layer counts do not match (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_CUBLAS_NULL:
			_snprintf(szErr, lMaxErr, "NN: The cublas handle is NULL! (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_CUDA_NOTSUPPORED_ON_DISPLAYGPU:
			_snprintf(szErr, lMaxErr, "CUDA: The function you are attempting to run is not supported on the display GPU (only supported on headless gpus)! (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_CUDA_MISSING_NCCL64DLL:
			_snprintf(szErr, lMaxErr, "CUDA: The 'nccl64' DLL is missing from the executable directory!  For example when using the version 134 for CUDA 10.0, the file 'nccl64_134.10.0.dll' should be in the same directory as the executable. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_TSNE:
			_snprintf(szErr, lMaxErr, "TSNE: A general TSN-E error occurred. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_TSNE_NO_DISTANCES_FOUND:
			_snprintf(szErr, lMaxErr, "TSNE: No differences found between the images - they may all be the same. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD:
			_snprintf(szErr, lMaxErr, "SSD: A general SSD error occurred. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_NOT_INITIALIZED:
			_snprintf(szErr, lMaxErr, "SSD: The SSD is not initialized. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_CODE_TYPE:
			_snprintf(szErr, lMaxErr, "SSD: The SSD code type specified is invalid. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_BBOX_DIMENSION:
			_snprintf(szErr, lMaxErr, "SSD: The SSD bbox dimension (width or height) is invalid (e.g. < 0). (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_HOST_TYPE_NOT_SUPPORTED:
			_snprintf(szErr, lMaxErr, "SSD: The HOST type specified is not supported for this function. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_BAD_MATCH:
			_snprintf(szErr, lMaxErr, "SSD: The current matching is bad, expected a match index of -1. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_GT_LABEL_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "SSD: The ground truth label is out of range. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_PRIOR_VARIANCE_COUNT:
			_snprintf(szErr, lMaxErr, "SSD: The prior variances count does not match the prior bbox count. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_BACKGROUND_LABEL_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "SSD: The background label id is out of range. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_INCORRECT:
			_snprintf(szErr, lMaxErr, "SSD: The match_index should equal the number of priors in the compute conf loss calculation. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_GT_MISSING_ITEM:
			_snprintf(szErr, lMaxErr, "SSD: The ground-truths are missing an expected itemId in the compute conf loss calculation. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "SSD: The match index is out of range of the ground-truths in the compute conf loss calculation. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_INVALID_LABEL:
			_snprintf(szErr, lMaxErr, "SSD: The label in the compute conf loss calculation is invalid. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_NOT_SUPPORTED_IN_HALF_BBOX:
			_snprintf(szErr, lMaxErr, "SSD: The requested query is not supported by the half Bbox - only full BBox's support this type of query. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_LOC_PRED_LABEL_NOT_FOUND:
			_snprintf(szErr, lMaxErr, "SSD: Could not find an expected label in the loc predictions. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_SAMPLE_SIZE_TOO_SMALL:
			_snprintf(szErr, lMaxErr, "SSD: The sample size is too small and must be > 0. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_NUM_CLASSES:
			_snprintf(szErr, lMaxErr, "SSD: The number of classes is incorrect (e.g. when using map to agnostic, only 2 classes are valid for backgroundLabel >= 0, otherwise only 1 class is valid). (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_CONF_LOSS_TYPE:
			_snprintf(szErr, lMaxErr, "SSD: The conf loss type is unknown and invalid. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_BACKGROUND_LABEL_IN_DATASET:
			_snprintf(szErr, lMaxErr, "SSD: The ground truth was found in the dataset. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_NUMLOCCLASSES_FOR_SHARED:
			_snprintf(szErr, lMaxErr, "SSD: The number of loc classes must be 1 when using shared location. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_LOCCOUNT_GTCOUNT:
			_snprintf(szErr, lMaxErr, "SSD: The loc pred and loc gt must be equal. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_LOC_LOSS_MATCH_COUNT:
			_snprintf(szErr, lMaxErr, "SSD: The loc loss match count is incorrect. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_INVALID_LOC_LOSS_TYPE:
			_snprintf(szErr, lMaxErr, "SSD: The loc loss type is invalid. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case ERROR_SSD_MINEHARDEXAMPLES_NO_MATCHES:
			_snprintf(szErr, lMaxErr, "SSD: No matches were found to mine hard examples. (%ld), Kernel = %ld", lErr, lKernel);
			return true;
	}

	return false;
}

bool GetCudaErrorString(long lKernel, long lErr, char* szErr, long lMaxErr)
{
	if (lErr == 0)
		return false;

	if ((lErr & ERROR_CUBLAS_OFFSET) == ERROR_CUBLAS_OFFSET)
	{
		lErr &= (~ERROR_CUBLAS_OFFSET);

		switch (lErr)
		{
		case CUBLAS_STATUS_NOT_INITIALIZED:
			_snprintf(szErr, lMaxErr, "cuBlas: The cuBlas library was not initialized propertly (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_ALLOC_FAILED:
			_snprintf(szErr, lMaxErr, "cuBlas: A resource allocation failed within the cuBlas library (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_INVALID_VALUE:
			_snprintf(szErr, lMaxErr, "cuBlas: An invalid parameter was passed to the function. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_ARCH_MISMATCH:
			_snprintf(szErr, lMaxErr, "cuBlas: The function requires functionality not supported by the current device architecture. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_MAPPING_ERROR:
			_snprintf(szErr, lMaxErr, "cuBlas: Access to the GPU memory failed possibly caused by a failure to bind to a texture. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_EXECUTION_FAILED:
			_snprintf(szErr, lMaxErr, "cuBlas: A cuBlas GPU kernel failed to execute. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_INTERNAL_ERROR:
			_snprintf(szErr, lMaxErr, "cuBlas: A failure occurred within cuBlas. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_NOT_SUPPORTED:
			_snprintf(szErr, lMaxErr, "cuBlas: The function called is not supported. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case CUBLAS_STATUS_LICENSE_ERROR:
			_snprintf(szErr, lMaxErr, "cuBlas: The functionality requested requires a license that is missing. (%ld), Kernel = %ld", lErr, lKernel);
			return true;
		}
	}
	else if ((lErr & ERROR_CUDNN_OFFSET) == ERROR_CUDNN_OFFSET)
	{
		lErr &= (~ERROR_CUDNN_OFFSET);

		switch (lErr)
		{
			case CUDNN_STATUS_NOT_INITIALIZED:
				_snprintf(szErr, lMaxErr, "cuDNN: The cuDNN library was not initialized propertly (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_ALLOC_FAILED:
				_snprintf(szErr, lMaxErr, "cuDNN: A resource allocation failed within the cuDNN library (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_BAD_PARAM:
				_snprintf(szErr, lMaxErr, "cuDNN: An incorrect parameter was passed to a function (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_INTERNAL_ERROR:
				_snprintf(szErr, lMaxErr, "cuDNN: An internal operation failed (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_INVALID_VALUE:
				_snprintf(szErr, lMaxErr, "cuDNN: An invalid value was detected (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_ARCH_MISMATCH:
				_snprintf(szErr, lMaxErr, "cuDNN: The function requires a feature not supported by the current GPU device - your device must have compute capability of 3.0 or greater (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_MAPPING_ERROR:
				_snprintf(szErr, lMaxErr, "cuDNN: An access to the GPU's memory space failed perhaps caused when binding to a texture (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_EXECUTION_FAILED:
				_snprintf(szErr, lMaxErr, "cuDNN: The current GPU program failed to execute (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_NOT_SUPPORTED:
				_snprintf(szErr, lMaxErr, "cuDNN: The functionality requested is not supported by this version of cuDNN (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_LICENSE_ERROR:
				_snprintf(szErr, lMaxErr, "cuDNN: The functionality requested requires a license that does not appear to exist (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
				_snprintf(szErr, lMaxErr, "cuDNN: The runtime library required by RNN calls (nvcuda.dll) cannot be found (%ld), Kernel = %ld", lErr, lKernel);
				return true;

#if CUDNN_MAJOR >= 7
			case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
				_snprintf(szErr, lMaxErr, "cuDNN: Some tasks in the user stream are still running (%ld), Kernel = %ld", lErr, lKernel);
				return true;

			case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
				_snprintf(szErr, lMaxErr, "cuDNN: A numerical overflow occurred while executing the GPU kernel (%ld), Kernel = %ld", lErr, lKernel);
				return true;
#endif
		}

		return false;
	}

	switch (lErr)
	{
		case cudaErrorMissingConfiguration:
			_snprintf(szErr, lMaxErr, "CUDA: Missing configuration error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorMemoryAllocation:
			_snprintf(szErr, lMaxErr, "CUDA: Memory allocation error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInitializationError:
			_snprintf(szErr, lMaxErr, "CUDA: Initialization error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorLaunchFailure:
			_snprintf(szErr, lMaxErr, "CUDA: Launch failure (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorPriorLaunchFailure:
			_snprintf(szErr, lMaxErr, "CUDA: Prior launch failure (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorLaunchTimeout:
			_snprintf(szErr, lMaxErr, "CUDA: Prior launch failure - timeout (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorLaunchOutOfResources:
			_snprintf(szErr, lMaxErr, "CUDA: Launch out of resources error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidDeviceFunction:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid device function (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidConfiguration:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid configuration for the device used (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidDevice:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid CUDA device (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidValue:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid parameter value (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidPitchValue:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid pitch parameter value (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidSymbol:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid symbol (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorMapBufferObjectFailed:
			_snprintf(szErr, lMaxErr, "CUDA: Map buffer object failed (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorUnmapBufferObjectFailed:
			_snprintf(szErr, lMaxErr, "CUDA: Unmap buffer object failed (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidHostPointer:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid host pointer (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidDevicePointer:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid device pointer (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidTexture:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid texture (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidTextureBinding:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid texture binding (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidChannelDescriptor:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid channel descriptor (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidMemcpyDirection:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid memcpy direction (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorAddressOfConstant:
			_snprintf(szErr, lMaxErr, "CUDA: Address of constant error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorTextureFetchFailed:
			_snprintf(szErr, lMaxErr, "CUDA: Texture fetch failed (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorTextureNotBound:
			_snprintf(szErr, lMaxErr, "CUDA: Texture not bound error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorSynchronizationError:
			_snprintf(szErr, lMaxErr, "CUDA: Synchronization error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidFilterSetting:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid filter setting (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidNormSetting:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid norm setting (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorMixedDeviceExecution:
			_snprintf(szErr, lMaxErr, "CUDA: Mixed device execution (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorCudartUnloading:
			_snprintf(szErr, lMaxErr, "CUDA: cuda runtime unloading (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorUnknown:
			_snprintf(szErr, lMaxErr, "CUDA: Unknown error condition (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorNotYetImplemented:
			_snprintf(szErr, lMaxErr, "CUDA: Function not yet implemented (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorMemoryValueTooLarge:
			_snprintf(szErr, lMaxErr, "CUDA: Memory value too large (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInvalidResourceHandle:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid resource handle (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorNotReady:
			_snprintf(szErr, lMaxErr, "CUDA: Not ready error (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorInsufficientDriver:
			_snprintf(szErr, lMaxErr, "CUDA: cuda runtime is newer than the installed NVIDIA CUDA driver (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorSetOnActiveProcess:
			_snprintf(szErr, lMaxErr, "CUDA: Set on active process error (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorInvalidSurface:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the surface parameter is invalid (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorNoDevice:
			_snprintf(szErr, lMaxErr, "CUDA: No available CUDA device (%ld), Kernel = %ld", lErr, lKernel);
			return true;
			
		case cudaErrorECCUncorrectable:
			_snprintf(szErr, lMaxErr, "CUDA: Uncorrectable ECC error detected (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorSharedObjectSymbolNotFound:
			_snprintf(szErr, lMaxErr, "CUDA: The link to to a shared object failed to resolve (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorSharedObjectInitFailed:
			_snprintf(szErr, lMaxErr, "CUDA: The initialization of a shared object failed (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorUnsupportedLimit:
			_snprintf(szErr, lMaxErr, "CUDA: The ::cudaLimit argument is not supported by the active device (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorDuplicateVariableName:
			_snprintf(szErr, lMaxErr, "CUDA: Inidcates that multiple global or constant variables share the same string name (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorDuplicateTextureName:
			_snprintf(szErr, lMaxErr, "CUDA: Inidcates that multiple texture variables share the same string name (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorDuplicateSurfaceName:
			_snprintf(szErr, lMaxErr, "CUDA: Inidcates that multiple surface variables share the same string name (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorDevicesUnavailable:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that all CUDA devices are busy or unavailable at the current time (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorInvalidKernelImage:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the device kernel image is invalid (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorNoKernelImageForDevice:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that there is no kernel image available that is suitable for the device (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorIncompatibleDriverContext:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the current context is not compatible with this CUDA Runtime (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorPeerAccessAlreadyEnabled:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a call to ::cudaDeviceEnablePeerAccess is trying to re-enable peer addressing from a context that already has peer addressing enabled (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorPeerAccessNotEnabled:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that ::cudaDeviceDisablePeerAccess is trying to disable peer addressing which has not been enabled yet (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorDeviceAlreadyInUse:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a call tried to access an exclusive-thread device that is already in use by a different thread (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorProfilerDisabled:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates profiler is not initialized for this run (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorAssert:
			_snprintf(szErr, lMaxErr, "CUDA: An assert triggered in device code during kernel execution (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorTooManyPeers:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the hardware resources required ot enable peer access have been exhaused for one or more of the devices (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorHostMemoryAlreadyRegistered:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the memory range specified has already been registered (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorHostMemoryNotRegistered:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the pointer specified does not correspond to any currently registered memory region (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorOperatingSystem:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that an OS call failed (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorPeerAccessUnsupported:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that P2P access is not supported across the given devices (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorLaunchMaxDepthExceeded:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a device runtime grid launch did not occur because  the depth of the child grid would exceed the maximum supported number of nested grid launches (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorLaunchFileScopedTex:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a grid launch did no occur because the kernel uses file-scoped textures which are unsupported by the device runtime (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorLaunchFileScopedSurf:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorSyncDepthExceeded:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a call to ::cudaDeviceSynchronize made from the device runtime failed becaue the call was made at grid depth greater than either the default (2 levels) or a user specified limit (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorLaunchPendingCountExceeded:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a device runtime grid launch failed because the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorNotPermitted:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates the attempted operation is not permitted (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorNotSupported:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates the attempted operation is not supported on the current system or device (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorHardwareStackError:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered an error in the call statck during kernel execution possibly due to stack corruption or exceeding the stack size limit (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorIllegalInstruction:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered an illegal instruction during kernel execution (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorMisalignedAddress:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered a load or storage instruction on a memory address which is not aligned (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorInvalidAddressSpace:
			_snprintf(szErr, lMaxErr, "CUDA: While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied an address not in those spaces (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorInvalidPc:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered an invalid program counter (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorIllegalAddress:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered a load or storage instruction on an invalid memory address (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorInvalidPtx:
			_snprintf(szErr, lMaxErr, "CUDA: A PTX compilation failed (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorInvalidGraphicsContext:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates an error with the OpenGL or DirectX context (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorNvlinkUncorrectable:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates an uncorrectable NVLink error was detected during the execution (%ld), Kernel = %ld", lErr, lKernel);
			return true;

		case cudaErrorStartupFailure:
			_snprintf(szErr, lMaxErr, "CUDA: Startup failure (%ld), Kernel = %ld", lErr, lKernel);
			return true;
	}

	return false;
}


//=============================================================================
//	Device Functions
//=============================================================================

//end util.cu