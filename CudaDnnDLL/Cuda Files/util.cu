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
			_snprintf(szErr, lMaxErr, "MEMORY: Expected device memory but received host memory (%ld)", lErr);
			return true;

		case ERROR_MEMORY_RANGE_EXCEEDED:
			_snprintf(szErr, lMaxErr, "MEMORY: Exceeded the maximum amount of memory size available as a chunk (%ld)", lErr);
			return true;

		case ERROR_MEMORY_MIXED_HALF_TYPES:
			_snprintf(szErr, lMaxErr, "MEMORY: You are using a mix of half types and non-half types.  All types for this function must be of the same type (%ld)", lErr);
			return true;

		case ERROR_MEMORY_HALF_TYPE_NOT_SUPPORTED:
			_snprintf(szErr, lMaxErr, "MEMORY: The GPU that you are using has limited half-type support.  Full half-type support is only available on Maxwell gpu's with compute 5.3 and above (%ld)", lErr);
			return true;

		case ERROR_MEMORY_OUT:
			_snprintf(szErr, lMaxErr, "MEMORY: Out of memory (%ld)", lErr);
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
			_snprintf(szErr, lMaxErr, "CUDA: The 'nccl64' DLL is missing from the executable directory!  For example when using the version 134 for CUDA 10.0, the file 'nccl64_134.10.0.dll' should be in the same directory as the executable. (%ld)", lErr);
			return true;

		case ERROR_TSNE:
			_snprintf(szErr, lMaxErr, "TSNE: A general TSN-E error occurred. (%ld)", lErr);
			return true;

		case ERROR_TSNE_NO_DISTANCES_FOUND:
			_snprintf(szErr, lMaxErr, "TSNE: No differences found between the images - they may all be the same. (%ld)", lErr);
			return true;

		case ERROR_SSD:
			_snprintf(szErr, lMaxErr, "SSD: A general SSD error occurred. (%ld)", lErr);
			return true;

		case ERROR_SSD_NOT_INITIALIZED:
			_snprintf(szErr, lMaxErr, "SSD: The SSD is not initialized. (%ld)", lErr);
			return true;

		case ERROR_SSD_INVALID_CODE_TYPE:
			_snprintf(szErr, lMaxErr, "SSD: The SSD code type specified is invalid. (%ld)", lErr);
			return true;

		case ERROR_SSD_INVALID_BBOX_DIMENSION:
			_snprintf(szErr, lMaxErr, "SSD: The SSD bbox dimension (width or height) is invalid (e.g. < 0). (%ld)", lErr);
			return true;

		case ERROR_SSD_HOST_TYPE_NOT_SUPPORTED:
			_snprintf(szErr, lMaxErr, "SSD: The HOST type specified is not supported for this function. (%ld)", lErr);
			return true;

		case ERROR_SSD_BAD_MATCH:
			_snprintf(szErr, lMaxErr, "SSD: The current matching is bad, expected a match index of -1. (%ld)", lErr);
			return true;

		case ERROR_SSD_GT_LABEL_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "SSD: The ground truth label is out of range. (%ld)", lErr);
			return true;

		case ERROR_SSD_PRIOR_VARIANCE_COUNT:
			_snprintf(szErr, lMaxErr, "SSD: The prior variances count does not match the prior bbox count. (%ld)", lErr);
			return true;

		case ERROR_SSD_BACKGROUND_LABEL_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "SSD: The background label id is out of range. (%ld)", lErr);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_INCORRECT:
			_snprintf(szErr, lMaxErr, "SSD: The match_index should equal the number of priors in the compute conf loss calculation. (%ld)", lErr);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_GT_MISSING_ITEM:
			_snprintf(szErr, lMaxErr, "SSD: The ground-truths are missing an expected itemId in the compute conf loss calculation. (%ld)", lErr);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_MATCH_INDEX_OUT_OF_RANGE:
			_snprintf(szErr, lMaxErr, "SSD: The match index is out of range of the ground-truths in the compute conf loss calculation. (%ld)", lErr);
			return true;

		case ERROR_SSD_COMPUTE_CONF_LOSS_INVALID_LABEL:
			_snprintf(szErr, lMaxErr, "SSD: The label in the compute conf loss calculation is invalid. (%ld)", lErr);
			return true;

		case ERROR_SSD_NOT_SUPPORTED_IN_HALF_BBOX:
			_snprintf(szErr, lMaxErr, "SSD: The requested query is not supported by the half Bbox - only full BBox's support this type of query. (%ld)", lErr);
			return true;

		case ERROR_SSD_LOC_PRED_LABEL_NOT_FOUND:
			_snprintf(szErr, lMaxErr, "SSD: Could not find an expected label in the loc predictions. (%ld)", lErr);
			return true;

		case ERROR_SSD_CODE_TYPE_UNKNOWN:
			_snprintf(szErr, lMaxErr, "SSD: Unknown Code-Type. (%ld)", lErr);
			return true;

		case ERROR_SSD_SAMPLE_SIZE_TOO_SMALL:
			_snprintf(szErr, lMaxErr, "SSD: The sample size is too small and must be > 0. (%ld)", lErr);
			return true;

		case ERROR_SSD_INVALID_NUM_CLASSES:
			_snprintf(szErr, lMaxErr, "SSD: The number of classes is incorrect (e.g. when using map to agnostic, only 2 classes are valid for backgroundLabel >= 0, otherwise only 1 class is valid). (%ld)", lErr);
			return true;

		case ERROR_SSD_CONF_LOSS_TYPE_UNKNOWN:
			_snprintf(szErr, lMaxErr, "SSD: The conf loss type is unknown and invalid. (%ld)", lErr);
			return true;

		case ERROR_SSD_BACKGROUND_LABEL_IN_DATASET:
			_snprintf(szErr, lMaxErr, "SSD: The ground truth was found in the dataset. (%ld)", lErr);
			return true;

		case ERROR_SSD_NUMLOCCLASSES_INVALID_FOR_SHARED:
			_snprintf(szErr, lMaxErr, "SSD: The number of loc classes must be 1 when using shared location. (%ld)", lErr);
			return true;
	}

	return false;
}

bool GetCudaErrorString(long lErr, char* szErr, long lMaxErr)
{
	if (lErr == 0)
		return false;

	if ((lErr & ERROR_CUBLAS_OFFSET) == ERROR_CUBLAS_OFFSET)
	{
		lErr &= (~ERROR_CUBLAS_OFFSET);

		switch (lErr)
		{
		case CUBLAS_STATUS_NOT_INITIALIZED:
			_snprintf(szErr, lMaxErr, "cuBlas: The cuBlas library was not initialized propertly (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_ALLOC_FAILED:
			_snprintf(szErr, lMaxErr, "cuBlas: A resource allocation failed within the cuBlas library (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_INVALID_VALUE:
			_snprintf(szErr, lMaxErr, "cuBlas: An invalid parameter was passed to the function. (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_ARCH_MISMATCH:
			_snprintf(szErr, lMaxErr, "cuBlas: The function requires functionality not supported by the current device architecture. (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_MAPPING_ERROR:
			_snprintf(szErr, lMaxErr, "cuBlas: Access to the GPU memory failed possibly caused by a failure to bind to a texture. (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_EXECUTION_FAILED:
			_snprintf(szErr, lMaxErr, "cuBlas: A cuBlas GPU kernel failed to execute. (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_INTERNAL_ERROR:
			_snprintf(szErr, lMaxErr, "cuBlas: A failure occurred within cuBlas. (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_NOT_SUPPORTED:
			_snprintf(szErr, lMaxErr, "cuBlas: The function called is not supported. (%ld)", lErr);
			return true;

		case CUBLAS_STATUS_LICENSE_ERROR:
			_snprintf(szErr, lMaxErr, "cuBlas: The functionality requested requires a license that is missing. (%ld)", lErr);
			return true;
		}
	}
	else if ((lErr & ERROR_CUDNN_OFFSET) == ERROR_CUDNN_OFFSET)
	{
		lErr &= (~ERROR_CUDNN_OFFSET);

		switch (lErr)
		{
			case CUDNN_STATUS_NOT_INITIALIZED:
				_snprintf(szErr, lMaxErr, "cuDNN: The cuDNN library was not initialized propertly (%ld)", lErr);
				return true;

			case CUDNN_STATUS_ALLOC_FAILED:
				_snprintf(szErr, lMaxErr, "cuDNN: A resource allocation failed within the cuDNN library (%ld)", lErr);
				return true;

			case CUDNN_STATUS_BAD_PARAM:
				_snprintf(szErr, lMaxErr, "cuDNN: An incorrect parameter was passed to a function (%ld)", lErr);
				return true;

			case CUDNN_STATUS_INTERNAL_ERROR:
				_snprintf(szErr, lMaxErr, "cuDNN: An internal operation failed (%ld)", lErr);
				return true;

			case CUDNN_STATUS_INVALID_VALUE:
				_snprintf(szErr, lMaxErr, "cuDNN: An invalid value was detected (%ld)", lErr);
				return true;

			case CUDNN_STATUS_ARCH_MISMATCH:
				_snprintf(szErr, lMaxErr, "cuDNN: The function requires a feature not supported by the current GPU device - your device must have compute capability of 3.0 or greater (%ld)", lErr);
				return true;

			case CUDNN_STATUS_MAPPING_ERROR:
				_snprintf(szErr, lMaxErr, "cuDNN: An access to the GPU's memory space failed perhaps caused when binding to a texture (%ld)", lErr);
				return true;

			case CUDNN_STATUS_EXECUTION_FAILED:
				_snprintf(szErr, lMaxErr, "cuDNN: The current GPU program failed to execute (%ld)", lErr);
				return true;

			case CUDNN_STATUS_NOT_SUPPORTED:
				_snprintf(szErr, lMaxErr, "cuDNN: The functionality requested is not supported by this version of cuDNN (%ld)", lErr);
				return true;

			case CUDNN_STATUS_LICENSE_ERROR:
				_snprintf(szErr, lMaxErr, "cuDNN: The functionality requested requires a license that does not appear to exist (%ld)", lErr);
				return true;

			case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
				_snprintf(szErr, lMaxErr, "cuDNN: The runtime library required by RNN calls (nvcuda.dll) cannot be found (%ld)", lErr);
				return true;

#if CUDNN_MAJOR >= 7
			case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
				_snprintf(szErr, lMaxErr, "cuDNN: Some tasks in the user stream are still running (%ld)", lErr);
				return true;

			case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
				_snprintf(szErr, lMaxErr, "cuDNN: A numerical overflow occurred while executing the GPU kernel (%ld)", lErr);
				return true;
#endif
		}

		return false;
	}

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
			_snprintf(szErr, lMaxErr, "CUDA: Prior launch failure - timeout (%ld)", lErr);
			return true;
			
		case cudaErrorLaunchOutOfResources:
			_snprintf(szErr, lMaxErr, "CUDA: Launch out of resources error (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidDeviceFunction:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid device function (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidConfiguration:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid configuration for the device used (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidDevice:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid CUDA device (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidValue:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid parameter value (%ld)", lErr);
			return true;
			
		case cudaErrorInvalidPitchValue:
			_snprintf(szErr, lMaxErr, "CUDA: Invalid pitch parameter value (%ld)", lErr);
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
			_snprintf(szErr, lMaxErr, "CUDA: cuda runtime is newer than the installed NVIDIA CUDA driver (%ld)", lErr);
			return true;
			
		case cudaErrorSetOnActiveProcess:
			_snprintf(szErr, lMaxErr, "CUDA: Set on active process error (%ld)", lErr);
			return true;

		case cudaErrorInvalidSurface:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the surface parameter is invalid (%ld)", lErr);
			return true;

		case cudaErrorNoDevice:
			_snprintf(szErr, lMaxErr, "CUDA: No available CUDA device (%ld)", lErr);
			return true;
			
		case cudaErrorECCUncorrectable:
			_snprintf(szErr, lMaxErr, "CUDA: Uncorrectable ECC error detected (%ld)", lErr);
			return true;

		case cudaErrorSharedObjectSymbolNotFound:
			_snprintf(szErr, lMaxErr, "CUDA: The link to to a shared object failed to resolve (%ld)", lErr);
			return true;

		case cudaErrorSharedObjectInitFailed:
			_snprintf(szErr, lMaxErr, "CUDA: The initialization of a shared object failed (%ld)", lErr);
			return true;

		case cudaErrorUnsupportedLimit:
			_snprintf(szErr, lMaxErr, "CUDA: The ::cudaLimit argument is not supported by the active device (%ld)", lErr);
			return true;

		case cudaErrorDuplicateVariableName:
			_snprintf(szErr, lMaxErr, "CUDA: Inidcates that multiple global or constant variables share the same string name (%ld)", lErr);
			return true;

		case cudaErrorDuplicateTextureName:
			_snprintf(szErr, lMaxErr, "CUDA: Inidcates that multiple texture variables share the same string name (%ld)", lErr);
			return true;

		case cudaErrorDuplicateSurfaceName:
			_snprintf(szErr, lMaxErr, "CUDA: Inidcates that multiple surface variables share the same string name (%ld)", lErr);
			return true;

		case cudaErrorDevicesUnavailable:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that all CUDA devices are busy or unavailable at the current time (%ld)", lErr);
			return true;

		case cudaErrorInvalidKernelImage:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the device kernel image is invalid (%ld)", lErr);
			return true;

		case cudaErrorNoKernelImageForDevice:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that there is no kernel image available that is suitable for the device (%ld)", lErr);
			return true;

		case cudaErrorIncompatibleDriverContext:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the current context is not compatible with this CUDA Runtime (%ld)", lErr);
			return true;

		case cudaErrorPeerAccessAlreadyEnabled:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a call to ::cudaDeviceEnablePeerAccess is trying to re-enable peer addressing from a context that already has peer addressing enabled (%ld)", lErr);
			return true;

		case cudaErrorPeerAccessNotEnabled:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that ::cudaDeviceDisablePeerAccess is trying to disable peer addressing which has not been enabled yet (%ld)", lErr);
			return true;

		case cudaErrorDeviceAlreadyInUse:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a call tried to access an exclusive-thread device that is already in use by a different thread (%ld)", lErr);
			return true;

		case cudaErrorProfilerDisabled:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates profiler is not initialized for this run (%ld)", lErr);
			return true;

		case cudaErrorAssert:
			_snprintf(szErr, lMaxErr, "CUDA: An assert triggered in device code during kernel execution (%ld)", lErr);
			return true;

		case cudaErrorTooManyPeers:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the hardware resources required ot enable peer access have been exhaused for one or more of the devices (%ld)", lErr);
			return true;

		case cudaErrorHostMemoryAlreadyRegistered:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the memory range specified has already been registered (%ld)", lErr);
			return true;

		case cudaErrorHostMemoryNotRegistered:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that the pointer specified does not correspond to any currently registered memory region (%ld)", lErr);
			return true;

		case cudaErrorOperatingSystem:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that an OS call failed (%ld)", lErr);
			return true;

		case cudaErrorPeerAccessUnsupported:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that P2P access is not supported across the given devices (%ld)", lErr);
			return true;

		case cudaErrorLaunchMaxDepthExceeded:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a device runtime grid launch did not occur because  the depth of the child grid would exceed the maximum supported number of nested grid launches (%ld)", lErr);
			return true;

		case cudaErrorLaunchFileScopedTex:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a grid launch did no occur because the kernel uses file-scoped textures which are unsupported by the device runtime (%ld)", lErr);
			return true;

		case cudaErrorLaunchFileScopedSurf:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. (%ld)", lErr);
			return true;

		case cudaErrorSyncDepthExceeded:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a call to ::cudaDeviceSynchronize made from the device runtime failed becaue the call was made at grid depth greater than either the default (2 levels) or a user specified limit (%ld)", lErr);
			return true;

		case cudaErrorLaunchPendingCountExceeded:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates that a device runtime grid launch failed because the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount (%ld)", lErr);
			return true;

		case cudaErrorNotPermitted:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates the attempted operation is not permitted (%ld)", lErr);
			return true;

		case cudaErrorNotSupported:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates the attempted operation is not supported on the current system or device (%ld)", lErr);
			return true;

		case cudaErrorHardwareStackError:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered an error in the call statck during kernel execution possibly due to stack corruption or exceeding the stack size limit (%ld)", lErr);
			return true;

		case cudaErrorIllegalInstruction:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered an illegal instruction during kernel execution (%ld)", lErr);
			return true;

		case cudaErrorMisalignedAddress:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered a load or storage instruction on a memory address which is not aligned (%ld)", lErr);
			return true;

		case cudaErrorInvalidAddressSpace:
			_snprintf(szErr, lMaxErr, "CUDA: While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied an address not in those spaces (%ld)", lErr);
			return true;

		case cudaErrorInvalidPc:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered an invalid program counter (%ld)", lErr);
			return true;

		case cudaErrorIllegalAddress:
			_snprintf(szErr, lMaxErr, "CUDA: Device encountered a load or storage instruction on an invalid memory address (%ld)", lErr);
			return true;

		case cudaErrorInvalidPtx:
			_snprintf(szErr, lMaxErr, "CUDA: A PTX compilation failed (%ld)", lErr);
			return true;

		case cudaErrorInvalidGraphicsContext:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates an error with the OpenGL or DirectX context (%ld)", lErr);
			return true;

		case cudaErrorNvlinkUncorrectable:
			_snprintf(szErr, lMaxErr, "CUDA: Indicates an uncorrectable NVLink error was detected during the execution (%ld)", lErr);
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