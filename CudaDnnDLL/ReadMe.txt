========================================================================
    DYNAMIC LINK LIBRARY : CudaDnnDLL Project Overview
========================================================================

The CudaDnnDLL provides all low-level CUDA and cuDNN supporting functions including
all device level kernels used to interact with CUDA to perform AI operations at
high speed using the massively parallel nature of CUDA.

The main entry point to the DLL are the DLL_InvokeFloat and DLL_InvokeDouble functions
which then delegate each call to the appropriate function.

C# applications interact with the CudaDnnDLL using the CudaControl found on NuGet at:
https://www.nuget.org/packages/CudaControl or by searching 
https://www.nuget.org/packages?q=CudaControl

Several builds of the CudaDnnDLL are created to support different versions of CUDA
and in some cases support different compute levels.  The following variantes are
currently built.

CudaDnnDll.11.0 - supports CUDA 11.0 with compute level 5.3+
CudaDnnDll.10.2 - supports CUDA 10.2 with compute level 5.3+
CudaDnnDll.10.2.3_5 - supports CUDA 10.2 with compute level 3.5
CudaDnnDll.10.1 - supports CUDA 10.1 with compute level 5.3+
CudaDnnDll.10.1.3_5 - supports CUDA 10.1 with compute level 3.5
CudaDnnDll.10.0 - supports CUDA 10.0
CudaDnnDll.9.2  - supports CUDA 9.2

/////////////////////////////////////////////////////////////////////////////
