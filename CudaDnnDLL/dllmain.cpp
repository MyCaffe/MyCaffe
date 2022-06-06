// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include "Cuda Files\main.h"

HMODULE g_hModule = NULL;
Kernel<double>* g_rgdwDoubleKernelTable[MAX_KERNELS];
Kernel<float>* g_rgdwFloatKernelTable[MAX_KERNELS];
DWORD g_dwMaxKernelCount = MAX_KERNELS;
DWORD g_dwLastKernelDoubleIndex = 1;
DWORD g_dwLastKernelFloatIndex = 1;
CRITICAL_SECTION g_DoubleKernelTableLock;
bool m_bDoubleKernelTableLockInit = false;
CRITICAL_SECTION g_FloatKernelTableLock;
bool m_bFloatKernelTableLockInit = false;

void initializeKernelTables();
void freeKernelTables();

BOOL APIENTRY DllMain( HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved )
{
	switch (ul_reason_for_call)
	{
		case DLL_PROCESS_ATTACH:
			g_hModule = hModule;
			initializeKernelTables();
			break;

		case DLL_THREAD_ATTACH:
			break;

		case DLL_THREAD_DETACH:
			break;

		case DLL_PROCESS_DETACH:
			freeKernelTables();
			g_hModule = NULL;
			break;
	}

	return TRUE;
}

bool allocateTlsIndex(DWORD* pdwIdx)
{
	if (*pdwIdx == 0)
	{
		if ((*pdwIdx = TlsAlloc()) == TLS_OUT_OF_INDEXES)
			return FALSE;
	}

	return TRUE;
}

void initializeKernelTables()
{
	//-------------------------------------------
	//	Setup the double kernel table.
	//-------------------------------------------

	if (!InitializeCriticalSectionAndSpinCount(&g_DoubleKernelTableLock, 0x00000400))
		return;

	m_bDoubleKernelTableLockInit = true;
	EnterCriticalSection(&g_DoubleKernelTableLock);

	for (DWORD dwIdx = 0; dwIdx < MAX_KERNELS; dwIdx++)
	{
		g_rgdwDoubleKernelTable[dwIdx] = NULL;
	}

	LeaveCriticalSection(&g_DoubleKernelTableLock);


	//-------------------------------------------
	//	Setup the float kernel table.
	//-------------------------------------------

	if (!InitializeCriticalSectionAndSpinCount(&g_FloatKernelTableLock, 0x00000400))
		return;

	m_bFloatKernelTableLockInit = true;
	EnterCriticalSection(&g_FloatKernelTableLock);

	for (DWORD dwIdx = 0; dwIdx < MAX_KERNELS; dwIdx++)
	{
		g_rgdwFloatKernelTable[dwIdx] = NULL;
	}

	LeaveCriticalSection(&g_FloatKernelTableLock);
}

void freeKernelTables()
{
	if (m_bDoubleKernelTableLockInit)
	{
		EnterCriticalSection(&g_DoubleKernelTableLock);

		// Only delete the global kernel for the others
		// are deleted by the respective owners.

		try
		{
			for (DWORD dwIdx = 0; dwIdx < MAX_KERNELS; dwIdx++)
			{
				if (g_rgdwDoubleKernelTable[dwIdx] != NULL)
				{
					delete g_rgdwDoubleKernelTable[dwIdx];
					g_rgdwDoubleKernelTable[dwIdx] = NULL;
				}
			}
		}
		catch (...)
		{
		}

		m_bDoubleKernelTableLockInit = false;
		LeaveCriticalSection(&g_DoubleKernelTableLock);

		try
		{
			if (g_DoubleKernelTableLock.LockCount == -1)
				DeleteCriticalSection(&g_DoubleKernelTableLock);
			else
				OutputDebugStringA("CudaDnnDLL Double CS still locked.");
		}
		catch (...)
		{
		}
	}
	else
	{
		OutputDebugStringA("CudaDnnDLL Double CS NOT INIT!");
	}

	if (m_bFloatKernelTableLockInit)
	{
		EnterCriticalSection(&g_FloatKernelTableLock);

		try
		{
			for (DWORD dwIdx = 0; dwIdx < MAX_KERNELS; dwIdx++)
			{
				if (g_rgdwFloatKernelTable[dwIdx] != NULL)
				{
					delete g_rgdwFloatKernelTable[dwIdx];
					g_rgdwFloatKernelTable[dwIdx] = NULL;
				}
			}
		}
		catch (...)
		{
		}

		m_bFloatKernelTableLockInit = false;
		LeaveCriticalSection(&g_FloatKernelTableLock);

		try
		{
			if (g_FloatKernelTableLock.LockCount == -1)
				DeleteCriticalSection(&g_FloatKernelTableLock);
			else
				OutputDebugStringA("CudaDnnDLL Float CS still locked.");
		}
		catch (...)
		{
		}
	}
	else
	{
		OutputDebugStringA("CudaDnnDLL Float CS NOT INIT!");
	}
}
