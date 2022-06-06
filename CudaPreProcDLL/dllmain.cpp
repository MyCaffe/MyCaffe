// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include "Cuda Files\main.h"

HMODULE g_hModule;
Kernel<double>* g_pKernelD;
Kernel<float>* g_pKernelF;
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

void initializeKernelTables()
{
	//-------------------------------------------
	//	Setup the double kernel table.
	//-------------------------------------------

	if (!InitializeCriticalSectionAndSpinCount(&g_DoubleKernelTableLock, 0x00000400))
		return;

	m_bDoubleKernelTableLockInit = true;
	EnterCriticalSection(&g_DoubleKernelTableLock);
	g_pKernelD = new Kernel<double>();
	LeaveCriticalSection(&g_DoubleKernelTableLock);


	//-------------------------------------------
	//	Setup the float kernel table.
	//-------------------------------------------

	if (!InitializeCriticalSectionAndSpinCount(&g_FloatKernelTableLock, 0x00000400))
		return;

	m_bFloatKernelTableLockInit = true;
	EnterCriticalSection(&g_FloatKernelTableLock);
	g_pKernelF = new Kernel<float>();
	LeaveCriticalSection(&g_FloatKernelTableLock);
}

void freeKernelTables()
{
	if (m_bDoubleKernelTableLockInit)
	{
		EnterCriticalSection(&g_DoubleKernelTableLock);

		try
		{
			if (g_pKernelD != NULL)
			{
				delete g_pKernelD;
				g_pKernelD = NULL;
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
				OutputDebugStringA("CudaPreProcDLL Double CS still locked.");
		}
		catch (...)
		{
		}
	}
	else
	{
		OutputDebugStringA("CudaPreProcDLL Double CS NOT INIT!");
	}

	if (m_bFloatKernelTableLockInit)
	{
		EnterCriticalSection(&g_FloatKernelTableLock);

		try
		{
			if (g_pKernelF != NULL)
			{
				delete g_pKernelF;
				g_pKernelF = NULL;
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
				OutputDebugStringA("CudaPreProcDLL Float CS still locked.");
		}
		catch (...)
		{
		}
	}
	else
	{
		OutputDebugStringA("CudaPreProcDLL Float CS NOT INIT!");
	}
}
