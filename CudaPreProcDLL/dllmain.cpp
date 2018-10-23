// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include "Cuda Files\main.h"

HMODULE g_hModule;
Kernel<double>* g_pKernelD;
Kernel<float>* g_pKernelF;
CRITICAL_SECTION g_DoubleKernelTableLock;
CRITICAL_SECTION g_FloatKernelTableLock;

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

	EnterCriticalSection(&g_DoubleKernelTableLock);
	g_pKernelD = new Kernel<double>();
	LeaveCriticalSection(&g_DoubleKernelTableLock);


	//-------------------------------------------
	//	Setup the float kernel table.
	//-------------------------------------------

	if (!InitializeCriticalSectionAndSpinCount(&g_FloatKernelTableLock, 0x00000400))
		return;

	EnterCriticalSection(&g_FloatKernelTableLock);
	g_pKernelF = new Kernel<float>();
	LeaveCriticalSection(&g_FloatKernelTableLock);
}

void freeKernelTables()
{
	EnterCriticalSection(&g_DoubleKernelTableLock);
	if (g_pKernelD != NULL)
	{
		delete g_pKernelD;
		g_pKernelD = NULL;
	}
	LeaveCriticalSection(&g_DoubleKernelTableLock);
	DeleteCriticalSection(&g_DoubleKernelTableLock);

	EnterCriticalSection(&g_FloatKernelTableLock);
	if (g_pKernelF != NULL)
	{
		delete g_pKernelF;
		g_pKernelF = NULL;
	}
	LeaveCriticalSection(&g_FloatKernelTableLock);
	DeleteCriticalSection(&g_FloatKernelTableLock);
}
