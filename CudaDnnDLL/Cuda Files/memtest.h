//=============================================================================
//	FILE:	memtest.h
//
//	DESC:	This file is used to manage the memory testing.
//=============================================================================
#ifndef __MEMTEST_CU__
#define __MEMTEST_CU__

#include "util.h"
#include "math.h"

//=============================================================================
//	Types
//=============================================================================

enum MEMTEST_TYPE
{
	MOV_INV_8 = 1
};


//=============================================================================
//	Classes
//=============================================================================

template <class T>
class Memory;


//-----------------------------------------------------------------------------
//	MemoryTest Handle Class
//
//	This class stores the Memory Test description information.
//-----------------------------------------------------------------------------
template <class T>
class memtestHandle
{
	void* m_pTestingStartAddress;
	size_t m_szAddressesTested;
	size_t m_szAddressSize;
	Memory<T>* m_pMem;
	unsigned int m_rgerr_count_host[2];
	size_t* m_rgerr_addr_host[2];
	size_t* m_rgerr_addr_host1;
	size_t* m_rgerr_addr_host2;
	unsigned long* m_rgerr_expect_host;
	unsigned long* m_rgerr_current_host;
	unsigned long* m_rgerr_second_read_host;
	size_t* m_rgerr_addr;
	unsigned long* m_rgerr_expect;
	unsigned long* m_rgerr_current;
	unsigned long* m_rgerr_second_read;
	unsigned int* m_perr_count;
	unsigned char* m_pTestMem;
	size_t m_szTotalNumBlocks;

	long allocate_small_mem();
	long reset_small_mem();
	long run_move_inv_8_test(size_t szStartOffset, size_t szCount, bool bWrite, bool bReadWrite, bool bRead);
	long move_inv_test(size_t szStartOffset, size_t szCount, unsigned int p1, unsigned int p2, bool bWrite, bool bReadWrite, bool bRead);
	long error_checking(int nHostIdx, size_t blockidx, unsigned int* pErr);
	long load_error_data(long* plCount, T** ppfData, bool bVerbose);
	long alloc_mem(void** pp, size_t sz);
	long free_mem(void* p);

public:
	
	memtestHandle()
	{
		m_pTestingStartAddress = NULL;
		m_szAddressesTested = 0;
		m_pMem = NULL;
		m_rgerr_count_host[0] = 0;
		m_rgerr_count_host[1] = 0;
		m_rgerr_addr_host[0] = NULL;
		m_rgerr_addr_host[1] = NULL;
		m_rgerr_addr_host1 = NULL;
		m_rgerr_addr_host2 = NULL;
		m_rgerr_expect_host = NULL;
		m_rgerr_current_host = NULL;
		m_rgerr_second_read_host = NULL;
		m_rgerr_addr = NULL;
		m_rgerr_expect = NULL;
		m_rgerr_current = NULL;
		m_rgerr_second_read = NULL;
		m_perr_count = NULL;
		m_pTestMem = NULL;
		m_szTotalNumBlocks = 0;
	}

	long Initialize(Memory<T>* pMem, T fPctToAllocate, size_t* szTotalNumBlocks, T* pfMemAllocated, T* pfMemStartAddress, T* pfMemBlockSize); 
	long Run(MEMTEST_TYPE testType, size_t szStartOffset, size_t szCount, long* plCount, T** ppfData, bool bVerbose, bool bWrite, bool bReadWrite, bool bRead);
	long CleanUp();
};


//=============================================================================
//	Inline Methods
//=============================================================================

#endif