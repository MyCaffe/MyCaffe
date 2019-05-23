//=============================================================================
//	FILE:	memtest.cu
//
//	DESC:	This file implements the base class used to manage the memory testing
//			run on the underlying GPU device.
//
//	Using portions from CUDA Gpu memtest created by Innovative Systems Lab, 
//	located at https://sourceforge.net/p/cudagpumemtest/code/HEAD/tree/ and
//	used under the licence:
//
// -----------------------------
//  Illinois Open Source License
//
//  University of Illinois / NCSA
//  Open Source License
//
//  Copyright © 2009, University of Illinois.All rights reserved.
//
//  [original code] Developed by :
//
//  Innovative Systems Lab
//  National Center for Supercomputing Applications
//  http ://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html
//	
//	Permission is hereby granted, free of charge, to any person obtaining a copy of
//	this software and associated documentation files(the "Software"), to deal with
//	the Software without restriction, including without limitation the rights to use,
//	copy, modify, merge, publish, distribute, sublicense, and / or sell copies of the
//	Software, and to permit persons to whom the Software is furnished to do so, subject
//	to the following conditions :
//
//  * Redistributions of source code must retain the above copyright notice, this list
//  of conditions and the following disclaimers.
//
//  * Redistributions in binary form must reproduce the above copyright notice, this list
//  of conditions and the following disclaimers in the documentation and / or other materials
//  provided with the distribution.
//
//  * Neither the names of the Innovative Systems Lab, the National Center for Supercomputing
//  Applications, nor the names of its contributors may be used to endorse or promote products
//  derived from this Software without specific prior written permission.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE
//  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//  OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS WITH THE SOFTWARE.
//=============================================================================

#include "util.h"
#include "memory.h"
#include "memtest.h"

//=============================================================================
//	Defines and Constants
//=============================================================================

const unsigned long BLOCKSIZE = ((unsigned long)1024 * (unsigned long)1024);
const unsigned long GRIDSIZE = 128;
const unsigned long MAX_ERR_RECORD_COUNT = BLOCKSIZE;

#define MIN(x, y) (x < y ? x : y)


//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long memtestHandle<T>::allocate_small_mem()
{
	LONG lErr;

	if (lErr = cudaMalloc((void**)&m_perr_count, sizeof(unsigned int) * GRIDSIZE))
		return lErr;

	if (lErr = cudaMalloc((void**)&m_rgerr_addr, sizeof(size_t) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMalloc((void**)&m_rgerr_expect, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMalloc((void**)&m_rgerr_current, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMalloc((void**)&m_rgerr_second_read, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMallocHost((void**)&m_rgerr_addr_host1, sizeof(size_t) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMallocHost((void**)&m_rgerr_addr_host2, sizeof(size_t) * MAX_ERR_RECORD_COUNT))
		return lErr;

	m_rgerr_addr_host[0] = m_rgerr_addr_host1;
	m_rgerr_addr_host[1] = m_rgerr_addr_host2;

	//if (lErr = cudaMallocHost((void**)&m_rgerr_expect_host, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
	//	return lErr;

	//if (lErr = cudaMallocHost((void**)&m_rgerr_current_host, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
	//	return lErr;

	//if (lErr = cudaMallocHost((void**)&m_rgerr_second_read_host, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
	//	return lErr;

	reset_small_mem();

	return 0;
}

template <class T>
long memtestHandle<T>::reset_small_mem()
{
	LONG lErr;

	if (lErr = cudaMemset(m_perr_count, 0, sizeof(unsigned int)))
		return lErr;

	if (lErr = cudaMemset(m_rgerr_addr, 0, sizeof(size_t) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMemset(m_rgerr_expect, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMemset(m_rgerr_current, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
		return lErr;

	if (lErr = cudaMemset(m_rgerr_second_read, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
		return lErr;

	memset(m_rgerr_addr_host1, 0, sizeof(size_t) * MAX_ERR_RECORD_COUNT);
	memset(m_rgerr_addr_host2, 0, sizeof(size_t) * MAX_ERR_RECORD_COUNT);

	m_rgerr_count_host[0] = 0;
	m_rgerr_count_host[1] = 0;

	//if (lErr = memset(m_rgerr_expect_host, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
	//	return lErr;

	//if (lErr = memset(m_rgerr_current_host, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
	//	return lErr;

	//if (lErr = memset(m_rgerr_second_read_host, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT))
	//	return lErr;

	return 0;
}

template <class T>
long memtestHandle<T>::Initialize(Memory<T>* pMem, T fPctToAllocate, size_t* pszTotalNumBlocks, T* pfMemAllocated, T* pfMemStartAddress, T* pfMemBlockSize)
{
	LONG lErr;
	int nDeviceID;

	m_pMem = pMem;

	try
	{
		if (lErr = cudaGetDevice(&nDeviceID))
			throw lErr;

		cudaDeviceProp prop;
		if (lErr = cudaGetDeviceProperties(&prop, nDeviceID))
			throw lErr;

		// need to leave a little headroom or later calls will fail
		m_szTotalNumBlocks = prop.totalGlobalMem / BLOCKSIZE - 16;

		if (lErr = cudaStreamSynchronize(0))
			throw lErr;

		if (lErr = allocate_small_mem())
			throw lErr;

		size_t szFree;
		size_t szTotal;

		if (lErr = cudaMemGetInfo(&szFree, &szTotal))
			throw lErr;

		m_szTotalNumBlocks = MIN(m_szTotalNumBlocks, szFree / BLOCKSIZE - 16);

		if (fPctToAllocate > 0 && fPctToAllocate < 1)
			m_szTotalNumBlocks = (size_t)(m_szTotalNumBlocks * fPctToAllocate);

		if (m_szTotalNumBlocks > 48200)
			m_szTotalNumBlocks -= 100;

		do
		{
			m_szTotalNumBlocks -= 16; // magic number 16MB

			if (m_szTotalNumBlocks <= 0)
				throw ERROR_OUTOFMEMORY;

			size_t szBytes = m_szTotalNumBlocks * BLOCKSIZE;
			lErr = cudaMalloc((void**)&m_pTestMem, szBytes);
		} while (lErr != cudaSuccess);

		*pszTotalNumBlocks = m_szTotalNumBlocks;
		*pfMemAllocated = T(m_szTotalNumBlocks / 1000.0);
		*pfMemStartAddress = T((long long)m_pTestMem);
		*pfMemBlockSize = T((long long)BLOCKSIZE);
	}
	catch (LONG lErrEx)
	{
		CleanUp();
		return lErrEx;
	}

	return 0;
}

template long memtestHandle<double>::Initialize(Memory<double>* pMem, double dfPctToAllocate, size_t* pszTotalNumBlocks, double* pfMemAllocated, double* pfMemStartAddr, double* pfMemBlockSize);
template long memtestHandle<float>::Initialize(Memory<float>* pMem, float fPctToAllocate, size_t* pszTotalNumBlocks, float* pfMemAllocated, float* pfMemStartAddr, float* pfMemBlockSize);


template <class T>
long memtestHandle<T>::CleanUp()
{
	m_pTestingStartAddress = NULL;
	m_szAddressesTested = 0;

	if (m_pTestMem != NULL)
	{
		cudaFree(m_pTestMem);
		m_pTestMem = NULL;
	}

	m_szTotalNumBlocks = 0;

	if (m_rgerr_addr != NULL)
	{
		cudaFree(m_rgerr_addr);
		m_rgerr_addr = NULL;
	}

	if (m_rgerr_expect != NULL)
	{
		cudaFree(m_rgerr_expect);
		m_rgerr_expect = NULL;
	}

	if (m_rgerr_current != NULL)
	{
		cudaFree(m_rgerr_current);
		m_rgerr_current = NULL;
	}

	if (m_rgerr_second_read != NULL)
	{
		cudaFree(m_rgerr_second_read);
		m_rgerr_second_read = NULL;
	}

	if (m_perr_count != NULL)
	{
		cudaFree(m_perr_count);
		m_perr_count = NULL;
	}

	if (m_rgerr_addr_host1 != NULL)
	{
		cudaFreeHost(m_rgerr_addr_host1);
		m_rgerr_addr_host1 = NULL;
	}

	if (m_rgerr_addr_host2 != NULL)
	{
		cudaFreeHost(m_rgerr_addr_host2);
		m_rgerr_addr_host2 = NULL;
	}

	if (m_rgerr_expect_host != NULL)
	{
		cudaFreeHost(m_rgerr_expect_host);
		m_rgerr_expect_host = NULL;
	}

	if (m_rgerr_current_host != NULL)
	{
		cudaFreeHost(m_rgerr_current_host);
		m_rgerr_current_host = NULL;
	}

	if (m_rgerr_second_read_host != NULL)
	{
		cudaFreeHost(m_rgerr_second_read_host);
		m_rgerr_second_read_host = NULL;
	}

	m_rgerr_addr_host[0] = NULL;
	m_rgerr_addr_host[1] = NULL;
	m_rgerr_count_host[0] = 0;
	m_rgerr_count_host[1] = 0;

	return 0;
}

template long memtestHandle<double>::CleanUp();
template long memtestHandle<float>::CleanUp();


template <class T>
long memtestHandle<T>::Run(MEMTEST_TYPE testType, size_t szStartOffset, size_t szCount, long* plCount, T** ppfData, bool bVerbose, bool bWrite, bool bReadWrite, bool bRead)
{
	LONG lErr = 0;

	reset_small_mem();

	switch (testType)
	{
		case MOV_INV_8:
			lErr = run_move_inv_8_test(szStartOffset, szCount, bWrite, bReadWrite, bRead);
			break;

		default:
			lErr = ERROR_PARAM_OUT_OF_RANGE;
			break;
	}

	if (lErr = load_error_data(plCount, ppfData, bVerbose))
		return lErr;

	return lErr;
}

template long memtestHandle<double>::Run(MEMTEST_TYPE testType, size_t szStartOffset, size_t szCount, long* plCount, double** ppfData, bool bVerbose, bool bWrite, bool bReadWrite, bool bRead);
template long memtestHandle<float>::Run(MEMTEST_TYPE testType, size_t szStartOffset, size_t szCount, long* plCount, float** ppfData, bool bVerbose, bool bWrite, bool bReadWrite, bool bRead);


template <class T>
long memtestHandle<T>::alloc_mem(void** pp, size_t sz)
{
	LONG lErr;

#ifdef USE_PINNED_HOST_MEM
	if (lErr = cudaMallocHost(pp, sz))
		return lErr;
#else
	*pp = malloc(sz);
	if (*pp == NULL)
		return ERROR_MEMORY_OUT;
#endif
	return 0;
}

template long memtestHandle<double>::alloc_mem(void** pp, size_t sz);
template long memtestHandle<float>::alloc_mem(void** pp, size_t sz);


template <class T>
long memtestHandle<T>::free_mem(void* p)
{
	if (p != NULL)
	{
#ifdef USE_PINNED_HOST_MEM
		cudaFreeHost(p);
#else
		free(p);
#endif
	}

	return 0;
}

template long memtestHandle<double>::free_mem(void *p);
template long memtestHandle<float>::free_mem(void *p);


template <class T>
long memtestHandle<T>::load_error_data(long* plCount, T** ppfData, bool bVerbose)
{
	LONG lErr;
	T* pfDataFinal = NULL;
	T* pfData = NULL;
	int nIdx = 0;

	try
	{	
		int nMax = m_rgerr_count_host[0] + m_rgerr_count_host[1];

		if (bVerbose && nMax > 0)
		{
			if (lErr = alloc_mem((void**)&pfData, nMax * 2 * sizeof(T)))
				throw lErr;
		}

		unsigned int nIdx1 = 0;
		unsigned int nIdx2 = 0;

		while (nIdx1 < m_rgerr_count_host[0] && nIdx2 < m_rgerr_count_host[1])
		{
			if (nIdx1 < m_rgerr_count_host[0] && nIdx2 < m_rgerr_count_host[1])
			{
				if (m_rgerr_addr_host1[nIdx1] < m_rgerr_addr_host2[nIdx2])
				{
					if (pfData != NULL)
						pfData[nIdx] = T(m_rgerr_addr_host1[nIdx1]);
					nIdx1++;
				}
				else if (m_rgerr_addr_host2[nIdx2] < m_rgerr_addr_host1[nIdx1])
				{
					if (pfData != NULL)
						pfData[nIdx] = T(m_rgerr_addr_host2[nIdx2]);
					nIdx2++;
				}
				else
				{
					if (pfData != NULL)
						pfData[nIdx] = T(m_rgerr_addr_host1[nIdx1]);
					nIdx1++;
					nIdx2++;
				}
			}
			else if (nIdx1 < m_rgerr_count_host[0])
			{
				if (pfData != NULL)	
					pfData[nIdx] = T(m_rgerr_addr_host1[nIdx1]);
				nIdx1++;
			}
			else if (nIdx2 < m_rgerr_count_host[1])
			{
				if (pfData != NULL)
					pfData[nIdx] = T(m_rgerr_addr_host2[nIdx2]);
				nIdx2++;
			}

			nIdx++;
		}

		int nCount = 3;

		if (pfData != NULL)
			nCount += nIdx;

		if (lErr = alloc_mem((void**)&pfDataFinal, nCount * sizeof(T)))
			throw lErr;

		pfDataFinal[0] = T((size_t)m_pTestingStartAddress);
		pfDataFinal[1] = T(m_szAddressesTested);
		pfDataFinal[2] = T(nIdx);

		if (pfData != NULL)
		{
			if (lErr = cudaMemcpy(&pfDataFinal[3], pfData, nIdx * sizeof(T), cudaMemcpyHostToHost))
				throw lErr;

			free_mem(pfData);
			pfData = NULL;
		}

		*plCount = nCount;
		*ppfData = pfDataFinal;
	}
	catch (LONG lErr2)
	{
		if (pfData != NULL)
			free_mem(pfData);

		if (pfDataFinal != NULL)
			free_mem(pfData);

		return lErr2;
	}

	return 0;
}

template long memtestHandle<double>::load_error_data(long* plCount, double** ppfData, bool bVerbose);
template long memtestHandle<float>::load_error_data(long* plCount, float** ppfData, bool bVerbose);


template <class T>
long memtestHandle<T>::run_move_inv_8_test(size_t szStartOffset, size_t szCount, bool bWrite, bool bReadWrite, bool bRead)
{
	long lErr;
	unsigned int p0 = 0x80;
	unsigned int p1 = p0 | (p0 << 8) | (p0 << 16) | (p0 << 24);
	unsigned int p2 = ~p1;

	if (lErr = move_inv_test(szStartOffset, szCount, p1, p2, bWrite, bReadWrite, bRead))
		return lErr;

	return 0;
}

template long memtestHandle<double>::run_move_inv_8_test(size_t szStartOffset, size_t szCount, bool bWrite, bool bReadWrite, bool bRead);
template long memtestHandle<float>::run_move_inv_8_test(size_t szStartOffset, size_t szCount, bool bWrite, bool bReadWrite, bool bRead);


template<typename T>
__global__ void kernel_move_inv_write(unsigned char* _ptr, unsigned char* end_ptr, unsigned int pattern)
{
	const int SEGMENT = BLOCKSIZE / GRIDSIZE;
	unsigned int* ptr = (unsigned int*)(_ptr + blockIdx.x * SEGMENT);

	if (ptr >= (unsigned int*)end_ptr)
		return;

	const int nCount = SEGMENT / sizeof(unsigned int);
	for (int i = 0; i < nCount; i++)
	{
		ptr[i] = pattern;
	}

	return;
}

template<typename T>
__global__ void kernel_move_inv_readwrite(unsigned char* _ptr, unsigned char* end_ptr, unsigned int p1, unsigned int p2, unsigned int* err, size_t* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
	const int SEGMENT = BLOCKSIZE / GRIDSIZE;
	unsigned int* ptr = (unsigned int*)(_ptr + blockIdx.x * SEGMENT);

	if (ptr >= (unsigned int*)end_ptr)
		return;

	const int nCount = SEGMENT / sizeof(unsigned int);
	for (int i = 0; i < nCount; i++)
	{
		if (ptr[i] != p1)
		{
			unsigned int idx = atomicAdd(err, 1);
			idx = idx % MAX_ERR_RECORD_COUNT;
			err_addr[idx] = (size_t)&ptr[i];
			err_expect[idx] = (unsigned long)p1;
			err_current[idx] = (unsigned long)ptr[i];
			err_second_read[idx] = (unsigned long)*&ptr[i];
		}

		ptr[i] = p2;
	}

	return;
}

template<typename T>
__global__ void kernel_move_inv_read(unsigned char* _ptr, unsigned char* end_ptr, unsigned int pattern, unsigned int* err, size_t* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
	const int SEGMENT = BLOCKSIZE / GRIDSIZE;
	unsigned int* ptr = (unsigned int*)(_ptr + blockIdx.x * SEGMENT);

	if (ptr >= (unsigned int*)end_ptr)
		return;

	const int nCount = SEGMENT / sizeof(unsigned int);
	for (int i = 0; i < nCount; i++)
	{
		if (ptr[i] != pattern)
		{
			unsigned int idx = atomicAdd(err, 1);
			idx = idx % MAX_ERR_RECORD_COUNT;
			err_addr[idx] = (size_t)&ptr[i];
			err_expect[idx] = (unsigned long)pattern;
			err_current[idx] = (unsigned long)ptr[i];
			err_second_read[idx] = (unsigned long)*&ptr[i];
		}
	}

	return;
}

template <class T>
long memtestHandle<T>::move_inv_test(size_t szStartOffset, size_t szCount, unsigned int p1, unsigned int p2, bool bWrite, bool bReadWrite, bool bRead)
{
	long lErr;
	unsigned int errVal = 0;
	unsigned int err = 0;
	unsigned char* ptr = m_pTestMem;
	unsigned char* end_ptr = ptr + m_szTotalNumBlocks * BLOCKSIZE;
	dim3 grid;

	grid.x = GRIDSIZE;

	m_pTestingStartAddress = ptr + (szStartOffset * BLOCKSIZE);
	m_szAddressesTested = szCount * BLOCKSIZE;

	if (bWrite)
	{
		for (size_t i = szStartOffset; i < szStartOffset + szCount && i < m_szTotalNumBlocks; i += GRIDSIZE)
		{
			size_t szOffset = i * BLOCKSIZE;

			kernel_move_inv_write<T> << <grid, 1 >> > (ptr + szOffset, end_ptr, p1);
			if (lErr = cudaStreamSynchronize(0))
				return lErr;

			if (lErr = cudaGetLastError())
				return lErr;
		}
	}

	if (bReadWrite)
	{
		for (size_t i = szStartOffset; i < szStartOffset + szCount && i < m_szTotalNumBlocks; i += GRIDSIZE)
		{
			size_t szOffset = i * BLOCKSIZE;

			kernel_move_inv_readwrite<T> << <grid, 1 >> > (ptr + szOffset, end_ptr, p1, p2, m_perr_count, m_rgerr_addr, m_rgerr_expect, m_rgerr_current, m_rgerr_second_read);
			if (lErr = cudaStreamSynchronize(0))
				return lErr;

			if (lErr = cudaGetLastError())
				return lErr;

			if (lErr = error_checking(0, i, &errVal))
				return lErr;

			err += errVal;
		}
	}

	if (bRead)
	{
		for (size_t i = szStartOffset; i < szStartOffset + szCount && i < m_szTotalNumBlocks; i += GRIDSIZE)
		{
			size_t szOffset = i * BLOCKSIZE;

			kernel_move_inv_read<T> << <grid, 1 >> > (ptr + szOffset, end_ptr, p2, m_perr_count, m_rgerr_addr, m_rgerr_expect, m_rgerr_current, m_rgerr_second_read);
			if (lErr = cudaStreamSynchronize(0))
				return lErr;

			if (lErr = cudaGetLastError())
				return lErr;

			if (lErr = error_checking(1, i, &errVal))
				return lErr;

			err += errVal;
		}
	}

	return 0;
}

template long memtestHandle<double>::move_inv_test(size_t szStartOffset, size_t szCount, unsigned int p1, unsigned int p2, bool bWrite, bool bReadWrite, bool bRead);
template long memtestHandle<float>::move_inv_test(size_t szStartOffset, size_t szCount, unsigned int p1, unsigned int p2, bool bWrite, bool bReadWrite, bool bRead);


template <class T>
long memtestHandle<T>::error_checking(int nHostIdx, size_t blockidx, unsigned int* pErr)
{
	long lErr;
	unsigned int err = 0;

	if (lErr = cudaMemcpy((void*)&err, (void*)m_perr_count, sizeof(unsigned int), cudaMemcpyDeviceToHost))
		return lErr;

	m_rgerr_count_host[nHostIdx] = err;

	if (err > 0)
	{
		if (lErr = cudaMemcpy((void*)m_rgerr_addr_host[nHostIdx], (void*)m_rgerr_addr, sizeof(size_t) * err, cudaMemcpyDeviceToHost))
			return lErr;

		//if (lErr = cudaMemcpy((void*)&m_rgerr_expect_host[0], (void*)m_rgerr_expect, sizeof(unsigned long) * err, cudaMemcpyDeviceToHost))
		//	return lErr;

		//if (lErr = cudaMemcpy((void*)&m_rgerr_current_host[0], (void*)m_rgerr_current, sizeof(unsigned long) * err, cudaMemcpyDeviceToHost))
		//	return lErr;

		//if (lErr = cudaMemcpy((void*)&m_rgerr_second_read_host[0], (void*)m_rgerr_second_read, sizeof(unsigned long) * err, cudaMemcpyDeviceToHost))
		//	return lErr;
	}

	*pErr = err;

	return 0;
}

template long memtestHandle<double>::error_checking(int nHostIdx, size_t blockidx, unsigned int* pErr);
template long memtestHandle<float>::error_checking(int nHostIdx, size_t blockidx, unsigned int* pErr);

// end