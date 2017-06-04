//=============================================================================
//	FILE:	tsne_g.cu
//
//	DESC:	This file implements TSNE gradient algorithm.
//
//  Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
//  All rights reserved.
//   
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//  3. All advertising materials mentioning features or use of this software
//     must display the following acknowledgement:
//     This product includes software developed by the Delft University of Technology.
//  4. Neither the name of the Delft University of Technology nor the names of 
//     its contributors may be used to endorse or promote products derived from 
//     this software without specific prior written permission.
// 
//  THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
//  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
//  EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
//  OF SUCH DAMAGE.
//=============================================================================

#include "util.h"
#include "memory.h"
#include "tsne_g.h"
#include <algorithm>
#include <vector>
#include <queue>
#include <stack>
#include <limits>
#include <cmath>
#include <cstdlib>


//=============================================================================
//	Local Classes
//=============================================================================

template <class T>
class StackItem
{
	T* m_pData;
	StackItem<T>* m_pNext;
	StackItem<T>* m_pPrev;

public:
	StackItem(T* pData)
	{
		m_pNext = NULL;
		m_pPrev = NULL;
		m_pData = pData;
	}

	StackItem<T>* Next()
	{
		return m_pNext;
	}

	StackItem<T>* Prev()
	{
		return m_pPrev;
	}

	T* Data()
	{
		return m_pData;
	}

	void SetNext(StackItem<T>* pNext)
	{
		m_pNext = pNext;
	}

	void SetPrev(StackItem<T>* pPrev)
	{
		m_pPrev = pPrev;
	}
};

template <class T>
class Stack
{
	StackItem<T>* m_pTop;
	int m_nCount;

public:
	Stack()
	{
		m_pTop = NULL;
		m_nCount = 0;
	}

	~Stack()
	{
		T* pItem = Pop();

		while (pItem != NULL)
		{
			pItem = Pop();
		}
	}

	int Count()
	{
		return m_nCount;
	}

	void Push(T* pData)
	{
		StackItem<T>* pNewItem = new StackItem<T>(pData);
		
		if (m_pTop == NULL)
		{
			m_pTop = pNewItem;
			m_nCount = 1;
		}
		else
		{
			m_pTop->SetNext(pNewItem);
			pNewItem->SetPrev(m_pTop);
			m_pTop = pNewItem;
			m_nCount++;
		}
	}

	T* Pop()
	{
		T* pData = NULL;

		if (m_pTop != NULL)
		{
			StackItem<T>* pItem = m_pTop;
			pData = pItem->Data();			
			m_pTop = pItem->Prev();
			m_nCount--;

			if (m_pTop != NULL)
				m_pTop->SetNext(NULL);

			delete pItem;
		}

		return pData;
	}

	T* Peek()
	{
		if (m_pTop == NULL)
			return NULL;

		return m_pTop->Data();
	}
};



template <class T>
class Cell
{
	unsigned int m_nDimension;
	T* m_pfCorner;
	T* m_pfWidth;

public:
	Cell(unsigned int dim, T* pfCorner = NULL, T* pfWidth = NULL)
	{
		m_nDimension = dim;
		m_pfCorner = (T*)malloc(dim * sizeof(T));
		m_pfWidth = (T*)malloc(dim * sizeof(T));

		if (pfCorner != NULL && pfWidth != NULL)
		{
			for (unsigned int d=0; d<dim; d++)
			{
				setCorner(d, pfCorner[d]);
				setWidth(d, pfWidth[d]);
			}
		}
	}

	~Cell()
	{
		if (m_pfCorner != NULL)
			free(m_pfCorner);

		if (m_pfWidth != NULL)
			free(m_pfWidth);
	}

	T getCorner(unsigned int d)
	{
		return m_pfCorner[d];
	}

	void setCorner(unsigned int d, T fVal)
	{
		m_pfCorner[d] = fVal;
	}

	T getWidth(unsigned int d)
	{
		return m_pfWidth[d];
	}

	void setWidth(unsigned int d, T fVal)
	{
		m_pfWidth[d] = fVal;
	}

	bool containsPoint(T rgPoint[])
	{
		for (unsigned int d=0; d<m_nDimension; d++)
		{
			if (m_pfCorner[d] - m_pfWidth[d] > rgPoint[d])
				return false;

			if (m_pfCorner[d] + m_pfWidth[d] < rgPoint[d])
				return false;
		}

		return true;
	}
};

template <class T>
class SpTree
{
	// Fixed constants
	static const unsigned int QT_NODE_CAPACITY = 1;

	// A buffer we use when doing force computations
	T* m_pfBuff;

	// Properties of this node in the tree
	SpTree<T>* m_parent;
	unsigned int m_nDimension;
	bool m_bIsLeaf;
	unsigned int m_nSize;
	unsigned int m_nCumSize;

	// Axis-aligned bounding box stored as a center with half-dimensions to 
	// represent the boundaries of this quad tree.
	Cell<T>* m_boundary;

	// Indices in this space-partitioning tree node, corresponding
	// center-of-mass, and list of all children.
	T* m_pfData;
	T* m_pfCenterOfMass;
	unsigned int m_rgIndex[QT_NODE_CAPACITY];

	// Children
	SpTree<T>** m_rgpChildren;
	unsigned int m_nNoChildren;

	T fmax(T a, T b)
	{
		if (a > b)
			return a;
		else
			return b;
	}

	void init(SpTree<T>* parent, unsigned int D, T* data, T* corner, T* width);

	void fill(unsigned int N)
	{
		for (unsigned int i=0; i<N; i++)
		{
			insert(i);
		}
	}

	bool insert(unsigned int new_index);
	bool subdivide();

public:
	SpTree(unsigned int D, T* data, unsigned int N);
	
	SpTree(SpTree<T>* parent, unsigned int D, T* data, T* corner, T* width)
	{
		init(parent, D, data, corner, width);
	}

	~SpTree();

	void computeNonEdgeForces(unsigned int nPointIndex, T fTheta, T* neg_f, T* pSumQ);
	void computeEdgeForces(T* rowP, T* colP, T* valP, unsigned int N, T* pos_f);
};


template <class T>
SpTree<T>::SpTree(unsigned int D, T* data, unsigned int N)
{
	// Compute mean, width and height of the current map (boundaries of SpTree)
	int nD = 0;
	T fMax = (sizeof(T) == 4) ? FLT_MAX : DBL_MAX;
	T* mean_Y = (T*)calloc(D, sizeof(T));
	T* min_Y = (T*)malloc(D * sizeof(T));
	T* max_Y = (T*)malloc(D * sizeof(T));

	for (unsigned int d=0; d<D; d++)
	{
		min_Y[d] = fMax;
		max_Y[d] = -fMax;
	}

	for (unsigned int n = 0; n<N; n++)
	{
		for (unsigned int d=0; d<D; d++)
		{
			mean_Y[d] += data[n * D + d];

			if (data[nD + d] < min_Y[d])
				min_Y[d] = data[nD + d];

			if (data[nD + d] > max_Y[d])
				max_Y[d] = data[nD + d];
		}

		nD += D;
	}

	for (unsigned int d=0; d<D; d++)
	{
		mean_Y[d] /= T(N);
	}

	// Construct the SpTree
	T* pfWidth = (T*)malloc(D * sizeof(T));

	for (unsigned int d=0; d<D; d++)
	{
		pfWidth[d] = fmax(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + T(1e-5);
	}

	init(NULL, D, data, mean_Y, pfWidth);
	fill(N);

	// Clean-up memory
	free(mean_Y);
	free(max_Y);
	free(min_Y);
	free(pfWidth);
}

template <class T>
SpTree<T>::~SpTree()
{
	for (unsigned int i=0; i<m_nNoChildren; i++)
	{
		if (m_rgpChildren[i] != NULL)
			delete m_rgpChildren[i];
	}

	if (m_rgpChildren != NULL)
		free(m_rgpChildren);

	if (m_pfCenterOfMass != NULL)
		free(m_pfCenterOfMass);

	if (m_pfBuff != NULL)
		free(m_pfBuff);

	if (m_boundary != NULL)
		delete m_boundary;
}

template <class T>
void SpTree<T>::init(SpTree<T>* parent, unsigned int D, T* pfData, T* pfCorner, T* pfWidth)
{
	m_parent = parent;
	m_nDimension = D;
	m_nNoChildren = 2;

	for (unsigned int d=1; d<D; d++)
	{
		m_nNoChildren *= 2;
	}

	m_pfData = pfData;
	m_bIsLeaf = true;
	m_nSize = 0;
	m_nCumSize = 0;

	m_boundary = new Cell<T>(m_nDimension);

	for (unsigned int d = 0; d<D; d++)
	{
		m_boundary->setCorner(d, pfCorner[d]);
		m_boundary->setWidth(d, pfWidth[d]);
	}

	m_rgpChildren = (SpTree<T>**)malloc(m_nNoChildren * sizeof(SpTree<T>*));

	for (unsigned int i=0; i<m_nNoChildren; i++)
	{
		m_rgpChildren[i] = NULL;
	}

	m_pfCenterOfMass = (T*)calloc(D, sizeof(T));
	m_pfBuff = (T*)malloc(D * sizeof(T));
}

// Insert a point into the SpTree
//template <class T>
//bool SpTree<T>::insert(unsigned int new_index)
//{
//	// Ignore objects which do not belong in this quad tree
//	T* pfPoint = m_pfData + new_index * m_nDimension;
//	if (!m_boundary->containsPoint(pfPoint))
//		return false;
//
//	// Online update of cumulative size and center-of-mass
//	m_nCumSize++;
//	T fMult1 = T(m_nCumSize - 1)/T(m_nCumSize);
//	T fMult2 = T(1.0) / T(m_nCumSize);
//
//	for (unsigned int d=0; d<m_nDimension; d++)
//	{
//		m_pfCenterOfMass[d] *= fMult1;
//		m_pfCenterOfMass[d] += fMult2 * pfPoint[d];
//	}
//
//	// If there is space in this quad tree and it is a leaf, add the object here.
//	if (m_bIsLeaf && m_nSize < QT_NODE_CAPACITY)
//	{
//		m_rgIndex[m_nSize] = new_index;
//		m_nSize++;
//		return true;
//	}
//
//	// Don't add duplicates for now (this is not very nice)
//	for (unsigned int n = 0; n<m_nSize; n++)
//	{
//		bool bDuplicate = true;
//		for (unsigned int d=0; d<m_nDimension; d++)
//		{
//			if (pfPoint[d] != m_pfData[m_rgIndex[n] * m_nDimension + d])
//			{
//				bDuplicate = false;
//				break;
//			}
//		}
//			
//		if (bDuplicate)
//			return true;
//	}
//
//	// Otherwise, we need to subdivide the current cell
//	if (m_bIsLeaf)
//		subdivide();
//
//	// Find out where the point can be inserted.
//	for (unsigned int i=0; i<m_nNoChildren; i++)
//	{
//		if (m_rgpChildren[i]->insert(new_index))
//			return true;
//	}
//
//	// Otherwise, the point cannot be inserted (this should never happen)
//	return false;
//}

// Non recursive version (recursive version causes stack overflow)
template <class T>
bool SpTree<T>::insert(unsigned int new_index)
{
	Stack<SpTree<T>> rgStack;

	rgStack.Push(this);

	while (rgStack.Count() > 0)
	{
		SpTree<T>* pTree = rgStack.Pop();

		// Ignore objects which do not belong in this quad tree
		T* pfPoint = pTree->m_pfData + new_index * m_nDimension;
		if (pTree->m_boundary->containsPoint(pfPoint))
		{
			// Online update of cumulative size and center-of-mass
			pTree->m_nCumSize++;
			T fMult1 = T(pTree->m_nCumSize - 1) / T(pTree->m_nCumSize);
			T fMult2 = T(1.0) / T(pTree->m_nCumSize);

			for (unsigned int d = 0; d < pTree->m_nDimension; d++)
			{
				pTree->m_pfCenterOfMass[d] *= fMult1;
				pTree->m_pfCenterOfMass[d] += fMult2 * pfPoint[d];
			}

			// If there is space in this quad tree and it is a leaf, add the object here.
			if (pTree->m_bIsLeaf && pTree->m_nSize < QT_NODE_CAPACITY)
			{
				pTree->m_rgIndex[m_nSize] = new_index;
				pTree->m_nSize++;
				return true;
			}
			else
			{
				bool bDuplicate = false;

				// Don't add duplicates for now (this is not very nice)
				for (unsigned int n = 0; n < pTree->m_nSize; n++)
				{
					bDuplicate = true;
					for (unsigned int d = 0; d < pTree->m_nDimension; d++)
					{
						if (pfPoint[d] != pTree->m_pfData[pTree->m_rgIndex[n] * pTree->m_nDimension + d])
						{
							bDuplicate = false;
							break;
						}
					}

					if (bDuplicate)
						break;
				}

				if (!bDuplicate)
				{
					bool bSubdivided = true;

					// Otherwise, we need to subdivide the current cell
					if (pTree->m_bIsLeaf)
						bSubdivided = pTree->subdivide();	// returns false on nan

					// Find out where the point can be inserted.
					if (bSubdivided)
					{
						for (int i = pTree->m_nNoChildren - 1; i >= 0; i--)
						{
							rgStack.Push(pTree->m_rgpChildren[i]);
						}
					}
				}
			}
		}
	}

	// Otherwise, the point cannot be inserted (this should never happen)
	return false;
}

// Create four children which fully divide this cell into four quads of equal area.
template <class T>
bool SpTree<T>::subdivide()
{
	// Create new children
	T* pfNewCorner = (T*)malloc(m_nDimension * sizeof(T));
	T* pfNewWidth = (T*)malloc(m_nDimension * sizeof(T));

	for (unsigned int i=0; i<m_nNoChildren; i++)
	{
		unsigned int div = 1;

		for (unsigned int d=0; d<m_nDimension; d++)
		{
			T fWid = m_boundary->getWidth(d);
			if (fWid != fWid) // check for nan
			{
				free(pfNewCorner);
				free(pfNewWidth);
				return false;
			}

			pfNewWidth[d] = T(0.5) * fWid;
			if (pfNewWidth[d] != pfNewWidth[d]) // check for nan
			{
				free(pfNewCorner);
				free(pfNewWidth);
				return false;
			}

			T fCorner = m_boundary->getCorner(d);
			if (fCorner != fCorner)	// check for nan
			{
				free(pfNewCorner);
				free(pfNewWidth);
				return false;
			}

			if ((i / div) % 2 == 1)
				pfNewCorner[d] = fCorner - pfNewWidth[d];
			else
				pfNewCorner[d] = fCorner + pfNewWidth[d];

			div *= 2;
		}

		m_rgpChildren[i] = new SpTree<T>(this, m_nDimension, m_pfData, pfNewCorner, pfNewWidth);
	}

	free(pfNewCorner);
	free(pfNewWidth);

	// Move existing points to correct children
	for (unsigned int i=0; i<m_nSize; i++)
	{
		bool bSuccess = false;

		for (unsigned int j=0; j<m_nNoChildren; j++)
		{
			if (!bSuccess)
				bSuccess = m_rgpChildren[i]->insert(m_rgIndex[i]);
		}

		m_rgIndex[i] = (unsigned int)-1;
	}

	// Empty parent node
	m_nSize = 0;
	m_bIsLeaf = false;

	return true;
}


template <class T>
void SpTree<T>::computeNonEdgeForces(unsigned int nPointIndex, T fTheta, T* neg_f, T* pSumQ)
{
	// Make sure that we spend no time on empty nodes or self-interactions
	if (m_nCumSize == 0 || (m_bIsLeaf && m_nSize == 1 && m_rgIndex[0] == nPointIndex))
		return;

	// Compute distance between point and center-of-mass
	T fD = T(0);
	unsigned int nIdx = nPointIndex * m_nDimension;

	for (unsigned int d=0; d<m_nDimension; d++)
	{
		m_pfBuff[d] = m_pfData[nIdx + d] - m_pfCenterOfMass[d];
		fD += m_pfBuff[d] * m_pfBuff[d];
	}

	// Check whether we can use this node as a 'summary'
	T fMaxWidth = T(0);
	T fCurWidth;

	for (unsigned int d=0; d<m_nDimension; d++)
	{
		fCurWidth = m_boundary->getWidth(d);
		
		if (fCurWidth > fMaxWidth)
			fMaxWidth = fCurWidth;
	}

	if (m_bIsLeaf || fMaxWidth / sqrt(fD) < fTheta)
	{
		// Compute and add t-SNE force between poitn and current node.
		fD = T(1) / (T(1) + fD);
		T fMult = m_nCumSize * fD;
		*pSumQ += fMult;
		fMult *= fD;

		for (unsigned int d=0; d<m_nDimension; d++)
		{
			neg_f[d] += fMult * m_pfBuff[d];
		}
	}
	else
	{
		// Recursively apply Barnes-Hut to children
		for (unsigned int i = 0; i<m_nNoChildren; i++)
		{
			m_rgpChildren[i]->computeNonEdgeForces(nPointIndex, fTheta, neg_f, pSumQ);
		}
	}
}


template <class T>
void SpTree<T>::computeEdgeForces(T* rowP, T* colP, T* valP, unsigned int N, T* pos_f)
{
	// Loop over all edges in the graph.
	unsigned int nIdx1 = 0;
	unsigned int nIdx2 = 0;
	T fD;

	for (unsigned int n=0; n<N; n++)
	{
		unsigned int nRow1 = (unsigned int)rowP[n];
		unsigned int nRow2 = (unsigned int)rowP[n + 1];

		for (unsigned int i=nRow1; i<nRow2; i++)
		{
			// Compute pairwise distance and Q-value
			fD = T(1);
			nIdx2 = (unsigned int)colP[i] * m_nDimension;

			for (unsigned int d=0; d<m_nDimension; d++)
			{
				m_pfBuff[d] = m_pfData[nIdx1 + d] - m_pfData[nIdx2 + d];
				fD += m_pfBuff[d] * m_pfBuff[d];
			}

			fD = valP[i] / fD;

			// Sum positive force
			for (unsigned int d=0; d<m_nDimension; d++)
			{
				pos_f[nIdx1 + d] += fD * m_pfBuff[d];
			}
		}

		nIdx1 += m_nDimension;
	}
}


//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long tsnegHandle<T>::Initialize(Memory<T>* pMem, Math<T>* pMath)
{
	LONG lErr;
	int nDeviceID;

	if (lErr = cudaGetDevice(&nDeviceID))
		return lErr;

	m_pMem = pMem;
	m_pMath = pMath;
	m_nCurrentIteration = 0;

	try
	{
		//------------------------------------------------
		//	Get the device memory pointers.
		//------------------------------------------------

		if (m_hY > 0)
		{
			if ((m_pY_on_host = m_pMem->GetMemoryToHost(m_hY)) == NULL)
				throw ERROR_MEMORY_OUT;
		}

		if (m_hdC > 0)
		{
			if ((m_pdC_on_host = m_pMem->GetMemoryToHost(m_hdC)) == NULL)
				throw ERROR_MEMORY_OUT;
		}

		if ((m_pValP_on_host = m_pMem->GetMemoryToHost(m_hValP)) == NULL)
			throw ERROR_MEMORY_OUT;

		if ((m_pRowP_on_host = m_pMem->GetHostBuffer(m_hRowP)->Data()) == NULL)
			throw ERROR_MEMORY_OUT;

		if ((m_pColP_on_host = m_pMem->GetHostBuffer(m_hColP)->Data()) == NULL)
			throw ERROR_MEMORY_OUT;

		if (lErr = m_pMem->AllocHost(m_nD, &m_pBuff_on_host, NULL, false))
			throw lErr;

		if (lErr = m_pMem->AllocHost(m_nN * m_nD, &m_pPosF_on_host, NULL, false))
			throw lErr;

		if (lErr = m_pMem->AllocHost(m_nN * m_nD, &m_pNegF_on_host, NULL, false))
			throw lErr;
	}
	catch (LONG lErrEx)
	{
		CleanUp();
		return lErrEx;
	}

	return 0;
}

template long tsnegHandle<double>::Initialize(Memory<double>* pMem, Math<double>* pMath);
template long tsnegHandle<float>::Initialize(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long tsnegHandle<T>::CleanUp()
{
	if (m_pY_on_host != NULL)
	{
		m_pMem->FreeHost(m_pY_on_host);
		m_pY_on_host = NULL;
	}

	if (m_pdC_on_host != NULL)
	{
		m_pMem->FreeHost(m_pdC_on_host);
		m_pdC_on_host = NULL;
	}

	if (m_pValP_on_host != NULL)
	{
		m_pMem->FreeHost(m_pValP_on_host);
		m_pValP_on_host = NULL;
	}

	if (m_pBuff_on_host != NULL)
	{
		m_pMem->FreeHost(m_pBuff_on_host);
		m_pBuff_on_host = NULL;
	}

	if (m_pPosF_on_host != NULL)
	{
		m_pMem->FreeHost(m_pPosF_on_host);
		m_pPosF_on_host = NULL;
	}

	if (m_pNegF_on_host != NULL)
	{
		m_pMem->FreeHost(m_pNegF_on_host);
		m_pNegF_on_host = NULL;
	}

	return 0;
}

template long tsnegHandle<double>::CleanUp();
template long tsnegHandle<float>::CleanUp();


template <class T>
long tsnegHandle<T>::ComputeGradient(bool bValPUpdated)
{
	LONG lErr;

	if (lErr = m_pMem->SetMemoryToHost(m_hY, m_pY_on_host))
		return lErr;

	if (bValPUpdated)
	{
		if (lErr = m_pMem->SetMemoryToHost(m_hValP, m_pValP_on_host))
			return lErr;
	}

	if (lErr = computeGradient(m_pRowP_on_host, m_pColP_on_host, m_pValP_on_host, m_pY_on_host, m_nN, m_nD, m_pdC_on_host, m_fTheta))
		return lErr;

	if (lErr = m_pMem->SetMemory(m_hdC, m_pdC_on_host, -1, -1))
		return lErr;

	if (lErr = m_pMem->SetMemory(m_hY, m_pY_on_host, -1, -1))
		return lErr;

	return 0;
}

template long tsnegHandle<double>::ComputeGradient(bool bValPUpdated);
template long tsnegHandle<float>::ComputeGradient(bool bValPUpdated);


template <class T>
long tsnegHandle<T>::computeGradient(T* rowP, T* colP, T* valP, T* Y, unsigned int N, unsigned int D, T* dC, T fTheta)
{
	// Construct space-partitioning tree on current map
	SpTree<T>* pTree = new SpTree<T>(D, Y, N);

	// Compute all terms required for t-SNE gradient.
	T fSumQ = T(0);
	memset(m_pPosF_on_host, 0, sizeof(T) * m_nN * m_nD);
	memset(m_pNegF_on_host, 0, sizeof(T) * m_nN * m_nD);

	pTree->computeEdgeForces(rowP, colP, valP, N, m_pPosF_on_host);

	for (unsigned int n=0; n<m_nN; n++)
	{
		pTree->computeNonEdgeForces(n, m_fTheta, m_pNegF_on_host + n * D, &fSumQ);
	}

	// Compute final t-SNE gradient
	for (unsigned int i=0; i<m_nN * m_nD; i++)
	{
		dC[i] = m_pPosF_on_host[i] - (m_pNegF_on_host[i] / fSumQ);
	}

	delete pTree;
	
	return 0;
}

template <class T>
long tsnegHandle<T>::EvaluateError(T* pfErr)
{
	LONG lErr;

	if (lErr = m_pMem->SetMemoryToHost(m_hY, m_pY_on_host))
		return lErr;

	if (lErr = m_pMem->SetMemoryToHost(m_hValP, m_pValP_on_host))
		return lErr;

	return evaluateError(m_pRowP_on_host, m_pColP_on_host, m_pValP_on_host, m_pY_on_host, m_nN, m_nD, m_fTheta, pfErr);
}

template long tsnegHandle<double>::EvaluateError(double* pfErr);
template long tsnegHandle<float>::EvaluateError(float* pfErr);


template <class T>
long tsnegHandle<T>::evaluateError(T* rowP, T* colP, T* valP, T* Y, unsigned int N, unsigned int D, T fTheta, T* pfErr)
{
	// Get estimate of normalization term
	SpTree<T>* pTree = new SpTree<T>(D, Y, N);
	
	memset(m_pBuff_on_host, 0, sizeof(T) * m_nD);
	T fSumQ = T(0);

	for (unsigned int n=0; n<m_nN; n++)
	{
		pTree->computeNonEdgeForces(n, fTheta, m_pBuff_on_host, &fSumQ);
	}


	// Loop over all edges to compute t-SNE error
	unsigned int nIdx1;
	unsigned int nIdx2;
	T fC = T(0);

	for (unsigned int n=0; n<m_nN; n++)
	{
		nIdx1 = n * m_nD;

		unsigned int nRow1 = (unsigned int)m_pRowP_on_host[n];
		unsigned int nRow2 = (unsigned int)m_pRowP_on_host[n + 1];

		for (unsigned int i=nRow1; i < nRow2; i++)
		{
			T fQ = T(0);
			nIdx2 = (unsigned int)m_pColP_on_host[i] * m_nD;

			for (unsigned int d = 0; d<m_nD; d++)
			{
				m_pBuff_on_host[d] = m_pY_on_host[nIdx1 + d] - m_pY_on_host[nIdx2 + d];
				fQ += m_pBuff_on_host[d] * m_pBuff_on_host[d];
			}

			fQ = (T(1.0) / (T(1.0) + fQ)) / fSumQ;
			fC += m_pValP_on_host[i] * log((m_pValP_on_host[i] + FLT_MIN) / (fQ + FLT_MIN));
		}
	}

	delete pTree;

	*pfErr = fC;

	return 0;
}

template <class T>
long tsnegHandle<T>::symmetrizeMatrix(T* row_P, T* col_P, T* val_P, unsigned int* pnRowCount)
{
	LONG lErr = 0;
	int* rgRowCounts = NULL;
	unsigned int* sym_row_P = NULL;
	unsigned int* sym_col_P = NULL;
	T* sym_val_P = NULL;
	unsigned int* offset = NULL;


	// Get sparse matrix
	try
	{
		// Count number of elements and row counts of symmetrix matrix
		rgRowCounts = (int*)calloc(m_nN, sizeof(int));
		if (rgRowCounts == NULL)
			throw ERROR_MEMORY_OUT;

		for (unsigned int n=0; n<m_nN; n++)
		{
			unsigned int nRow1 = (unsigned int)row_P[n];
			unsigned int nRow2 = (unsigned int)row_P[n+1];

			for (unsigned int i=nRow1; i<nRow2; i++)
			{
				// Check whether element (col_P[i], n) is present
				bool bPresent = false;
				unsigned int nColP = (unsigned int)col_P[i];
				unsigned int nRowA = (unsigned int)row_P[nColP];
				unsigned int nRowB = (unsigned int)row_P[nColP + 1];

				for (unsigned int m = nRowA; m<nRowB; m++)
				{
					if ((unsigned int)col_P[m] == n)
					{
						bPresent = true;
						break;
					}
				}

				rgRowCounts[n]++;

				if (!bPresent)
					rgRowCounts[nColP]++;
			}
		}

		unsigned int nNoElm = 0;

		for (unsigned int n=0; n<m_nN; n++)
		{
			nNoElm += rgRowCounts[n];
		}

		// Allocate memory for symmetrized matrix
		if ((sym_row_P = (unsigned int*)malloc((m_nN + 1) * sizeof(unsigned int))) == NULL)
			throw ERROR_MEMORY_OUT;

		if ((sym_col_P = (unsigned int*)malloc(nNoElm * sizeof(unsigned int))) == NULL)
			throw ERROR_MEMORY_OUT;

		if ((sym_val_P = (T*)malloc(nNoElm * sizeof(T))) == NULL)
			throw ERROR_MEMORY_OUT;


		// Construct new row indices for symmetric matrix
		sym_row_P[0] = 0;

		for (unsigned int n=0; n<m_nN; n++)
		{
			sym_row_P[n + 1] = sym_row_P[n] + (unsigned int)rgRowCounts[n];
		}

		// Fill the result matrix.
		if ((offset = (unsigned int*)calloc(m_nN, sizeof(unsigned int))) == NULL)
			throw ERROR_MEMORY_OUT;

		for (unsigned int n = 0; n<m_nN; n++)
		{
			unsigned int nRow1 = (unsigned int)row_P[n];
			unsigned int nRow2 = (unsigned int)row_P[n+1];

			// Considering element (n, col_P[i])
			for (unsigned int i=nRow1; i<nRow2; i++)
			{
				// Check whether element (col_P[i], n) is present
				bool bPresent = false;
				unsigned int nColP = (unsigned int)col_P[i];
				unsigned int nRowA = (unsigned int)row_P[nColP];
				unsigned int nRowB = (unsigned int)row_P[nColP+1];

				for (unsigned int m = nRowA; m<nRowB; m++)
				{
					if ((unsigned int)col_P[m] == n)
					{
						bPresent = true;

						// make sure we do not add elements twice
						if (n <= nColP)
						{
							sym_col_P[sym_row_P[n]	   + offset[n]]	    = nColP;
							sym_col_P[sym_row_P[nColP] + offset[nColP]] = n;
							sym_val_P[sym_row_P[n]     + offset[n]]     = val_P[i] + val_P[m];
							sym_val_P[sym_row_P[nColP] + offset[nColP]] = val_P[i] + val_P[m];
						}
					}
				}

				// If (col_P[i], n) is not present, there is no addition involved
				if (!bPresent)
				{
					sym_col_P[sym_row_P[n]	   + offset[n]]	    = nColP;
					sym_col_P[sym_row_P[nColP] + offset[nColP]] = n;
					sym_val_P[sym_row_P[n]     + offset[n]]     = val_P[i];
					sym_val_P[sym_row_P[nColP] + offset[nColP]] = val_P[i];
				}

				//  Update offsets
				if (!bPresent || (bPresent && n <= nColP))
				{
					offset[n]++;

					if (nColP != n)
						offset[nColP]++;
				}
			}
		}

		// Divide the result by two
		for (unsigned int i=0; i<nNoElm; i++)
		{
			sym_val_P[i] /= T(2.0);
		}

		// Return symmetrized matrices

		for (unsigned int n=0; n<m_nN+1; n++)
		{
			row_P[n] = T(sym_row_P[n]);
		}

		for (unsigned int n=0; n<nNoElm; n++)
		{
			col_P[n] = T(sym_col_P[n]);
			val_P[n] = sym_val_P[n];
		}

		*pnRowCount = (unsigned int)row_P[m_nN];
	}
	catch (LONG lErrEx)
	{
		lErr = lErrEx;
	}

	if (sym_val_P != NULL)
		free(sym_val_P);

	if (sym_col_P != NULL)
		free(sym_col_P);

	if (sym_row_P != NULL)
		free(sym_row_P);

	if (rgRowCounts != NULL)
		free(rgRowCounts);

	if (offset != NULL)
		free(offset);

	return lErr;
}


template <class T>
long tsnegHandle<T>::SymmetrizeMatrix(unsigned int* pnRowCount)
{
	LONG lErr;

	if (lErr = symmetrizeMatrix(m_pRowP_on_host, m_pColP_on_host, m_pValP_on_host, pnRowCount))
		return lErr;

	return m_pMem->SetMemory(m_hValP, m_pValP_on_host, -1, -1);
}

template long tsnegHandle<double>::SymmetrizeMatrix(unsigned int* pnRowCount);
template long tsnegHandle<float>::SymmetrizeMatrix(unsigned int* pnRowCount);

// end