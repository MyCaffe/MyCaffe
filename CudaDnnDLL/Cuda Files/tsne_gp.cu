//=============================================================================
//	FILE:	tsne_gp.cu
//
//	DESC:	This file implements TSNE gaussian perplexity algorithm.
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
#include "tsne_gp.h"
#include <algorithm>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>


//=============================================================================
//	Local Classes
//=============================================================================

template <class T>
class DataPoint
{
	unsigned int m_nIdx;

public:
	T* m_x;
	unsigned int m_nD;

	DataPoint()
	{
		m_x = NULL;
		m_nD = 1;
		m_nIdx = (unsigned int)-1;
	}

	DataPoint(unsigned int nD, unsigned int nIdx, T* x)
	{
		m_nD = nD;
		m_nIdx = nIdx;
		m_x = (T*)malloc(m_nD * sizeof(T));
		memcpy(m_x, x, m_nD * sizeof(T));
	}

	DataPoint(const DataPoint& d)
	{
		if (this != &d)
		{
			m_nD = d.m_nD;
			m_nIdx = d.m_nIdx;
			m_x = (T*)malloc(m_nD * sizeof(T));
			memcpy(m_x, d.m_x, m_nD * sizeof(T));
		}
	}

	~DataPoint()
	{
		if (m_x != NULL)
			free(m_x);
	}

	DataPoint& operator=(const DataPoint& d)
	{
		if (this != &d)
		{
			m_nD = d.m_nD;
			m_nIdx = d.m_nIdx;
			m_x = (T*)malloc(m_nD * sizeof(T));
			memcpy(m_x, d.m_x, m_nD * sizeof(T));
		}

		return *this;
	}

	unsigned int index() const
	{
		return m_nIdx;
	}

	unsigned int dimensionality() const
	{
		return m_nD;
	}

	T x(unsigned int nIdx) const
	{
		return m_x[nIdx];
	}

	T euclidean_distance(const DataPoint<T>* d) const
	{
		T fDD = 0;
		T fDiff;
		T* x1 = m_x;
		T* x2 = d->m_x;

		for (unsigned int d=0; d<m_nD; d++)
		{	
			fDiff = (x1[d] - x2[d]);
			fDD += fDiff * fDiff;
		}

		return sqrt(fDD);
	}
};

template <class T>
class VpTree
{
private:
	std::vector<DataPoint<T>*> m_items;
	T m_fTau;

	// Single node of VP tree (has a point and radius; left children are closer to point than the radius)
	struct Node
	{
		unsigned int index;		// index of point in node.
		T threshold;			// radius.
		Node* left;				// points closer by than threshold.
		Node* right;			// points farther away than threshold.

		Node()
		{
			index = 0;
			threshold = 0;
			left = NULL;
			right = NULL;
		}

		~Node()
		{
			if (left != NULL)
				delete left;

			if (right != NULL)
				delete right;
		}
	}* m_root;

	// An item on the intermediate result queue.
	struct HeapItem
	{
		unsigned int index;
		T dist;

		HeapItem(unsigned int nIdx, T fDist)
		{
			index = nIdx;
			dist = fDist;
		}

		bool operator<(const HeapItem& item) const
		{
			return dist < item.dist;
		}
	};

	// Distance comparer for use in std::nth_element
	struct DistanceComparator
	{
		const DataPoint<T>* m_item;

		DistanceComparator(const DataPoint<T>* item) : m_item(item) {}

		bool operator()(const DataPoint<T>* a, const DataPoint<T>* b)
		{
			return m_item->euclidean_distance(a) < m_item->euclidean_distance(b);
		}
	};

	// Function that (recursively) fills the tree
	Node* buildFromPoints(unsigned int lower, unsigned int upper)
	{
		// Indicates that we're done here!
		if (upper == lower)
			return NULL;

		// Lower index is center of current node
		Node* node = new Node();
		node->index = lower;

		// if we did not arrive at leaf yet
		if (upper - lower > 1)
		{
			// Choose an arbitrary point and move it to the start.
			unsigned int i = (unsigned int)((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
			std::swap(m_items[lower], m_items[i]);

			// Partition around the median distance
			unsigned int median = (upper + lower) / 2;
			std::nth_element(m_items.begin() + lower + 1,
				             m_items.begin() + median,
							 m_items.begin() + upper,
							 DistanceComparator(m_items[lower]));

			// Threshold of the new node will be the distance to the median
			node->threshold = m_items[lower]->euclidean_distance(m_items[median]);

			// Recursively build tree.
			node->index = lower;
			node->left = buildFromPoints(lower + 1, median);
			node->right = buildFromPoints(median, upper);
		}

		return node;
	}

	// Helper function that searches the tree.
	void search(Node* node, const DataPoint<T>* target, unsigned int k, std::priority_queue<HeapItem>& heap)
	{
		// Indicates that we're done here!
		if (node == NULL)
			return;

		// compute distance between target and the current node.
		T fDist = m_items[node->index]->euclidean_distance(target);

		// If current node within radious tau
		if (fDist < m_fTau)
		{
			// Remove furthest node from result list (if we already have k results)
			if (heap.size() == k)
				heap.pop();

			// Add current node to result list.
			heap.push(HeapItem(node->index, fDist));

			// Update value of tau (furthest point in result list)
			if (heap.size() == k)
				m_fTau = heap.top().dist;
		}

		// Return if we have arrived at a leaf.
		if (node->left == NULL && node->right == NULL)
			return;

		// If the target lies within the radius of ball
		if (fDist < node->threshold)
		{
			// If there can still be neighbors inside the ball,
			// recursively search left child first
			if (fDist - m_fTau <= node->threshold)
				search(node->left, target, k, heap);

			// If there can still be neighbors outside the ball,
			// recursively search the right child first.
			if (fDist + m_fTau >= node->threshold)
				search(node->right, target, k, heap);
		}
		else
		{
			// If there can still be neighbors outside the ball,
			// recursively search right child first
			if (fDist + m_fTau >= node->threshold)
				search(node->right, target, k, heap);

			// If there can still be neighbors inside the ball,
			// recursively search the left child first.
			if (fDist - m_fTau <= node->threshold)
				search(node->left, target, k, heap);
		}
	}

public:
	// Default constructor
	VpTree()
	{
		m_root = NULL;
	}

	// Destructor
	~VpTree()
	{
		if (m_root != NULL)
			delete m_root;
	}

	// Function to create a new VpTree from data.
	void create(const std::vector<DataPoint<T>*>& items)
	{
		if (m_root != NULL)
			delete m_root;

		m_items = items;
		m_root = buildFromPoints(0, (unsigned int)m_items.size());
	}

	// Function that uses the tree to find the k nearest neighbors of target.
	void search(const DataPoint<T>* target, unsigned int k, std::vector<DataPoint<T>*>* results, std::vector<T>* distances)
	{
		// Use a priority queue to store intermediate results on
		std::priority_queue<HeapItem> heap;

		// Variable that tracks the distance to the farthest point in our results
		m_fTau = (sizeof(T) == 4) ? FLT_MAX : DBL_MAX;

		// Perform the search
		search(m_root, target, k, heap);

		// Gather final results
		results->clear();
		distances->clear();

		while (!heap.empty())
		{
			results->push_back(m_items[heap.top().index]);
			distances->push_back(heap.top().dist);
			heap.pop();
		}

		// Results are in reverse order
		std::reverse(results->begin(), results->end());
		std::reverse(distances->begin(), distances->end());
	}
};



//=============================================================================
//	Class Methods
//=============================================================================

template <class T>
long tsnegpHandle<T>::Initialize(Memory<T>* pMem, Math<T>* pMath)
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
		srand((unsigned int)time(0));

		//------------------------------------------------
		//	Get the device memory pointers.
		//------------------------------------------------

		m_pRowP = pMem->GetHostBuffer(m_hRowPonhost)->Data();
		m_pColP = pMem->GetHostBuffer(m_hColPonhost)->Data();
		
		if ((m_pX = pMem->GetMemoryToHost(m_hX)) == NULL)
			throw ERROR_MEMORY_OUT;

		if ((m_pValP = pMem->GetMemoryToHost(m_hValP)) == NULL)
			throw ERROR_MEMORY_OUT;

		if ((m_pCurP = pMem->GetMemoryToHost(m_hCurP)) == NULL)
			throw ERROR_MEMORY_OUT;

		m_pRowP[0] = 0;

		for (unsigned int n=0; n<m_nN; n++)
		{
			m_pRowP[n+1] = m_pRowP[n] + m_nK;
		}

		m_pTree = new VpTree<T>();

		for (unsigned int n=0; n<m_nN; n++)
		{
			m_rgObjX[n] = new DataPoint<T>(m_nD, n, m_pX + n * m_nD);
		}

		m_pTree->create(m_rgObjX);
	}
	catch (LONG lErrEx)
	{
		CleanUp();
		return lErrEx;
	}

	return 0;
}

template long tsnegpHandle<double>::Initialize(Memory<double>* pMem, Math<double>* pMath);
template long tsnegpHandle<float>::Initialize(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long tsnegpHandle<T>::CleanUp()
{
	m_pRowP = NULL;
	m_pColP = NULL;

	if (m_pX != NULL)
		m_pMem->FreeHost(m_pX);
	
	if (m_pValP != NULL)
	{
		m_pMem->SetMemory(m_hValP, m_pValP, -1, -1);
		m_pMem->FreeHost(m_pValP);
	}

	if (m_pCurP != NULL)
	{
		m_pMem->SetMemory(m_hCurP, m_pCurP, -1, -1);
		m_pMem->FreeHost(m_pCurP);
	}

	for (unsigned int n=0; n<m_nN; n++)
	{
		delete m_rgObjX[n];
	}

	if (m_pTree != NULL)
		delete m_pTree;

	return 0;
}

template long tsnegpHandle<double>::CleanUp();
template long tsnegpHandle<float>::CleanUp();


template <class T>
long tsnegpHandle<T>::Run(bool *pbDone, int* pnCurrentIteration, int* pnMaxIteration)
{
	cublasHandle_t cublas = m_pMath->GetCublasHandle();


	//-------------------------------------------------
	//	Run one iteration.
	//-------------------------------------------------

	unsigned int n = (unsigned int)m_nCurrentIteration;

	// Find nearest neighbors
	std::vector<DataPoint<T>*> rgIndices;
	std::vector<T> rgDistances;

	m_pTree->search(m_rgObjX[n], m_nK + 1, &rgIndices, &rgDistances);
	if (rgDistances.size() == 0)
		return ERROR_TSNE_NO_DISTANCES_FOUND;

	// Initialize some variables for binary search.
	bool found = false;
	T fBeta = 1.0;
	T fMinBeta = -m_fMax;
	T fMaxBeta = m_fMax;
	T fTol = T(1e-5);

	// Iterate until we find a good perplexity
	int iter = 0;
	T sum_P = 0;
	while (!found && iter < 200)
	{
		// Compute Gaussian kernel row
		for (unsigned int m=0; m<m_nK; m++)
		{
			m_pCurP[m] = exp(-fBeta * rgDistances[m + 1] * rgDistances[m + 1]);
		}

		// Compute entropy of current row
		sum_P = m_fMin;
		for (unsigned int m=0; m<m_nK; m++)
		{
			sum_P += m_pCurP[m];
		}

		T H = T(0);
		for (unsigned int m=0; m<m_nK; m++)
		{
			H += fBeta * (rgDistances[m + 1] * rgDistances[m + 1] * m_pCurP[m]);
		}

		H = (H / sum_P) + log(sum_P);

		// Evaluate whether the entropy is within the tolerance level
		T Hdiff = H - log(m_fPerplexity);
		
		if (Hdiff < fTol && -Hdiff < fTol)
		{
			found = true;
		}
		else
		{
			if (Hdiff > 0)
			{
				fMinBeta = fBeta;
				if (fMaxBeta == m_fMax || fMaxBeta == -m_fMax)
					fBeta *= T(2);
				else
					fBeta = (fBeta + fMaxBeta) / T(2);
			}
			else
			{
				fMaxBeta = fBeta;
				if (fMinBeta == -m_fMax || fMinBeta == m_fMax)
					fBeta /= T(2);
				else
					fBeta = (fBeta + fMinBeta) / T(2);
			}
		}

		// Update iteration counter
		iter++;
	}

	// Row-normalize current row of P and store in matrix
	for (unsigned int m=0; m<m_nK; m++)
	{
		m_pCurP[m] /= sum_P;
	}

	for (unsigned int m=0; m<m_nK; m++)
	{
		m_pColP[(unsigned int)m_pRowP[n] + m] = T(rgIndices[m + 1]->index());
		m_pValP[(unsigned int)m_pRowP[n] + m] = m_pCurP[m];
	}


	//-------------------------------------------------
	//	Collect the return parameters.
	//-------------------------------------------------

	*pbDone = FALSE;
	m_nCurrentIteration ++;
	
	if (m_nCurrentIteration == m_nN)
		*pbDone = TRUE;

	if (pnCurrentIteration != NULL)
		*pnCurrentIteration = m_nCurrentIteration;

	if (pnMaxIteration != NULL)
		*pnMaxIteration = m_nN;

	return 0;
}

template long tsnegpHandle<double>::Run(bool *pbDone, int* pnCurrentIteration, int* pnMaxIteration);
template long tsnegpHandle<float>::Run(bool *pbDone, int* pnCurrentIteration, int* pnMaxIteration);

// end