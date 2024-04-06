//=============================================================================
//	FILE:	fused_comp.cu
//
//	DESC:	This file implements fused computation functions.
//=============================================================================
#define NV_CUDNN_DISABLE_EXCEPTION 1
#include <limits>
#include <cudnn_frontend.h>

#include "util.h"
#include "fused_comp.h"
#include "memory.h"
#include <string>
#include <iostream>
#include <fstream>
#include <map>

//=============================================================================
//	Function Definitions
//=============================================================================

//=============================================================================
//	Private Classes
//=============================================================================

template <class T>
class FusedCompData
{
protected:
	Memory<T>* m_pMem;
	Math<T>* m_pMath;
	cudnn_frontend::graph::Graph* m_pGraph;
	map<long, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> m_tensor_map;
	map<long, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> m_tensor_map_transposed;
	int m_nOpCount;
	cudnnHandle_t m_cuda;
	cublasHandle_t m_cublas;
	int m_nGpuID;
	int m_nLocalID;
	static std::unordered_map <int, cudnn_frontend::graph::Graph*> m_prebuilt_graphs;
	static std::unordered_map <int, int> m_prebuild_graphs_ref;

	LONG add_op_matmul(DataType dtCompute, T dfPadding, long hA, long hB, long* phC);
	LONG transpose(T* data, T* dataT, long nS1, long nS2, long nS3);


	long get_compute_capability(size_t* cc)
	{
		LONG lErr;
		struct cudaDeviceProp prop;
		if (lErr = cudaGetDeviceProperties(&prop, 0))
			return lErr;
		*cc = prop.major * 10 + prop.minor;
		return 0;
	}

	bool is_arch_supported_by_cudnn()
	{
		size_t cc;
		if (get_compute_capability(&cc))
			return false;

		// Hopper and Ada architecture is supported by CUDNN starting from version 8.6
		if (cudnnGetVersion() < 8600 && (90 <= cc || cc == 89))
			return false;

		return true;
	}

public:
	FusedCompData(Memory<T>* pMem, Math<T>* pMath) : m_tensor_map()
	{
		m_pGraph = NULL;
		m_nLocalID = -1;
		m_nGpuID = 0;
		m_nOpCount = 0;
		m_cuda = NULL;
		m_pMem = pMem;
		m_pMath = pMath;
	}

	~FusedCompData()
	{
	}

	Memory<T>* GetMemory()
	{
		return m_pMem;
	}

	Math<T>* GetMath()
	{
		return m_pMath;
	}

	LONG Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
	void CleanUp();

	LONG AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTranspose, long* phTensorHandle, long* phTensorWorkspace);
	LONG GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose);
	LONG AddOp(FusedCompOp nOp, DataType dtCompute, T fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);
	LONG Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* pWorkspaceSize);
	LONG Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount);
};

std::unordered_map <int, cudnn_frontend::graph::Graph*> FusedCompData<double>::m_prebuilt_graphs;
std::unordered_map <int, int> FusedCompData<double>::m_prebuild_graphs_ref;

std::unordered_map <int, cudnn_frontend::graph::Graph*> FusedCompData<float>::m_prebuilt_graphs;
std::unordered_map <int, int> FusedCompData<float>::m_prebuild_graphs_ref;

//=============================================================================
//	Class Methods - FusedCompData
//=============================================================================

template <class T>
long FusedCompData<T>::Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace)
{
	LONG lErr;

	if (!is_arch_supported_by_cudnn())
		return ERROR_PARAM_OUT_OF_RANGE;

	m_cuda = m_pMem->GetCuDNN(hCuda);
	m_pGraph = new cudnn_frontend::graph::Graph();
	if (m_pGraph == NULL)
		return ERROR_MEMORY_OUT;

	m_pGraph->set_io_data_type((cudnn_frontend::DataType_t)dtIo);
	m_pGraph->set_intermediate_data_type((cudnn_frontend::DataType_t)dtIntermediate);
	m_pGraph->set_compute_data_type((cudnn_frontend::DataType_t)dtCompute);
	m_nGpuID = nGpuID;

	if (lErr = cublasCreate(&m_cublas))
		return lErr | ERROR_CUBLAS_OFFSET;

	return 0;
}

template long FusedCompData<double>::Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
template long FusedCompData<float>::Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);


template <class T>
void FusedCompData<T>::CleanUp()
{
	if (m_cublas != NULL)
		cublasDestroy(m_cublas);

	m_cublas = NULL;
	m_cuda = NULL;
	m_tensor_map.clear();
	m_tensor_map_transposed.clear();

	if (m_nLocalID >= 0)
	{
		if (m_prebuild_graphs_ref.find(m_nLocalID) != m_prebuild_graphs_ref.end())
		{
			m_prebuild_graphs_ref[m_nLocalID]--;
			if (m_prebuild_graphs_ref[m_nLocalID] == 0)
			{
				m_prebuild_graphs_ref.erase(m_nLocalID);

				cudnn_frontend::graph::Graph* pGraph = m_prebuilt_graphs[m_nLocalID];
				delete pGraph;

				m_prebuilt_graphs.erase(m_nLocalID);
			}
		}
	}

	if (m_pGraph != NULL)
	{
		delete m_pGraph;
		m_pGraph = NULL;
	}
}

template void FusedCompData<double>::CleanUp();
template void FusedCompData<float>::CleanUp();


template <class T>
long FusedCompData<T>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTranspose, long* phTensorHandle, long* phTensorWorkspace)
{
	LONG lErr;
	std::vector<int64_t> dim = { nS1 };

	if (nS2 > 1)
		dim[0] *= nS2;

	if (nS3 > 0)
		dim.push_back(nS3);

	if (nS4 > 0)
		dim.push_back(nS4);

	int ndim = dim.size();

	auto props = cudnn_frontend::graph::Tensor_attributes();
	props.set_data_type((cudnn_frontend::DataType_t)dt);
	props.set_is_virtual(false);
	props.set_is_pass_by_value(false);
	props.set_dim(dim);
	props.set_name("Tensor" + std::to_string(m_tensor_map.size() + 1) + ((bTranspose) ? ".t" : ""));

	//std::vector<int64_t> stride = { nS3 * nS4, nS4, 1 };
	auto stride_order = cudnn_frontend::detail::generate_row_major_stride_order(ndim);
	std::vector<int64_t> stride = cudnn_frontend::detail::generate_stride(dim, stride_order);

	props.set_stride(stride);

	std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> tensor = m_pGraph->tensor(props);
	long hTensor = m_tensor_map.size() + 1;

	*phTensorWorkspace = 0;
	m_tensor_map[hTensor] = tensor;

	if (bTranspose)
	{
		if (dim.size() != 3)
			return ERROR_PARAM_OUT_OF_RANGE;

		size_t lWorksapceItems = nS1 * max(1,nS2) * max(1,nS3) * max(1,nS4);
		if (lErr = m_pMem->AllocMemory(m_nGpuID, false, lWorksapceItems, NULL, 0, phTensorWorkspace))
			return lErr;

		m_tensor_map_transposed[hTensor] = tensor;
	}

	*phTensorHandle = hTensor;

	return 0;
}

template long FusedCompData<double>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTranspose, long* phTensorHandle, long* phTensorWorkspace);
template long FusedCompData<float>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTranspose, long* phTensorHandle, long* phTensorWorkspace);


template <class T>
long FusedCompData<T>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose)
{
	if (m_tensor_map.find(hTensorHandle) == m_tensor_map.end())
		return ERROR_PARAM_OUT_OF_RANGE;

	auto tensor = m_tensor_map[hTensorHandle];

	*pnS1 = tensor->get_dim()[0];
	*pnS2 = tensor->get_dim().size() > 1 ? tensor->get_dim()[1] : 0;
	*pnS3 = tensor->get_dim().size() > 2 ? tensor->get_dim()[2] : 0;
	*pnS4 = tensor->get_dim().size() > 3 ? tensor->get_dim()[3] : 0;
	*pdt = (DataType)tensor->get_data_type();
	*pbTranspose = false;

	if (m_tensor_map_transposed.find(hTensorHandle) != m_tensor_map.end())
		*pbTranspose = true;
	
	return 0;
}

template long FusedCompData<double>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose);
template long FusedCompData<float>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose);


template <class T>
long FusedCompData<T>::AddOp(FusedCompOp nOp, DataType dtCompute, T fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor)
{
	LONG lErr;

	switch (nOp)
	{
		case FusedCompOp::FUSED_COMP_OP_MATMUL:
			if (hTensor1 == 0 || hTensor2 == 0)
				return ERROR_PARAM_NULL;

			if (lErr = add_op_matmul(dtCompute, fPadding, hTensor1, hTensor2, plIntermediateTensor))
				return lErr;
			break;

		defualt:
			return ERROR_PARAM_OUT_OF_RANGE;
	}

	return 0;
}

template long FusedCompData<double>::AddOp(FusedCompOp nOp, DataType dtCompute, double dfPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);
template long FusedCompData<float>::AddOp(FusedCompOp nOp, DataType dtCompute, float fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);


template <class T>
LONG FusedCompData<T>::add_op_matmul(DataType dtCompute, T fPadding, long hA, long hB, long* phC)
{
	if (m_tensor_map.find(hA) == m_tensor_map.end() || m_tensor_map.find(hB) == m_tensor_map.end())
		return ERROR_PARAM_OUT_OF_RANGE;

	m_nOpCount++;
	auto attributes = cudnn_frontend::graph::Matmul_attributes();
	attributes.set_compute_data_type((cudnn_frontend::DataType_t)dtCompute);
	attributes.set_name("Matmul" + std::to_string(m_nOpCount));

	auto tensorA = m_tensor_map[hA];
	auto tensorB = m_tensor_map[hB];

	if (m_tensor_map_transposed.find(hA) != m_tensor_map_transposed.end())
		tensorA = m_tensor_map_transposed[hA];

	if (m_tensor_map_transposed.find(hB) != m_tensor_map_transposed.end())
		tensorB = m_tensor_map_transposed[hB];

	std::shared_ptr <cudnn_frontend::graph::Tensor_attributes> tensorC = m_pGraph->matmul(tensorA, tensorB, attributes);
	tensorC->set_output(true);
	tensorC->set_data_type((cudnn_frontend::DataType_t)dtCompute);

	long nIdx = (long)(m_tensor_map.size() + 1);
	m_tensor_map[nIdx] = tensorC;
	*phC = nIdx;

	return 0;
}

template long FusedCompData<double>::add_op_matmul(DataType dtCompute, double dfPadding, long hA, long hB, long* phC);
template long FusedCompData<float>::add_op_matmul(DataType dtCompute, float fPadding, long hA, long hB, long* phC);


template <>
long FusedCompData<float>::transpose(float* data, float* dataT, long nS1, long nS2, long nS3)
{
	LONG lErr;
	int rows = nS2 * nS1;
	int cols = nS3;
	const float alpha = 1.0f;
	const float beta = 0.0f;

	if (lErr = cublasSgeam(m_cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, data, cols, &beta, NULL, rows, dataT, rows))
		return lErr | ERROR_CUBLAS_OFFSET;
	return 0;
}

template <>
long FusedCompData<double>::transpose(double* data, double* dataT, long nS1, long nS2, long nS3)
{
	LONG lErr;
	int rows = nS2 * nS1;
	int cols = nS3;
	const double alpha = 1.0f;
	const double beta = 0.0f;

	if (lErr = cublasDgeam(m_cublas, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, data, cols, &beta, NULL, rows, dataT, rows))
		return lErr | ERROR_CUBLAS_OFFSET;
	return 0;
}


template <class T>
long FusedCompData<T>::Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* plWorkspaceSize)
{
	cudnn_frontend::error_t status;
		
	try
	{
		cudnn_frontend::graph::Graph* pGraph;
		m_nLocalID = nLocalID;

		if (nLocalID < 0 || m_prebuilt_graphs.find(nLocalID) == m_prebuilt_graphs.end())
		{
			status = m_pGraph->validate();
			if (status.is_bad())
				return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

			status = m_pGraph->build_operation_graph(m_cuda);
			if (status.is_bad())
				return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

			std::vector<cudnn_frontend::HeurMode_t> heur_modes;
			heur_modes.push_back((cudnn_frontend::HeurMode_t)heur1);
			heur_modes.push_back((cudnn_frontend::HeurMode_t)heur2);

			status = m_pGraph->create_execution_plans(heur_modes);
			if (status.is_bad())
				return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

			status = m_pGraph->check_support(m_cuda);
			if (status.is_bad())
				return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

			status = m_pGraph->build_plans(m_cuda);
			if (status.is_bad())
				return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

			if (nLocalID >= 0)
			{
				m_prebuilt_graphs.insert(std::make_pair(nLocalID, m_pGraph));
				m_pGraph = NULL;
				m_prebuild_graphs_ref.insert(std::make_pair(nLocalID, 0));
				pGraph = m_prebuilt_graphs[nLocalID];
			}
			else
			{
				pGraph = m_pGraph;
			}
		}

		if (nLocalID >= 0)
		{
			if (m_prebuild_graphs_ref.find(nLocalID) == m_prebuild_graphs_ref.end())
				return ERROR_FUSEDCOMP_NOT_INITIALIZED;

			m_prebuild_graphs_ref[nLocalID]++;
			pGraph = m_prebuilt_graphs[nLocalID];
		}

		*plWorkspaceSize = (long)pGraph->get_workspace_size();
	}
	catch(cudnn_frontend::cudnnException & e)
	{
		LONG lErr = (long)e.getCudnnStatus();
		return lErr | ERROR_CUDNN_OFFSET;
	}
	catch (...)
	{
		LONG lErr = (long)status.get_code();
		if (lErr == 0)
			return ERROR_FUSEDCOMP;

		return lErr | ERROR_CUDNNFE_OFFSET;
	}

	return 0;
}

template long FusedCompData<double>::Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* phWorkspace);
template long FusedCompData<float>::Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* phWorkspace);


template <class T>
long FusedCompData<T>::Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount)
{
	LONG lErr;
	MemoryCollection* pMemCol = m_pMem->GetMemoryCollection();
	std::unordered_map<int64_t, void*> var_pack;

	T* workspace = NULL;
	if (hWorkspace != 0)
	{
		MemoryItem* pWorkspace;
		if (lErr = pMemCol->GetData(hWorkspace, &pWorkspace))
			return lErr;

		workspace = (T*)pWorkspace->Data();
	}

	for (int i = 0; i < lCount; i++)
	{
		long hTensor = (long)rghTensor[i];
		long hTensorData = (long)rghTensorData[i];
		long hTensorWorkspaceData = (long)rghTensorWorkspaceData[i];

		if (m_tensor_map.find(hTensor) == m_tensor_map.end())
			return ERROR_PARAM_OUT_OF_RANGE;

		auto tensor = m_tensor_map[hTensor];
		int64_t uid = (int64_t)tensor->get_uid();

		MemoryItem* pData;
		if (lErr = pMemCol->GetData(hTensorData, &pData))
			return lErr;

		T* data = (T*)pData->Data();

		if (m_tensor_map_transposed.find(hTensor) != m_tensor_map_transposed.end())
		{
			if (hTensorWorkspaceData == 0)
				return ERROR_PARAM_NULL;

			auto tensor = m_tensor_map_transposed[hTensor];
			int64_t uid = (int64_t)tensor->get_uid();

			MemoryItem* pDataT;
			if (lErr = pMemCol->GetData(hTensorWorkspaceData, &pDataT))
				return lErr;

			T* dataT = (T*)pDataT->Data();

			if (lErr = transpose(data, dataT, tensor->get_dim()[0], tensor->get_dim()[1], tensor->get_dim()[2]))
				return lErr | ERROR_CUBLAS_OFFSET;

			var_pack[uid] = (void*)dataT;
		}
		else
		{
			var_pack[uid] = (void*)data;
		}
	}

	cudnn_frontend::graph::Graph* pGraph;

	if (nLocalID >= 0)
	{
		if (m_prebuilt_graphs.find(nLocalID) == m_prebuilt_graphs.end())
			return ERROR_FUSEDCOMP_NOT_INITIALIZED;

		pGraph = m_prebuilt_graphs[nLocalID];
	}
	else
	{
		pGraph = m_pGraph;
	}

	cudnn_frontend::error_t status = pGraph->execute(m_cuda, var_pack, workspace);
	if (status.is_bad())
		return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

	return 0;
}

template long FusedCompData<double>::Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount);
template long FusedCompData<float>::Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount);


//=============================================================================
//	Class Methods - LayerNorm
//=============================================================================

template <class T>
long fusedcompHandle<T>::Update(Memory<T>* pMem, Math<T>* pMath)
{
	m_pMem = pMem;
	m_pMath = pMath;
	m_nRefCount++;

	m_pData = new FusedCompData<T>(pMem, pMath);

	if (m_pData == NULL)
		return ERROR_MEMORY_OUT;

	return 0;
}

template long fusedcompHandle<double>::Update(Memory<double>* pMem, Math<double>* pMath);
template long fusedcompHandle<float>::Update(Memory<float>* pMem, Math<float>* pMath);


template <class T>
long fusedcompHandle<T>::Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWorkspace)
{
	long lErr;

	if (lErr = m_pData->Initialize(hCuda, nGpuID, dtIo, dtIntermediate, dtCompute, preBuilt, phWorkspace))
	{
		m_pData->CleanUp();
		return lErr;
	}

	return 0;
}

template long fusedcompHandle<double>::Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
template long fusedcompHandle<float>::Initialize(long hCuda, int nGpuID, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);


template <class T>
long fusedcompHandle<T>::CleanUp()
{
	m_nRefCount--;

	if (m_nRefCount == 0)
	{
		if (m_pData != NULL)
		{
			m_pData->CleanUp();
			delete m_pData;
			m_pData = NULL;
		}
	}

	return 0;
}

template long fusedcompHandle<double>::CleanUp();
template long fusedcompHandle<float>::CleanUp();


template <class T>
long fusedcompHandle<T>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTransform, long* phTensorHandle, long* phTensorWorkspace)
{
	return m_pData->AddTensor(dt, nS1, nS2, nS3, nS4, bTransform, phTensorHandle, phTensorWorkspace);
}

template long fusedcompHandle<double>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTransform, long* phTensorHandle, long* phTensorWorkspace);
template long fusedcompHandle<float>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, bool bTransform, long* phTensorHandle, long* phTensorWorkspace);


template <class T>
long fusedcompHandle<T>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose)
{
	return m_pData->GetTensor(hTensorHandle, pdt, pnS1, pnS2, pnS3, pnS4, pbTranspose);
}

template long fusedcompHandle<double>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose);
template long fusedcompHandle<float>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4, bool* pbTranspose);


template <class T>
long fusedcompHandle<T>::AddOp(FusedCompOp nOp, DataType dtCompute, T fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor)
{
	return m_pData->AddOp(nOp, dtCompute, fPadding, hTensor1, hTensor2, hTensor3, hTensor4, plIntermediateTensor);
}

template long fusedcompHandle<double>::AddOp(FusedCompOp nOp, DataType dtCompute, double dfPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);
template long fusedcompHandle<float>::AddOp(FusedCompOp nOp, DataType dtCompute, float fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);

template <class T>
long fusedcompHandle<T>::Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* phWorkspace)
{
	return m_pData->Build(nLocalID, heur1, heur2, phWorkspace);
}

template long fusedcompHandle<double>::Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* phWokspace);
template long fusedcompHandle<float>::Build(int nLocalID, HeurMode heur1, HeurMode heur2, long* phWokspace);

template <class T>
long fusedcompHandle<T>::Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount)
{
	return m_pData->Execute(nLocalID, hWorkspace, rghTensor, rghTensorData, rghTensorWorkspaceData, lCount);
}

template long fusedcompHandle<double>::Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount);
template long fusedcompHandle<float>::Execute(int nLocalID, long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, LONGLONG* rghTensorWorkspaceData, long lCount);

// end