//=============================================================================
//	FILE:	fused_comp.cu
//
//	DESC:	This file implements fused computation functions.
//=============================================================================

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
	cudnn_frontend::graph::Graph m_graph;
	map<long, std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>> m_tensor_map;
	int m_nOpCount;
	cudnnHandle_t m_cuda;

	LONG add_op_matmul(DataType dtCompute, T dfPadding, long hA, long hB, long* phC);

public:
	FusedCompData(Memory<T>* pMem, Math<T>* pMath) : m_tensor_map()
	{
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

	LONG Initialize(long hCuda, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
	void CleanUp();

	LONG AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);
	LONG GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4);
	LONG AddOp(FusedCompOp nOp, DataType dtCompute, T fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);
	LONG Build(HeurMode heur1, HeurMode heur2, long* pWorkspaceSize);
	LONG Execute(long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, long lCount);
};

//=============================================================================
//	Class Methods - FusedCompData
//=============================================================================

template <class T>
long FusedCompData<T>::Initialize(long hCuda, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace)
{
	m_cuda = m_pMem->GetCuDNN(hCuda);
	m_graph.set_io_data_type((cudnn_frontend::DataType_t)dtIo);
	m_graph.set_intermediate_data_type((cudnn_frontend::DataType_t)dtIntermediate);
	m_graph.set_compute_data_type((cudnn_frontend::DataType_t)dtCompute);

	return 0;
}

template long FusedCompData<double>::Initialize(long hCuda, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
template long FusedCompData<float>::Initialize(long hCuda, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);


template <class T>
void FusedCompData<T>::CleanUp()
{
}

template void FusedCompData<double>::CleanUp();
template void FusedCompData<float>::CleanUp();


template <class T>
long FusedCompData<T>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle)
{
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
	props.set_name("Tensor" + std::to_string(m_tensor_map.size() + 1));

	std::vector<int64_t> stride = { nS3 * nS4, nS4, 1 };
	props.set_stride(stride);

	long hTensor = m_tensor_map.size() + 1;
	m_tensor_map[hTensor] = m_graph.tensor(props);

	*phTensorHandle = hTensor;

	return 0;
}

template long FusedCompData<double>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);
template long FusedCompData<float>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);


template <class T>
long FusedCompData<T>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4)
{
	if (m_tensor_map.find(hTensorHandle) == m_tensor_map.end())
		return ERROR_PARAM_OUT_OF_RANGE;

	auto tensor = m_tensor_map[hTensorHandle];

	*pnS1 = tensor->get_dim()[0];
	*pnS2 = tensor->get_dim().size() > 1 ? tensor->get_dim()[1] : 0;
	*pnS3 = tensor->get_dim().size() > 2 ? tensor->get_dim()[2] : 0;
	*pnS4 = tensor->get_dim().size() > 3 ? tensor->get_dim()[3] : 0;
	*pdt = (DataType)tensor->get_data_type();
	
	return 0;
}

template long FusedCompData<double>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4);
template long FusedCompData<float>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4);


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
	std::shared_ptr <cudnn_frontend::graph::Tensor_attributes> tensorC = m_graph.matmul(tensorA, tensorB, attributes);
	tensorC->set_output(true);
	tensorC->set_data_type((cudnn_frontend::DataType_t)dtCompute);

	long nIdx = (long)(m_tensor_map.size() + 1);
	m_tensor_map[nIdx] = tensorC;
	*phC = nIdx;

	return 0;
}

template long FusedCompData<double>::add_op_matmul(DataType dtCompute, double dfPadding, long hA, long hB, long* phC);
template long FusedCompData<float>::add_op_matmul(DataType dtCompute, float fPadding, long hA, long hB, long* phC);


template <class T>
long FusedCompData<T>::Build(HeurMode heur1, HeurMode heur2, long* plWorkspaceSize)
{
	cudnn_frontend::error_t status = m_graph.validate();
	if (status.is_bad())
		return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

	status = m_graph.build_operation_graph(m_cuda);
	if (status.is_bad())
		return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

	std::vector<cudnn_frontend::HeurMode_t> heur_modes;
	heur_modes.push_back((cudnn_frontend::HeurMode_t)heur1);
	heur_modes.push_back((cudnn_frontend::HeurMode_t)heur2);

	status = m_graph.create_execution_plans(heur_modes);
	if (status.is_bad())
		return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

	status = m_graph.check_support(m_cuda);
	if (status.is_bad())
		return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

	status = m_graph.build_plans(m_cuda);
	if (status.is_bad())
		return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

	*plWorkspaceSize = (long)m_graph.get_workspace_size();

	return 0;
}

template long FusedCompData<double>::Build(HeurMode heur1, HeurMode heur2, long* phWorkspace);
template long FusedCompData<float>::Build(HeurMode heur1, HeurMode heur2, long* phWorkspace);


template <class T>
long FusedCompData<T>::Execute(long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, long lCount)
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

		if (m_tensor_map.find(hTensor) == m_tensor_map.end())
			return ERROR_PARAM_OUT_OF_RANGE;

		auto tensor = m_tensor_map[hTensor];
		int64_t uid = (int64_t)tensor->get_uid();

		MemoryItem* pData;
		if (lErr = pMemCol->GetData(hTensorData, &pData))
			return lErr;

		T* data = (T*)pData->Data();
		var_pack[uid] = (void*)data;
	}

	cudnn_frontend::error_t status = m_graph.execute(m_cuda, var_pack, workspace);
	if (status.is_bad())
		return (long)status.get_code() | ERROR_CUDNNFE_OFFSET;

	return 0;
}

template long FusedCompData<double>::Execute(long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, long lCount);
template long FusedCompData<float>::Execute(long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, long lCount);




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
long fusedcompHandle<T>::Initialize(long hCuda, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWorkspace)
{
	long lErr;

	if (lErr = m_pData->Initialize(hCuda, dtIo, dtIntermediate, dtCompute, preBuilt, phWorkspace))
	{
		m_pData->CleanUp();
		return lErr;
	}

	return 0;
}

template long fusedcompHandle<double>::Initialize(long hCuda, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);
template long fusedcompHandle<float>::Initialize(long hCuda, DataType dtIo, DataType dtIntermediate, DataType dtCompute, PreBuiltFusedComp preBuilt, long* phWokspace);


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
long fusedcompHandle<T>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle)
{
	return m_pData->AddTensor(dt, nS1, nS2, nS3, nS4, phTensorHandle);
}

template long fusedcompHandle<double>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);
template long fusedcompHandle<float>::AddTensor(DataType dt, long nS1, long nS2, long nS3, long nS4, long* phTensorHandle);


template <class T>
long fusedcompHandle<T>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4)
{
	return m_pData->GetTensor(hTensorHandle, pdt, pnS1, pnS2, pnS3, pnS4);
}

template long fusedcompHandle<double>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4);
template long fusedcompHandle<float>::GetTensor(long hTensorHandle, DataType* pdt, long* pnS1, long* pnS2, long* pnS3, long* pnS4);


template <class T>
long fusedcompHandle<T>::AddOp(FusedCompOp nOp, DataType dtCompute, T fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor)
{
	return m_pData->AddOp(nOp, dtCompute, fPadding, hTensor1, hTensor2, hTensor3, hTensor4, plIntermediateTensor);
}

template long fusedcompHandle<double>::AddOp(FusedCompOp nOp, DataType dtCompute, double dfPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);
template long fusedcompHandle<float>::AddOp(FusedCompOp nOp, DataType dtCompute, float fPadding, long hTensor1, long hTensor2, long hTensor3, long hTensor4, long* plIntermediateTensor);

template <class T>
long fusedcompHandle<T>::Build(HeurMode heur1, HeurMode heur2, long* phWorkspace)
{
	return m_pData->Build(heur1, heur2, phWorkspace);
}

template long fusedcompHandle<double>::Build(HeurMode heur1, HeurMode heur2, long* phWokspace);
template long fusedcompHandle<float>::Build(HeurMode heur1, HeurMode heur2, long* phWokspace);

template <class T>
long fusedcompHandle<T>::Execute(long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, long lCount)
{
	return m_pData->Execute(hWorkspace, rghTensor, rghTensorData, lCount);
}

template long fusedcompHandle<double>::Execute(long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, long lCount);
template long fusedcompHandle<float>::Execute(long hWorkspace, LONGLONG* rghTensor, LONGLONG* rghTensorData, long lCount);

// end