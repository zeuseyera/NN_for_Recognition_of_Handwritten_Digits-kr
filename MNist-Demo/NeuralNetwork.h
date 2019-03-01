// NeuralNetwork.h: interface for the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_)
#define AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <math.h>
#include <vector>

using namespace std;

#define SIGMOID(x) ( 1.7159 * tanh(0.66666667 * x) )
// derivative of the sigmoid as a function of the sigmoid's output
// �ñ׸��̵� ����Լ��� �ñ׸��̵��� �̺а��� ���Ѵ�
#define DSIGMOID(S) ( 0.66666667 / 1.7159 * (1.7159 + (S)) * (1.7159 - (S)) )


// forward declarations
class NNLayer;		//�Ű�� �� Ŭ����
class NNWeight;		//�Ű�� ����ġ Ŭ����
class NNNeuron;		//�Ű�� ���� Ŭ����
class NNConnection;	//�Ű�� ���� Ŭ����


// helpful typedef's
typedef std::vector<NNLayer*>		VectorLayers;		//������ ������
typedef std::vector<NNWeight*>		VectorWeights;		//����ġ���� ������
typedef std::vector<NNNeuron*>		VectorNeurons;		//�������� ������
typedef std::vector<NNConnection>	VectorConnections;	//���Ấ��
typedef std::basic_string<TCHAR>	tstring;			//�̸�

//==============================================================================
// �Ű�� Ŭ����
class NeuralNetwork  
{
public:
	volatile double m_etaLearningRatePrevious;
	volatile double m_etaLearningRate;

	volatile UINT m_cBackprops;  // counter used in connection with Weight sanity check
	void PeriodicWeightSanityCheck();

	void Calculate(	double*	inputVector,	//�Է°�
					UINT	count,			//�Է°���
					double*	outputVector = NULL,	//��°�
					UINT	oCount = 0,		//��°���
					std::vector<std::vector<double>>* pNeuronOutputs = NULL );

	void Backpropagate( double*	actualOutput,	//��°�
						double*	desiredOutput,	//��ǥ��
						UINT	count,			//��´��� ����
						std::vector<std::vector<double>>* pMemorizedNeuronOutputs );

	void EraseHessianInformation();
	void DivideHessianInformationBy( double divisor );
	void BackpropagateSecondDervatives( double*	actualOutputVector,
										double*	targetOutputVector,
										UINT	count );

	void Serialize(CArchive &ar);

	NeuralNetwork();
	virtual ~NeuralNetwork();
	void Initialize();

	VectorLayers m_Layers;	//�Ű�� ��
};


//==============================================================================
// �Ű�� �� Ŭ����
class NNLayer
{
public:
	void PeriodicWeightSanityCheck();  // check if weights are "reasonable"
	void Calculate();
	void Backpropagate( std::vector<double>& dErr_wrt_dXn,		/* in */ 
						std::vector<double>& dErr_wrt_dXnm1,	/* out */ 
						std::vector<double>* thisLayerOutput,	// memorized values of this layer's output
						std::vector<double>* prevLayerOutput,	// memorized values of previous layer's output
						double etaLearningRate );

	void EraseHessianInformation();
	void DivideHessianInformationBy( double divisor );
	void BackpropagateSecondDerivatives(std::vector<double>& dErr_wrt_dXn /* in */, 
										std::vector<double>& dErr_wrt_dXnm1 /* out */);

	void Serialize(CArchive& ar );

	NNLayer();
	NNLayer( LPCTSTR str, NNLayer* pPrev = NULL );
	virtual ~NNLayer();

	VectorWeights m_Weights;	//�� ����ġ
	VectorNeurons m_Neurons;	//�� ����

	tstring		label;			//�� �̸�
	NNLayer*	m_pPrevLayer;	//������ ������

	// flag for one-time warning (per layer) about potential floating point overflow
	// �ε��Ҽ��� ���� ���ɼ��� ���� 1ȸ ��� �÷���
	bool m_bFloatingPointWarning;

protected:
	void Initialize();
};


//==============================================================================
// �Ű�� ����Ŭ����
class NNConnection
{
public: 
	NNConnection(UINT neuron = ULONG_MAX, UINT weight = ULONG_MAX)
		: NeuronIndex( neuron )
		, WeightIndex( weight ) {};

	virtual ~NNConnection() {};

	UINT	NeuronIndex,	//���� ���� �ĺ���
			WeightIndex;	//���� ����ġ �ĺ���
};


//==============================================================================
// �Ű�� ����ġ Ŭ����
class NNWeight
{
public:
	NNWeight();
	NNWeight( LPCTSTR str, double val = 0.0 );
	virtual ~NNWeight();

	tstring label;	//����ġ �̸� : ����ȣ_����ġ��ȣ_����ġ��ü����
	double	value;	//����ġ ��
	double	diagHessian;	//����ġ �밢��þ�

protected:
	void Initialize();
};


//==============================================================================
// �Ű�� ����Ŭ����
class NNNeuron
{
public:
	NNNeuron();
	NNNeuron( LPCTSTR str );
	virtual ~NNNeuron();

	void AddConnection( UINT iNeuron, UINT iWeight );
	void AddConnection( NNConnection const & conn );


	tstring	label;	//���� �̸� : ����ȣ_������ȣ_������ü����
	double	output;	//���� ���

	VectorConnections m_Connections;	//���� ����

///	VectorWeights m_Weights;
///	VectorNeurons m_Neurons;
	
protected:
	void Initialize();

};


#endif // !defined(AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_)
