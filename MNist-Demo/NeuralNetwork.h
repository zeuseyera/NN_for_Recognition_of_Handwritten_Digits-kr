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
// 시그모이드 출력함수로 시그모이드의 미분값을 구한다
#define DSIGMOID(S) ( 0.66666667 / 1.7159 * (1.7159 + (S)) * (1.7159 - (S)) )


// forward declarations
class NNLayer;		//신경망 층 클래스
class NNWeight;		//신경망 가중치 클래스
class NNNeuron;		//신경망 뉴런 클래스
class NNConnection;	//신경망 연결 클래스


// helpful typedef's
typedef std::vector<NNLayer*>		VectorLayers;		//층벡터 포인터
typedef std::vector<NNWeight*>		VectorWeights;		//가중치벡터 포인터
typedef std::vector<NNNeuron*>		VectorNeurons;		//뉴런벡터 포인터
typedef std::vector<NNConnection>	VectorConnections;	//연결벡터
typedef std::basic_string<TCHAR>	tstring;			//이름

//==============================================================================
// 신경망 클래스
class NeuralNetwork  
{
public:
	volatile double m_etaLearningRatePrevious;
	volatile double m_etaLearningRate;

	volatile UINT m_cBackprops;  // counter used in connection with Weight sanity check
	void PeriodicWeightSanityCheck();

	void Calculate(	double*	inputVector,	//입력값
					UINT	count,			//입력개수
					double*	outputVector = NULL,	//출력값
					UINT	oCount = 0,		//출력개수
					std::vector<std::vector<double>>* pNeuronOutputs = NULL );

	void Backpropagate( double*	actualOutput,	//출력값
						double*	desiredOutput,	//목표값
						UINT	count,			//출력뉴런 개수
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

	VectorLayers m_Layers;	//신경망 층
};


//==============================================================================
// 신경망 층 클래스
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

	VectorWeights m_Weights;	//층 가중치
	VectorNeurons m_Neurons;	//층 뉴런

	tstring		label;			//층 이름
	NNLayer*	m_pPrevLayer;	//이전층 지시자

	// flag for one-time warning (per layer) about potential floating point overflow
	// 부동소수점 범람 가능성에 대한 1회 경고 플래그
	bool m_bFloatingPointWarning;

protected:
	void Initialize();
};


//==============================================================================
// 신경망 연결클래스
class NNConnection
{
public: 
	NNConnection(UINT neuron = ULONG_MAX, UINT weight = ULONG_MAX)
		: NeuronIndex( neuron )
		, WeightIndex( weight ) {};

	virtual ~NNConnection() {};

	UINT	NeuronIndex,	//연결 뉴런 식별자
			WeightIndex;	//연결 가중치 식별자
};


//==============================================================================
// 신경망 가중치 클래스
class NNWeight
{
public:
	NNWeight();
	NNWeight( LPCTSTR str, double val = 0.0 );
	virtual ~NNWeight();

	tstring label;	//가중치 이름 : 층번호_가중치번호_가중치전체순번
	double	value;	//가중치 값
	double	diagHessian;	//가중치 대각헤시안

protected:
	void Initialize();
};


//==============================================================================
// 신경망 뉴런클래스
class NNNeuron
{
public:
	NNNeuron();
	NNNeuron( LPCTSTR str );
	virtual ~NNNeuron();

	void AddConnection( UINT iNeuron, UINT iWeight );
	void AddConnection( NNConnection const & conn );


	tstring	label;	//뉴런 이름 : 층번호_뉴런번호_뉴런전체순번
	double	output;	//뉴런 출력

	VectorConnections m_Connections;	//뉴런 연결

///	VectorWeights m_Weights;
///	VectorNeurons m_Neurons;
	
protected:
	void Initialize();

};


#endif // !defined(AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_)
