// NeuralNetwork.cpp: implementation of the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "NeuralNetwork.h"
#include "MNist.h"  // for the _Intelocked functions
#include <malloc.h>  // for the _alloca function


#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif


///////////////////////////////////////////////////////////////////////
// NeuralNetwork 클래스
//  NeuralNetwork class definition

NeuralNetwork::NeuralNetwork()
{
	Initialize();
}

void NeuralNetwork::Initialize()
{
	// delete all layers
	VectorLayers::iterator it;
	
	for( it=m_Layers.begin(); it<m_Layers.end(); it++ )
	{
		delete *it;
	}
	
	m_Layers.clear();
	
	// arbitrary, so that brand-new NNs can be serialized with a non-ridiculous number
	// 임의, 그래서 아주 새로운 신경망은 엉뚱하지않은 숫자로 일련화할 수 있다
	m_etaLearningRate = .001;
	m_cBackprops = 0;
	
}

NeuralNetwork::~NeuralNetwork()
{
	// call Initialize(); 생각하면 의미가 있다. makes sense if you think
	Initialize();
}


//------------------------------------------------------------------------------
// 신경망의 순방향 전파를 진행한다
void NeuralNetwork::Calculate(double*	inputVector,				//입력뉴런(841)
							  UINT		iCount,						//입력개수(841)
							  double*	outputVector,	/* =NULL */	//출력값 저장소(10)
							  UINT		oCount,			/* =0 */	//출력개수(10)
							  //모든 층의 출력을 저장하기 위한 벡터
							  std::vector<std::vector<double>>* pNeuronOutputs) /* =NULL */
{
	VectorLayers::iterator lit = m_Layers.begin();	//층 반복자
	VectorNeurons::iterator nit;					//뉴런 반복자

	//--------------------------------------------------------------------------
	// first layer is input layer: directly set outputs of all of its neurons to the input vector
	// 1) 첫번째(L0) 층은 입력층: 입력벡터의 모든 뉴런을 직접 설정한다.
	if ( lit<m_Layers.end() )  
	{
		nit = (*lit)->m_Neurons.begin();
		UINT	wichi = 0;

		// there should be exactly one neuron per input
		// 이 층은 반드시 하나의 입력당 하나의 뉴런이 있어야 한다.
		ASSERT( iCount == (*lit)->m_Neurons.size() );

		// 모든 뉴런을 반복하여 입력값을 뉴런의 출력값에 넣어준다
		while( ( nit < (*lit)->m_Neurons.end() ) && ( wichi < iCount ) )
		{
			//입력값을 가중치 적용 없이 뉴런의 출력에 넣어준다
			(*nit)->output = inputVector[ wichi ];
			nit++;		//뉴런의 위치를 증가시키다
			wichi++;	//입력의 위치를 증가시킨다
		}
	}

	//--------------------------------------------------------------------------
	// 2) 나머지(L1~L4) 층을 반복한다, Calculate() 함수를 호출하여
	for( lit++; lit<m_Layers.end(); lit++ )
	{
		(*lit)->Calculate();	//각 층의 Calculate() 함수를 호출한다
	}
	
	//--------------------------------------------------------------------------
	// 3) 결과를 출력벡터에 적재한다. load up output vector with results
	if ( outputVector != NULL )
	{
		lit = m_Layers.end();
		lit--;	//마지막층을 지시하기 위해
		
		nit = (*lit)->m_Neurons.begin();

		//최종 출력뉴런의 출력값을 출력벡터에 저장한다
		for ( UINT ii=0; ii<oCount; ++ii )
		{
			//뉴런의 출력을 출력값 저장소에 복사한다 
			outputVector[ ii ] = (*nit)->output;
			nit++;	//뉴런의 위치를 증가시킨다
		}
	}
	
	//--------------------------------------------------------------------------
	// 4) 결과를 뉴런의 출력값에 적재한다. load up neuron output values with results
	if ( pNeuronOutputs != NULL )
	{
		// 처음 사용을 위해 확인(재사용이 예상된다) check for first time use (re-use is expected)
		if ( pNeuronOutputs->empty() != FALSE )
		{
			// it's empty, so allocate memory for its use
			// 비어있다, 그래서 사용을 위해 메모리를 할당한다.

			pNeuronOutputs->clear();  // 보호를 위해 for safekeeping
			
			UINT ii = 0;
			// 모든 층을 반복한다
			for( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
			{
				std::vector<double> layerOut;

				// 모든 뉴런을 반복한다
				for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
				{
					layerOut.push_back( (*lit)->m_Neurons[ ii ]->output );
				}
				
				pNeuronOutputs->push_back( layerOut );
			}
		}
		else
		{
			// it's not empty, so assume it's been used in a past iteration and memory for
			// it has already been allocated internally.  Simply store the values
			// 비어있지 않다, 그래서 이것은 반복 이전에 사용되었고 메모리는
			// 이미 내부적으로 할당되었다고 가정한다. 단순히 값을 저장한다.
			
			UINT ii, jj = 0;
			// 모든 층을 반복한다
			for( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
			{
				// 모든 뉴런을 반복한다
				for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
				{
					(*pNeuronOutputs)[ jj ][ ii ] = (*lit)->m_Neurons[ ii ]->output;
				}
				
				++jj;
			}
		}
	}
}

//==============================================================================
// 역전파를 수행한다
void NeuralNetwork::Backpropagate(double*	actualOutput,	//출력값
								  double*	desiredOutput,	//목표값
								  UINT		count,			//출력개수
								  std::vector<std::vector<double>>* pMemorizedNeuronOutputs)
{
	// 신경망을 통해 역전파를 한다. backpropagates through the neural net
	
	ASSERT( ( actualOutput != NULL ) && ( desiredOutput != NULL ) && ( count < 256 ) );
	
	ASSERT( m_Layers.size() >= 2 );  // there must be at least two layers in the net
	
	if ( ( actualOutput == NULL ) || ( desiredOutput == NULL ) || ( count >= 256 ) )
		return;
	
	
	// 가중치의 온전성(상태)확인 시간이 되었다면 확인 check if it's time for a weight sanity check
	m_cBackprops++;
	
	if ( (m_cBackprops % 10000) == 0 )
	{
		// 매 10000 역전파 마다 every 10000 backprops
		PeriodicWeightSanityCheck();
	}

	// proceed from the last layer to the first, iteratively
	// We calculate the last layer separately, and first, since it provides the needed derviative
	// (i.e., dErr_wrt_dXnm1) for the previous layers
	// 마지막 층에서 부터 첫번째 층으로 진행한다, 반복적으로
	// 마지막 층을 별도로 계산한다, 먼저, 이전 층을 위해 필요한
	// 편미분(즉, dErr_wrt_dXnm1)을 제공하기 때문에
	
	// 명명법 nomenclature:
	//
	// Err	신경망 전체의 출력에러 is output error of the entire neural net
	// Xn	n번째층의 출력벡터 is the output vector on the n-th layer
	// Xnm1 이전층의 출력벡터 is the output vector of the previous layer
	// Wn	n번째층의 가중치 벡터 is the vector of weights of the n-th layer
	// Yn	n번째층의 활성화 값, 즉, 찌그림 함수 적용 전의 입력의 가중합
	//		is the activation value of the n-th layer, i.e., the weighted sum of inputs 
	//		BEFORE the squashing function is applied
	// F	찌그림 함수 is the squashing function: Xn = F(Yn)
	// F'	찌그림 함수의 편미분 is the derivative of the squashing function
	//		단순히, F = tanh 라면, 그러면 F'(Yn) = 1 - Xn^2,
	//		즉, 편미분은 출력으로부터 계산된다, 알고있는 입력 없이
	//		Conveniently, for F = tanh, then F'(Yn) = 1 - Xn^2,
	//		i.e., the derivative can be calculated from the output, without knowledge of the input

	VectorLayers::iterator lit = m_Layers.end() - 1;
	
	std::vector<double> dErr_wrt_dXlast( (*lit)->m_Neurons.size() );
	std::vector<std::vector<double>> differentials;
	
	UINT iSize = m_Layers.size();
	
	differentials.resize( iSize );
	
	UINT ii;
	
	// start the process by calculating dErr_wrt_dXn for the last layer.
	// for the standard MSE Err function (i.e., 0.5*sumof( (actual-target)^2 ), this differential is simply
	// the difference between the target and the actual
	// 마지막층에 대한 dErr_wrt_dXn 계산으로 절차를 시작한다.
	// 표준 MSE 에러 함수(즉, 0.5*sumof( (actual-target)^2), 이 미분은 간단하다
	// 목표와 실제 사이의 차이
	for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
	{
		dErr_wrt_dXlast[ ii ] = actualOutput[ ii ] - desiredOutput[ ii ];
	}

	// store Xlast and reserve memory for the remaining vectors stored in differentials
	// Xlast를 저장하고 미분에 저장된 나머지 벡터를 위한 메모리를 예약한다 
	
	differentials[ iSize-1 ] = dErr_wrt_dXlast;  // last one
	
	for ( ii=0; ii<iSize-1; ++ii )
	{
		differentials[ ii ].resize( m_Layers[ii]->m_Neurons.size(), 0.0 );
	}
	
	// now iterate through all layers including the last but excluding the first, and ask each of
	// them to backpropagate error and adjust their weights, and to return the differential
	// dErr_wrt_dXnm1 for use as the input value of dErr_wrt_dXn for the next iterated layer
	// 이제 첫번째를 제외하고 마지막을 포함한 모든 층을 반복한다, 그리고
	// 역전파를 하기위한 각각의 에러를 구하고, 가중치를 조정하고,
	// 다음 반복층을 위한 dErr_wrt_dXn의 입력값으로 사용하기 위한 미분 dErr_wrt_dXnm1를 반환한다.
	
	BOOL bMemorized = ( pMemorizedNeuronOutputs != NULL );	//출력층 이면

	// re-initialized to last layer for clarity, although it should already be this value
	// 명확성을 위해 마지막층을 다시 초기화, 비록 이미 이 값일 지라도
	lit = m_Layers.end() - 1;
	
	ii = iSize - 1;
	for ( lit; lit>m_Layers.begin(); lit--)
	{	//이전 층으로 이동하며
		if ( bMemorized != FALSE )
		{	//출력 및 은닉층을 역전파 한다
			(*lit)->Backpropagate(differentials[ ii ],
								  differentials[ ii - 1 ], 
								  &(*pMemorizedNeuronOutputs)[ ii ],
								  &(*pMemorizedNeuronOutputs)[ ii - 1 ],
								  m_etaLearningRate );
		}
		else
		{	//입력층을 역전파 한다
			(*lit)->Backpropagate(differentials[ ii ],
								  differentials[ ii - 1 ], 
								  NULL,
								  NULL,
								  m_etaLearningRate );
		}
		
		--ii;
	}

	differentials.clear();
}

								  
void NeuralNetwork::PeriodicWeightSanityCheck()
{
	// fucntion that simply goes through all weights, and tests them against an arbitrary
	// "reasonable" upper limit.  If the upper limit is exceeded, a warning is displayed
	// 함수는 모든 가중치를 단순히 통과한다, 그리고 임의의 "적당한" 상한과
	// 비교하여 시험한다. 상한을 초과하면, 경고를 표시한다
	
	VectorLayers::iterator lit;
	
	for ( lit=m_Layers.begin(); lit<m_Layers.end(); lit++)
	{
		(*lit)->PeriodicWeightSanityCheck();
	}
}


void NeuralNetwork::EraseHessianInformation()
{
	// controls each layer to erase (set to value of zero) all its diagonal Hessian info
	// 모든 대각 헤시안 정보를 지우기위해(0의 값으로 설정) 각 층을 제어한다
	VectorLayers::iterator lit;
	
	for ( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
	{
		(*lit)->EraseHessianInformation();
	}
}


void NeuralNetwork::DivideHessianInformationBy( double divisor )
{
	// controls each layer to divide its current diagonal Hessian info by a common divisor. 
	// A check is also made to ensure that each Hessian is strictly zero-positive
	// 공약수로 현재의 대각 헤시안 정보를 나누기 위해 각 층을 제어한다
	// 확인은 또한 각 헤시안이 오로지 양의 0을 보장하기 위한 것이다
	VectorLayers::iterator lit;
	
	for ( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
	{
		(*lit)->DivideHessianInformationBy( divisor );
	}
	
}


void NeuralNetwork::BackpropagateSecondDervatives(double* actualOutputVector, 
												  double* targetOutputVector, UINT count )
{
	// calculates the second dervatives (for diagonal Hessian) and backpropagates
	// them through neural net
	// 두번째 편미분(대각 헤시안에 대한)을 계산하고 신경망을 통해 편미분을 역전파한다
	ASSERT( ( actualOutputVector != NULL ) && ( targetOutputVector != NULL ) && ( count < 256 ) );
	
	ASSERT( m_Layers.size() >= 2 );  // 망은 적어도 2층이 있어야 한다 there must be at least two layers in the net
	
	if ( ( actualOutputVector == NULL ) || ( targetOutputVector == NULL ) || ( count >= 256 ) )
		return;
	
	// we use nearly the same nomenclature as above (e.g., "dErr_wrt_dXnm1") even though everything here
	// is actually second derivatives and not first derivatives, since otherwise the ASCII would 
	// become too confusing.  To emphasize that these are second derivatives, we insert a "2"
	// such as "d2Err_wrt_dXnm1".  We don't insert the second "2" that's conventional for designating
	// second derivatives
	// 우리는 위와 거의 동일한 용어(예를들면, dErr_wrt_dXnm1)를 사용한다 비록 여기의
	// 모든것은 실제로 두번째 편미분(미분한 결과)이고 첫번째 편미분(미분한 결과)가 아니다,
	// 그렇지 않으면 ASCII가 너무 혼동될 것이기 때문이다. 이 두번째 편미분(미분한 결과)를
	// 강조하기 위해, 우리는 "d2Err_wrt_dXnm1"처럼 "2"를 삽입한다. 우리는 두번째 편미분을
	// 지정하기 위해 통상적인 두번째 "2"를 삽입하지 않았다.
	
	VectorLayers::iterator lit;
	
	lit = m_Layers.end() - 1;  // 마지막 층을 설정 set to last layer
	
	std::vector<double> d2Err_wrt_dXlast( (*lit)->m_Neurons.size() );
	std::vector<std::vector<double>> differentials;
	
	UINT iSize = m_Layers.size();
	
	differentials.resize( iSize );
	
	UINT ii;
	
	// start the process by calculating the second derivative dErr_wrt_dXn for the last layer.
	// for the standard MSE Err function (i.e., 0.5*sumof( (actual-target)^2 ), this differential is 
	// exactly one
	// 마지막 층을 위한 두번째 편미분 dErr_wrt_dXn 계산으로 절차를 시작한다
	// 표준 MSE 에러함수(즉, 0.5*sumof( (actual-target)^2)에 대해서, 편미분은 단 하나이다.
	lit = m_Layers.end() - 1;  // 마지막층을 지시한다 point to last layer
	
	for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
	{
		d2Err_wrt_dXlast[ ii ] = 1.0;
	}

	// store Xlast and reserve memory for the remaining vectors stored in differentials
	// Xlast를 저장하고 미분을 저장한 나머지 벡터를 위한 메모리를 예약한다 
	differentials[ iSize-1 ] = d2Err_wrt_dXlast;  // last one
	
	for ( ii=0; ii<iSize-1; ++ii )
	{
		differentials[ ii ].resize( m_Layers[ii]->m_Neurons.size(), 0.0 );
	}

	// now iterate through all layers including the last but excluding the first, starting from
	// the last, and ask each of
	// them to backpropagate the second derviative and accumulate the diagonal Hessian, and also to
	// return the second dervative
	// d2Err_wrt_dXnm1 for use as the input value of dErr_wrt_dXn for the next iterated layer (which
	// is the previous layer spatially)
	// 이제 첫번째를 제외하고 마지막을 포함한 모든 층을 반복한다, 마지막부터 시작한다,
	// 그리고 역전파를 위한 두번째 편미분을 구하고, 대각 헤시안을 모은다, 그리고
	// 또한 반환하기 위한 두번째 편미분 d2Err_wrt_dXnm1을 구한다
	// 다음 반복층을 위한 dErr_wrt_dXn의 입력값으로 사용하기 위하여 (이전층은 공간적이다)

	// re-initialized to last layer for clarity, although it should already be this value
	// 명확성을 위해 마지막층을 다시 초기화, 비록 이미 이 값일 지라도
	lit = m_Layers.end() - 1;
	
	ii = iSize - 1;
	for ( lit; lit>m_Layers.begin(); lit--)
	{	// 두번째 미분한결과를 역전파한다
		(*lit)->BackpropagateSecondDerivatives( differentials[ ii ], differentials[ ii - 1 ] );
		
		--ii;
	}
	
	differentials.clear();
}

//==============================================================================
// 신경망의 저장 및 탑재
void NeuralNetwork::Serialize(CArchive &ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장코드 추가 add storing code here
		
		ar << m_etaLearningRate;	//학습율
		
		ar << m_Layers.size();		//층 개수
		
		VectorLayers::iterator lit;	//층 반복자

		//층을 반복하여 호출한다
		for( lit=m_Layers.begin(); lit<m_Layers.end(); lit++ )
		{
			(*lit)->Serialize( ar );
		}
	}
	else
	{
		// TODO: 여기에 탑재코드 추가 add loading code here
		
		double eta; 
		ar >> eta;
		// two-step storage is needed since m_etaLearningRate is "volatile"
		// m_etaLearningRate 은 휘발성 이기때문에 두단게 저장이 필요하다
		m_etaLearningRate = eta;	//학습율
		
		int nLayers;
		NNLayer* pLayer = NULL;
		
		ar >> nLayers;	//층 개수

		//층을 반복하여 호출한다
		for ( int ii=0; ii<nLayers; ++ii )
		{
			pLayer = new NNLayer( _T(""), pLayer );	//층을 생성하고
			
			m_Layers.push_back( pLayer );			//추가하고
			
			pLayer->Serialize( ar );				//신경망을 탑재한다
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
// 신경망 층 클래스
//  NNLayer class definition

//==============================================================================
// 신경망 층 생성자
NNLayer::NNLayer() :
label( _T("") ), m_pPrevLayer( NULL )
{
	Initialize();
}

//==============================================================================
// 신경망 층 소멸자
NNLayer::NNLayer( LPCTSTR str, NNLayer* pPrev /* =NULL */ ) :
label( str ), m_pPrevLayer( pPrev )
{
	Initialize();
}

//==============================================================================
// 신경망 층 초기화
void NNLayer::Initialize()
{
	VectorWeights::iterator wit;
	VectorNeurons::iterator nit;
	
	for( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		delete *nit;
	}
	
	for( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		delete *wit;
	}
	
	m_Weights.clear();
	m_Neurons.clear();
	
	m_bFloatingPointWarning = false;
}


NNLayer::~NNLayer()
{
	// call Initialize(); makes sense if you think
	Initialize();
}

//==============================================================================
// 층의 모든 뉴런의 출력을 구한다
void NNLayer::Calculate()
{
	ASSERT( m_pPrevLayer != NULL );
	
	VectorNeurons::iterator nit;		//뉴런 반복자
	VectorConnections::iterator cit;	//연결 반복자
	
	double dSum;

	// 모든 뉴런을 반복하여
	for( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // 용어를 쉽게하기 위하여 to ease the terminology
		
		cit = n.m_Connections.begin();	//뉴런에서 첫번째 연결위치를 가져온다
		
		ASSERT( (*cit).WeightIndex < m_Weights.size() );

		// weight of the first connection is the bias; neuron is ignored
		// 첫번째 연결의 가중치는 편향이다; 뉴런은 무시한다
		dSum = m_Weights[ (*cit).WeightIndex ]->value;

		// 모든 연결을 반복하여 출력의 합을 구한다
		for ( cit++ ; cit<n.m_Connections.end(); cit++ )
		{
			ASSERT( (*cit).WeightIndex < m_Weights.size() );
			ASSERT( (*cit).NeuronIndex < m_pPrevLayer->m_Neurons.size() );

			dSum += ( m_Weights[ (*cit).WeightIndex ]->value ) * 
					( m_pPrevLayer->m_Neurons[ (*cit).NeuronIndex ]->output );
		}
		
		n.output = SIGMOID( dSum );	//출력을 시그모이드 적용
	}
}

//==============================================================================
// 뉴런층을 역전파 한다
void NNLayer::Backpropagate(std::vector<double>& dErr_wrt_dXn,		/* in */ 
							std::vector<double>& dErr_wrt_dXnm1,	/* out */
							std::vector<double>* thisLayerOutput,	// memorized values of this layer's output
							std::vector<double>* prevLayerOutput,	// memorized values of previous layer's output
							double etaLearningRate )
{
	// 명명법(NeuralNetwork 클래스에서 반복) nomenclature (repeated from NeuralNetwork class):
	//
	// Err	신경망 전체의 출력에러 is output error of the entire neural net
	// Xn	n번째층의 출력벡터 is the output vector on the n-th layer
	// Xnm1 이전층의 출력벡터 is the output vector of the previous layer
	// Wn	n번째층의 가중치 벡터 is the vector of weights of the n-th layer
	// Yn	n번째층의 활성화 값, 즉, 찌그림 함수 적용 전의 입력의 가중합
	//		is the activation value of the n-th layer, i.e., the weighted sum of inputs 
	//		BEFORE the squashing function is applied
	// F	찌그림 함수 is the squashing function: Xn = F(Yn)
	// F'	찌그림 함수의 편미분 is the derivative of the squashing function
	//		단순히, F = tanh 라면, 그러면 F'(Yn) = 1 - Xn^2,
	//		즉, 편미분은 출력으로부터 계산된다, 알고있는 입력 없이
	//		Conveniently, for F = tanh, then F'(Yn) = 1 - Xn^2,
	//		i.e., the derivative can be calculated from the output, without knowledge of the input

	ASSERT( dErr_wrt_dXn.size() == m_Neurons.size() );
	ASSERT( m_pPrevLayer != NULL );
	ASSERT( dErr_wrt_dXnm1.size() == m_pPrevLayer->m_Neurons.size() );
	
	UINT ii, jj;	//  [2015.12.26 12:21 Jobs]
	UINT kk;
	UINT nIndex;	//  [2015.12.26 12:21 Jobs]
	double output;
	
	std::vector< double > dErr_wrt_dYn( m_Neurons.size() );
	//
	// std::vector<double> dErr_wrt_dWn( m_Weights.size(), 0.0 );  // important to initialize to zero
	//////////////////////////////////////////////////
	//
	///// DESIGN TRADEOFF: REVIEW !!
	// We would prefer (for ease of coding) to use STL vector for the array "dErr_wrt_dWn", which is the 
	// differential of the current pattern's error wrt weights in the layer.  However, for layers with
	// many weights, such as fully-connected layers, there are also many weights.  The STL vector
	// class's allocator is remarkably stupid when allocating large memory chunks, and causes a remarkable 
	// number of page faults, with a consequent slowing of the application's overall execution time.
	// 우리는 "dErr_wrt_dWn" 배열을 위하여 STL 벡터 사용을 선호한다, 층에서 현재
	// 패턴들의 에러 wrt 가중치의 미분이다. 하지만, 많은 가중치와 층을 위하여,
	// 이를테면 완전히 연결된 층, 이는 또한 많은 가중치가 있다. STL 벡터 클래스
	// 할당자는 대용량 메모리 덩어리를 할당할 때 현저하게 둔하다, 그리고 페이지
	// 결함의 뚜렷한 원인이다, 응용프로그램의 전체 실행 시간의 둔화의 결과를 동반한다.

	// To fix this, I tried using a plain-old C array, by new'ing the needed space from the heap, and 
	// delete[]'ing it at the end of the function. However, this caused the same number of page-fault
	// errors, and did not improve performance.
	// 이 문제를 해결하기 위해, 나는 기존 C 배열을 시도했다, 힙에서 필요한 공간을
	// new'ing 하여, 그리고 함수의 종료시 delete[]'ing 한다. 하지만, 이것은 페이지
	// 결함 오류의 동일한 하나의 원인이다, 그리고 성능이 향상되지 않았다.
	
	// So I tried a plain-old C array allocated on the stack (i.e., not the heap).  Of course I could not
	// write a statement like 
	//    double dErr_wrt_dWn[ m_Weights.size() ];
	// since the compiler insists upon a compile-time known constant value for the size of the array.  
	// To avoid this requirement, I used the _alloca function, to allocate memory on the stack.
	// The downside of this is excessive stack usage, and there might be stack overflow probelms.  That's why
	// this comment is labeled "REVIEW"
	// 그래서 나는 스텍에 기존 C 배열 할당을 시도했다(즉, 힙이 아니다). 물론
	// 내가 좋아하는 문장 double dErr_wrt_dWn[ m_Weights.size() ] 을 쓸수 없었다;
	// 컴파일러는 배열의 크기에 대하여 알려진 상수값을 컴파일시간을 우선하기
	// 때문이다. 이 요구를 방지하기 위해, 나는 _alloca 함수를 사용했다, 스택에
	// 메모리를 할당하기 위해. 이것의 단점은 과도한 스택 사용이다, 그리고
	// 스택 오버플로우 문제가 있을 수 있다. 그래서 이 주석을 "REVIEW"로 붙였다.
	
	double* dErr_wrt_dWn = (double*)( _alloca( sizeof(double) *  m_Weights.size() ) );
	
	for ( ii=0; ii<m_Weights.size(); ++ii )
	{
		dErr_wrt_dWn[ ii ] =0.0;
	}

	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;

	BOOL bMemorized = ( thisLayerOutput != NULL ) && ( prevLayerOutput != NULL );

	// 식(3) calculate dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn
	for ( ii=0; ii<m_Neurons.size(); ++ii )
	{
		ASSERT( ii<dErr_wrt_dYn.size() );
		ASSERT( ii<dErr_wrt_dXn.size() );
		
		if ( bMemorized != FALSE )
		{
			output = (*thisLayerOutput)[ ii ];
		}
		else
		{
			output = m_Neurons[ ii ]->output;
		}
		
		dErr_wrt_dYn[ ii ] = DSIGMOID( output ) * dErr_wrt_dXn[ ii ];
	}
	
	// 식(4) calculate dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn
	// For each neuron in this layer, go through the list of connections from the prior layer, and
	// update the differential for the corresponding weight
	// 이 층에서 각 뉴런을 위하여, 이전 층에서 연결의 목록을 통해 진행한다,
	// 그리고 해당하는 가중치를 위하여 미분을 갱신한다

	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron &n = *(*nit);  // 용어를 쉽게하기 위하여 for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk = (*cit).NeuronIndex;
			if ( kk == ULONG_MAX )
			{
				output = 1.0;  // 편향 가중치 이다 this is the bias weight
			}
			else
			{
				ASSERT( kk<m_pPrevLayer->m_Neurons.size() );
				
				if ( bMemorized != FALSE )
				{
					output = (*prevLayerOutput)[ kk ];
				}
				else
				{
					output = m_pPrevLayer->m_Neurons[ kk ]->output;
				}
			}
			
			// since after changing dErr_wrt_dWn to a C-style array,
			// the size() function this won't work
			// ASSERT( (*cit).WeightIndex < dErr_wrt_dWn.size() );
			ASSERT( ii<dErr_wrt_dYn.size() );
			dErr_wrt_dWn[ (*cit).WeightIndex ] += dErr_wrt_dYn[ ii ] * output;
		}
		
		ii++;
	}

	// 식(5) calculate dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn,
	// which is needed as the input value of
	// dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
	// For each neuron in this layer
	// dErr_wrt_Xn의 입력값이 필요하다
	// 다음층(즉, 이전)의 역전파를 위하여
	// 이 층에서 각 뉴런을 위하여
	
	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron &n = *(*nit);  // 용어를 쉽게하기 위하여 for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk=(*cit).NeuronIndex;
			if ( kk != ULONG_MAX )
			{
				// we exclude ULONG_MAX, which signifies the phantom bias neuron with
				// constant output of "1", since we cannot train the bias neuron
				// ULONG_MAX는 제외한다, 이는 "1"의 고정출력으로 가상 편향뉴런을
				// 의미한다, 편향뉴런을 훈련할 수 없기 때문이다
				
				nIndex = kk;
				
				ASSERT( nIndex<dErr_wrt_dXnm1.size() );
				ASSERT( ii<dErr_wrt_dYn.size() );
				ASSERT( (*cit).WeightIndex<m_Weights.size() );
				
				dErr_wrt_dXnm1[ nIndex ] += dErr_wrt_dYn[ ii ] * m_Weights[ (*cit).WeightIndex ]->value;
			}
		}
		
		ii++;  // ii 뉴런 반복자를 추적한다 ii tracks the neuron iterator
	}
	
	struct DOUBLE_UNION
	{
		union 
		{
			double dd;
			unsigned __int64 ullong;
		};
	};
	
	DOUBLE_UNION oldValue, newValue;
	
	// finally, update the weights of this layer neuron using dErr_wrt_dW and the learning rate eta
	// Use an atomic compare-and-exchange operation, which means that another thread might be in 
	// the process of backpropagation and the weights might have shifted slightly
	// 끝으로, dErr_wrt_dW를 사용 이 층의 뉴런의 가중치를 갱신한다 그리고
	// 학습율 eta는 극소의 비교 및 교환 작업을 사용한다, 이는 다른 스레드가
	// 역전파의 과정에 있을 수 있다는 것을 의미하고 가중치는 약간 변화됐을 수 있다
	
	double dMicron = ::GetPreferences().m_dMicronLimitParameter;
	double epsilon, divisor;
	
	for ( jj=0; jj<m_Weights.size(); ++jj )
	{
		divisor = m_Weights[ jj ]->diagHessian + dMicron ; 
		
		// 공약수로 현재의 대각 헤시안 정보를 나누기 위해 각 층을 제어한다
		// 확인은 또한 각 헤시안이 오로지 양의 0을 보장하기 위한 것이다
		
		// the following code has been rendered unnecessary, since the value of the Hessian has been
		// verified when it was created, so as to ensure that it is strictly
		// zero-positive.  Thus, it is impossible for the diagHessian to be less than zero,
		// and it is impossible for the divisor to be less than dMicron
		// 다음코드는 불필요하다, 이것은 생성되었을 때 헤시안 값이 확인되었기
		// 때문이다, 그래서 오로지 양의 0을 보장하기 위한 것이다. 그러므로,
		// diagHessian 이 0 보다 작아지면 안된다 그리고 divisor 가 dMicron 보다
		// 작아지면 안된다
		/*
		if ( divisor < dMicron )  
		{
		// it should not be possible to reach here, since everything in the second derviative equations 
		// is strictly zero-positive, and thus "divisor" should definitely be as large as MICRON.
		
		  ASSERT( divisor >= dMicron );
		  divisor = 1.0 ;  // this will limit the size of the update to the same as the size of gloabal eta
		  }
		*/
		
		epsilon = etaLearningRate / divisor;
		oldValue.dd = m_Weights[ jj ]->value;
		newValue.dd = oldValue.dd - epsilon * dErr_wrt_dWn[ jj ];
		
		
		while ( oldValue.ullong !=
			_InterlockedCompareExchange64((unsigned __int64*)(&m_Weights[ jj ]->value), 
										   newValue.ullong, oldValue.ullong ) ) 
		{
			// another thread must have modified the weight.  Obtain its new value, adjust it, and try again
			// 다른 쓰레드는 가중치를 수정해야 한다. 새로운 값을 구하고, 조정하고,
			// 그리고 다시 시도한다
			oldValue.dd = m_Weights[ jj ]->value;
			newValue.dd = oldValue.dd - epsilon * dErr_wrt_dWn[ jj ];
		}
	}
}


//==============================================================================
//
void NNLayer::PeriodicWeightSanityCheck()
{
	// called periodically by the neural net, to request a check on the "reasonableness" of the 
	// weights.  The warning message is given only once per layer
	// 신경망으로 주기적 호출, 가중치의 "적당함"을 확인하기 위해.
	// 경고 메시지는 층당 한번만 한다
	
	VectorWeights::iterator wit;
	
	for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		NNWeight& ww = *(*wit);
		double val = fabs( ww.value );
		
		if ( (val>100.0) && (m_bFloatingPointWarning == false) )
		{
			// 100.0 is an arbitrary value, that no reasonable weight should ever exceed
			// 100.0은 임의의 값이다, 적당한 가중치는 절대 초과하지 않는다

			CString strMess;
			strMess.Format( _T( "주의: 가중치가 무한히 커졌다 \n" )
				_T( "층: %s \n가중치: %s \n가중치 값 = %g \n가중치 헤시안 = %g\n\n" )
				_T( "이것은 역전차와 조사를 그만두는 것을 말한다" ),
				label.c_str(), ww.label.c_str(), ww.value, ww.diagHessian );
			
			::MessageBox( NULL, strMess, _T( "가중치 문제" ), MB_ICONEXCLAMATION | MB_OK );
			
			m_bFloatingPointWarning = true;
		}
	}
}


//==============================================================================
//
void NNLayer::EraseHessianInformation()
{
	// goes through all the weights associated with this layer, and sets each of their
	// diagHessian value to zero
	// 이 층과 관련된 모든 가중치를 통해 이뤄진다, 그리고 
	// diagHessian의 값을 각각 0으로 설정한다
	
	VectorWeights::iterator wit;
	
	for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		(*wit)->diagHessian = 0.0;
	}
	
}


//==============================================================================
//
void NNLayer::DivideHessianInformationBy(double divisor)
{
	// goes through all the weights associated with this layer, and divides each of their
	// diagHessian value by the indicated divisor
	// 이 층과 관련된 모든 가중치를 통해 이뤄진다, 그리고 
	// diagHessian의 값을 각각 지정된 제수로 나눈다

	VectorWeights::iterator wit;
	double dTemp;
	
	for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
	{
		dTemp = (*wit)->diagHessian;
		
		if ( dTemp < 0.0 )
		{
			// it should not be possible to reach here, since all calculations for the second
			// derviative are strictly zero-positive.  However, there are some early indications 
			// that this check is necessary anyway
			// 여기에 도달하지 않아야 한다, 온전한 양의 0은 두번째 편미분을 위해
			// 모두 계산하기 때문이다. 하지만, 어쨌든 이 확인에 필요한 일부 초기 징후가 있다
			
			ASSERT ( dTemp >= 0.0 );  // 디버그 모드에서 중단된다 will break in debug mode
			dTemp = 0.0;
		}
		
		(*wit)->diagHessian = dTemp / divisor ;
	}
}


//==============================================================================
//
void NNLayer::BackpropagateSecondDerivatives(std::vector<double>& d2Err_wrt_dXn /* in */, 
											 std::vector<double>& d2Err_wrt_dXnm1 /* out */)
{
	// 명명법(NeuralNetwork 클래스에서 반복) nomenclature (repeated from NeuralNetwork class):
	// NOTE: even though we are addressing SECOND derivatives ( and not first derivatives),
	// we use nearly the same notation as if there were first derivatives, since otherwise the
	// ASCII look would be confusing.  We add one "2" but not two "2's", such as "d2Err_wrt_dXn",
	// to give a gentle emphasis that we are using second derivatives
	//
	// Err	신경망 전체의 출력에러 is output error of the entire neural net
	// Xn	n번째층의 출력벡터 is the output vector on the n-th layer
	// Xnm1 이전층의 출력벡터 is the output vector of the previous layer
	// Wn	n번째층의 가중치 벡터 is the vector of weights of the n-th layer
	// Yn	n번째층의 활성화 값, 즉, 찌그림 함수 적용 전의 입력의 가중합
	//		is the activation value of the n-th layer, i.e., the weighted sum of inputs 
	//		BEFORE the squashing function is applied
	// F	찌그림 함수 is the squashing function: Xn = F(Yn)
	// F'	찌그림 함수의 편미분 is the derivative of the squashing function
	//		단순히, F = tanh 라면, 그러면 F'(Yn) = 1 - Xn^2,
	//		즉, 편미분은 출력으로부터 계산된다, 알고있는 입력 없이
	//		Conveniently, for F = tanh, then F'(Yn) = 1 - Xn^2,
	//		i.e., the derivative can be calculated from the output, without knowledge of the input
	
	ASSERT( d2Err_wrt_dXn.size() == m_Neurons.size() );
	ASSERT( m_pPrevLayer != NULL );
	ASSERT( d2Err_wrt_dXnm1.size() == m_pPrevLayer->m_Neurons.size() );

	UINT ii, jj;	//  [2015.12.26 12:22 Jobs]
	UINT kk;
	UINT nIndex;	//  [2015.12.26 12:22 Jobs]
	double output;
	double dTemp;
		
	std::vector<double> d2Err_wrt_dYn( m_Neurons.size() );
	//
	// std::vector<double> d2Err_wrt_dWn( m_Weights.size(), 0.0 );  // important to initialize to zero
	//////////////////////////////////////////////////
	//
	///// DESIGN TRADEOFF: REVIEW !!
	//
	// Note that the reasoning of this comment is identical to that in the NNLayer::Backpropagate() 
	// function, from which the instant BackpropagateSecondDerivatives() function is derived from
	//
	// We would prefer (for ease of coding) to use STL vector for the array "d2Err_wrt_dWn", which is the 
	// second differential of the current pattern's error wrt weights in the layer.  However, for layers with
	// many weights, such as fully-connected layers, there are also many weights.  The STL vector
	// class's allocator is remarkably stupid when allocating large memory chunks, and causes a remarkable 
	// number of page faults, with a consequent slowing of the application's overall execution time.
	// 우리는 "dErr_wrt_dWn" 배열을 위하여 STL 벡터 사용을 선호한다, 층에서 현재
	// 패턴들의 에러 wrt 가중치의 미분이다. 하지만, 많은 가중치와 층을 위하여,
	// 이를테면 완전히 연결된 층, 이는 또한 많은 가중치가 있다. STL 벡터 클래스
	// 할당자는 대용량 메모리 덩어리를 할당할 때 현저하게 둔하다, 그리고 페이지
	// 결함의 뚜렷한 원인이다, 응용프로그램의 전체 실행 시간의 둔화의 결과를 동반한다.
	
	// To fix this, I tried using a plain-old C array, by new'ing the needed space from the heap, and 
	// delete[]'ing it at the end of the function.  However, this caused the same number of page-fault
	// errors, and did not improve performance.
	// 이 문제를 해결하기 위해, 나는 기존 C 배열을 시도했다, 힙에서 필요한 공간을
	// new'ing 하여, 그리고 함수의 종료시 delete[]'ing 한다. 하지만, 이것은 페이지
	// 결함 오류의 동일한 하나의 원인이다, 그리고 성능이 향상되지 않았다.
	
	// So I tried a plain-old C array allocated on the stack (i.e., not the heap).  Of course I could not
	// write a statement like 
	//    double d2Err_wrt_dWn[ m_Weights.size() ];
	// since the compiler insists upon a compile-time known constant value for the size of the array.  
	// To avoid this requirement, I used the _alloca function, to allocate memory on the stack.
	// The downside of this is excessive stack usage, and there might be stack overflow probelms.  That's why
	// this comment is labeled "REVIEW"
	// 그래서 나는 스텍에 기존 C 배열 할당을 시도했다(즉, 힙이 아니다). 물론
	// 내가 좋아하는 문장 double dErr_wrt_dWn[ m_Weights.size() ] 을 쓸수 없었다;
	// 컴파일러는 배열의 크기에 대하여 알려진 상수값을 컴파일시간을 우선하기
	// 때문이다. 이 요구를 방지하기 위해, 나는 _alloca 함수를 사용했다, 스택에
	// 메모리를 할당하기 위해. 이것의 단점은 과도한 스택 사용이다, 그리고
	// 스택 오버플로우 문제가 있을 수 있다. 그래서 이 주석을 "REVIEW"로 붙였다.

	double* d2Err_wrt_dWn = (double*)( _alloca( sizeof(double) *  m_Weights.size() ) );
	
	for ( ii=0; ii<m_Weights.size(); ++ii )
	{
		d2Err_wrt_dWn[ ii ] =0.0;
	}

	VectorNeurons::iterator nit;
	VectorConnections::iterator cit;

	// calculate d2Err_wrt_dYn = ( F'(Yn) )^2 * dErr_wrt_Xn (where dErr_wrt_Xn is actually a second derivative )
	
	for ( ii=0; ii<m_Neurons.size(); ++ii )
	{
		ASSERT( ii<d2Err_wrt_dYn.size() );
		ASSERT( ii<d2Err_wrt_dXn.size() );
		
		output = m_Neurons[ ii ]->output;
		
		dTemp = DSIGMOID( output ) ;
		d2Err_wrt_dYn[ ii ] = d2Err_wrt_dXn[ ii ] * dTemp * dTemp;
	}
	
	// calculate d2Err_wrt_Wn = ( Xnm1 )^2 * d2Err_wrt_Yn (where dE2rr_wrt_Yn is actually a second derivative)
	// For each neuron in this layer, go through the list of connections from the prior layer, and
	// update the differential for the corresponding weight
	
	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk = (*cit).NeuronIndex;
			if ( kk == ULONG_MAX )
			{
				output = 1.0;  // this is the bias connection; implied neuron output of "1"
			}
			else
			{
				ASSERT( kk<m_pPrevLayer->m_Neurons.size() );
				
				output = m_pPrevLayer->m_Neurons[ kk ]->output;
			}
			
			////////////	ASSERT( (*cit).WeightIndex < d2Err_wrt_dWn.size() );  // since after changing d2Err_wrt_dWn to a C-style array, the size() function this won't work
			ASSERT( ii<d2Err_wrt_dYn.size() );
			d2Err_wrt_dWn[ (*cit).WeightIndex ] += d2Err_wrt_dYn[ ii ] * output * output ;
		}
		
		ii++;
	}
	
	
	// calculate d2Err_wrt_Xnm1 = ( Wn )^2 * d2Err_wrt_dYn (where d2Err_wrt_dYn is a second derivative not a first).
	// d2Err_wrt_Xnm1 is needed as the input value of
	// d2Err_wrt_Xn for backpropagation of second derivatives for the next (i.e., previous spatially) layer
	// For each neuron in this layer
	
	ii = 0;
	for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
	{
		NNNeuron& n = *(*nit);  // for simplifying the terminology
		
		for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
		{
			kk=(*cit).NeuronIndex;
			if ( kk != ULONG_MAX )
			{
				// we exclude ULONG_MAX, which signifies the phantom bias neuron with
				// constant output of "1", since we cannot train the bias neuron
				
				nIndex = kk;
				
				ASSERT( nIndex<d2Err_wrt_dXnm1.size() );
				ASSERT( ii<d2Err_wrt_dYn.size() );
				ASSERT( (*cit).WeightIndex<m_Weights.size() );
				
				dTemp = m_Weights[ (*cit).WeightIndex ]->value ; 
				
				d2Err_wrt_dXnm1[ nIndex ] += d2Err_wrt_dYn[ ii ] * dTemp * dTemp ;
			}
			
		}
		
		ii++;  // ii tracks the neuron iterator
		
	}
	
	struct DOUBLE_UNION
	{
		union 
		{
			double dd;
			unsigned __int64 ullong;
		};
	};
	
	DOUBLE_UNION oldValue, newValue;
	
	// finally, update the diagonal Hessians for the weights of this layer neuron using dErr_wrt_dW.
	// By design, this function (and its iteration over many (approx 500 patterns) is called while a 
	// single thread has locked the nueral network, so there is no possibility that another
	// thread might change the value of the Hessian.  Nevertheless, since it's easy to do, we
	// use an atomic compare-and-exchange operation, which means that another thread might be in 
	// the process of backpropagation of second derivatives and the Hessians might have shifted slightly
	
	for ( jj=0; jj<m_Weights.size(); ++jj )
	{
		oldValue.dd = m_Weights[ jj ]->diagHessian;
		newValue.dd = oldValue.dd + d2Err_wrt_dWn[ jj ];
		
		while ( oldValue.ullong != _InterlockedCompareExchange64( (unsigned __int64*)(&m_Weights[ jj ]->diagHessian), 
			newValue.ullong, oldValue.ullong ) ) 
		{
			// another thread must have modified the weight.  Obtain its new value, adjust it, and try again
			
			oldValue.dd = m_Weights[ jj ]->diagHessian;
			newValue.dd = oldValue.dd + d2Err_wrt_dWn[ jj ];
		}
		
	}
	
}

//==============================================================================
// 층의 저장소 주소를 전달받아 층, 뉴런, 연결, 가중치 정보를 저장 또는 탑재
void NNLayer::Serialize(CArchive &ar)
{
	VectorNeurons::iterator		nit;	//뉴런 반복자
	VectorWeights::iterator		wit;	//가중치 반복자
	VectorConnections::iterator	cit;	//연결 반복자
	
	int ii, jj;
	
	if (ar.IsStoring())
	{
		// TODO: 저장코드는 여기에 추가 add storing code here

		ar.WriteString( label.c_str() );	//층이름
		// ar.ReadString will look for \r\n when loading from the archive
		// ar.ReadString은 저장소에서 탑재할때 \r\n을 찾는다
		ar.WriteString( _T("\r\n") );	//행바꿈
		ar << m_Neurons.size();	//뉴런 개수
		ar << m_Weights.size();	//가중치 개수

		//뉴런을 반복하면서 저장한다
		for ( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
		{
			NNNeuron& n = *(*nit);
			ar.WriteString( n.label.c_str() );	//뉴런 이름 : 층번호_뉴런번호_뉴런전체순번
			ar.WriteString( _T("\r\n") );	//행바꿈
			ar << n.m_Connections.size();	//연결유전자 개수

			//연결을 반복하면서 저장한다
			for ( cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
			{
				ar << (*cit).NeuronIndex;	//연결 뉴런 식별자
				ar << (*cit).WeightIndex;	//연결 가중치 식별자
			}
		}

		//가중치를 반복하면서 저장한다
		for ( wit=m_Weights.begin(); wit<m_Weights.end(); wit++ )
		{
			ar.WriteString( (*wit)->label.c_str() );	//가중치 이름 : 층번호_가중치번호_가중치전체순번
			ar.WriteString( _T("\r\n") );	//행바꿈
			ar << (*wit)->value;			//가중치
		}
	}
	else
	{
		// TODO: 탑재코드는 여기에 추가 add loading code here

		CString str;
		ar.ReadString( str );	//층이름을 탑재한다
		
		label = str;	//층이름을 저장한다
		
		int iNumNeurons, iNumWeights, iNumConnections;
		double value;
		
		NNNeuron* pNeuron;
		NNWeight* pWeight;
		NNConnection conn;
		
		ar >> iNumNeurons;	//뉴런 개수
		ar >> iNumWeights;	//가중치 개수
		
		//반복하면서 뉴런을 탑재한다
		for ( ii=0; ii<iNumNeurons; ++ii )
		{
			ar.ReadString( str );	//뉴런이름
			pNeuron = new NNNeuron( (LPCTSTR)str );	//뉴런이름으로 뉴런생성
			m_Neurons.push_back( pNeuron );	//뉴런을 추가한다
			
			ar >> iNumConnections;	//연결유전자 개수

			//반복하면서 연결을 탑재한다
			for ( jj=0; jj<iNumConnections; ++jj )
			{
				ar >> conn.NeuronIndex;	//연결 뉴런 식별자
				ar >> conn.WeightIndex;	//연결 가중치 식별자
				
				pNeuron->AddConnection( conn );	//연결정보를 추가한다
			}
		}
		
		//반복하면서 가중치를 탑재한다
		for ( jj=0; jj<iNumWeights; ++jj )
		{
			ar.ReadString( str );	//가중치 이름
			ar >> value;			//가중치 값
			
			pWeight = new NNWeight( (LPCTSTR)str, value ); //가중치 생성
			m_Weights.push_back( pWeight );	//가중치를 추가한다
		}
	}
}


///////////////////////////////////////////////////////////////////////
//
// NNWeight

//==============================================================================
// NNWeight 생성자
NNWeight::NNWeight()
	: label( _T("") )	//가중치 이름 : 층번호_가중치번호_가중치전체순번
	, value( 0.0 )
	, diagHessian( 0.0 )
{
	Initialize();
}

//==============================================================================
// NNWeight 생성자
NNWeight::NNWeight( LPCTSTR str, double val /* =0.0 */ )
	: label( str )	//가중치 이름 : 층번호_가중치번호_가중치전체순번
	, value( val )
	, diagHessian( 0.0 )
{
	Initialize();
}

//==============================================================================
// NNWeight 초기화
void NNWeight::Initialize()
{	
}

//==============================================================================
// NNWeight 소멸자
NNWeight::~NNWeight()
{
}


////////////////////////////////////////////////////////////////////////////////
//
// NNNeuron

//==============================================================================
// NNNeuron 생성자
NNNeuron::NNNeuron()
	: label( _T("") )	//뉴런 이름 : 층번호_뉴런번호_뉴런전체순번
	, output( 0.0 )
{
	Initialize();
}

//==============================================================================
// NNNeuron 생성자
NNNeuron::NNNeuron( LPCTSTR str )
	: label( str )	//뉴런 이름 : 층번호_뉴런번호_뉴런전체순번
	, output( 0.0 )
{
	Initialize();
}

//==============================================================================
// NNNeuron 초기화
void NNNeuron::Initialize()
{
	m_Connections.clear();
}

//==============================================================================
// NNNeuron 소멸자
NNNeuron::~NNNeuron()
{
	Initialize();
}

//==============================================================================
// NNConnection벡터에 연결정보(뉴런, 가중치)를 추가한다
void NNNeuron::AddConnection( UINT iNeuron, UINT iWeight )
{
	m_Connections.push_back( NNConnection( iNeuron, iWeight ) );
}

//==============================================================================
// NNConnection벡터에 인자주소의 연결정보(뉴런, 가중치)를 추가한다
void NNNeuron::AddConnection( NNConnection const & conn )
{
	m_Connections.push_back( conn );
}


