// Preferences.h: interface for the CPreferences class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_PREFERENCES_H__47D8F8EB_8E88_4B05_B51E_14332D5DD2EC__INCLUDED_)
#define AFX_PREFERENCES_H__47D8F8EB_8E88_4B05_B51E_14332D5DD2EC__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <afxcoll.h>  // for CStringList


class CPreferences  
{
public:
	int		m_cNumBackpropThreads;		//역전파에 사용할 쓰레드 개수

	int		m_nMagicTrainingLabels;		//훈련라벨 번호
	int		m_nMagicTrainingImages;		//훈련이미지 번호

	UINT	m_nItemsTrainingLabels;		//훈련라벨 개수
	UINT	m_nItemsTrainingImages;		//훈련이미지 개수

	int		m_cNumTestingThreads;		//시험에 사용할 쓰레드 개수

	int		m_nMagicTestingLabels;		//시험라벨 번호
	int		m_nMagicTestingImages;		//시험이미지 번호

	UINT	m_nItemsTestingLabels;		//시험라벨 개수
	UINT	m_nItemsTestingImages;		//시험이미지 개수

	int		m_nRowsImages;	//이미지 행수
	int		m_nColsImages;	//이미지 열수

	int		m_nMagWindowSize;			//돋보기 창크기
	int		m_nMagWindowMagnification;	//돋보기 배율

	double	m_dInitialEtaLearningRate;	//초기 학습율
	double	m_dLearningRateDecay;		//학습율에 대한 감소율
	double	m_dMinimumEtaLearningRate;	//최소 학습율
	UINT	m_nAfterEveryNBackprops;	//n번 역전파이후 감소율을 적용한다

	// for limiting the step size in backpropagation, since we are using second order
	// "Stochastic Diagonal Levenberg-Marquardt" update algorithm.  See Yann LeCun 1998
	// "Gradianet-Based Learning Applied to Document Recognition" at page 41
	// 역전파에서 보폭 제한을 위하여, 우리가 사용하는 두번째 갱신(확률 대각
	// Levenberg-Marquardt) 알고리즘이기 때문에

	double	m_dMicronLimitParameter;	//학습율 확장에대한 제수(미크론) 제한
	UINT	m_nNumHessianPatterns;		//패턴번호는 헤시안 계산에 사용한다

	// for distortions of the input image, in an attempt to improve generalization
	// 입력이미지의 왜곡을 위하여, 개선하기 위한 정규화의 범위 

	double m_dMaxScaling;  // as a percentage, such as 20.0 for plus/minus 20%
	double m_dMaxRotation;  // in degrees, such as 20.0 for plus/minus rotations of 20 degrees
	double m_dElasticSigma;  // one sigma value for randomness in Simard's elastic distortions
	double m_dElasticScaling;  // after-smoohting scale factor for Simard's elastic distortions
	
	void ReadIniFile(CWinApp* pApp);
	CWinApp* m_pMainApp;
	CPreferences();
	virtual ~CPreferences();

protected:
	void Get( LPCTSTR strSection, LPCTSTR strEntry, UINT& uiVal );
	void Get( LPCTSTR strSection, LPCTSTR strEntry, int &iVal );
	void Get( LPCTSTR strSection, LPCTSTR strEntry, float &fVal );
	void Get( LPCTSTR strSection, LPCTSTR strEntry, double &dVal );
	void Get( LPCTSTR strSection, LPCTSTR strEntry, LPTSTR pStrVal );
	void Get( LPCTSTR strSection, LPCTSTR strEntry, CString &strVal );
	void Get( LPCTSTR strSection, LPCTSTR strEntry, bool &bVal );

};

#endif // !defined(AFX_PREFERENCES_H__47D8F8EB_8E88_4B05_B51E_14332D5DD2EC__INCLUDED_)
