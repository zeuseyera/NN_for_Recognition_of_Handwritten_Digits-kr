// MNistDoc.h : interface of the CMNistDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_MNISTDOC_H__33D2ECF2_FA6C_44B8_A44A_004C2D073AA6__INCLUDED_)
#define AFX_MNISTDOC_H__33D2ECF2_FA6C_44B8_A44A_004C2D073AA6__INCLUDED_


// disable the template warning C4786 : identifier was truncated to '255' characters in the browser information

#pragma warning( push )
#pragma warning( disable : 4786 )

#include "NeuralNetwork.h"	// 클래스뷰로 추가 Added by ClassView

using namespace std;

#include <vector>

#include <afxmt.h>  // 중요 부분을 위하여, 멀티쓰레드 등 for critical section, multi-threaded etc

#define GAUSSIAN_FIELD_SIZE ( 21 )  // 오로지 홀수 strictly odd number

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000



class CMNistDoc : public COleDocument
{
protected: // create from serialization only
	CMNistDoc();
	DECLARE_DYNCREATE(CMNistDoc)

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CMNistDoc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	virtual void DeleteContents();
	virtual BOOL CanCloseFrame(CFrameWnd* pFrame);
	//}}AFX_VIRTUAL

// Implementation
public:

	void ApplyDistortionMap( double* inputVector );
	void GenerateDistortionMap( double severityFactor = 1.0 );
	double* m_DispH;  // 수평 왜곡맵 배열 horiz distortion map array
	double* m_DispV;  // 수직 왜곡맵 배열 vert distortion map array
	double m_GaussianKernel[ GAUSSIAN_FIELD_SIZE ] [ GAUSSIAN_FIELD_SIZE ];

	int m_cCols;  // 왜곡맵의 크기 size of the distortion maps
	int m_cRows;
	int m_cCount;
	// zero-based indices, starting at bottom-left
	inline double& At( double* p, int row, int col )
	{
		int location = row * m_cCols + col;
		ASSERT(location>=0 &&
			   location<m_cCount &&
			   row<m_cRows &&
			   row>=0 &&
			   col<m_cCols
			   && col>=0 );
		return p[ location ];
	}


	double GetCurrentEta();
	double GetPreviousEta();
	void BackpropagateNeuralNet(double* inputVector,
								int iCount,
								double* targetOutputVector, 
								double* actualOutputVector,
								int oCount, 
								std::vector<std::vector<double>>* pMemorizedNeuronOutputs, 
								BOOL bDistort);
	void CalculateNeuralNet(double* inputVector,
							int count,
							double* outputVector = NULL, 
							int oCount = 0,
							std::vector<std::vector<double>>* pNeuronOutputs = NULL,
							BOOL bDistort = FALSE);
	void CalculateHessian();
	// mutex to guard access to neural net; the MFC CMutex is not necessarily trustworthy
	// 신경망의 접근을 보호하기 위한 상호배제개체
	HANDLE m_utxNeuralNet;

	//--------------------------------------------------------------------------
	// 역전파와 훈련에 관련된 멤버 backpropagation and training-related members

	volatile UINT m_cBackprops;		//역전파중인 패턴위치
	volatile BOOL m_bNeedHessian;
	
	HWND m_hWndForBackpropPosting;
	UINT m_nAfterEveryNBackprops;
	double m_dEtaDecay;
	double m_dMinimumEta;
	// this number will be changed by one thread and used by others
	// 이 수는 쓰레드에 의해 변경되고 다른것에 의해 사용된다
	volatile double m_dEstimatedCurrentMSE;

	// static member used by threads to identify themselves
	// 정적 멤버는 자신을 식별하기 위하여 쓰레드에 의해 사용된다
	static UINT m_iBackpropThreadIdentifier;	//쓰레드 번호
	
	UINT m_iNumBackpropThreadsRunning;		//실행중인 쓰레드 개수
	CWinThread* m_pBackpropThreads[100];	//쓰레드 포인터[배열]
	BOOL m_bDistortTrainingPatterns;		//훈련패턴 왜곡
	BOOL m_bBackpropThreadsAreRunning;		//TRUE 있음, FALSE 없음
	volatile BOOL m_bBackpropThreadAbortFlag;//FALSE 실행, TRUE 멈춤
	void StopBackpropagation();
	BOOL StartBackpropagation(UINT		iStartPattern = 0,
							  UINT		iNumThreads = 2,
							  HWND		hWnd = NULL,
							  double	initialEta = 0.005,
							  double	minimumEta = 0.000001,
							  double	etaDecay = 0.990,
							  UINT		nAfterEvery = 1000,
							  BOOL		bDistortPatterns = TRUE,
							  double	estimatedCurrentMSE = 1.0 );
	static UINT BackpropagationThread( LPVOID pVoid);

	// critical section for "get next pattern"-like operations;
	// the MFC CCriticalSection is only marginally more convenient
	// "get next pattern"과 같은 동작에 대한 중요 부분
	// MFC CCriticalSection은 단지 조금 더 편리하다
	CRITICAL_SECTION m_csTrainingPatterns;

	volatile UINT m_iNextTrainingPattern;
	volatile UINT m_iRandomizedTrainingPatternSequence[ 60000 ];
	void RandomizeTrainingPatternSequence();
	UINT GetCurrentTrainingPatternNumber( BOOL bFromRandomizedPatternSequence = FALSE );
	// 배열에 훈련패턴을 가져온다
	void GetTrainingPatternArrayValues( int iNumImage = 0,				//이미지 고유번호
										unsigned char* pArray = NULL,	//이미지 저장 배열
										int* pLabel = NULL,				//이미지 라벨
										BOOL bFlipGrayscale = TRUE );
	UINT GetNextTrainingPattern(unsigned char* pArray = NULL,
								int* pLabel = NULL,
								BOOL bFlipGrayscale = TRUE,
								BOOL bFromRandomizedPatternSequence = TRUE,
								UINT* iSequenceNum = NULL );
	UINT GetRandomTrainingPattern(unsigned char* pArray=NULL,
								  int* pLabel=NULL,
								  BOOL bFlipGrayscale=TRUE);

	//--------------------------------------------------------------------------
	// 시험과 관련된 멤버 testing-related members

	UINT m_iNumTestingThreadsRunning;
	CWinThread* m_pTestingThreads[100];
	BOOL m_bDistortTestingPatterns;
	BOOL m_bTestingThreadsAreRunning;
	HWND m_hWndForTestingPosting;
	// 0 == training set; 1 == testing set (which is the default
	UINT m_iWhichImageSet;

	// static member used by threads to identify themselves
	// 정적 멤버는 자신을 식별하기 위하여 쓰레드에 의해 사용된다
	static UINT m_iTestingThreadIdentifier;

	static UINT TestingThread( LPVOID pVoid );
	void StopTesting();
	BOOL StartTesting(UINT iStartingPattern,
					  UINT iNumThreads,
					  HWND hWnd,
					  BOOL bDistortPatterns,
					  UINT iWhichImageSet = 1 );
	volatile BOOL m_bTestingThreadAbortFlag;

	// critical section for "get next pattern"-like operations;
	// the MFC CCriticalSection is only marginally more convenient
	// "get next pattern"과 같은 동작에 대한 중요 부분
	// MFC CCriticalSection은 단지 조금 더 편리하다
	CRITICAL_SECTION m_csTestingPatterns;
	
	volatile UINT m_iNextTestingPattern;
	UINT GetNextTestingPatternNumber();
	void GetTestingPatternArrayValues(int iNumImage = 0,			//이미지 고유번호
									  unsigned char* pArray = NULL,	//이미지 저장 배열
									  int* pLabel = NULL,			//이미지 라벨
									  BOOL bFlipGrayscale = TRUE );

	// returns TRUE to signify roll-over back to zero-th pattern
	UINT GetNextTestingPattern( unsigned char* pArray=NULL,
								int* pLabel=NULL,
								BOOL bFlipGrayscale=TRUE);
	
	CFile m_fileTrainingLabels;	//훈련라벨 파일
	CFile m_fileTrainingImages;	//훈련이미지 파일
	CFile m_fileTestingLabels;	//시험라벨 파일
	CFile m_fileTestingImages;	//시험이미지 파일
	void CloseMnistFiles();
	BOOL m_bFilesOpen;
	virtual ~CMNistDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:
	NeuralNetwork m_NN;

protected:
	//--------------------------------------------------------------------------
	// CAutoCS: a helper class for automatically locking and unlocking a critical section
	// CAutoCS: 자동으로 중요부분을 잠금과 잠금해제를 위한 도우미 클래스
	class CAutoCS
	{
	public:
		CAutoCS(CRITICAL_SECTION& rcs) : m_rcs(rcs)
		{ ::EnterCriticalSection( &m_rcs ); }
		virtual ~CAutoCS() { ::LeaveCriticalSection( &m_rcs ); }
	private:
		CRITICAL_SECTION& m_rcs;
	};	// class CAutoCS

	//--------------------------------------------------------------------------
	// class CAutoMutex: a helper class for automatically obtaining and releasing ownership of a mutex
	// CAutoMutex: 자동으로 상호배제개체의 소유권을 획득과 해제를 위한 도우미 클래스
	class CAutoMutex
	{
	public:
		CAutoMutex( HANDLE& rutx ) : m_rutx( rutx )
		{  ASSERT( rutx != NULL );
		   DWORD dwRet = ::WaitForSingleObject( m_rutx, INFINITE );
		   ASSERT( dwRet == WAIT_OBJECT_0 );
		}
		virtual ~CAutoMutex() { ::ReleaseMutex( m_rutx ); }
	private:
		HANDLE& m_rutx;
	};	// CAutoMutex



// Generated message map functions
protected:
	//{{AFX_MSG(CMNistDoc)
	afx_msg void OnButtonOpenMnistFiles();
	afx_msg void OnButtonCloseMnistFiles();

	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

// re-enable warning C4786 re : identifier was truncated to '255' characters in the browser information

#pragma warning( pop )


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_MNISTDOC_H__33D2ECF2_FA6C_44B8_A44A_004C2D073AA6__INCLUDED_)
