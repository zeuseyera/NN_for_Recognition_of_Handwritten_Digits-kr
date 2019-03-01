#if !defined(AFX_DLGBACKPROPPARAMETERS_H__58993721_5E08_4AB0_85D1_D00AE597CCC7__INCLUDED_)
#define AFX_DLGBACKPROPPARAMETERS_H__58993721_5E08_4AB0_85D1_D00AE597CCC7__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// DlgBackpropParameters.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CDlgBackpropParameters dialog

class CDlgBackpropParameters : public CDialog
{
// Construction
public:
	CDlgBackpropParameters(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
	//{{AFX_DATA(CDlgBackpropParameters)
	enum { IDD = IDD_DIALOG_BACKPROP_PARAMETERS };
	UINT	m_AfterEvery;				//n번 역전파이후 감소율을 적용한다
	double	m_EtaDecay;					//학습율에 대한 감소율
	double	m_InitialEta;				//초기 학습율
	double	m_MinimumEta;				//최소 학습율
	CString	m_strInitialEtaMessage;
	CString	m_strStartingPatternNum;
	UINT	m_StartingPattern;			//역전파 시작패턴 번고
	UINT	m_cNumThreads;				//역전파에 사용할 쓰레드 개수
	BOOL	m_bDistortPatterns;			//패턴왜곡 여부
	double	m_EstimatedCurrentMSE;		//현재 예상 MSE
	//}}AFX_DATA


// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CDlgBackpropParameters)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:

	// Generated message map functions
	//{{AFX_MSG(CDlgBackpropParameters)
		// NOTE: the ClassWizard will add member functions here
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_DLGBACKPROPPARAMETERS_H__58993721_5E08_4AB0_85D1_D00AE597CCC7__INCLUDED_)
