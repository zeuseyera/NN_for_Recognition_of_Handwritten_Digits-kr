// DlgCharacterImage.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "MNist.h"
#include "DlgCharacterImage.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

//------------------------------------------------------------------------------
// CDlgCharacterImage 대화 상자입니다.


CDlgCharacterImage::CDlgCharacterImage(CWnd* pParent /*=NULL*/)
	: CDialog(CDlgCharacterImage::IDD, pParent)
	, m_pDoc( NULL )
{
	//{{AFX_DATA_INIT(CDlgCharacterImage)
	// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
}


void CDlgCharacterImage::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CDlgCharacterImage)
	DDX_Control(pDX, IDC_RADIO_TRAINING_SET, m_ctlRadioWhichSet);
	DDX_Control(pDX, IDC_CHECK_PATTERN_DISTORTION, m_ctlCheckDistortInputPattern);
	DDX_Control(pDX, IDC_EDIT_VALUE, m_ctlEditLabelValue);
	DDX_Control(pDX, IDC_EDIT_IMAGE_NUM, m_ctlEditImageNumber);
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CDlgCharacterImage, CDialog)
	//{{AFX_MSG_MAP(CDlgCharacterImage)
	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_BUTTON_GET, OnButtonGetImageData)
	ON_BN_CLICKED(IDC_BUTTON_NEXT, OnButtonGetNextImageData)
	ON_BN_CLICKED(IDC_BUTTON_PREVIOUS, OnButtonGetPreviousImageData)
	ON_BN_CLICKED(IDC_BUTTON_NN_CALCULATE, OnButtonNnCalculate)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

//******************************************************************************
// CDlgCharacterImage 메시지 처리기입니다.

//==============================================================================
//
BOOL CDlgCharacterImage::OnInitDialog() 
{
	CDialog::OnInitDialog();

	ASSERT( m_pDoc != NULL );

	// create the character image window, using static placeholder from the dialog template,
	// and set its size to the global size of the character image
	// 문자 이미지창 생성
	// 대화상자 템플릿에서 정적 placeholder 사용
	// 문자이미지의 전체 크기로 크기 설정
	CRect rcPlace;
	CWnd* pPlaceholder = GetDlgItem( IDC_CHARACTER_IMAGE );
	if ( pPlaceholder != NULL )
	{
		// 화면 좌표에 in screen coords
		pPlaceholder->GetWindowRect( &rcPlace );
		// 화면에서 이 윈도우 좌표로 맵 map from screen to this window's coords
		::MapWindowPoints( NULL, m_hWnd, (POINT*)&rcPlace, 2 );

		// WS_EX_STATIC 가장자리의 폭을 +2 허용 +2 allows for the width of WS_EX_STATIC edge
		rcPlace.right = rcPlace.left + g_cImageSize+2;
		// +2 allows for the width of WS_EX_STATIC edge
		rcPlace.bottom = rcPlace.top + g_cImageSize+2;

		m_wndCharImage.CreateEx( WS_EX_STATICEDGE,  NULL, _T("문자이미지"),
			WS_CHILD|WS_VISIBLE, rcPlace, this, IDC_CHARACTER_IMAGE );

		// close placeholder window since it's no longer needed
		// placeholder창 닫음(더이상 필요없음)
		pPlaceholder->DestroyWindow();
	}

	// create the neuron viewer image window, using static placeholder from the dialog template
	// 뉴런 뷰어 이미지 창 생성, 
	// 대화상자 템플릿에서 정적 placeholder 사용
	pPlaceholder = GetDlgItem( IDC_NEURON_VIEWER );
	if ( pPlaceholder != NULL )
	{
		// 화면 좌표에 in screen coords
		pPlaceholder->GetWindowRect( &rcPlace );
		// 화면에서 이 윈도우 좌표로 맵 map from screen to this window's coords
		::MapWindowPoints( NULL, m_hWnd, (POINT*)&rcPlace, 2 );

		m_wndNeuronViewer.CreateEx( WS_EX_STATICEDGE,  NULL, _T("뉴런보기"),
			WS_CHILD|WS_VISIBLE, rcPlace, this, IDC_NEURON_VIEWER );

		// close placeholder window since it's no longer needed
		// placeholder창 닫음(더이상 필요없음)
		pPlaceholder->DestroyWindow();
	}

	// 에딧콘트롤 값 초기화 initialize values of edit controls
	m_ctlEditImageNumber.SetWindowText( _T("0") );
	m_ctlEditLabelValue.SetWindowText( _T("0") );

	// initialize state of "Distort input pattern" check box
	// "Distort input pattern" 체크상자의 상태 초기화
	m_ctlCheckDistortInputPattern.SetCheck( 1 );  // 1 == checked

	// initialize state of "Training Pattern/Testing Pattern" radio buttons
	// "Training Pattern/Testing Pattern" 라디오버튼의 상태 초기화
	m_ctlRadioWhichSet.SetCheck( 1 );  // 1 == checked (which implies that testing is un-checked

	// initialize resize helper
	// 헬퍼 재조정크기 초기화
	m_resizeHelper.Init( m_hWnd );
	m_resizeHelper.Fix(
		IDC_CHARACTER_IMAGE,
		CResizeHelper_Dlg::kLeftRight,
		CResizeHelper_Dlg::kTopBottom );

	// clear m_NeuronOutputs
	// m_NeuronOutputs 청소
	m_NeuronOutputs.clear();

	return TRUE;  // return TRUE, 컨트롤에 포커스를 설정하지 않은 경우
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

//==============================================================================
//
void CDlgCharacterImage::OnOK()
{
	// ordinarily do nothing, to prevent the dialog from closing when user hits the "Enter key
	// However, check if the user hit "return" after edting the pattern number window
	// 일반적으로 아무것도 하지 않음, 대화창 닫기를 방지하기 위해, 사용자가 "Enter"키를 눌렀을때
	// 그러나, 사용자가 패턴번호창을 수정한 후에 "Enter"키를 눌렀는지 확인한다.
	CWnd* pWnd = GetFocus();

	if ( pWnd->m_hWnd == m_ctlEditImageNumber.m_hWnd )
	{
		// current focus is the edit control that contains the index for the image number, and
		// user has just hit the "return" button.  We assume that he wants to calculate the neural
		// net, so we set the focus tot he Calculate button and call the corresponding
		// calculate functions
		// 현재 포커스는 이미지 번호에 대한 인덱스를 포함하는 에딧컨트롤이다,
		// 그리고 사용자가 "Enter"키를 눌렀다. 사용자가 신경망을 계산하기를 원한다고 가정,
		// 그래서 "계산"에 포커스를 설정하고 해당 계산함수를 호출한다.

		//----------------------------------------------------------------------------
		// CODE REVIEW: There's no need to shift focus away from the edit.  This allows the user to 
		// continuously enter numbers and hit enter, without the need to re-click the edit window
		// 코드 검토: 에딧에서 포커스를 멀리 이동할 필요가 없다. 사용자가 지속적으로
		// 숫자를 입력 입력하고 엔터키를 누를 수 있게한다, 편집창을 다시 클릭할 필요 없이
		// CWnd* pAnotherWnd = GetDlgItem( IDC_BUTTON_NN_CALCULATE );
		// if ( pAnotherWnd != NULL )
		//		pAnotherWnd->SetFocus();

		// no point in SetSel here: the OnButtonNnCalculate function (which calls UpdateCharacterImageData)
		// eventually sends WM_SETTEXT to the edit control, which un-selects all text.  
		// We will SetSel right after the WM_SETTEXT, in the UpdateCharacterImageData function
		// OnButtonNnCalculate 함수(이는 UpdateCharacterImageData를 호출)는 결국
		// 에딧컨트롤에 WM_SETTEXT를 보낸다, 모든 문자를 선택취소.
		// WM_SETTEXT 후에 SetSel을 할 것이다, UpdateCharacterImageData 함수에서
		// m_ctlEditImageNumber.SetSel( 0, -1 );  // select all text in control

		OnButtonNnCalculate();
	}
}

//==============================================================================
//
void CDlgCharacterImage::OnCancel()
{
	// do nothing, to override default behavior, which is to close the dialog when the ESC key is pressed
	// 아무것도 하지 마시오, 기본 동작을 재정의 하기 위해, "Esc"키를 눌렀을때 대화창을 닫는것 이다.
}

//==============================================================================
//
void CDlgCharacterImage::OnSize(UINT nType, int cx, int cy) 
{
	CDialog::OnSize(nType, cx, cy);

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.

	m_resizeHelper.OnSize();	
}

//==============================================================================
//
void CDlgCharacterImage::OnButtonGetImageData() 
{
	// gets image corresponding to value in edit control
	// 에딧창에서 값에 해당하는 이미지를 가져온다

	CString strNum;
	m_ctlEditImageNumber.GetWindowText( strNum );

	UINT iNum = _ttoi( strNum );	//  [2015.12.26 12:25 Jobs]

	UpdateCharacterImageData( iNum );
}

//==============================================================================
//
void CDlgCharacterImage::OnButtonGetNextImageData() 
{
	// increments the value of the edit control and gets corresponding image
	// 에딧컨트롤의 값을 증가시키고 해당 이미지를 가져온다

	CString strNum;
	m_ctlEditImageNumber.GetWindowText( strNum );

	UINT iNum = _ttoi( strNum );	//  [2015.12.26 12:25 Jobs]
	++iNum;

	UpdateCharacterImageData( iNum );	
}

//==============================================================================
//
void CDlgCharacterImage::OnButtonGetPreviousImageData() 
{
	// decrements the value of the edit control and gets corresponding image
	// 에딧컨트롤의 값을 감소시키고 해당 이미지를 가져온다

	CString strNum;
	m_ctlEditImageNumber.GetWindowText( strNum );

	UINT iNum = _ttoi( strNum );	//  [2015.12.26 12:26 Jobs]
	--iNum;

	UpdateCharacterImageData( iNum );		
}

//==============================================================================
// 선택한 고유번호의 이미지를 읽어온다
void CDlgCharacterImage::UpdateCharacterImageData(UINT &iNumImage)
{
	// updates the viewing window that contains the character's image
	// and adjusts the value of the int iNumImage so that it returns a value
	// that lies inside a valid range of the selected image set (training or testing)
	// 문자이미지가 포함된 보기창을 갱신하고 int형의 iNumImage 값을 조정한다
	// 그런다음 이 값은 반환한다
	// 선택한 이미지 집합(훈련 또는 시험)의 유효범위 안에서

	unsigned char grayArray[ g_cImageSize * g_cImageSize ] = {0};
	int label = 0;

	// determine whether we are looking for the training patterns or the testing patterns
	// 훈련패턴 또는 시험패턴을 찾고 어느 것인지 결정한다

	BOOL bTraining = ( 1 == m_ctlRadioWhichSet.GetCheck() );  // 1 == checked

	if ( bTraining != FALSE )
	{
		// the training set has been selected
		// adjust numeric value of pattern so that it lies inside a valid range
		// 훈련집합을 선택했다
		// 유효범위 안에 있게 패턴의 숫자값을 조정한다
		if ( iNumImage < 0 ) iNumImage = 0;
		if ( iNumImage >= ::GetPreferences().m_nItemsTrainingImages )
			iNumImage = ::GetPreferences().m_nItemsTrainingImages - 1;

		// get gray-scale values for character image, and numeric value of its "answer"
		// 문자이미지에 대한 단조값을 가진다, 그리고 "answer"의 숫자값
		m_pDoc->GetTrainingPatternArrayValues( iNumImage, grayArray, &label, TRUE );
	}
	else
	{
		// testing set has been selected
		// adjust numeric value of pattern so that it lies inside a valid range
		// 시험집합을 선택했다
		// 유효범위 안에 있게 패턴의 숫자값을 조정한다
		if ( iNumImage < 0 ) iNumImage = 0;
		if ( iNumImage >= ::GetPreferences().m_nItemsTestingImages )
			iNumImage = ::GetPreferences().m_nItemsTestingImages - 1;

		// get gray-scale values for character image, and numeric value of its "answer"
		// 문자이미지에 대한 단조값을 가진다, 그리고 "answer"의 숫자값
		m_pDoc->GetTestingPatternArrayValues( iNumImage, grayArray, &label, TRUE );
	}

	// 대화창의 모습을 갱신한다 update appearance of the dialog

	CString strNum;

	strNum.Format( _T("%i"), label );		// 이미지 라벨
	m_ctlEditLabelValue.SetWindowText( strNum );

	strNum.Format( _T("%i"), iNumImage );	//이미지 고유번호
	m_ctlEditImageNumber.SetWindowText( strNum );
	// select all text in control; will be highlighted only if edit also has focus
	// 컨트롤에서 모든 문자를 선택한다; 포커스가 에딧에 있는 경우에만 강조표시가 된다.
	m_ctlEditImageNumber.SetSel( 0, -1 );

	// 입력데이타를 화면 표시용 빗맵으로 변환한다
	m_wndCharImage.BuildBitmapFromGrayValues( grayArray );
	m_wndCharImage.Invalidate( );
}

//==============================================================================
// 계산 버튼을 눌러 입력데이타로 신경망 출력을 구하고 표시한다
void CDlgCharacterImage::OnButtonNnCalculate() 
{
	// runs the current character image through the neural net
	// 신경망을 통해 현재 문자이미지를 실행한다

	// first, get image corresponding to value in edit control
	// note that this code is identical to OnButtonGetImageData,
	// which is deliberate because the user might have changed the content
	// of the image label, but which is somewhat redundant if he didn't
	// 먼저, 에딧컨트롤의 기록에 OnButtonGetImageData로 동일한 코드에
	// 해당하는 이미지를 가진다, 사용자가 의도적으로 이미지라벨의 내용을
	// 변경할 수 있기 때문에, 하지만 이는 사용자가 유효범위 안에 있도록
	// 패턴의 숫자값을 조정하지 않은 경우라면 다소 불필요함.

	CString strNum;
	m_ctlEditImageNumber.GetWindowText( strNum );	//원하는 패턴 고유번호

	UINT iNum = _ttoi( strNum );	//숫자로 변환	//  [2015.12.26 12:26 Jobs]

	// 고유번호의 이미지를 읽어온다
	UpdateCharacterImageData( iNum );

	// now get image data, based on whether the training set or the testing set has been 
	// selected (this is the part that's redundant to the above call to UpdateCharacterImageData)
	// 이제 이미지 데이타를 얻는다, 훈련집합 또는 시험집합이 선택되었는지에 따라
	// (UpdateCharacterImageData를 위에서 호출하여 불필요/중복되는 부분이다)

	unsigned char grayArray[ g_cImageSize * g_cImageSize ] = {0};
	int label = 0;

	BOOL bTraining = ( 1 == m_ctlRadioWhichSet.GetCheck() );  // 1 == checked

	if ( bTraining != FALSE )
	{
		// the training set has been selected
		// no need to adjust numeric value of the pattern since UpdateCharacterImageData() has 
		// already done this for us
		// 훈련집합을 선택했다
		// UpdateCharacterImageData()를 이미 했으므로 패턴의 숫자값을 조정할 필요가 없다

		// get gray-scale values for character image, and numeric value of its "answer"
		// 문자이미지에 대한 단조값을 가진다, 그리고 "answer"의 숫자값
		m_pDoc->GetTrainingPatternArrayValues( iNum, grayArray, &label, TRUE );

	}
	else
	{
		// testing set has been selected
		// 시험집합을 선택했다

		// get gray-scale values for character image, and numeric value of its "answer"
		// 문자이미지에 대한 단조값을 가진다, 그리고 "answer"의 숫자값
		m_pDoc->GetTestingPatternArrayValues( iNum, grayArray, &label, TRUE );
	}

	// pad the values to 29x29, convert to double precision, and run through the neural net
	// This operation is timed and the result is displayed
	// 29x29로 값을 채운다, double형의 정밀도로 변환한다, 그리고 신경망을 통해 실행한다
	// 이 동작은 시간이 제한되었고 결과를 표시한다

	DWORD tick = ::GetTickCount();

	int ii, jj;
	double inputVector[841];	// 29x29 픽셀

	for ( ii=0; ii<841; ++ii )
	{
		inputVector[ ii ] = 1.0;  // 1은 백색, -1은 흑색 one is white, -one is black
	}

	// top row of inputVector is left as zero, left-most column is left as zero 
	// inputVector의 위쪽끝 행은 0, 왼쪽끝 열은 0
	for ( ii=0; ii<g_cImageSize; ++ii )
	{
		for ( jj=0; jj<g_cImageSize; ++jj )
		{
			inputVector[ 1 + jj + 29*(ii+1) ] =
				(double)((int)(unsigned char)grayArray[ jj + g_cImageSize*ii ]) /
				128.0 - 1.0;  // 1은 백색, -1은 흑색 one is white, -one is black
		}
	}

	// get state of "Distort input pattern" check box
	// "Distort input pattern" 체크박스의 상태를 가져온다
	BOOL bDistort = ( 1 == m_ctlCheckDistortInputPattern.GetCheck() );  // 1 == checked

	double outputVector[10] = {0.0};
	double targetOutputVector[10] = {0.0};

	// initialize target output vector (i.e., desired values)
	// 목표출력 벡터를 초기화 한다 (즉, 원하는 값)
	for ( ii=0; ii<10; ++ii )
	{
		targetOutputVector[ ii ] = -1.0;
	}

	if ( label > 9 ) label = 9;
	if ( label < 0 ) label = 0;

	targetOutputVector[ label ] = 1.0;	//목표값을 설정

	// 신경망을 계산한다 calculate neural net
	m_pDoc->CalculateNeuralNet( inputVector, 841, outputVector, 10, &m_NeuronOutputs, bDistort );

	DWORD diff = ::GetTickCount() - tick;

	// write numerical result (and time taken to get them) to results window
	// 결과창에 수치결과(그리고 수행 시간)를 기록한다
	CString strLine, strResult;
	double dTemp, sampleMse = 0.0;

	strResult.Format( _T("결과:\n") );

	// 묵표값과 출력값으로 MSE를 구한다
	for ( ii=0; ii<10; ii++ )
	{
		// 화면에 표시하기 위하여 모든 출력값을 문자로 변환한다
		strLine.Format( _T(" %2i = %+6.3f \n"), ii, outputVector[ii] );
		strResult += strLine;

		dTemp = targetOutputVector[ ii ] - outputVector[ ii ];
		sampleMse += dTemp * dTemp;
	}

	sampleMse = 0.5 * sampleMse;
	strLine.Format( _T("\n이미지 오류\n Ep = %g\n\n%i mSecs"), sampleMse, diff );
	strResult += strLine;
	CWnd* pWnd = GetDlgItem(IDC_STATIC_TIME);
	if ( pWnd != NULL )
		pWnd->SetWindowText( strResult );

	// 뉴런 출력 보기를 갱신한다. update the view of the outputs of the neurons
	m_wndNeuronViewer.BuildBitmapFromNeuronOutputs( m_NeuronOutputs );
	m_wndNeuronViewer.Invalidate();
}
