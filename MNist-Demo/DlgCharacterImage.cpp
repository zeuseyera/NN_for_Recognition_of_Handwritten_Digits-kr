// DlgCharacterImage.cpp : ���� �����Դϴ�.
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
// CDlgCharacterImage ��ȭ �����Դϴ�.


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
// CDlgCharacterImage �޽��� ó�����Դϴ�.

//==============================================================================
//
BOOL CDlgCharacterImage::OnInitDialog() 
{
	CDialog::OnInitDialog();

	ASSERT( m_pDoc != NULL );

	// create the character image window, using static placeholder from the dialog template,
	// and set its size to the global size of the character image
	// ���� �̹���â ����
	// ��ȭ���� ���ø����� ���� placeholder ���
	// �����̹����� ��ü ũ��� ũ�� ����
	CRect rcPlace;
	CWnd* pPlaceholder = GetDlgItem( IDC_CHARACTER_IMAGE );
	if ( pPlaceholder != NULL )
	{
		// ȭ�� ��ǥ�� in screen coords
		pPlaceholder->GetWindowRect( &rcPlace );
		// ȭ�鿡�� �� ������ ��ǥ�� �� map from screen to this window's coords
		::MapWindowPoints( NULL, m_hWnd, (POINT*)&rcPlace, 2 );

		// WS_EX_STATIC �����ڸ��� ���� +2 ��� +2 allows for the width of WS_EX_STATIC edge
		rcPlace.right = rcPlace.left + g_cImageSize+2;
		// +2 allows for the width of WS_EX_STATIC edge
		rcPlace.bottom = rcPlace.top + g_cImageSize+2;

		m_wndCharImage.CreateEx( WS_EX_STATICEDGE,  NULL, _T("�����̹���"),
			WS_CHILD|WS_VISIBLE, rcPlace, this, IDC_CHARACTER_IMAGE );

		// close placeholder window since it's no longer needed
		// placeholderâ ����(���̻� �ʿ����)
		pPlaceholder->DestroyWindow();
	}

	// create the neuron viewer image window, using static placeholder from the dialog template
	// ���� ��� �̹��� â ����, 
	// ��ȭ���� ���ø����� ���� placeholder ���
	pPlaceholder = GetDlgItem( IDC_NEURON_VIEWER );
	if ( pPlaceholder != NULL )
	{
		// ȭ�� ��ǥ�� in screen coords
		pPlaceholder->GetWindowRect( &rcPlace );
		// ȭ�鿡�� �� ������ ��ǥ�� �� map from screen to this window's coords
		::MapWindowPoints( NULL, m_hWnd, (POINT*)&rcPlace, 2 );

		m_wndNeuronViewer.CreateEx( WS_EX_STATICEDGE,  NULL, _T("��������"),
			WS_CHILD|WS_VISIBLE, rcPlace, this, IDC_NEURON_VIEWER );

		// close placeholder window since it's no longer needed
		// placeholderâ ����(���̻� �ʿ����)
		pPlaceholder->DestroyWindow();
	}

	// ������Ʈ�� �� �ʱ�ȭ initialize values of edit controls
	m_ctlEditImageNumber.SetWindowText( _T("0") );
	m_ctlEditLabelValue.SetWindowText( _T("0") );

	// initialize state of "Distort input pattern" check box
	// "Distort input pattern" üũ������ ���� �ʱ�ȭ
	m_ctlCheckDistortInputPattern.SetCheck( 1 );  // 1 == checked

	// initialize state of "Training Pattern/Testing Pattern" radio buttons
	// "Training Pattern/Testing Pattern" ������ư�� ���� �ʱ�ȭ
	m_ctlRadioWhichSet.SetCheck( 1 );  // 1 == checked (which implies that testing is un-checked

	// initialize resize helper
	// ���� ������ũ�� �ʱ�ȭ
	m_resizeHelper.Init( m_hWnd );
	m_resizeHelper.Fix(
		IDC_CHARACTER_IMAGE,
		CResizeHelper_Dlg::kLeftRight,
		CResizeHelper_Dlg::kTopBottom );

	// clear m_NeuronOutputs
	// m_NeuronOutputs û��
	m_NeuronOutputs.clear();

	return TRUE;  // return TRUE, ��Ʈ�ѿ� ��Ŀ���� �������� ���� ���
	// ����: OCX �Ӽ� �������� FALSE�� ��ȯ�ؾ� �մϴ�.
}

//==============================================================================
//
void CDlgCharacterImage::OnOK()
{
	// ordinarily do nothing, to prevent the dialog from closing when user hits the "Enter key
	// However, check if the user hit "return" after edting the pattern number window
	// �Ϲ������� �ƹ��͵� ���� ����, ��ȭâ �ݱ⸦ �����ϱ� ����, ����ڰ� "Enter"Ű�� ��������
	// �׷���, ����ڰ� ���Ϲ�ȣâ�� ������ �Ŀ� "Enter"Ű�� �������� Ȯ���Ѵ�.
	CWnd* pWnd = GetFocus();

	if ( pWnd->m_hWnd == m_ctlEditImageNumber.m_hWnd )
	{
		// current focus is the edit control that contains the index for the image number, and
		// user has just hit the "return" button.  We assume that he wants to calculate the neural
		// net, so we set the focus tot he Calculate button and call the corresponding
		// calculate functions
		// ���� ��Ŀ���� �̹��� ��ȣ�� ���� �ε����� �����ϴ� ������Ʈ���̴�,
		// �׸��� ����ڰ� "Enter"Ű�� ������. ����ڰ� �Ű���� ����ϱ⸦ ���Ѵٰ� ����,
		// �׷��� "���"�� ��Ŀ���� �����ϰ� �ش� ����Լ��� ȣ���Ѵ�.

		//----------------------------------------------------------------------------
		// CODE REVIEW: There's no need to shift focus away from the edit.  This allows the user to 
		// continuously enter numbers and hit enter, without the need to re-click the edit window
		// �ڵ� ����: �������� ��Ŀ���� �ָ� �̵��� �ʿ䰡 ����. ����ڰ� ����������
		// ���ڸ� �Է� �Է��ϰ� ����Ű�� ���� �� �ְ��Ѵ�, ����â�� �ٽ� Ŭ���� �ʿ� ����
		// CWnd* pAnotherWnd = GetDlgItem( IDC_BUTTON_NN_CALCULATE );
		// if ( pAnotherWnd != NULL )
		//		pAnotherWnd->SetFocus();

		// no point in SetSel here: the OnButtonNnCalculate function (which calls UpdateCharacterImageData)
		// eventually sends WM_SETTEXT to the edit control, which un-selects all text.  
		// We will SetSel right after the WM_SETTEXT, in the UpdateCharacterImageData function
		// OnButtonNnCalculate �Լ�(�̴� UpdateCharacterImageData�� ȣ��)�� �ᱹ
		// ������Ʈ�ѿ� WM_SETTEXT�� ������, ��� ���ڸ� �������.
		// WM_SETTEXT �Ŀ� SetSel�� �� ���̴�, UpdateCharacterImageData �Լ�����
		// m_ctlEditImageNumber.SetSel( 0, -1 );  // select all text in control

		OnButtonNnCalculate();
	}
}

//==============================================================================
//
void CDlgCharacterImage::OnCancel()
{
	// do nothing, to override default behavior, which is to close the dialog when the ESC key is pressed
	// �ƹ��͵� ���� ���ÿ�, �⺻ ������ ������ �ϱ� ����, "Esc"Ű�� �������� ��ȭâ�� �ݴ°� �̴�.
}

//==============================================================================
//
void CDlgCharacterImage::OnSize(UINT nType, int cx, int cy) 
{
	CDialog::OnSize(nType, cx, cy);

	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.

	m_resizeHelper.OnSize();	
}

//==============================================================================
//
void CDlgCharacterImage::OnButtonGetImageData() 
{
	// gets image corresponding to value in edit control
	// ����â���� ���� �ش��ϴ� �̹����� �����´�

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
	// ������Ʈ���� ���� ������Ű�� �ش� �̹����� �����´�

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
	// ������Ʈ���� ���� ���ҽ�Ű�� �ش� �̹����� �����´�

	CString strNum;
	m_ctlEditImageNumber.GetWindowText( strNum );

	UINT iNum = _ttoi( strNum );	//  [2015.12.26 12:26 Jobs]
	--iNum;

	UpdateCharacterImageData( iNum );		
}

//==============================================================================
// ������ ������ȣ�� �̹����� �о�´�
void CDlgCharacterImage::UpdateCharacterImageData(UINT &iNumImage)
{
	// updates the viewing window that contains the character's image
	// and adjusts the value of the int iNumImage so that it returns a value
	// that lies inside a valid range of the selected image set (training or testing)
	// �����̹����� ���Ե� ����â�� �����ϰ� int���� iNumImage ���� �����Ѵ�
	// �׷����� �� ���� ��ȯ�Ѵ�
	// ������ �̹��� ����(�Ʒ� �Ǵ� ����)�� ��ȿ���� �ȿ���

	unsigned char grayArray[ g_cImageSize * g_cImageSize ] = {0};
	int label = 0;

	// determine whether we are looking for the training patterns or the testing patterns
	// �Ʒ����� �Ǵ� ���������� ã�� ��� ������ �����Ѵ�

	BOOL bTraining = ( 1 == m_ctlRadioWhichSet.GetCheck() );  // 1 == checked

	if ( bTraining != FALSE )
	{
		// the training set has been selected
		// adjust numeric value of pattern so that it lies inside a valid range
		// �Ʒ������� �����ߴ�
		// ��ȿ���� �ȿ� �ְ� ������ ���ڰ��� �����Ѵ�
		if ( iNumImage < 0 ) iNumImage = 0;
		if ( iNumImage >= ::GetPreferences().m_nItemsTrainingImages )
			iNumImage = ::GetPreferences().m_nItemsTrainingImages - 1;

		// get gray-scale values for character image, and numeric value of its "answer"
		// �����̹����� ���� �������� ������, �׸��� "answer"�� ���ڰ�
		m_pDoc->GetTrainingPatternArrayValues( iNumImage, grayArray, &label, TRUE );
	}
	else
	{
		// testing set has been selected
		// adjust numeric value of pattern so that it lies inside a valid range
		// ���������� �����ߴ�
		// ��ȿ���� �ȿ� �ְ� ������ ���ڰ��� �����Ѵ�
		if ( iNumImage < 0 ) iNumImage = 0;
		if ( iNumImage >= ::GetPreferences().m_nItemsTestingImages )
			iNumImage = ::GetPreferences().m_nItemsTestingImages - 1;

		// get gray-scale values for character image, and numeric value of its "answer"
		// �����̹����� ���� �������� ������, �׸��� "answer"�� ���ڰ�
		m_pDoc->GetTestingPatternArrayValues( iNumImage, grayArray, &label, TRUE );
	}

	// ��ȭâ�� ����� �����Ѵ� update appearance of the dialog

	CString strNum;

	strNum.Format( _T("%i"), label );		// �̹��� ��
	m_ctlEditLabelValue.SetWindowText( strNum );

	strNum.Format( _T("%i"), iNumImage );	//�̹��� ������ȣ
	m_ctlEditImageNumber.SetWindowText( strNum );
	// select all text in control; will be highlighted only if edit also has focus
	// ��Ʈ�ѿ��� ��� ���ڸ� �����Ѵ�; ��Ŀ���� ������ �ִ� ��쿡�� ����ǥ�ð� �ȴ�.
	m_ctlEditImageNumber.SetSel( 0, -1 );

	// �Էµ���Ÿ�� ȭ�� ǥ�ÿ� �������� ��ȯ�Ѵ�
	m_wndCharImage.BuildBitmapFromGrayValues( grayArray );
	m_wndCharImage.Invalidate( );
}

//==============================================================================
// ��� ��ư�� ���� �Էµ���Ÿ�� �Ű�� ����� ���ϰ� ǥ���Ѵ�
void CDlgCharacterImage::OnButtonNnCalculate() 
{
	// runs the current character image through the neural net
	// �Ű���� ���� ���� �����̹����� �����Ѵ�

	// first, get image corresponding to value in edit control
	// note that this code is identical to OnButtonGetImageData,
	// which is deliberate because the user might have changed the content
	// of the image label, but which is somewhat redundant if he didn't
	// ����, ������Ʈ���� ��Ͽ� OnButtonGetImageData�� ������ �ڵ忡
	// �ش��ϴ� �̹����� ������, ����ڰ� �ǵ������� �̹������� ������
	// ������ �� �ֱ� ������, ������ �̴� ����ڰ� ��ȿ���� �ȿ� �ֵ���
	// ������ ���ڰ��� �������� ���� ����� �ټ� ���ʿ���.

	CString strNum;
	m_ctlEditImageNumber.GetWindowText( strNum );	//���ϴ� ���� ������ȣ

	UINT iNum = _ttoi( strNum );	//���ڷ� ��ȯ	//  [2015.12.26 12:26 Jobs]

	// ������ȣ�� �̹����� �о�´�
	UpdateCharacterImageData( iNum );

	// now get image data, based on whether the training set or the testing set has been 
	// selected (this is the part that's redundant to the above call to UpdateCharacterImageData)
	// ���� �̹��� ����Ÿ�� ��´�, �Ʒ����� �Ǵ� ���������� ���õǾ������� ����
	// (UpdateCharacterImageData�� ������ ȣ���Ͽ� ���ʿ�/�ߺ��Ǵ� �κ��̴�)

	unsigned char grayArray[ g_cImageSize * g_cImageSize ] = {0};
	int label = 0;

	BOOL bTraining = ( 1 == m_ctlRadioWhichSet.GetCheck() );  // 1 == checked

	if ( bTraining != FALSE )
	{
		// the training set has been selected
		// no need to adjust numeric value of the pattern since UpdateCharacterImageData() has 
		// already done this for us
		// �Ʒ������� �����ߴ�
		// UpdateCharacterImageData()�� �̹� �����Ƿ� ������ ���ڰ��� ������ �ʿ䰡 ����

		// get gray-scale values for character image, and numeric value of its "answer"
		// �����̹����� ���� �������� ������, �׸��� "answer"�� ���ڰ�
		m_pDoc->GetTrainingPatternArrayValues( iNum, grayArray, &label, TRUE );

	}
	else
	{
		// testing set has been selected
		// ���������� �����ߴ�

		// get gray-scale values for character image, and numeric value of its "answer"
		// �����̹����� ���� �������� ������, �׸��� "answer"�� ���ڰ�
		m_pDoc->GetTestingPatternArrayValues( iNum, grayArray, &label, TRUE );
	}

	// pad the values to 29x29, convert to double precision, and run through the neural net
	// This operation is timed and the result is displayed
	// 29x29�� ���� ä���, double���� ���е��� ��ȯ�Ѵ�, �׸��� �Ű���� ���� �����Ѵ�
	// �� ������ �ð��� ���ѵǾ��� ����� ǥ���Ѵ�

	DWORD tick = ::GetTickCount();

	int ii, jj;
	double inputVector[841];	// 29x29 �ȼ�

	for ( ii=0; ii<841; ++ii )
	{
		inputVector[ ii ] = 1.0;  // 1�� ���, -1�� ��� one is white, -one is black
	}

	// top row of inputVector is left as zero, left-most column is left as zero 
	// inputVector�� ���ʳ� ���� 0, ���ʳ� ���� 0
	for ( ii=0; ii<g_cImageSize; ++ii )
	{
		for ( jj=0; jj<g_cImageSize; ++jj )
		{
			inputVector[ 1 + jj + 29*(ii+1) ] =
				(double)((int)(unsigned char)grayArray[ jj + g_cImageSize*ii ]) /
				128.0 - 1.0;  // 1�� ���, -1�� ��� one is white, -one is black
		}
	}

	// get state of "Distort input pattern" check box
	// "Distort input pattern" üũ�ڽ��� ���¸� �����´�
	BOOL bDistort = ( 1 == m_ctlCheckDistortInputPattern.GetCheck() );  // 1 == checked

	double outputVector[10] = {0.0};
	double targetOutputVector[10] = {0.0};

	// initialize target output vector (i.e., desired values)
	// ��ǥ��� ���͸� �ʱ�ȭ �Ѵ� (��, ���ϴ� ��)
	for ( ii=0; ii<10; ++ii )
	{
		targetOutputVector[ ii ] = -1.0;
	}

	if ( label > 9 ) label = 9;
	if ( label < 0 ) label = 0;

	targetOutputVector[ label ] = 1.0;	//��ǥ���� ����

	// �Ű���� ����Ѵ� calculate neural net
	m_pDoc->CalculateNeuralNet( inputVector, 841, outputVector, 10, &m_NeuronOutputs, bDistort );

	DWORD diff = ::GetTickCount() - tick;

	// write numerical result (and time taken to get them) to results window
	// ���â�� ��ġ���(�׸��� ���� �ð�)�� ����Ѵ�
	CString strLine, strResult;
	double dTemp, sampleMse = 0.0;

	strResult.Format( _T("���:\n") );

	// ��ǥ���� ��°����� MSE�� ���Ѵ�
	for ( ii=0; ii<10; ii++ )
	{
		// ȭ�鿡 ǥ���ϱ� ���Ͽ� ��� ��°��� ���ڷ� ��ȯ�Ѵ�
		strLine.Format( _T(" %2i = %+6.3f \n"), ii, outputVector[ii] );
		strResult += strLine;

		dTemp = targetOutputVector[ ii ] - outputVector[ ii ];
		sampleMse += dTemp * dTemp;
	}

	sampleMse = 0.5 * sampleMse;
	strLine.Format( _T("\n�̹��� ����\n Ep = %g\n\n%i mSecs"), sampleMse, diff );
	strResult += strLine;
	CWnd* pWnd = GetDlgItem(IDC_STATIC_TIME);
	if ( pWnd != NULL )
		pWnd->SetWindowText( strResult );

	// ���� ��� ���⸦ �����Ѵ�. update the view of the outputs of the neurons
	m_wndNeuronViewer.BuildBitmapFromNeuronOutputs( m_NeuronOutputs );
	m_wndNeuronViewer.Invalidate();
}
