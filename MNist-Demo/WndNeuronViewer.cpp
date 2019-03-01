// WndNeuronViewer.cpp : implementation file
//

#include "stdafx.h"
#include "MNist.h"
#include "WndNeuronViewer.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CWndNeuronViewer

CWndNeuronViewer::CWndNeuronViewer()
{
}

CWndNeuronViewer::~CWndNeuronViewer()
{
}


BEGIN_MESSAGE_MAP(CWndNeuronViewer, CWnd)
	//{{AFX_MSG_MAP(CWndNeuronViewer)
	ON_WM_CREATE()
	ON_WM_NCDESTROY()
	ON_WM_PAINT()
	ON_WM_MOUSEMOVE()
	ON_MESSAGE( WM_MOUSELEAVE, OnMouseLeave )
	ON_WM_ERASEBKGND()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// CWndNeuronViewer message handlers

//==============================================================================
//
int CWndNeuronViewer::OnCreate(LPCREATESTRUCT lpCreateStruct) 
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;
	
	// TODO: Add your specialized creation code here

	// create the magnifier window, and make certain it's hidden

	BOOL bRet = m_wndMagnifier.CreateEx( NULL, AfxRegisterWndClass( CS_DBLCLKS | CS_HREDRAW | CS_VREDRAW ),
		_T("NeuronOutputMagnifier"), WS_CHILD, CRect(0,0,0,0), ::AfxGetMainWnd(), 0x1345 );

	ASSERT( bRet != FALSE );

	m_wndMagnifier.ShowWindow( SW_HIDE );

	// allocate memory for DIB bitmap values, and create default for DDB bitmap
	// we have a fixed "viewer" size of 120x120 pixels

#define DIB_VIEWER_SIZE (230)

	m_cRows = DIB_VIEWER_SIZE;
	m_cCols = DIB_VIEWER_SIZE;

	m_cPixels = m_cRows * m_cCols;
	
	m_pValues = new COLORREF[ m_cPixels ];

	int ii;

	for ( ii=0; ii<m_cPixels; ++ii )
	{
		m_pValues[ii] = RGB_TO_BGRQUAD(255,255,255);
	}

	// create device dependent bitmap and store for future OnPaints

	CClientDC dc(this);

	m_bmDisplayedBitmap.CreateCompatibleBitmap( &dc, m_cCols, m_cRows );

	BITMAPINFO bmInfo;
	BITMAPINFOHEADER& bmInfoHeader = bmInfo.bmiHeader;

	::memset( &bmInfo, 0, sizeof(BITMAPINFO) );
	bmInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmInfoHeader.biWidth = m_cCols;
	bmInfoHeader.biHeight = -m_cRows;  // positive number for bottom-up DIB, negative for top-down
	bmInfoHeader.biPlanes = 1;
	bmInfoHeader.biBitCount = 32;
	bmInfoHeader.biCompression = BI_RGB;
	bmInfoHeader.biSizeImage = 0;
	bmInfoHeader.biXPelsPerMeter = 100;  // arbitrary value
	bmInfoHeader.biYPelsPerMeter = 100;  // arbitrary
	bmInfoHeader.biClrUsed = 0;
	bmInfoHeader.biClrImportant = 0;

	int iRet = ::SetDIBits( (HDC)dc,
							(HBITMAP)m_bmDisplayedBitmap,
							0,
							m_cRows,
							(LPVOID)m_pValues,
							&bmInfo,
							DIB_RGB_COLORS );

	return 0;
}

void CWndNeuronViewer::OnNcDestroy() 
{
	
	delete[] m_pValues;
	m_bmDisplayedBitmap.DeleteObject();


	CWnd::OnNcDestroy();
	
	// TODO: Add your message handler code here
	
}

//==============================================================================
//
void CWndNeuronViewer::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	
	// TODO: Add your message handler code here

	// draw out the pre-stored DDB bitmap

	CDC memDC;
	memDC.CreateCompatibleDC( &dc );
	CBitmap* pOldBitmap = memDC.SelectObject( &m_bmDisplayedBitmap );

	CRect rc;
	GetClientRect( &rc );

	int left = rc.Width()/2 - DIB_VIEWER_SIZE/2;
	int top = rc.Height()/2 - DIB_VIEWER_SIZE/2;

	if ( left<0 ) left = 0;
	if ( top<0 ) top = 0;

	dc.BitBlt( left, top, m_cCols, m_cRows, &memDC, 0, 0, SRCCOPY );

	memDC.SelectObject( pOldBitmap );
	memDC.DeleteDC();
	
	// Do not call CWnd::OnPaint() for painting messages
}

//==============================================================================
// 뉴런 출력으로 빗맵을 만든다
void CWndNeuronViewer::BuildBitmapFromNeuronOutputs( std::vector<std::vector<double>>& neuronOutputs )
{
	ASSERT( ::IsWindow( this->m_hWnd ) );

	// zero out current DIB bitmap values

	int ii, jj;
	DWORD whiteness = RGB_TO_BGRQUAD(255,255,255);

	for ( ii=0; ii<m_cPixels; ++ii )
	{
		m_pValues[ ii ] = whiteness;
	}

	// go through each layer's output individually

	std::vector<double>::iterator it;

	int iSize = neuronOutputs.size();  // should be the same as the number of layers

	//--------------------------------------------------------------------------
	// draw the output of the neurons from the zero-th layer, which should be identical to 
	// input image
	// 0번째 층에서 출력을 그린다, 입력 이미지와 동일해야 한다
	it = neuronOutputs[ 0 ].begin();

	// 출력을 그린다
	DrawOutputBox( 10, 10, 29, 29, it );

	ASSERT( it == neuronOutputs[ 0 ].end() );

	//--------------------------------------------------------------------------
	// draw the output of the neurons from the first layer.
	// there are 6 feature maps in the first layer, each 13x13
	// 첫번째 층에서 뉴런의 출력을 그린다
	// 첫번째 층에는 6개의 특징맵이 있다, 각 13x13

	it = neuronOutputs[ 1 ].begin();

	int offV = 10;
	int offH = 44;

	// 첫번째 층의 출력을 그린다
	for ( ii=0; ii<6; ++ii )
	{
		DrawOutputBox( offH, offV, 13, 13, it );
		offV += 14;
	}

	ASSERT( it == neuronOutputs[ 1 ].end() );

	//--------------------------------------------------------------------------
	// draw the output of the neurons from the second layer.
	// There are 50 feature maps in the second layer, each 5x5
	// We will draw these in a block 5 across by 10 down
	// 두번째 층에서 뉴런의 출력을 그린다.
	// 두번째 층에는 50개의 특징맵이 있다, 각 5x5
	// 아래로 10개 가로로 5개의 블럭으로 그린다
	it = neuronOutputs[ 2 ].begin();

	offV = 10;
	offH = 61;

	// 두번째 층의 출력을 그린다
	for ( ii=0; ii<5; ++ii )
	{
		for ( jj=0; jj<10; ++jj )
		{		
			DrawOutputBox( offH, offV, 5, 5, it );
			offV += 6;
		}

		offV = 10;
		offH += 6;
	}

	ASSERT( it == neuronOutputs[ 2 ].end() );

	//--------------------------------------------------------------------------
	// draw the outputs of the neurons in the third layer.
	// The third layer is a fully-connected layer in which there are 100 neurons,
	// which will be displayed in 1x1 patches, in a single column of 100 down
	// 세번째 층에서 뉴런의 출력을 그린다
	// 세번째 층은 100개의 뉴런이 완전히 연결된 층이다,
	// 1x1 단위로 표시한다, 100개를 아래로 1열로
	it = neuronOutputs[ 3 ].begin();

	offV = 10;
	offH = 94;

	for ( ii=0; ii<100; ++ii )
	{
		DrawOutputBox( offH, offV, 1, 1, it );
		offV += 2;
	}

	ASSERT( it == neuronOutputs[ 3 ].end() );

	//--------------------------------------------------------------------------
	// The fourth layer is the final output layer, and we draw it in a different way
	// We output the digits 0 though 9, with an intensity that codes the output of
	// the corresponding neuron
	// 4번째 층은 마지막 출력층이다, 그리고 다른 방법으로 그린다
	// 숫자 0 ~ 9, 해당 뉴런의 출력 코드를 강도로
	it = neuronOutputs[ 4 ].begin();

	double dGray;
	int iGray;
	CString str;
	DWORD dwDigitValues[ 20*20 ];

	UINT iBestIndex = 0;
	double dBestChoice = -99.0;

	CClientDC dc( this );
	CBitmap bm;
	bm.CreateCompatibleBitmap( &dc, 20, 20 );

	CDC memDC;
	memDC.CreateCompatibleDC( &dc );

	LOGFONT lf = {0};
	lf.lfHeight = 20;
	lf.lfWeight = FW_BOLD;
	lf.lfPitchAndFamily = FF_SWISS;
	//_stprintf( lf.lfFaceName, _T("Arial") );
	sprintf_s( lf.lfFaceName, _T("Arial") );	//  [2015.12.26 12:10 Jobs]

	CFont font;
	font.CreateFontIndirect( &lf );

	CBitmap* pOldBitmap = memDC.SelectObject( &bm );
	CFont* pOldFont = memDC.SelectObject( &font );

	offV = 10;
	offH = 99;
	
	for ( ii=0; ii<10; ++ii )
	{
		memDC.FillSolidRect( 0,0,20,20, RGB(255,255,255) );
		memDC.SetBkMode( TRANSPARENT );
		
		dGray = *it;
		iGray = (int)( (dGray + 1.0)/2.0 * 255.0 );
		iGray = 255 - iGray;
		
		if ( iGray > 255 )
			iGray = 255;
		if ( iGray < 0 )
			iGray = 0;

		if ( dGray > dBestChoice )
		{
			dBestChoice = dGray;
			iBestIndex = ii;
		}
		
		it++;
		memDC.SetTextColor( RGB(iGray,iGray,iGray) );
		
		str.Format( _T("%d"), ii );
		memDC.DrawText( str, 1, CRect(0,0,19,19), DT_SINGLELINE | DT_CENTER | DT_VCENTER );
		
		// de-select the bitmap so that we can extract the DIB bits from it
		// 빗맵 선택을 취소한다 그래서 it 에서 DIB 비트를 추출할 수 있다
		memDC.SelectObject( pOldBitmap );

		// bm now contains a DDB bitmap of the desired digit.  Getting the DIB bitmap values
		// from it is a two-step process, in which we call GetDIBits twice, the first time to 
		// populate a BITMAPINFO structure, and the second time to get the actual DIB values
		// bm은 이제 원하는 숫자의 DDB 빗맵을 담을 수있다, 두 단계의 절차로 DIB
		// 빗맵 값을 구한다, 첫번째 BITMAPINFO 구조로 채운다, 그리고 두번째 실제
		// DIB 값을 구한다

		BITMAPINFO bmInfo;
		BITMAPINFOHEADER& bmInfoHeader = bmInfo.bmiHeader;
		
		::memset( &bmInfo, 0, sizeof(BITMAPINFO) );
		bmInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
		
		// call GetDIBits with lots of NULL values to populate the bmInfo structure 
		// bmInfo 구조를 채우기 위해 NULL 값으로 GetDIBits를 호출한다
		int iRet = ::GetDIBits( (HDC)dc, (HBITMAP)bm, 0, 0, NULL, &bmInfo, DIB_RGB_COLORS );
		
		//////////////////////
		//  apparently, we cannot force the gdi to give us a top-down bitmap.  It insists on giving bottom-up and
		// if we try to force it to give a top-down by setting biHeight to a negative value, the GetDIBits function fails
		// 하향식 빗맵을 제공하기 위해 gdi를 억제할 수 없다. 이것은 상향식 제공을
		// 고집한다 그리고 우리는 음의 값으로 biHeight를 설정하여 하향식 제공으로
		// 억제하기 위하여 시도했다
/*
		if ( bmInfoHeader.biHeight > 0 )
		{
			bmInfoHeader.biHeight =  bmInfoHeader.biHeight;  // negative forces a top-down DIB, positive a bottom-up
		}
*/
		// force Windows to give us a plain-old array of BRGA color values
		bmInfoHeader.biCompression = BI_RGB;
		// force Windows to give us a full 32 bit color w/o a color table 
		bmInfoHeader.biBitCount = 32;

		// 실제 빗맵 비트를 구하기위해 GetDIBits를 호출한다 call GetDIBits to get actual bitmap bits
		iRet = ::GetDIBits( (HDC)dc, (HBITMAP)bm, 0, bmInfoHeader.biHeight, (LPVOID)dwDigitValues, &bmInfo, DIB_RGB_COLORS );

		// if biHeight was positive, then we got a bottom-up bitmap
		// Since we need a top-down, we must reverse it
		// 만약 biHeight가 양 이면, 그러면 상향식 빗맵을 가진다
		// 우리는 하향식이 필요하기 때문에, 이것을 반전해야 한다
		if ( bmInfoHeader.biHeight > 0 )
		{
			double dTemp;
			for ( int kk=0; kk<10; ++kk )
			{
				for ( int mm=0; mm<20; ++mm )
				{
				dTemp = dwDigitValues[ mm + 20*kk ];
				dwDigitValues[ mm + 20*kk ] = dwDigitValues[ mm + 20*(19-kk) ];
				dwDigitValues[ mm + 20*(19-kk) ] = dTemp;
				}
			}
		}

		// finally, dwDigitValues contains the BGR values for the intensity-coded digit
		// copy it to the output
		// 끝으로, dwDigitValues는 출력을 숫자로 복사한 강도를 위하여 BGR 값을 포함한다
		DrawOutputBox( offH, offV, 20, 20, dwDigitValues, 400 );
		
		offV += 21;
		
		// re-select thing into the dc to ready the dc for the next iteration
		// 다음 반복을 위하여 dc를 준비하기 위해 dc를 다시 선택한다
		pOldBitmap = memDC.SelectObject( &bm );
	}

	// clean up
	memDC.SelectObject( pOldBitmap );
	bm.DeleteObject();
	font.DeleteObject();
	memDC.DeleteDC();

	// put a triangular pointer next to the "best" choice
	// "최고"선택에 다음의 삼각 지시기를 넣는다
#define BEST_CHOICE_COLOR (RGB_TO_BGRQUAD(255,0,0))

	offH = 122;
	offV = 20 + iBestIndex*21;

	for ( ii=0; ii<6; ++ii )
	{
		for ( jj=0; jj<=ii; ++jj )
		{
			At( m_pValues, offV+jj, offH+ii ) = BEST_CHOICE_COLOR;
			At( m_pValues, offV-jj, offH+ii ) = BEST_CHOICE_COLOR;
		}
	}

#undef BEST_CHOICE_COLOR

	// create device dependent bitmap and store for future OnPaints

	// CClientDC dc(this);

	BITMAPINFO bmInfo;
	BITMAPINFOHEADER& bmInfoHeader = bmInfo.bmiHeader;

	::memset( &bmInfo, 0, sizeof(BITMAPINFO) );
	bmInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmInfoHeader.biWidth = m_cCols;
	bmInfoHeader.biHeight = -m_cRows;  // positive number for bottom-up DIB, negative for top-down
	bmInfoHeader.biPlanes = 1;
	bmInfoHeader.biBitCount = 32;
	bmInfoHeader.biCompression = BI_RGB;
	bmInfoHeader.biSizeImage = 0;
	bmInfoHeader.biXPelsPerMeter = 100;  // arbitrary value
	bmInfoHeader.biYPelsPerMeter = 100;  // arbitrary
	bmInfoHeader.biClrUsed = 0;
	bmInfoHeader.biClrImportant = 0;

	int iRet = ::SetDIBits( (HDC)dc, (HBITMAP)m_bmDisplayedBitmap, 0, m_cRows, (LPVOID)m_pValues, &bmInfo, DIB_RGB_COLORS );
}

//==============================================================================
// 출력에 테두리를 그린다
void CWndNeuronViewer::DrawOutputBox(UINT left,
									 UINT top,
									 UINT clientWidth,
									 UINT clientHeight,
									 std::vector<double>::iterator& it)
{
	// draws gray values into a client box surrounded by a one-pixel-width bright gray frame
	// 
// similar to cornflowerblue (whose exact value is RGB( 100, 149, 237 ));
// COLORREF value is 0x00FFC0AA; web value is #AAC0FF
#define FRAMECOLOR (RGB_TO_BGRQUAD(170, 192, 255 ))
	
	UINT windowWidth = clientWidth + 2;
	UINT windowHeight = clientHeight + 2;

	int row, col;

	double dGray;
	int iGray;

	// draw top and bottom of frame
	for ( col=left; col<left+windowWidth; ++col )
	{
		At( m_pValues, top, col ) = FRAMECOLOR;
		At( m_pValues, top+windowHeight-1, col ) = FRAMECOLOR;
	}

	// draw left and right of frame
	for ( row=top; row<top+windowHeight; ++row )
	{
		At( m_pValues, row, left ) = FRAMECOLOR;
		At( m_pValues, row, left+windowWidth-1 ) = FRAMECOLOR;
	}

	for ( row=top; row<top+clientHeight; ++row )
	{
		for ( col=left; col<left+clientWidth; ++col )
		{
			dGray = *it;
			iGray = (int)( (dGray + 1.0)/2.0 * 255.0 );

			if ( iGray > 255 )
				iGray = 255;
			if ( iGray < 0 )
				iGray = 0;

			At( m_pValues, row+1, col+1 ) = RGB_TO_BGRQUAD( iGray, iGray, iGray );
			it++;
		}
	}

#undef FRAMECOLOR
}

//==============================================================================
//
void CWndNeuronViewer::DrawOutputBox(UINT left,
									 UINT top,
									 UINT clientWidth,
									 UINT clientHeight,
									 DWORD* pArray,
									 int count )
{
	// silly, but we convert to a std::vector of doubles and then call the stl version
	// 유치한, 하지만 더블형의 std::vector로 변환한다 그런다음 stl 버전을 호출한다
	std::vector<double> temp;
	double d;
	int gg;

	for ( int ii=0; ii<count; ++ii )
	{
		// extract green value, as most representative of gray-scale
		// 녹색 값을 추출한다, 단조의 가장 대표되는 것으로
		gg = ( pArray[ii] & 0x0000FF00 ) >> 8;
		d = (double)(gg-128)/128.0;
		temp.push_back( d );
	}

	std::vector<double>::iterator it = temp.begin();

	DrawOutputBox( left, top, clientWidth, clientHeight, it );
}

//==============================================================================
// 마우스 이동시 확대된 이미지를 출력한다
void CWndNeuronViewer::OnMouseMove(UINT nFlags, CPoint point) 
{
	CWnd::OnMouseMove(nFlags, point);

	// blt magnified image into m_wndMagnifier, and re-position and show it
	// m_wndMagnifier에 확대된 이미지를 순간복사한다, 그리고 위치를 수정하고 보인다
	CRect rc, rc1;
	GetClientRect( rc );
	ClientToScreen( rc );
	::AfxGetMainWnd()->ScreenToClient( rc );

	rc1 = rc;

	int nSize = ::GetPreferences().m_nMagWindowSize * g_cImageSize;

	rc.OffsetRect( -40, 110 );

	m_wndMagnifier.SetWindowPos(&wndTopMost,
								rc.left,
								rc.top,
								nSize,
								nSize, 
								SWP_SHOWWINDOW | SWP_NOZORDER );
		
	CClientDC dcMag( &m_wndMagnifier );

	CDC memDC;
	memDC.CreateCompatibleDC( &dcMag );
	CBitmap* pOldBitmap = memDC.SelectObject( &m_bmDisplayedBitmap );

	int divisor = ::GetPreferences().m_nMagWindowMagnification;
	if ( divisor < 1 ) divisor = 1;
	int delta = nSize/divisor;
	int originX = (rc1.Width() - DIB_VIEWER_SIZE)/2;
	int originY = (rc1.Height() - DIB_VIEWER_SIZE)/2;

	
	dcMag.StretchBlt(0,
					 0,
					 nSize,
					 nSize,
					 &memDC,
					 point.x-delta/2-originX,
					 point.y-delta/2-originY,
					 delta,
					 delta,
					 SRCCOPY );

	memDC.SelectObject( pOldBitmap );

	// track mouse for mouse-leave message
	// note the complementary handler for OnMouseLeave based on _TrackMouseEvent

	TRACKMOUSEEVENT tme;
	tme.cbSize = sizeof( tme );
	tme.dwFlags = TME_LEAVE;
	tme.hwndTrack = m_hWnd;
		
	_TrackMouseEvent( &tme );
}

//==============================================================================
// 돋보기 보기/숨기기
afx_msg LRESULT CWndNeuronViewer::OnMouseLeave(WPARAM, LPARAM)
{
	// hides magnification window as mouse leaves the window area
	// 윈도우 영역을 떠날 때 돋보기 창을 감춘다
	m_wndMagnifier.ShowWindow( SW_HIDE );

	return (0L);
}

//==============================================================================
// 배경을 지운다
BOOL CWndNeuronViewer::OnEraseBkgnd(CDC* pDC) 
{
	// TODO: Add your message handler code here and/or call default
	
	CWnd::OnEraseBkgnd(pDC);

	CRect rc;
	GetClientRect( &rc );

	pDC->FillSolidRect( &rc, RGB(255,255,255) );  // fill the client with whiteness

	return TRUE;
}
