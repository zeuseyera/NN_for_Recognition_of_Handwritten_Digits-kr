// MNistDoc.cpp : implementation of the CMNistDoc class
//

#include "stdafx.h"
#include "MNist.h"

#include "MNistDoc.h"
#include "CntrItem.h"

extern CMNistApp theApp;

UINT CMNistDoc::m_iBackpropThreadIdentifier = 0;  // static member used by threads to identify themselves
UINT CMNistDoc::m_iTestingThreadIdentifier = 0;  

#include "SHLWAPI.H"	// for the path functions
#pragma comment( lib, "shlwapi.lib" )

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CMNistDoc

IMPLEMENT_DYNCREATE(CMNistDoc, COleDocument)

BEGIN_MESSAGE_MAP(CMNistDoc, COleDocument)
//{{AFX_MSG_MAP(CMNistDoc)
ON_BN_CLICKED(IDC_BUTTON_OPEN_MNIST_FILES,	OnButtonOpenMnistFiles)
ON_BN_CLICKED(IDC_BUTTON_CLOSE_MNIST_FILES,	OnButtonCloseMnistFiles)

//}}AFX_MSG_MAP
// Enable default OLE container implementation
ON_UPDATE_COMMAND_UI(ID_EDIT_PASTE,			COleDocument::OnUpdatePasteMenu)
ON_UPDATE_COMMAND_UI(ID_EDIT_PASTE_LINK,	COleDocument::OnUpdatePasteLinkMenu)
ON_UPDATE_COMMAND_UI(ID_OLE_EDIT_CONVERT,	COleDocument::OnUpdateObjectVerbMenu)
ON_COMMAND(ID_OLE_EDIT_CONVERT,				COleDocument::OnEditConvert)
ON_UPDATE_COMMAND_UI(ID_OLE_EDIT_LINKS,		COleDocument::OnUpdateEditLinksMenu)
ON_COMMAND(ID_OLE_EDIT_LINKS,				COleDocument::OnEditLinks)
ON_UPDATE_COMMAND_UI_RANGE(ID_OLE_VERB_FIRST, ID_OLE_VERB_LAST, COleDocument::OnUpdateObjectVerbMenu)
END_MESSAGE_MAP()

//==============================================================================
// CMNistDoc 생성자
CMNistDoc::CMNistDoc()
{
	// Use OLE compound files
	EnableCompoundFile();
	
	// TODO: add one-time construction code here
	
	m_bFilesOpen = FALSE;
	m_bBackpropThreadAbortFlag = FALSE;
	m_bBackpropThreadsAreRunning = FALSE;
	m_cBackprops = 0;
	m_nAfterEveryNBackprops = 1;
	
	m_bTestingThreadsAreRunning = FALSE;
	m_bTestingThreadAbortFlag = FALSE;
	
	m_iNextTestingPattern = 0;
	m_iNextTrainingPattern = 0;
	
	::InitializeCriticalSection( &m_csTrainingPatterns );
	::InitializeCriticalSection( &m_csTestingPatterns );

	// anonymous mutex which is unowned initially	
	m_utxNeuralNet = ::CreateMutex( NULL, FALSE, NULL );
	
	ASSERT( m_utxNeuralNet != NULL );

	// 왜곡맵을 저장하기위한 메모리 할당 allocate memory to store the distortion maps
	
	m_cCols = 29;
	m_cRows = 29;
	
	m_cCount = m_cCols * m_cRows;
	
	m_DispH = new double[ m_cCount ];
	m_DispV = new double[ m_cCount ];

	// create a gaussian kernel, which is constant, for use in generating elastic distortions
	// 가우시안 커널 생성, 상수 이다, 변동왜곡 생성에서 사용하기 위하여
	int iiMid = GAUSSIAN_FIELD_SIZE/2;  // GAUSSIAN_FIELD_SIZE 오로지 홀수 is strictly odd
	
	double twoSigmaSquared = 2.0 * (::GetPreferences().m_dElasticSigma) * (::GetPreferences().m_dElasticSigma);
	twoSigmaSquared = 1.0 /  twoSigmaSquared;
	double twoPiSigma = 1.0 / (::GetPreferences().m_dElasticSigma) * sqrt( 2.0 * 3.1415926535897932384626433832795 );
	
	for ( int col=0; col<GAUSSIAN_FIELD_SIZE; ++col )
	{
		for ( int row=0; row<GAUSSIAN_FIELD_SIZE; ++row )
		{
			m_GaussianKernel[ row ][ col ] = twoPiSigma * 
				( exp(- ( ((row-iiMid)*(row-iiMid) + (col-iiMid)*(col-iiMid)) * twoSigmaSquared ) ) );
		}
	}
}

//==============================================================================
// CMNistDoc 소멸자
CMNistDoc::~CMNistDoc()
{
	if ( m_bFilesOpen != FALSE )
	{
		CloseMnistFiles();
	}
	
	::DeleteCriticalSection( &m_csTrainingPatterns );
	::DeleteCriticalSection( &m_csTestingPatterns );
	
	::CloseHandle( m_utxNeuralNet );
	
	// delete memory of the distortion maps, allocated in constructor
	// 왜곡맵의 메모리를 삭제한다, 생성자에서 할당한다
	delete[] m_DispH;
	delete[] m_DispV;
	
	
}

//DEL BOOL CMNistDoc::OnOpenDocument(LPCTSTR lpszPathName) 
//DEL {
//DEL 	if (!COleDocument::OnOpenDocument(lpszPathName))
//DEL 		return FALSE;
//DEL 
//DEL 	
//DEL 	// TODO: Add your specialized creation code here
//DEL 	
//DEL 	return TRUE;
//DEL }

//==============================================================================
// CMNistDoc::DeleteContents
void CMNistDoc::DeleteContents() 
{
	// TODO: Add your specialized code here and/or call the base class
	
	COleDocument::DeleteContents();
	
	m_NN.Initialize();
}

//==============================================================================
// CMNistDoc::OnNewDocument
BOOL CMNistDoc::OnNewDocument()
{
	if (!COleDocument::OnNewDocument())
		return FALSE;
	
	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)
	
	// grab the mutex for the neural network
	// 신경망을 위하여 상호배제개체(mutex)를 수집한다
	CAutoMutex tlo( m_utxNeuralNet );

	//--------------------------------------------------------------------------
	// 신경망을 초기화하고 만든다 initialize and build the neural net
	
	NeuralNetwork& NN = m_NN;  // 용어를 쉽게하기 위하여 for easier nomenclature
	NN.Initialize();
	
	NNLayer* pLayer;
	
	int ii, jj, kk;
	int icNeurons = 0;	//뉴런순번(신경망 전체에서)
	int icWeights = 0;	//가중치순번(신경망 전체에서)
	double initWeight;
	CString label;

	//--------------------------------------------------------------------------
	// 0층 layer zero, the input layer.
	// Create neurons: exactly the same number of neurons as the input
	// vector of 29x29=841 pixels, and no weights/connections
	
	pLayer = new NNLayer( _T("Layer00") );
	NN.m_Layers.push_back( pLayer );
	
	for ( int ii=0; ii<841; ++ii )
	{	// 뉴런이름 : 층번호_뉴런번호_뉴런전체순번
		label.Format( _T("Layer00_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )
	
	//--------------------------------------------------------------------------
	// 1층 layer one:
	// This layer is a convolutional layer that has 6 feature maps.  Each feature 
	// map is 13x13, and each unit in the feature maps is a 5x5 convolutional kernel
	// of the input layer.
	// So, there are 13x13x6 = 1014 neurons, (5x5+1)x6 = 156 weights
	
	pLayer = new NNLayer( _T("Layer01"), pLayer );
	NN.m_Layers.push_back( pLayer );
	
	for ( int ii=0; ii<1014; ++ii )
	{	// 뉴런이름 : 층번호_뉴런번호_뉴런전체순번
		label.Format( _T("Layer01_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	for ( ii=0; ii<156; ++ii )
	{	// 가중치이름 : 층번호_가중치번호_가중치전체순번
		//label.Format( _T("Layer01_Weight%04d_Num%06d"), ii, icWeights );
		label.Format( _T("Layer01_Weight%04d_Num%06d"), ii, icWeights++ );	//  [2015.12.26 13:11 Jobs]
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}
	
	// interconnections with previous layer: this is difficult
	// The previous layer is a top-down bitmap image that has been padded to size 29x29
	// Each neuron in this layer is connected to a 5x5 kernel in its feature map, which 
	// is also a top-down bitmap of size 13x13.  We move the kernel by TWO pixels, i.e., we
	// skip every other pixel in the input image
	
	int kernelTemplate[25] = {	// 29x29 입력에 대한 5x5 커널
		0,  1,  2,  3,  4,		// 1행
		29, 30, 31, 32, 33,		// 2행
		58, 59, 60, 61, 62,		// 3행
		87, 88, 89, 90, 91,		// 4행
		116,117,118,119,120 };	// 5행
		
	int iNumWeight;
		
	int fm;

	// 6개의 특징맵에 대한 연결을 챙성한다
	for ( fm=0; fm<6; ++fm)
	{
		for ( ii=0; ii<13; ++ii )		//행 증가
		{
			for ( jj=0; jj<13; ++jj )	//열 증가
			{
				iNumWeight = fm * 26;  // 26 is the number of weights per feature map
				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*13 + fm*169 ] );
				
				n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight
				
				for ( kk=0; kk<25; ++kk )	// 이전층의 출력에 대한 연결
				{
					// note: max val of index == 840, corresponding to 841 neurons in prev layer
					n.AddConnection( 2*jj + 58*ii + kernelTemplate[kk], iNumWeight++ );
				}
			}
		}
	}

	//--------------------------------------------------------------------------
	// 2층 layer two:
	// This layer is a convolutional layer that has 50 feature maps.  Each feature 
	// map is 5x5, and each unit in the feature maps is a 5x5 convolutional kernel
	// of corresponding areas of all 6 of the previous layers, each of which is a 13x13 feature map
	// So, there are 5x5x50 = 1250 neurons, (5x5+1)x6x50 = 7800 weights
	
	pLayer = new NNLayer( _T("Layer02"), pLayer );
	NN.m_Layers.push_back( pLayer );
	
	for ( ii=0; ii<1250; ++ii )
	{
		label.Format( _T("Layer02_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	//for ( ii=0; ii<7800; ++ii )
	for ( ii=0; ii<7550; ++ii )	// 가중치개수: ((5x5)x6 + 1) x 50 = 7550 [2015.12.26 1:50 Jobs]
	{
		//label.Format( _T("Layer02_Weight%04d_Num%06d"), ii, icWeights );
		label.Format( _T("Layer02_Weight%04d_Num%06d"), ii, icWeights++ );	//  [2015.12.26 13:13 Jobs]
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}
	
	// Interconnections with previous layer: this is difficult
	// Each feature map in the previous layer is a top-down bitmap image whose size
	// is 13x13, and there are 6 such feature maps.  Each neuron in one 5x5 feature map of this 
	// layer is connected to a 5x5 kernel positioned correspondingly in all 6 parent
	// feature maps, and there are individual weights for the six different 5x5 kernels.  As
	// before, we move the kernel by TWO pixels, i.e., we
	// skip every other pixel in the input image.  The result is 50 different 5x5 top-down bitmap
	// feature maps
	
	int kernelTemplate2[25] = { // 13x13 입력에 대한 5x5 커널
		0,  1,  2,  3,  4,		// 1행
		13, 14, 15, 16, 17,		// 2행
		26, 27, 28, 29, 30,		// 3행
		39, 40, 41, 42, 43,		// 4행
		52, 53, 54, 55, 56   };	// 5행

	// 50개의 특징맵에 대한 연결을 생성한다
	for ( int fm=0; fm<50; ++fm)
	{
		for ( int ii=0; ii<5; ++ii )		// 행 증가
		{
			for ( int jj=0; jj<5; ++jj )	// 열 증가
			{
				//iNumWeight = fm * 26;  // 26 is the number of weights per feature map
				//iNumWeight = fm * 156;  // 156 is the number of weights per feature map	// billconan 제안
				iNumWeight = fm * 151;  // 151 is the number of weights per feature map		//  [2015.12.26 13:49 Jobs]

				NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*5 + fm*25 ] );

				n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight
				
				for ( kk=0; kk<25; ++kk )	// 이전층 6개의 출력에 대한 연결
				{
					// note: max val of index == 1013, corresponding to 1014 neurons in prev layer
					n.AddConnection(       2*jj + 26*ii + kernelTemplate2[kk], iNumWeight++ );
					n.AddConnection( 169 + 2*jj + 26*ii + kernelTemplate2[kk], iNumWeight++ );
					n.AddConnection( 338 + 2*jj + 26*ii + kernelTemplate2[kk], iNumWeight++ );
					n.AddConnection( 507 + 2*jj + 26*ii + kernelTemplate2[kk], iNumWeight++ );
					n.AddConnection( 676 + 2*jj + 26*ii + kernelTemplate2[kk], iNumWeight++ );
					n.AddConnection( 845 + 2*jj + 26*ii + kernelTemplate2[kk], iNumWeight++ );
				}
			}
		}
	}

	//--------------------------------------------------------------------------
	// 3층 layer three:
	// This layer is a fully-connected layer with 100 units.  Since it is fully-connected,
	// each of the 100 neurons in the layer is connected to all 1250 neurons in
	// the previous layer.
	// So, there are 100 neurons and 100*(1250+1)=125100 weights
	
	pLayer = new NNLayer( _T("Layer03"), pLayer );
	NN.m_Layers.push_back( pLayer );
	
	for ( int ii=0; ii<100; ++ii )
	{
		label.Format( _T("Layer03_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	for ( int ii=0; ii<125100; ++ii )
	{
		//label.Format( _T("Layer03_Weight%04d_Num%06d"), ii, icWeights );
		label.Format( _T("Layer03_Weight%04d_Num%06d"), ii, icWeights++ );	//  [2015.12.26 13:15 Jobs]
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}
	
	// Interconnections with previous layer: fully-connected
	
	iNumWeight = 0;  // weights are not shared in this layer
	
	for ( int fm=0; fm<100; ++fm )
	{
		NNNeuron& n = *( pLayer->m_Neurons[ fm ] );
		n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight
		
		for ( int ii=0; ii<1250; ++ii )
		{
			n.AddConnection( ii, iNumWeight++ );
		}
	}

	//--------------------------------------------------------------------------
	// 4층 layer four, the final (output) layer:
	// This layer is a fully-connected layer with 10 units.  Since it is fully-connected,
	// each of the 10 neurons in the layer is connected to all 100 neurons in
	// the previous layer.
	// So, there are 10 neurons and 10*(100+1)=1010 weights
	
	pLayer = new NNLayer( _T("Layer04"), pLayer );
	NN.m_Layers.push_back( pLayer );
	
	for ( ii=0; ii<10; ++ii )
	{
		label.Format( _T("Layer04_Neuron%04d_Num%06d"), ii, icNeurons );
		pLayer->m_Neurons.push_back( new NNNeuron( (LPCTSTR)label ) );
		icNeurons++;
	}
	
	for ( ii=0; ii<1010; ++ii )
	{
		//label.Format( _T("Layer04_Weight%04d_Num%06d"), ii, icWeights );
		label.Format( _T("Layer04_Weight%04d_Num%06d"), ii, icWeights++ );	//  [2015.12.26 13:15 Jobs]
		initWeight = 0.05 * UNIFORM_PLUS_MINUS_ONE;
		pLayer->m_Weights.push_back( new NNWeight( (LPCTSTR)label, initWeight ) );
	}
	
	// Interconnections with previous layer: fully-connected
	
	iNumWeight = 0;  // weights are not shared in this layer
	
	for ( fm=0; fm<10; ++fm )
	{
		NNNeuron& n = *( pLayer->m_Neurons[ fm ] );
		n.AddConnection( ULONG_MAX, iNumWeight++ );  // bias weight

		// 연결을 추가한다
		for ( ii=0; ii<100; ++ii )
		{
			n.AddConnection( ii, iNumWeight++ );
		}
	}
	
	SetModifiedFlag( TRUE );
	
	return TRUE;
}

//==============================================================================
// CMNistDoc::serialization
void CMNistDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
	
	{
		// grab the mutex for the neural network in a local scope, and then serialize
		// 지역 영역에서 신경망에 대한 뮤텍스를 수집한다, 그리고 나열한다
		CAutoMutex tlo( m_utxNeuralNet );

		// 여기에서 신경망의 Serialize 를 호출하여 신경망 탑재 및 저장을 한다
		m_NN.Serialize( ar );
	}
	
	// Calling the base class COleDocument enables serialization
	//  of the container document's COleClientItem objects.
	COleDocument::Serialize(ar);
}

//******************************************************************************
// CMNistDoc diagnostics

#ifdef _DEBUG
//==============================================================================
//
void CMNistDoc::AssertValid() const
{
	COleDocument::AssertValid();
}

//==============================================================================
//
void CMNistDoc::Dump(CDumpContext& dc) const
{
	COleDocument::Dump(dc);
}
#endif //_DEBUG

//******************************************************************************
// CMNistDoc commands

//==============================================================================
// MNIST 파일열기 버튼을 눌렀다, 역전파 및 시험에 필요한 MNIST 파일을 연다
void CMNistDoc::OnButtonOpenMnistFiles() 
{
	// opens the MNIST image data file and the label file
	// MNIST 이미지 파일과 라벨파일을 연다
	// 파일 4개를 열고 헤더의 유효성 각각 확인한다

	struct FILEBEGINNING
	{
		union
		{
			char	raw0[ 4 ];	//32bit 공간(문자 1개는 8bit => 4 x 8 =32bit)
			int		nMagic;
		};
		union
		{
			char	raw1[ 4 ];
			UINT	nItems;
		};
		union
		{
			char	raw2[ 4 ];
			int		nRows;
		};
		union
		{
			char	raw3[ 4 ];
			int		nCols;
		};
	};
	
	if ( m_bFilesOpen != FALSE )
	{
		::MessageBox(NULL,
					 _T("파일을 이미 열었음!"),
					 _T("알림!"),
					 MB_ICONEXCLAMATION );
		return;
	}
	
	CWaitCursor wc;
	
	// 4개의 파일을 열어야 한다 we need to open four files:
	// (1) training images, (2) training labels, (3) testing images, (4) testing labels

	//--------------------------------------------------------------------------
	// (1) 훈련 이미지 Training images
	CFileDialog fd( TRUE );  // constructs an "open" dialog; we are happy with the remaining defaults
	
	fd.m_ofn.lpstrFilter = _T("All Files (*.*)\0*.*\0\0");
	//fd.m_ofn.lpstrTitle = _T("TRAINING Images");
	fd.m_ofn.lpstrTitle = _T("벼림이미지 파일선택");
	fd.m_ofn.lpstrInitialDir = theApp.m_sModulePath;
	
	if ( fd.DoModal() != IDOK ) return;
	
	// store chosen directory so that subsequent open-file dialogs can use it
	// 경로를 저장한다 그래서 후속 파일열기 다이얼로그를 사용할 수 있다
	CString newPath = fd.GetPathName();
	::PathMakePretty( newPath.GetBuffer(255) );
	::PathRemoveFileSpec( newPath.GetBuffer(255) );
	newPath.ReleaseBuffer();
	
	// open the file
	CFileException fe;
	
	if ( m_fileTrainingImages.Open( (LPCTSTR)fd.GetPathName(),
									CFile::modeRead | CFile::shareDenyNone,
									&fe ) != 0 )
	{
		// opened successfully
		// check for expected magic number and for expected dimensionality
		// 예상 고유번호과 예상 차원수를 확인한다
		FILEBEGINNING fileBegin;
		m_fileTrainingImages.Read( fileBegin.raw0, sizeof( fileBegin ) );
		
		// convert endian-ness (using winsock functions for convenience)
		
		fileBegin.nMagic = ::ntohl( fileBegin.nMagic );
		fileBegin.nItems = ::ntohl( fileBegin.nItems );
		fileBegin.nRows = ::ntohl( fileBegin.nRows );
		fileBegin.nCols = ::ntohl( fileBegin.nCols );
		
		// check against expected values
		
		if (fileBegin.nMagic != ::GetPreferences().m_nMagicTrainingImages ||
			fileBegin.nItems != ::GetPreferences().m_nItemsTrainingImages ||
			fileBegin.nRows != ::GetPreferences().m_nRowsImages ||
			fileBegin.nCols != ::GetPreferences().m_nColsImages )
		{
			// file is not configured as expected
			CString msg;
			msg.Format( _T("다음과 같이 훈련이미지의 파일헤더가 예상되지않은 값을 포함\n")
						_T("고유번호: 가짐 %i, 예상됨 %i \n")
						_T("항목개수: 가짐 %i, 예상됨 %i \n")
						_T("행 개수: 가짐 %i, 예상됨 %i \n")
						_T("열 개수: 가짐 %i, 예상됨 %i \n")
						_T("\n훈련이미지 대한 파일의 온전함을 확인하고/ 또는 INI 파일의 예상값을 수정한다."),
						fileBegin.nMagic, ::GetPreferences().m_nMagicTrainingImages,
						fileBegin.nItems, ::GetPreferences().m_nItemsTrainingImages,
						fileBegin.nRows, ::GetPreferences().m_nRowsImages,
						fileBegin.nCols, ::GetPreferences().m_nColsImages );
			
			::MessageBox( NULL, msg, _T("예상되지않은 형식!"), MB_OK|MB_ICONEXCLAMATION );
			
			// close all files
			m_fileTrainingImages.Close();
			
			return;
		}
	}
	else
	{
		// could not open image file
		TRACE(_T("%s(%i): 훈련이미지 파일을 열수 없음!\n"), __FILE__,__LINE__);
		
		CString msg;
		TCHAR szCause[255] = {0};
		
		fe.GetErrorMessage(szCause, 254);
		
		msg.Format( _T("벼림이미지에 대한 이미지파일을 열지 못함\n")
					_T("파일명: %s\n이유: %s\n오류코드: %d"), 
					fe.m_strFileName, szCause, fe.m_cause );
		
		::MessageBox( NULL
					, msg
					, _T("벼림이미지파일 열기 실패!!!")
					, MB_OK|MB_ICONEXCLAMATION );
		
		return;
	}
	
	//--------------------------------------------------------------------------
	// (2) 훈련 라벨 Training labels
	
	CFileDialog fd2( TRUE );  // constructs an "open" dialog; we are happy with the remaining defaults
	
	fd2.m_ofn.lpstrFilter = _T("All Files (*.*)\0*.*\0\0");
	fd2.m_ofn.lpstrTitle = _T("벼림딱지 파일선택");
	fd2.m_ofn.lpstrInitialDir = newPath;
	
	if ( fd2.DoModal() != IDOK ) 
	{
		// close the images file too
		
		m_fileTrainingImages.Close();
		return;
	}

	// open the file
	if ( m_fileTrainingLabels.Open((LPCTSTR)fd2.GetPathName(),
									CFile::modeRead | CFile::shareDenyNone,
									&fe ) != 0 )
	{
		// opened successfully
		// check for expected magic number and for expected dimensionality
		FILEBEGINNING fileBegin;
		m_fileTrainingLabels.Read( fileBegin.raw0, sizeof( fileBegin ) );
		
		// convert endian-ness (using winsock functions for convenience)
		fileBegin.nMagic = ::ntohl( fileBegin.nMagic );
		fileBegin.nItems = ::ntohl( fileBegin.nItems );
		
		// check against expected values
		if ( fileBegin.nMagic != ::GetPreferences().m_nMagicTrainingLabels ||
			fileBegin.nItems != ::GetPreferences().m_nItemsTrainingLabels )
		{
			// file is not configured as expected
			CString msg;
			msg.Format( _T("File header for training labels contains unexpected values as follows\n")
						_T("Magic number: got %i, expected %i \n")
						_T("Number of items: got %i, expected %i \n")
						_T("\nCheck integrity of file for training labels and/or adjust the expected values in the INI file"),
						fileBegin.nMagic, ::GetPreferences().m_nMagicTrainingLabels,
						fileBegin.nItems, ::GetPreferences().m_nItemsTrainingLabels );
			
			::MessageBox( NULL, msg, _T("Unexpected Format"), MB_OK|MB_ICONEXCLAMATION );
			
			// close all files
			m_fileTrainingImages.Close();
			m_fileTrainingLabels.Close();
			
			return;
		}
	}
	else
	{
		// could not open label file
		TRACE(_T("%s(%i): Could not open file for training labels \n"), __FILE__,__LINE__);
		
		CString msg;
		TCHAR szCause[255] = {0};
		
		fe.GetErrorMessage(szCause, 254);
		
		msg.Format( _T("Label file for training labels could not be opened\n")
					_T("File name: %s\nReason: %s\nError code: %d"), 
					fe.m_strFileName, szCause, fe.m_cause );
		
		::MessageBox(NULL,
					 msg,
					 _T("Failed to open file for training labels"),
					 MB_OK|MB_ICONEXCLAMATION );

		// close the already-opened files too
		m_fileTrainingImages.Close();
		
		return;
	}
	
	//--------------------------------------------------------------------------
	// (3) 시험이미지 Testing images
	CFileDialog fd3( TRUE );  // constructs an "open" dialog; we are happy with the remaining defaults
	
	fd3.m_ofn.lpstrFilter = _T("All Files (*.*)\0*.*\0\0");
	fd3.m_ofn.lpstrTitle = _T("평가이미지 파일선택");
	fd3.m_ofn.lpstrInitialDir = newPath;
	
	if ( fd3.DoModal() != IDOK ) 
	{
		// close the already-opened files too
		m_fileTrainingLabels.Close();
		m_fileTrainingImages.Close();
		
		return;
	}
	
	// open the file
	if ( m_fileTestingImages.Open( (LPCTSTR)fd3.GetPathName(),
									CFile::modeRead | CFile::shareDenyNone,
									&fe ) != 0 )
	{
		// opened successfully
		// check for expected magic number and for expected dimensionality
		FILEBEGINNING fileBegin;
		m_fileTestingImages.Read( fileBegin.raw0, sizeof( fileBegin ) );
		
		// convert endian-ness (using winsock functions for convenience)
		fileBegin.nMagic = ::ntohl( fileBegin.nMagic );
		fileBegin.nItems = ::ntohl( fileBegin.nItems );
		fileBegin.nRows = ::ntohl( fileBegin.nRows );
		fileBegin.nCols = ::ntohl( fileBegin.nCols );
		
		// check against expected values
		if (fileBegin.nMagic != ::GetPreferences().m_nMagicTestingImages ||
			fileBegin.nItems != ::GetPreferences().m_nItemsTestingImages ||
			fileBegin.nRows != ::GetPreferences().m_nRowsImages ||
			fileBegin.nCols != ::GetPreferences().m_nColsImages )
		{
			// file is not configured as expected
			CString msg;
			msg.Format( _T("File header for testing images contains unexpected values as follows\n")
						_T("Magic number: got %i, expected %i \n")
						_T("Number of items: got %i, expected %i \n")
						_T("Number of rows: got %i, expected %i \n")
						_T("Number of columns: got %i, expected %i \n")
						_T("\nCheck integrity of file for testing images and/or adjust the expected values in the INI file"),
						fileBegin.nMagic, ::GetPreferences().m_nMagicTestingImages,
						fileBegin.nItems, ::GetPreferences().m_nItemsTestingImages,
						fileBegin.nRows, ::GetPreferences().m_nRowsImages,
						fileBegin.nCols, ::GetPreferences().m_nColsImages );
			
			::MessageBox( NULL, msg, _T("Unexpected Format"), MB_OK|MB_ICONEXCLAMATION );

			// close all files
			m_fileTestingImages.Close();
			m_fileTrainingLabels.Close();
			m_fileTrainingImages.Close();
			
			return;
		}
	}
	else
	{
		// could not open image file
		TRACE(_T("%s(%i): Could not open file for testing images \n"), __FILE__,__LINE__);
		
		CString msg;
		TCHAR szCause[255] = {0};
		
		fe.GetErrorMessage(szCause, 254);
		
		msg.Format( _T("Image file for testing images could not be opened\n")
					_T("File name: %s\nReason: %s\nError code: %d"), 
					fe.m_strFileName, szCause, fe.m_cause );
		
		::MessageBox(NULL,
					 msg,
					 _T("Failed to open file for testing images"),
					 MB_OK|MB_ICONEXCLAMATION );
		
		
		// close the already-opened files too
		
		m_fileTrainingLabels.Close();
		m_fileTrainingImages.Close();
		
		return;
	}
	
	//--------------------------------------------------------------------------
	// (4) 시험라벨 Testing labels
	CFileDialog fd4( TRUE );  // constructs an "open" dialog; we are happy with the remaining defaults
	
	fd4.m_ofn.lpstrFilter = _T("All Files (*.*)\0*.*\0\0");
	fd4.m_ofn.lpstrTitle = _T("평가딱지 파일선택");
	fd4.m_ofn.lpstrInitialDir = newPath;
	
	if ( fd4.DoModal() != IDOK ) 
	{
		// close the already-opened files too
		
		m_fileTestingImages.Close();
		m_fileTrainingLabels.Close();
		m_fileTrainingImages.Close();
		
		return;
	}
	
	// open the file
	if ( m_fileTestingLabels.Open((LPCTSTR)fd4.GetPathName(),
								  CFile::modeRead | CFile::shareDenyNone,
								  &fe ) != 0 )
	{
		// opened successfully
		// check for expected magic number and for expected dimensionality
		
		FILEBEGINNING fileBegin;
		m_fileTestingLabels.Read( fileBegin.raw0, sizeof( fileBegin ) );
		
		// convert endian-ness (using winsock functions for convenience)
		
		fileBegin.nMagic = ::ntohl( fileBegin.nMagic );
		fileBegin.nItems = ::ntohl( fileBegin.nItems );
		
		// check against expected values
		
		if ( fileBegin.nMagic != ::GetPreferences().m_nMagicTestingLabels ||
			fileBegin.nItems != ::GetPreferences().m_nItemsTestingLabels )
		{
			// file is not configured as expected
			CString msg;
			msg.Format( _T("File header for testing labels contains unexpected values as follows\n")
						_T("Magic number: got %i, expected %i \n")
						_T("Number of items: got %i, expected %i \n")
						_T("\nCheck integrity of file for testing labels and/or adjust the expected values in the INI file"),
						fileBegin.nMagic,
						::GetPreferences().m_nMagicTestingLabels,
						fileBegin.nItems,
						::GetPreferences().m_nItemsTestingLabels );
			
			::MessageBox( NULL, msg, _T("Unexpected Format"), MB_OK|MB_ICONEXCLAMATION );
			
			// close all files
			m_fileTestingLabels.Close();
			m_fileTestingImages.Close();
			m_fileTrainingImages.Close();
			m_fileTrainingLabels.Close();
			
			return;
		}
	}
	else
	{
		// could not open label file
		TRACE(_T("%s(%i): Could not open file for testing labels \n"), __FILE__,__LINE__);
		
		CString msg;
		TCHAR szCause[255] = {0};
		
		fe.GetErrorMessage(szCause, 254);
		
		msg.Format( _T("Label file for testing labels could not be opened\n")
					_T("File name: %s\nReason: %s\nError code: %d"), 
					fe.m_strFileName, szCause, fe.m_cause );
		
		::MessageBox(NULL,
					 msg,
					 _T("Failed to open file for testing labels"),
					 MB_OK|MB_ICONEXCLAMATION );
		
		// close all already-opened files
		m_fileTestingImages.Close();
		m_fileTrainingImages.Close();
		m_fileTrainingLabels.Close();
		
		return;
	}
	
	// all four files are opened, and all four contain expected header information
	// 4개 파일 모두 열었다, 그리고 4개 모두 예상 헤더정보를 포함한다
	m_bFilesOpen = TRUE;
	
	ASSERT( g_cImageSize == 28 );
}

//==============================================================================
// MNIST 파일닫기 버튼을 눌렀다
void CMNistDoc::OnButtonCloseMnistFiles()
{
	CloseMnistFiles();
}

//==============================================================================
// 열린 MNIST 파일을 닫는다
void CMNistDoc::CloseMnistFiles()
{
	if (m_bFilesOpen != FALSE)
	{
		m_fileTestingImages.Close();
		m_fileTestingLabels.Close();
		m_fileTrainingImages.Close();
		m_fileTrainingLabels.Close();

		m_bFilesOpen = FALSE;
	}
	else
	{
		::MessageBox(NULL,
					_T("열린파일 없음"),
					_T("알림!"),
					MB_ICONEXCLAMATION );
		return;
	}
}

//==============================================================================
// 현재 학습율
double CMNistDoc::GetCurrentEta()
{
	return m_NN.m_etaLearningRate;
}

//==============================================================================
// 이전 학습율
double CMNistDoc::GetPreviousEta()
{
	// provided because threads might change the current eta before we are able to read it
	// 제공된 현재 학습율이 쓰레드에 의해 변경됐을 수 있기 때문에 그전에 읅을 수 있다
	return m_NN.m_etaLearningRatePrevious;
}

//==============================================================================
// 훈련 이미지의 현재 고유번호를 반환한다
UINT CMNistDoc::GetCurrentTrainingPatternNumber( BOOL bFromRandomizedPatternSequence /* =FALSE */ )
{
	// returns the current number of the training pattern, either from the straight sequence, or from
	// the randomized sequence
	
	UINT iRet;
	
	if ( bFromRandomizedPatternSequence == FALSE )
	{
		iRet = m_iNextTrainingPattern;
	}
	else
	{
		iRet = m_iRandomizedTrainingPatternSequence[ m_iNextTrainingPattern ];
	}
	
	return iRet;
}

// UNIFORM_ZERO_THRU_ONE gives a uniformly-distributed number between zero (inclusive) and one (exclusive)
// UNIFORM_ZERO_THRU_ONE 는 0(포함)과 1(제외)사이의 균일하게 분포된 수를 제공한다
#define UNIFORM_ZERO_THRU_ONE ( (double)(rand())/(RAND_MAX + 1 ) ) 

//==============================================================================
// 훈련 이미지의 순서를 섞는다
void CMNistDoc::RandomizeTrainingPatternSequence()
{
	// randomizes the order of m_iRandomizedTrainingPatternSequence, which is a UINT array
	// holding the numbers 0..59999 in random order
	
	CAutoCS tlo( m_csTrainingPatterns );
	
	UINT ii, jj, iiMax, iiTemp;
	
	iiMax = ::GetPreferences().m_nItemsTrainingImages;
	
	ASSERT( iiMax == 60000 );  // requirement of sloppy and unimaginative code
	
	// 순서대로 배열을 초기화 한다 initialize array in sequential order
	for ( ii=0; ii<iiMax; ++ii )
	{
		m_iRandomizedTrainingPatternSequence[ ii ] = ii;  
	}

	// now at each position, swap with a random position
	// 이제 각 위치에, 임의의 위치로 교체한다
	for ( ii=0; ii<iiMax; ++ii )
	{
		jj = (UINT)( UNIFORM_ZERO_THRU_ONE * iiMax );
		
		ASSERT( jj < iiMax );
		
		iiTemp = m_iRandomizedTrainingPatternSequence[ ii ];
		m_iRandomizedTrainingPatternSequence[ ii ] = m_iRandomizedTrainingPatternSequence[ jj ];
		m_iRandomizedTrainingPatternSequence[ jj ] = iiTemp;
	}
}

//==============================================================================
// 다음 훈련 이미지를 가져오고, 고유번호를 반환한다
UINT CMNistDoc::GetNextTrainingPattern(unsigned char* pArray,	/* =NULL */
									   int*		pLabel,			/* =NULL */ 
									   BOOL		bFlipGrayscale,	/* =TRUE */
									   BOOL		bFromRandomizedPatternSequence, /* =TRUE */
									   UINT*	iSequenceNum)	/* =NULL */
{
	// returns the number of the pattern corresponding to the pattern that will be stored in pArray
	// if BOOL bFromRandomizedPatternSequence is TRUE (which is the default) then the pattern
	// stored will be a pattern from the randomized sequence; otherwise the pattern will be a straight
	// sequential run through all the training patterns, from start to finish.  The sequence number,
	// which runs from 0..59999 monotonically, is returned in iSequenceNum (if it's not NULL)
	
	CAutoCS tlo( m_csTrainingPatterns );
	
	UINT iPatternNum;
	
	if ( bFromRandomizedPatternSequence == FALSE )
	{
		iPatternNum = m_iNextTrainingPattern;
	}
	else
	{
		iPatternNum = m_iRandomizedTrainingPatternSequence[ m_iNextTrainingPattern ];
	}

	ASSERT( iPatternNum < ::GetPreferences().m_nItemsTrainingImages );
	
	GetTrainingPatternArrayValues( iPatternNum, pArray, pLabel, bFlipGrayscale );
	
	if ( iSequenceNum != NULL )
	{
		*iSequenceNum = m_iNextTrainingPattern;
	}
	
	m_iNextTrainingPattern++;
	
	if ( m_iNextTrainingPattern >= ::GetPreferences().m_nItemsTrainingImages )
	{
		m_iNextTrainingPattern = 0;
	}
	
	return iPatternNum;	//고유번호
}

//==============================================================================
// 임의의 훈련 이미지를 가져온다
UINT CMNistDoc::GetRandomTrainingPattern(unsigned char* pArray,	/* =NULL */
										 int*	pLabel,			/* =NULL */
										 BOOL	bFlipGrayscale)	/* =TRUE */
{
	// returns the number of the pattern corresponding to the pattern stored in pArray
	// pArray에 저장된 패턴에 해당하는 패턴의 번호를 반환한다
	CAutoCS tlo( m_csTrainingPatterns );
	
	UINT patternNum = (UINT)( UNIFORM_ZERO_THRU_ONE *
					  (::GetPreferences().m_nItemsTrainingImages - 1) );
	
	GetTrainingPatternArrayValues( patternNum, pArray, pLabel, bFlipGrayscale );
	
	return patternNum;
}

//==============================================================================
// 열린 파일에서 훈련용 이미지값을 가져온다
void CMNistDoc::GetTrainingPatternArrayValues(int	iNumImage,		/* =0 */
											  unsigned char* pArray,/* =NULL */
											  int*	pLabel,			/* =NULL */
											  BOOL	bFlipGrayscale)	/* =TRUE */
{
	// fills an unsigned char array with gray values, corresponding to iNumImage, and also
	// returns the label for the image
	// 단조 값으로 unsigned char 배열을 채운다, iNumImage에 해당하는, 그리고 또한
	// 이미지에 대한 라벨을 반환한다
	
	CAutoCS tlo( m_csTrainingPatterns );
	
	int cCount = g_cImageSize*g_cImageSize;
	int fPos;
	
	if ( m_bFilesOpen != FALSE )
	{
		// pArray 배열에 이미지를 가져온다
		if ( pArray != NULL )
		{
			// 파일 헤더정보에 대한 16을 보상한다 16 compensates for file header info
			fPos = 16 + iNumImage*cCount;
			m_fileTrainingImages.Seek( fPos, CFile::begin );
			m_fileTrainingImages.Read( pArray, cCount );
			
			if ( bFlipGrayscale != FALSE )
			{
				for ( int ii=0; ii<cCount; ++ii )
				{
					pArray[ ii ] = 255 - pArray[ ii ];	// 255:백색 ~ 0:흑색
				}
			}
		}

		// 해당하는 이미지의 라벨(이미지에 기록된 숫자)을 가져온다
		if ( pLabel != NULL )
		{
			fPos = 8 + iNumImage;
			char r;
			m_fileTrainingLabels.Seek( fPos, CFile::begin );
			m_fileTrainingLabels.Read( &r, 1 );  // single byte
			
			*pLabel = r;
		}
	}
	else  // no files are open: return a simple gray wedge
	{
		if ( pArray != NULL )
		{
			for ( int ii=0; ii<cCount; ++ii )
			{
				pArray[ ii ] = ii*255/cCount;
			}
		}
		
		if ( pLabel != NULL )
		{
			*pLabel = INT_MAX;
		}
	}
}

//==============================================================================
//
UINT CMNistDoc::GetNextTestingPatternNumber()
{
	return m_iNextTestingPattern;
}

//==============================================================================
// 시험 이미지를 가져온다
UINT CMNistDoc::GetNextTestingPattern(unsigned char* pArray /* =NULL */,
									  int*	pLabel /* =NULL */,
									  BOOL	bFlipGrayscale /* =TRUE */ )
{
	// returns the number of the pattern corresponding to the pattern stored in pArray
	// pArray에 저장된 패턴에 해당하는 패턴의 번호를 반환한다
	CAutoCS tlo( m_csTestingPatterns );
	
	GetTestingPatternArrayValues( m_iNextTestingPattern, pArray, pLabel, bFlipGrayscale );
	
	UINT iRet = m_iNextTestingPattern;
	m_iNextTestingPattern++;
	
	if ( m_iNextTestingPattern >= ::GetPreferences().m_nItemsTestingImages )
	{
		m_iNextTestingPattern = 0;
	}
	
	return iRet ;
}

//==============================================================================
// 열린 파일에서 시험이미지를 배열에 가져온다
void CMNistDoc::GetTestingPatternArrayValues(int	iNumImage /* =0 */,
											 unsigned char* pArray /* =NULL */,
											 int*	pLabel /* =NULL */,
											 BOOL	bFlipGrayscale /* =TRUE */ )
{
	// fills an unsigned char array with gray values, corresponding to iNumImage, and also
	// returns the label for the image
	// 단조 값으로 unsigned char 배열을 채운다, iNumImage에 해당하는, 그리고 또한
	// 이미지에 대한 라벨을 반환한다

	CAutoCS tlo( m_csTestingPatterns );
	
	int cCount = g_cImageSize*g_cImageSize;	//실제 이미지 크기(28x28 = 784)
	int fPos;
	
	if ( m_bFilesOpen != FALSE )
	{
		if ( pArray != NULL )
		{
			// 파일 헤더정보에 대한 16을 보상한다 16 compensates for file header info
			fPos = 16 + iNumImage*cCount;
			m_fileTestingImages.Seek( fPos, CFile::begin );
			m_fileTestingImages.Read( pArray, cCount );
			
			if ( bFlipGrayscale != FALSE )
			{
				for ( int ii=0; ii<cCount; ++ii )
				{
					pArray[ ii ] = 255 - pArray[ ii ];
				}
			}
		}
		
		if ( pLabel != NULL )
		{
			// 파일 헤더정보에 대한 8을 보상한다
			fPos = 8 + iNumImage;
			char r;
			m_fileTestingLabels.Seek( fPos, CFile::begin );
			m_fileTestingLabels.Read( &r, 1 );  // single byte
			
			*pLabel = r;
		}
	}
	else  // no files are open: return a simple gray wedge
	{
		if ( pArray != NULL )
		{
			for ( int ii=0; ii<cCount; ++ii )
			{
				pArray[ ii ] = ii*255/cCount;
			}
		}
		
		if ( pLabel != NULL )
		{
			*pLabel = INT_MAX;
		}
	}
}

//==============================================================================
//
void CMNistDoc::GenerateDistortionMap( double severityFactor /* =1.0 */ )
{
	// generates distortion maps in each of the horizontal and vertical directions
	// Three distortions are applied: a scaling, a rotation, and an elastic distortion
	// Since these are all linear tranformations, we can simply add them together, after calculation
	// one at a time
	
	// The input parameter, severityFactor, let's us control the severity of the distortions relative
	// to the default values.  For example, if we only want half as harsh a distortion, set
	// severityFactor == 0.5
	
	// First, elastic distortion, per Patrice Simard, "Best Practices For Convolutional Neural Networks..."
	// at page 2.
	// Three-step process: seed array with uniform randoms, filter with a gaussian kernel, normalize (scale)
	
	int row, col;
	double* uniformH = new double[ m_cCount ];
	double* uniformV = new double[ m_cCount ];
	
	
	for ( col=0; col<m_cCols; ++col )
	{
		for ( row=0; row<m_cRows; ++row )
		{
			At( uniformH, row, col ) = UNIFORM_PLUS_MINUS_ONE;
			At( uniformV, row, col ) = UNIFORM_PLUS_MINUS_ONE;
		}
	}
	
	// filter with gaussian
	
	double fConvolvedH, fConvolvedV;
	double fSampleH, fSampleV;
	double elasticScale = severityFactor * ::GetPreferences().m_dElasticScaling;
	int xxx, yyy, xxxDisp, yyyDisp;
	int iiMid = GAUSSIAN_FIELD_SIZE/2;  // GAUSSIAN_FIELD_SIZE is strictly odd
	
	for ( col=0; col<m_cCols; ++col )
	{
		for ( row=0; row<m_cRows; ++row )
		{
			fConvolvedH = 0.0;
			fConvolvedV = 0.0;
			
			for ( xxx=0; xxx<GAUSSIAN_FIELD_SIZE; ++xxx )
			{
				for ( yyy=0; yyy<GAUSSIAN_FIELD_SIZE; ++yyy )
				{
					xxxDisp = col - iiMid + xxx;
					yyyDisp = row - iiMid + yyy;
					
					if ( xxxDisp<0 || xxxDisp>=m_cCols || yyyDisp<0 || yyyDisp>=m_cRows )
					{
						fSampleH = 0.0;
						fSampleV = 0.0;
					}
					else
					{
						fSampleH = At( uniformH, yyyDisp, xxxDisp );
						fSampleV = At( uniformV, yyyDisp, xxxDisp );
					}
					
					fConvolvedH += fSampleH * m_GaussianKernel[ yyy ][ xxx ];
					fConvolvedV += fSampleV * m_GaussianKernel[ yyy ][ xxx ];
				}
			}
			
			At( m_DispH, row, col ) = elasticScale * fConvolvedH;
			At( m_DispV, row, col ) = elasticScale * fConvolvedV;
		}
	}
	
	delete[] uniformH;
	delete[] uniformV;
	
	// next, the scaling of the image by a random scale factor
	// Horizontal and vertical directions are scaled independently
	double dSFHoriz = severityFactor *
					  ::GetPreferences().m_dMaxScaling / 
					  100.0 * 
					  UNIFORM_PLUS_MINUS_ONE;  // m_dMaxScaling is a percentage
	double dSFVert = severityFactor *
					 ::GetPreferences().m_dMaxScaling /
					 100.0 *
					 UNIFORM_PLUS_MINUS_ONE;  // m_dMaxScaling is a percentage

	int iMid = m_cRows/2;
	
	for ( row=0; row<m_cRows; ++row )
	{
		for ( col=0; col<m_cCols; ++col )
		{
			At( m_DispH, row, col ) += dSFHoriz * ( col-iMid );
			At( m_DispV, row, col ) -= dSFVert * ( iMid-row );  // negative because of top-down bitmap
		}
	}
	
	
	// finally, apply a rotation
	
	double angle = severityFactor * ::GetPreferences().m_dMaxRotation * UNIFORM_PLUS_MINUS_ONE;
	angle = angle * 3.1415926535897932384626433832795 / 180.0;  // convert from degrees to radians
	
	double cosAngle = cos( angle );
	double sinAngle = sin( angle );
	
	for ( row=0; row<m_cRows; ++row )
	{
		for ( col=0; col<m_cCols; ++col )
		{
			At( m_DispH, row, col ) += ( col-iMid ) * ( cosAngle - 1 ) - ( iMid-row ) * sinAngle;
			At( m_DispV, row, col ) -= ( iMid-row ) * ( cosAngle - 1 ) + ( col-iMid ) * sinAngle;  // negative because of top-down bitmap
		}
	}
	
}

//==============================================================================
//
void CMNistDoc::ApplyDistortionMap(double *inputVector)
{
	// applies the current distortion map to the input vector
	
	// For the mapped array, we assume that 0.0 == background, and 1.0 == full intensity information
	// This is different from the input vector, in which +1.0 == background (white), and 
	// -1.0 == information (black), so we must convert one to the other
	
	std::vector< std::vector< double > >   mappedVector( m_cRows, std::vector< double >( m_cCols, 0.0 ));
	
	double sourceRow, sourceCol;
	double fracRow, fracCol;
	double w1, w2, w3, w4;
	double sourceValue;
	int row, col;
	int sRow, sCol, sRowp1, sColp1;
	BOOL bSkipOutOfBounds;
	
	for ( row=0; row<m_cRows; ++row )
	{
		for ( col=0; col<m_cCols; ++col )
		{
			// the pixel at sourceRow, sourceCol is an "phantom" pixel that doesn't really exist, and
			// whose value must be manufactured from surrounding real pixels (i.e., since 
			// sourceRow and sourceCol are floating point, not ints, there's not a real pixel there)
			// The idea is that if we can calculate the value of this phantom pixel, then its 
			// displacement will exactly fit into the current pixel at row, col (which are both ints)
			
			sourceRow = (double)row - At( m_DispV, row, col );
			sourceCol = (double)col - At( m_DispH, row, col );
			
			// weights for bi-linear interpolation
			
			fracRow = sourceRow - (int)sourceRow;
			fracCol = sourceCol - (int)sourceCol;
			
			
			w1 = ( 1.0 - fracRow ) * ( 1.0 - fracCol );
			w2 = ( 1.0 - fracRow ) * fracCol;
			w3 = fracRow * ( 1 - fracCol );
			w4 = fracRow * fracCol;
			
			
			// limit indexes

/*
			while (sourceRow >= m_cRows ) sourceRow -= m_cRows;
			while (sourceRow < 0 ) sourceRow += m_cRows;
			
			while (sourceCol >= m_cCols ) sourceCol -= m_cCols;
			while (sourceCol < 0 ) sourceCol += m_cCols;
*/
			bSkipOutOfBounds = FALSE;

			if ( (sourceRow + 1.0) >= m_cRows )	bSkipOutOfBounds = TRUE;
			if ( sourceRow < 0 )				bSkipOutOfBounds = TRUE;
			
			if ( (sourceCol + 1.0) >= m_cCols )	bSkipOutOfBounds = TRUE;
			if ( sourceCol < 0 )				bSkipOutOfBounds = TRUE;
			
			if ( bSkipOutOfBounds == FALSE )
			{
				// the supporting pixels for the "phantom" source pixel are all within the 
				// bounds of the character grid.
				// Manufacture its value by bi-linear interpolation of surrounding pixels
				
				sRow = (int)sourceRow;
				sCol = (int)sourceCol;
				
				sRowp1 = sRow + 1;
				sColp1 = sCol + 1;
				
				while (sRowp1 >= m_cRows ) sRowp1 -= m_cRows;
				while (sRowp1 < 0 ) sRowp1 += m_cRows;
				
				while (sColp1 >= m_cCols ) sColp1 -= m_cCols;
				while (sColp1 < 0 ) sColp1 += m_cCols;
				
				// perform bi-linear interpolation
				
				sourceValue =	w1 * At( inputVector, sRow  , sCol   ) +
					w2 * At( inputVector, sRow  , sColp1 ) +
					w3 * At( inputVector, sRowp1, sCol   ) +
					w4 * At( inputVector, sRowp1, sColp1 );
			}
			else
			{
				// At least one supporting pixel for the "phantom" pixel is outside the
				// bounds of the character grid. Set its value to "background"

				sourceValue = 1.0;  // "background" color in the -1 -> +1 range of inputVector
			}
			
			mappedVector[ row ][ col ] = 0.5 * ( 1.0 - sourceValue );  // conversion to 0->1 range we are using for mappedVector
			
		}
	}
	
	// now, invert again while copying back into original vector
	
	for ( row=0; row<m_cRows; ++row )
	{
		for ( col=0; col<m_cCols; ++col )
		{
			At( inputVector, row, col ) = 1.0 - 2.0 * mappedVector[ row ][ col ];
		}
	}			
	
}


//==============================================================================
//
void CMNistDoc::CalculateNeuralNet(double*	inputVector,				//입력뉴런(841)
								   int		count,						//입력개수(841)
								   double*	outputVector,/* =NULL */	//출력뉴런(10)
								   int		oCount,		 /* =0 */		//출력개수(10)
								   std::vector<std::vector<double>>* pNeuronOutputs, /* =NULL */
								   BOOL		bDistort) /* =FALSE */
{
	// wrapper function for neural net's Calculate() function, needed because the NN is a protected member
	// waits on the neural net mutex (using the CAutoMutex object, which automatically releases the
	// mutex when it goes out of scope) so as to restrict access to one thread at a time
	// 신경망 Calculate() 함수를 위한 래퍼함수이다, 신경망이 한번에 하나의 쓰레드로
	// 접근을 제한하기 위하여 신경망 뮤텍스(CAutoMutex 개체를, 범위를 벗어나면
	// 자동으로 뮤텍스를 해제하는 것)에서 보호된 멤버를 기다린다

	CAutoMutex tlo( m_utxNeuralNet );
	
	if ( bDistort != FALSE )
	{
		GenerateDistortionMap();
		ApplyDistortionMap( inputVector );
	}
	
	// 신경망의 출력을 구한다
	m_NN.Calculate( inputVector, count, outputVector, oCount, pNeuronOutputs );
}


//==============================================================================
// 신경망을 통한 역전파 함수
void CMNistDoc::BackpropagateNeuralNet(double*	inputVector,
									   int		iCount,
									   double*	targetOutputVector, 
									   double*	actualOutputVector,
									   int		oCount, 
									   std::vector<std::vector<double>>* pMemorizedNeuronOutputs, 
									   BOOL		bDistort )
{
	// function to backpropagate through the neural net. 
	
	ASSERT( (inputVector != NULL) && (targetOutputVector != NULL) && (actualOutputVector != NULL) );

	////////////////////////////////////////////////////////////////////////////
	//
	// CODE REVIEW NEEDED:
	//
	// It does not seem worthwhile to backpropagate an error that's very small.
	// "Small" needs to be defined and for now, "small" is set to a fixed size
	// of pattern error ErrP <= 0.10 * MSE, then there will not be a backpropagation
	// of the error.  The current MSE is updated from the neural net dialog CDlgNeuralNet
	// 아주 작은 에러는 역전파를 할 가치가 없어 보인다. "작다"를 지금 정의할 필요가 있다
	// "작다"는 ErrP <= 0.10 * MSE의 패턴에러를 고정된 크기로 설정한다, 그러면
	// 에러 역전파를 하지 않는다. 현재 MSE는 신경망 다이알로그 CDlgNeuralNet에서 갱신한다

	BOOL bWorthwhileToBackpropagate;  /////// part of code review
	
	{	
		// local scope for capture of the neural net, only during the forward calculation step,
		// i.e., we release neural net for other threads after the forward calculation, and after we
		// have stored the outputs of each neuron, which are needed for the backpropagation step
		// 신경망의 포착을 위한 지역 범위, 단지 순방향 계산 단계 동안만 이다,
		// 즉, 우리는 순방향 계산 후에 다른 쓰레드에 대해 신경망을 해제했다
		// 그리고 각 뉴런의 출력을 저장한 후에, 이것은 역전파 단계에 대해 필요하다

		CAutoMutex tlo( m_utxNeuralNet );
		
		// determine if it's time to adjust the learning rate
		// 학습률 조정을 하기 위한 시점인지 결정한다
		if ( (( m_cBackprops % m_nAfterEveryNBackprops ) == 0) && (m_cBackprops != 0) )
		{
			double eta = m_NN.m_etaLearningRate;
			eta *= m_dEtaDecay;
			if ( eta < m_dMinimumEta )
				eta = m_dMinimumEta;
			m_NN.m_etaLearningRatePrevious = m_NN.m_etaLearningRate;
			m_NN.m_etaLearningRate = eta;
		}
		
		// determine if it's time to adjust the Hessian (currently once per epoch)
		// 헤시안 조정을 하기 위한 시점인지 결정한다 (현재는 한 세대당 한번)
		if ( (m_bNeedHessian != FALSE) || (( m_cBackprops % ::GetPreferences().m_nItemsTrainingImages ) == 0) )
		{
			// adjust the Hessian.  This is a lengthy operation, since it must process approx 500 labels
			// 헤시안을 조정한다. 이것은 긴 작업이다, 
			CalculateHessian();
			
			m_bNeedHessian = FALSE;
		}
		
		// determine if it's time to randomize the sequence of training patterns (currently once per epoch)
		// 훈련패턴의 순서를 버무릴 시점인지 결정한다 (현재는 한 세대당 한번)
		if ( ( m_cBackprops % ::GetPreferences().m_nItemsTrainingImages ) == 0 )
		{
			RandomizeTrainingPatternSequence();
		}

		// increment counter for tracking number of backprops
		// 역전파 횟수를 추적하기 위하여 계수기를 증가한다
		m_cBackprops++;

		// 신경망을 통해 순방향 계산을 한다 forward calculate through the neural net
		CalculateNeuralNet( inputVector,
							iCount,
							actualOutputVector,
							oCount,
							pMemorizedNeuronOutputs,
							bDistort );

		// calculate error in the output of the neural net
		// note that this code duplicates that found in many other places, and it's probably sensible to 
		// define a (global/static ??) function for it
		// 신경망의 출력에서 에러를 계산한다
		// 이 코드는 많은 다른곳에서 발견되는 중복이 있다는 점에 주의한다, 그리고
		// 이것은 대개 함수 정의(global/static ??)의 그 의미가 있다 
		double dMSE = 0.0;
		for ( int ii=0; ii<10; ++ii )
		{
			dMSE += ( actualOutputVector[ii]-targetOutputVector[ii] ) * ( actualOutputVector[ii]-targetOutputVector[ii] );
		}
		dMSE /= 2.0;

		if ( dMSE <= ( 0.10 * m_dEstimatedCurrentMSE ) )
		{
			bWorthwhileToBackpropagate = FALSE;
		}
		else
		{
			bWorthwhileToBackpropagate = TRUE;
		}

		if ( (bWorthwhileToBackpropagate != FALSE) && (pMemorizedNeuronOutputs == NULL) )
		{
			// the caller has not provided a place to store neuron outputs, so we need to
			// backpropagate now, while the neural net is still captured.  Otherwise, another thread
			// might come along and call CalculateNeuralNet(), which would entirely change the neuron
			// outputs and thereby inject errors into backpropagation 
			// 호출자는 뉴런 출력을 저장할 장소를 제공하지 않는다, 그래서 지금 역전파를
			// 해야한다, 신경망을 점유하는 동안에. 그렇지 않으면, 다른 쓰레드가 와서
			// CalculateNeuralNet()를 호출할 지 모른다, 이는 완전히 뉴런 출력을 변경하고
			// 그때문에 역전파에 에러를 더한다
			m_NN.Backpropagate( actualOutputVector, targetOutputVector, oCount, NULL );
			
			SetModifiedFlag( TRUE );
			
			// we're done, so return
			return;
		}
	}
	
	// if we have reached here, then the mutex for the neural net has been released for other 
	// threads. The caller must have provided a place to store neuron outputs, which we can 
	// use to backpropagate, even if other threads call CalculateNeuralNet() and change the outputs
	// of the neurons
	// 여기에 도달했다면, 그러면 신경망에 대한 뮤텍스를 다른 쓰레드를 위하여
	// 해제해야 한다. 호출자는 신경망 출력을 저장하기 위한 장소를 제공해야 한다,
	// 이는 역전파에서 사용할 수 있다, 다른 쓰레드가 CalculateNeuralNet()를 호출하고
	// 뉴런의 출력을 변경을 했어도
	if ( (bWorthwhileToBackpropagate != FALSE) )
	{
		m_NN.Backpropagate( actualOutputVector,
							targetOutputVector,
							oCount,
							pMemorizedNeuronOutputs );
		
		// set modified flag to prevent closure of doc without a warning
		// 경고없이 doc의 닫기를 방지하기 위하여 플래그 수정을 설정한다
		SetModifiedFlag( TRUE );
	}
}

//==============================================================================
// 헤시안을 계산한다
void CMNistDoc::CalculateHessian()
{
	// controls the Neural network's calculation if the diagonal Hessian for the Neural net
	// This will be called from a thread, so although the calculation is lengthy,
	// it should not interfere with the UI
	// 
	// we need the neural net exclusively during this calculation, so grab it now
	// 이 계산을 하는동안 신경망을 독점할 필요가 있다, 그래서 바로 수집한다
	CAutoMutex tlo( m_utxNeuralNet );
	
	double inputVector[841] = {0.0};  // note: 29x29, not 28x28
	double targetOutputVector[10] = {0.0};
	double actualOutputVector[10] = {0.0};

	// 28x28 = 784 이기 때문에 grayLevels[784]
	unsigned char grayLevels[g_cImageSize * g_cImageSize] = { 0 };
	int label = 0;
	int ii, jj;
	UINT kk;
	
	// calculate the diagonal Hessian using 500 random patterns, per Yann LeCun
	// 1998 "Gradient-Based Learning Applied To Document Recognition"
	
	// message to dialog that we are commencing calculation of the Hessian
	// 헤시안의 계산 시작을 다이알로그에 전달한다
	if ( m_hWndForBackpropPosting != NULL )
	{
		// wParam == 4L -> related to Hessian, lParam == 1L -> commenced calculation
		::PostMessage( m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 4L, 1L );
	}

	// some of this code is similar to the BackpropagationThread() code
	// 이 코드의 일부는 BackpropagationThread() 코드와 비슷하다
	m_NN.EraseHessianInformation();
	
	UINT numPatternsSampled = ::GetPreferences().m_nNumHessianPatterns ;
	
	for ( kk=0; kk<numPatternsSampled; ++kk )
	{
		// grayLevels 배열에 이미지 데이타를 가져온다
		GetRandomTrainingPattern( grayLevels, &label, TRUE );
		
		if ( label < 0 ) label = 0;
		if ( label > 9 ) label = 9;
		
		// pad to 29x29, convert to double precision
		for ( ii=0; ii<841; ++ii )
		{
			inputVector[ ii ] = 1.0;  // one is white, -one is black
		}

		// 입력벡터에 입력이미지 데이타를 넣는다
		// top row of inputVector is left as zero, left-most column is left as zero 
		for ( ii=0; ii<g_cImageSize; ++ii )		//행을 증가시킨다
		{
			for ( jj=0; jj<g_cImageSize; ++jj )	//열을 증가시킨다
			{
				inputVector[ 1 + jj + 29*(ii+1) ] =	//30번째 부터 넣는다
					(double)((int)(unsigned char)grayLevels[jj + g_cImageSize*ii]) /
					128.0 - 1.0;  // one is white, -one is black
			}
		}
		
		// 원하는 출력벡터 desired output vector
		for ( ii=0; ii<10; ++ii )
		{
			targetOutputVector[ ii ] = -1.0;
		}
		targetOutputVector[ label ] = 1.0;

		// apply distortion map to inputVector.  It's not certain that this is needed or helpful.
		// The second derivatives do NOT rely on the output of the neural net (i.e., because the 
		// second derivative of the MSE function is exactly 1 (one), regardless of the actual output
		// of the net).  However, since the backpropagated second derivatives rely on the outputs of
		// each neuron, distortion of the pattern might reveal previously-unseen information about the
		// nature of the Hessian.  But I am reluctant to give the full distortion, so I set the
		// severityFactor to only 2/3 approx
		
		GenerateDistortionMap( 0.65 );
		ApplyDistortionMap( inputVector );
		
		// 신경망을 순방향 계산한다 forward calculate the neural network
		m_NN.Calculate( inputVector, 841, actualOutputVector, 10, NULL );
		
		// 두번째 미분값을 역전파한다 backpropagate the second derivatives
		m_NN.BackpropagateSecondDervatives( actualOutputVector, targetOutputVector, 10 );
		
		// progress message to dialog that we are calculating the Hessian
		// 
		if ( kk%50 == 0 )
		{
			// every 50 iterations ...
			if ( m_hWndForBackpropPosting != NULL )
			{
				// wParam == 4L -> related to Hessian, lParam == 2L -> progress indicator
				::PostMessage( m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 4L, 2L );
			}
		}
		
		if ( m_bBackpropThreadAbortFlag != FALSE )
			break;
		
	}
	
	m_NN.DivideHessianInformationBy( (double)numPatternsSampled );
	
	// message to dialog that we are finished calculating the Hessian
	if ( m_hWndForBackpropPosting != NULL )
	{
		// wParam == 4L -> related to Hessian, lParam == 4L -> finished calculation
		::PostMessage( m_hWndForBackpropPosting, UWM_BACKPROPAGATION_NOTIFICATION, 4L, 4L );
	}
	
}

//==============================================================================
//
BOOL CMNistDoc::CanCloseFrame(CFrameWnd* pFrame) 
{
	// check if any threads are running before we allow the main frame to close down
	
	BOOL bRet = TRUE;
	
	if ( (m_bBackpropThreadsAreRunning != FALSE) || (m_bTestingThreadsAreRunning != FALSE) )
	{
		CString str;
		int iRet;
		
		str.Format( _T( "This will stop backpropagation and/or testing threads \n" )
					_T( "Are you certain that you wish to close the application? \n\n" )
					_T( "Click \"Yes\" to stop all threads and close the application \n" )
					_T( "Click \"No\" or \"Cancel\" to continue running the threads and the application " ) );
		
		iRet = ::MessageBox(NULL,
							str,
							_T( "Threads Are Running" ),
							MB_ICONEXCLAMATION | MB_YESNOCANCEL );
		
		if ( (iRet == IDYES) || (iRet == IDOK) )
		{
			bRet = TRUE;
			
			if ( m_bBackpropThreadsAreRunning != FALSE )
			{
				StopBackpropagation();
			}
			
			if ( m_bTestingThreadsAreRunning != FALSE )
			{
				StopTesting();
			}
		}
		else
		{
			bRet = FALSE;
		}
	}
	
	if ( bRet != FALSE )
	{
		// only call the base class if, so far, it's still safe to close.  If we
		// always called the base class, then we would get needless reminders to save the
		// current document.  These reminders are not needed, since we're not closing 
		// anyway
		
		bRet &= COleDocument::CanCloseFrame(pFrame);
	}
	
	return bRet;
}

//==============================================================================
// 역전파 쓰레드를 설정하고 생성한다
BOOL CMNistDoc::StartBackpropagation(UINT	iStartPattern /* =0 */,
									 UINT	iNumThreads /* =2 */,
									 HWND	hWnd /* =NULL */,
									 double	initialEta /* =0.005 */,
									 double minimumEta /* =0.000001 */,
									 double etaDecay /* =0.990 */, 
									 UINT	nAfterEvery  /* =1000 */,
									 BOOL	bDistortPatterns /* =TRUE */,
									 double estimatedCurrentMSE /* =1.0 */)
{
	if ( m_bBackpropThreadsAreRunning != FALSE )
		return FALSE;	//역전파 쓰레드 동작중이면 FALSE를 반환한다

	//처음 역전파를 실행했다
	m_bBackpropThreadAbortFlag = FALSE;		//역전파쓰레드 실행 플래그 설정
	m_bBackpropThreadsAreRunning = TRUE;	//동작중인 쓰레드 있음 으로 설정
	m_iNumBackpropThreadsRunning = 0;
	m_iBackpropThreadIdentifier = 0;
	m_cBackprops = iStartPattern;		//훈련패턴 위치 설정
	m_bNeedHessian = TRUE;
	
	m_iNextTrainingPattern = iStartPattern;
	m_hWndForBackpropPosting = hWnd;
	
	if ( m_iNextTrainingPattern < 0 ) 
		m_iNextTrainingPattern = 0;
	if ( m_iNextTrainingPattern >= ::GetPreferences().m_nItemsTrainingImages )
		m_iNextTrainingPattern = ::GetPreferences().m_nItemsTrainingImages - 1;
	
	if ( iNumThreads < 1 ) 
		iNumThreads = 1;
	if ( iNumThreads > 10 )  // 10 is arbitrary upper limit
		iNumThreads = 10;
	
	m_NN.m_etaLearningRate = initialEta;
	m_NN.m_etaLearningRatePrevious = initialEta;
	m_dMinimumEta = minimumEta;
	m_dEtaDecay = etaDecay;
	m_nAfterEveryNBackprops = nAfterEvery;
	m_bDistortTrainingPatterns = bDistortPatterns;

	// estimated number that will define whether a forward calculation's error
	// is significant enough to warrant backpropagation
	// 순방향 계산 오류가 역전파에 타당한 정도의 의미가있는지 아닌지를 정의하는 평가 수치
	m_dEstimatedCurrentMSE = estimatedCurrentMSE;

	RandomizeTrainingPatternSequence();

	// 설정한 개수만큼 쓰레스를 생성한다
	for ( UINT ii=0; ii<iNumThreads; ++ii )
	{
		//2) BackpropagationThread를 호출하고, 역전파 쓰레드를 생성한다
		CWinThread* pThread = ::AfxBeginThread( BackpropagationThread,	//
												(LPVOID)this, 
												THREAD_PRIORITY_BELOW_NORMAL,
												0,
												CREATE_SUSPENDED,	//동작보류
												NULL );
		
		if ( pThread == NULL )
		{
			// creation failed; un-do everything
			StopBackpropagation();
			return FALSE;
		}
		
		pThread->m_bAutoDelete = FALSE;		 
		m_pBackpropThreads[ ii ] = pThread;	//생성한 쓰레드 포인터를 저장
		pThread->ResumeThread();			//쓰레드 동작 시작
		m_iNumBackpropThreadsRunning++;		//쓰레드수 계수
	}
	
	return TRUE;
}


//==============================================================================
//
void CMNistDoc::StopBackpropagation()
{
	// stops all the backpropagation threads
	// 모든 역전파 쓰레드를 종료한다
	if ( m_bBackpropThreadsAreRunning == FALSE )
	{
		// it's curious to select "stop" if no threads are running, but perform some
		// shutdown safeguards, just to be certain
		
		m_bBackpropThreadAbortFlag = TRUE;
		m_bBackpropThreadsAreRunning = FALSE;
		m_iNumBackpropThreadsRunning = 0;
		m_iBackpropThreadIdentifier = 0;
		m_cBackprops = 0;
		
		return;
	}
	
	m_bBackpropThreadAbortFlag = TRUE;
	
	HANDLE hThread[25];
	CString msg;
	DWORD dwTimeOut = 5000;  // 5 second default timeout
	UINT ii;
	
	while ( m_iNumBackpropThreadsRunning > 0 )
	{
		for ( ii=0; ii<m_iNumBackpropThreadsRunning; ++ii )
		{
			hThread[ ii ] = m_pBackpropThreads[ ii ]->m_hThread;
		}
		
		DWORD dwRet = ::WaitForMultipleObjects( m_iNumBackpropThreadsRunning,
												hThread,
												FALSE,
												dwTimeOut );
		DWORD dwErr = ::GetLastError();		// if an error occurred get its value as soon as possible
		
		if ( dwRet==WAIT_FAILED )
		{
			LPVOID lpMsgBuf;
			::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
							NULL,
							dwErr,
							MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
							(LPTSTR) &lpMsgBuf,
							0,
							NULL );
			
			::MessageBox(NULL,
						 (LPCTSTR)lpMsgBuf,
						 _T("Error Waiting For Backpropagation Thread Shutdown"),
						 MB_OK | MB_ICONINFORMATION );

			LocalFree( lpMsgBuf );
		}
		else if ( dwRet==WAIT_TIMEOUT )
		{
			// bad -- no threads are responding
			// give user option of waiting a bit more, or of terminating the threads forcefully
			
			msg.Format( _T("No thread has responded after waiting %d milliseconds\n\n")
						_T("Click \"Retry\" to wait another %d milliseconds and give a thread\n")
						_T("a chance to respond\n\n")
						_T("Click \"Cancel\" to terminate a thread forcibly\n\n")
						_T("Note: Forceful termination is not recommended and might cause memory leaks"),
						dwTimeOut, dwTimeOut );

			if ( IDCANCEL == ::MessageBox( NULL, msg, _T("No Thread Is Responding"), MB_RETRYCANCEL|MB_ICONEXCLAMATION|MB_DEFBUTTON1 ) )
			{
				// forceful thread termination was selected
				// pick first thread from list and terminate it
				
				::TerminateThread( hThread[0], 98 );	// specify exit code of "98"
			}
			
		}
		else if ( dwRet>=WAIT_ABANDONED_0 && dwRet<=(WAIT_ABANDONED_0+m_iNumBackpropThreadsRunning-1) )
		{
			msg.Format( _T("Thread reports mutex was abandoned") );
			::MessageBox( NULL, msg, _T("Mutex Abandoned"), MB_OK|MB_ICONEXCLAMATION );
		}
		else if ( dwRet>=WAIT_OBJECT_0 && dwRet<=(WAIT_OBJECT_0+m_iNumBackpropThreadsRunning-1) )
		{
			// the most common and expected return value
			// delete the object that signalled
			
			int nDex = dwRet - WAIT_OBJECT_0;
			
			delete m_pBackpropThreads[ nDex ];
			
			for ( ii=nDex; ii<m_iNumBackpropThreadsRunning-1; ++ii )
			{
				m_pBackpropThreads[ ii ] = m_pBackpropThreads[ ii+1 ];
			}
			
			m_iNumBackpropThreadsRunning--;
			
		}
		else
		{
			ASSERT( FALSE );	// shouldn't be able to get here
		}
	}
	
	// at this point, all threads have been terminated, so re-set flags to allow
	// for future re-start of the threads
	m_bBackpropThreadAbortFlag = TRUE;
	m_bBackpropThreadsAreRunning = FALSE;
	m_iNumBackpropThreadsRunning = 0;
	m_iBackpropThreadIdentifier = 0;
	m_cBackprops = 0;
}


//==============================================================================
// 여기서 쓰레드로 신경망 역전파 훈련을 주기적으로 수행한다
UINT CMNistDoc::BackpropagationThread(LPVOID pVoid)
{
	// thread for backpropagation training of NN
	// 
	// thread is "owned" by the doc, and accepts a pointer to the doc
	// continuously backpropagates until m_bThreadAbortFlag is set to TRUE
	// 쓰레드는 Doc의 소유이다, 그리고 m_bBackpropThreadAbortFlag를 TRUE로 설정할 때 까지
	// CMNistDoc는 지속적으로 BackpropagationThread에 대한 포인터를 받아들인다.
	
	CMNistDoc* pThis = reinterpret_cast< CMNistDoc* >( pVoid );
	
	ASSERT( pThis != NULL );
	
	// 쓰레드 이름을 설정한다 (디버깅시 도움) set thread name (helps during debugging)
	char str[25] = {0};  // must use chars, not TCHARs, for SetThreadname function
	//sprintf( str, "BACKP%02d", pThis->m_iBackpropThreadIdentifier++ );
	sprintf_s( str, "BACKP%02d", pThis->m_iBackpropThreadIdentifier++ );	//  [2015.12.26 12:06 Jobs]
	SetThreadName( -1, str );

	//--------------------------------------------------------------------------
	// 작업을 한다 do the work
	double inputVector[841] = {0.0};  // note: 29x29, not 28x28
	double targetOutputVector[10] = {0.0};
	double actualOutputVector[10] = {0.0};
	double dMSE;
	UINT scaledMSE;
	unsigned char grayLevels[g_cImageSize * g_cImageSize] = { 0 };
	int label = 0;
	int ii, jj;
	UINT iSequentialNum;
	
	std::vector<std::vector<double>> memorizedNeuronOutputs;

	//--------------------------------------------------------------------------
	//3) m_bBackpropThreadAbortFlag를 TRUE로 설정할 때까지 반복한다
	while ( pThis->m_bBackpropThreadAbortFlag == FALSE )
	{
		// 다음 훈련패턴을 가져온다
		int iRet = pThis->GetNextTrainingPattern(grayLevels,
												 &label,
												 TRUE,
												 TRUE,
												 &iSequentialNum );
		
		if ( label < 0 ) label = 0;
		if ( label > 9 ) label = 9;
		
		// post message to the dialog, telling it which pattern this thread is currently working on
		// 다이얼로그에 메시지를 보낸다, 이 쓰레드가 지금 작업중인 패턴이 무엇인지 알린다
		if ( pThis->m_hWndForBackpropPosting != NULL )
		{
			::PostMessage(pThis->m_hWndForBackpropPosting,
						  UWM_BACKPROPAGATION_NOTIFICATION,
						  1L,
						  (LPARAM)iSequentialNum );
		}

		//29x29 로 채운다, double형 정밀도로 변환한다 pad to 29x29, convert to double precision
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
					(double)((int)(unsigned char)grayLevels[ jj + g_cImageSize*ii ]) /
					128.0 - 1.0;  // 1은 백색, -1은 흑색 one is white, -one is black
			}
		}
		
		// 원하는 출력벡터 desired output vector
		for ( ii=0; ii<10; ++ii )
		{
			targetOutputVector[ ii ] = -1.0;
		}
		targetOutputVector[ label ] = 1.0;
		
		// 여기서 역전파 now backpropagate
		pThis->BackpropagateNeuralNet(inputVector,
									  841,
									  targetOutputVector,
									  actualOutputVector,
									  10, 
									  &memorizedNeuronOutputs,
									  pThis->m_bDistortTrainingPatterns );
		
		// calculate error for this pattern and post it to the hwnd so it can calculate a running 
		// estimate of MSE
		// 이 패턴에 대한 에러를 계산하고 hwnd로 보낸다 그래서 MSE 산출 계산을 할 수 있다
		dMSE = 0.0;
		for ( ii=0; ii<10; ++ii )
		{
			dMSE += ( actualOutputVector[ii]-targetOutputVector[ii] ) *
					( actualOutputVector[ii]-targetOutputVector[ii] );
		}
		dMSE /= 2.0;

		// arbitrary large pre-agreed upon scale factor; taking sqrt is simply to improve the scaling
		// 임의로 크게 미리정한 비율; sqrt 선택은 평가의 개선이 쉽다
		scaledMSE = (UINT)( sqrt( dMSE ) * 2.0e8 );
		
		if ( pThis->m_hWndForBackpropPosting != NULL )
		{
			::PostMessage(pThis->m_hWndForBackpropPosting,
						  UWM_BACKPROPAGATION_NOTIFICATION,
						  2L,
						  (LPARAM)scaledMSE );
		}

		// determine the neural network's answer, and compare it to the actual answer.
		// Post a message if the answer was incorrect, so the dialog can display mis-recognition
		// statistics
		// 신경망의 답을 확인한다, 그리고 실제 답을 비교한다.답이 일치하면
		// 메시지를 보낸다, 그래서 다이얼로그에 인식실패 통계를 포시할 수 있다
		int iBestIndex = 0;
		double maxValue = -99.0;
		
		for ( ii=0; ii<10; ++ii )
		{
			if ( actualOutputVector[ ii ] > maxValue )
			{
				iBestIndex = ii;
				maxValue = actualOutputVector[ ii ];
			}
		}

		if ( iBestIndex != label )
		{
			// pattern was mis-recognized.  Notify the testing dialog
			// 패턴 인식실패를 했다. 시럼 다이얼로그에 알린다
			if ( pThis->m_hWndForBackpropPosting != NULL )
			{
				::PostMessage(pThis->m_hWndForBackpropPosting,
							  UWM_BACKPROPAGATION_NOTIFICATION,
							  8L,
							  (LPARAM)0L );
			}
		}
		
	}  // 메인 반복의 끝 end of main "while not abort flag" loop
	
	return 0L;
}

//==============================================================================
// 신경망 시험을 위한 쓰레드
BOOL CMNistDoc::StartTesting(UINT iStartingPattern,
							 UINT iNumThreads,
							 HWND hWnd,
							 BOOL bDistortPatterns,
							 UINT iWhichImageSet /* =1 */ )
{
	//--------------------------------------------------------------------------
	// 시험 쓰레드의 생성과 시작 creates and starts testing threads
	if ( m_bTestingThreadsAreRunning != FALSE )
		return FALSE;
	
	m_bTestingThreadAbortFlag = FALSE;
	m_bTestingThreadsAreRunning = TRUE;
	m_iNumTestingThreadsRunning = 0;
	m_iTestingThreadIdentifier = 0;
	
	m_iNextTestingPattern = iStartingPattern;
	m_hWndForTestingPosting = hWnd;
	m_iWhichImageSet = iWhichImageSet;
	
	if ( m_iWhichImageSet > 1 )
		m_iWhichImageSet = 1;
	// which is not possible, since m_iWhichImageSet is a UINT
	// 불가능한 것이다, m_iWhichImageSet는 UINT형이기 때문에
	if ( m_iWhichImageSet < 0 )
		m_iWhichImageSet = 0;
	
	if ( m_iNextTestingPattern < 0 ) 
		m_iNextTestingPattern = 0;
	if ( m_iNextTestingPattern >= ::GetPreferences().m_nItemsTestingImages )
		m_iNextTestingPattern = ::GetPreferences().m_nItemsTestingImages - 1;
	
	if ( iNumThreads < 1 ) 
		iNumThreads = 1;
	if ( iNumThreads > 10 )  // 10은 임의의 상한 10 is arbitrary upper limit
		iNumThreads = 10;
	
	m_bDistortTestingPatterns = bDistortPatterns;
	
	for ( UINT ii=0; ii<iNumThreads; ++ii )
	{
		// 시험을 위한 쓰레드를 생성한다
		CWinThread* pThread = ::AfxBeginThread( TestingThread,
												(LPVOID)this, 
												THREAD_PRIORITY_BELOW_NORMAL,
												0,
												CREATE_SUSPENDED,
												NULL );
		
		if ( pThread == NULL )
		{
			// 생성 실패; 모두 복원 creation failed; un-do everything
			StopTesting();
			return FALSE;
		}
		
		pThread->m_bAutoDelete = FALSE;		 
		m_pTestingThreads[ ii ] = pThread;
		pThread->ResumeThread();
		m_iNumTestingThreadsRunning++;
	}
	
	return TRUE;
}


//==============================================================================
// 신경망 시험을 종료하고 모든 설정을 복원한다
void CMNistDoc::StopTesting()
{
	//--------------------------------------------------------------------------
	// 모든 시험 쓰레드를 멈춘다 stops all the testing threads
	if ( m_bTestingThreadsAreRunning == FALSE )
	{
		// it's curious to select "stop" if no threads are running, but perform some
		// shutdown safeguards, just to be certain
		// 만약 쓰레드가 동작중이 아닐때 "stop"을 선택하면 특이한 것이다, 하지만
		// 일부 셧다운 보호를 수행한다, 단지 확신한다
		m_bTestingThreadAbortFlag = TRUE;
		m_bTestingThreadsAreRunning = FALSE;
		m_iNumTestingThreadsRunning = 0;
		m_iTestingThreadIdentifier = 0;
		return;
	}
	
	m_bTestingThreadAbortFlag = TRUE;
	
	HANDLE hThread[25];
	CString msg;
	DWORD dwTimeOut = 5000;  // 5 second default timeout
	UINT ii;
	
	while ( m_iNumTestingThreadsRunning > 0 )
	{
		for ( ii=0; ii<m_iNumTestingThreadsRunning; ++ii )
		{
			hThread[ ii ] = m_pTestingThreads[ ii ]->m_hThread;
		}
		
		DWORD dwRet = ::WaitForMultipleObjects( m_iNumTestingThreadsRunning,
												hThread,
												FALSE,
												dwTimeOut );

		// if an error occurred get its value as soon as possible
		DWORD dwErr = ::GetLastError();
		
		if ( dwRet==WAIT_FAILED )
		{
			LPVOID lpMsgBuf;
			::FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,
							 NULL,
							 dwErr,
							 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
							 (LPTSTR) &lpMsgBuf,
							 0,
							 NULL );
			
			::MessageBox(NULL,
						 (LPCTSTR)lpMsgBuf,
						 _T("Error Waiting For Testing Thread Shutdown"),
						 MB_OK | MB_ICONINFORMATION );

			LocalFree( lpMsgBuf );
		}
		else if ( dwRet==WAIT_TIMEOUT )
		{
			// bad -- no threads are responding
			// give user option of waiting a bit more, or of terminating the threads forcefully
			
			msg.Format( _T("No thread has responded after waiting %d milliseconds\n\n")
				_T("Click \"Retry\" to wait another %d milliseconds and give a thread\n")
				_T("a chance to respond\n\n")
				_T("Click \"Cancel\" to terminate a thread forcibly\n\n")
				_T("Note: Forceful termination is not recommended and might cause memory leaks"),
				dwTimeOut,
				dwTimeOut );

			if ( IDCANCEL == ::MessageBox(NULL,
										  msg,
										  _T("No Thread Is Responding"),
										  MB_RETRYCANCEL|MB_ICONEXCLAMATION|MB_DEFBUTTON1) )
			{
				// forceful thread termination was selected
				// pick first thread from list and terminate it
				
				::TerminateThread( hThread[0], 98 );	// specify exit code of "98"
			}
			
		}
		else if ( dwRet>=WAIT_ABANDONED_0 && dwRet<=(WAIT_ABANDONED_0+m_iNumTestingThreadsRunning-1) )
		{
			msg.Format( _T("Thread reports mutex was abandoned") );
			::MessageBox(NULL,
						 msg,
						 _T("Mutex Abandoned"),
						 MB_OK|MB_ICONEXCLAMATION );
		}
		else if ( dwRet>=WAIT_OBJECT_0 && dwRet<=(WAIT_OBJECT_0+m_iNumTestingThreadsRunning-1) )
		{
			// the most common and expected return value
			// delete the object that signalled
			
			int nDex = dwRet - WAIT_OBJECT_0;
			
			delete m_pTestingThreads[ nDex ];
			
			for ( ii=nDex; ii<m_iNumTestingThreadsRunning-1; ++ii )
			{
				m_pTestingThreads[ ii ] = m_pTestingThreads[ ii+1 ];
			}
			
			m_iNumTestingThreadsRunning--;
			
		}
		else
		{
			ASSERT( FALSE );	// shouldn't be able to get here
		}
	}
	
	
	// at this point, all threads have been terminated, so re-set flags to allow
	// for future re-start of the threads
	
	m_bTestingThreadAbortFlag = TRUE;
	m_bTestingThreadsAreRunning = FALSE;
	m_iNumTestingThreadsRunning = 0;
	m_iTestingThreadIdentifier = 0;
	
	
}

//==============================================================================
// 시험 쓰레드
UINT CMNistDoc::TestingThread(LPVOID pVoid)
{
	// thread for testing of Neural net
	// Continuously get the doc's next pattern, puts it through the neural net, and
	// inspects the output.  As the thread goes through the patterns, it post messages to the
	// m_hWndForTestingPosting, which presumably is the dialog that shows testing results,
	// advising it of the current pattern being tested.  If the actual output from the 
	// neural net differs from the desired output, another message is posted, advising the 
	// m_hWndForTestingPosting of the identity of the mis-recognized pattern
	// 신셩망의 시험을 위한 쓰레드
	// 지속적으로 Doc의 다음 패턴을 얻고, 신경망을 통해 패턴을 넣는다, 그리고
	// 출력을 검사한다. 쓰레드는 패턴을 통과하면서, m_hWndForTestingPosting 에
	// 메시지를 전달한다, 시험 결과를 보이기 다이알로그이다, 시험된 현재 패턴을
	// 알리는 것이다. 신경망의 실제 출력이 원하는 출력과 다른 경우, 다른 메시지가
	// 전달된다

	// thread is owned by the doc and accepts a pointer to the doc as a parameter

	CMNistDoc* pThis = reinterpret_cast< CMNistDoc* >( pVoid );
	
	ASSERT( pThis != NULL );
	
	// 쓰레드 이름 설정 set thread name (helps during debugging)
	
	char str[25] = {0};  // must use chars, not TCHARs, for SetThreadname function
	//sprintf( str, "TEST%02d", pThis->m_iTestingThreadIdentifier++ );
	sprintf_s( str, "TEST%02d", pThis->m_iTestingThreadIdentifier++ );	//  [2015.12.26 12:06 Jobs]
	SetThreadName( -1, str );
	
	// do the work
	
	double inputVector[841] = {0.0};  // note: 29x29, not 28x28
	double targetOutputVector[10] = {0.0};
	double actualOutputVector[10] = {0.0};

	double dPatternMSE = 0.0;
	double dTotalMSE = 0.0;
	UINT scaledMSE = 0;
	UINT iPatternsProcessed = 0;

	// 실제 이미지를 저장할 메모리를 할당한다
	unsigned char grayLevels[g_cImageSize * g_cImageSize] = { 0 };
	int label = 0;
	//int ii, jj;	//  [2015.12.26 12:08 Jobs]
	UINT iPatNum, iSequentialNum;
	
	while ( pThis->m_bTestingThreadAbortFlag == FALSE )
	{
		// 시험 이미지 집합 또는 훈련 이미지 집합 testing image set or training image set
		
		if ( pThis->m_iWhichImageSet == 1 )
		{
			// 시험집합 testing set
			
			iPatNum = pThis->GetNextTestingPattern( grayLevels, &label, TRUE );
			
			// post message to the dialog, telling it which pattern this thread is currently working on
			
			if ( pThis->m_hWndForTestingPosting != NULL )
			{
				::PostMessage(pThis->m_hWndForTestingPosting,
							  UWM_TESTING_NOTIFICATION,
							  1L,
							  (LPARAM)iPatNum );
			}
		}
		else
		{
			// 훈련집합 training set
			
			iPatNum = pThis->GetNextTrainingPattern(grayLevels,
													&label,
													TRUE,
													FALSE,
													&iSequentialNum );
			
			// post message to the dialog, telling it which pattern this thread is currently working on
			
			if ( pThis->m_hWndForTestingPosting != NULL )
			{
				::PostMessage(pThis->m_hWndForTestingPosting,
							  UWM_TESTING_NOTIFICATION,
							  1L,
							  (LPARAM)iSequentialNum );
			}
		}
		
		
		if ( label < 0 ) label = 0;
		if ( label > 9 ) label = 9;
		
		// pad to 29x29, convert to double precision
		
		for ( int ii=0; ii<841; ++ii )
		{
			inputVector[ ii ] = 1.0;  // one is white, -one is black
		}
		
		// top row of inputVector is left as zero, left-most column is left as zero 
		
		for ( UINT ii=0; ii<g_cImageSize; ++ii )
		{
			for ( UINT jj=0; jj<g_cImageSize; ++jj )
			{
				inputVector[ 1 + jj + 29*(ii+1) ] =
					(double)((int)(unsigned char)grayLevels[ jj + g_cImageSize*ii ]) /
					128.0 - 1.0;  // one is white, -one is black
			}
		}
		
		// desired output vector
		
		for ( int ii=0; ii<10; ++ii )
		{
			targetOutputVector[ ii ] = -1.0;
		}
		targetOutputVector[ label ] = 1.0;
		
		//----------------------------------------------------------------------
		// now calculate output of neural network
		pThis->CalculateNeuralNet(inputVector,			//입력뉴런(이미지)
								  841,					//입력뉴런 개수
								  actualOutputVector,	//출력뉴런
								  10,					//출력 개수
								  NULL,					//출력 값
								  pThis->m_bDistortTestingPatterns );//왜곡적용 여부

		
		// calculate error for this pattern and accumulate it for posting of
		// total MSE of all patterns when thread is exiting
		
		dPatternMSE = 0.0;
		for ( int ii=0; ii<10; ++ii )
		{
			dPatternMSE += ( actualOutputVector[ii]-targetOutputVector[ii] ) *
						   ( actualOutputVector[ii]-targetOutputVector[ii] );
		}
		dPatternMSE /= 2.0;

		dTotalMSE += dPatternMSE;
		++iPatternsProcessed;		
		

		// determine the neural network's answer, and compare it to the actual answer
		
		int iBestIndex = 0;
		double maxValue = -99.0;
		UINT code;
		
		for ( int ii=0; ii<10; ++ii )
		{
			if ( actualOutputVector[ ii ] > maxValue )
			{
				iBestIndex = ii;
				maxValue = actualOutputVector[ ii ];
			}
		}
		
		
		// moment of truth: Did neural net get the correct answer
		
		if ( iBestIndex != label )
		{
			// pattern was mis-recognized.  Notify the testing dialog
			
			// lParam is built to contain a coded bit pattern, as follows:
			//
			//  0          1          2         3
			//  0123456 7890123 456789012345678901
			// |  act  |  tar  |    pattern num   |
			//
			// where act == actual output of the neural net, and tar == target
			// this gives 2^7 = 128 possible outputs (only 10 are needed here... future expansion??)
			// and 2^18 = 262144 possible pattern numbers ( only 10000 are needed here )
			
			code  = ( iPatNum    & 0x0003FFFF );
			code |= ( label      & 0x0000007F ) << 18;
			code |= ( iBestIndex & 0x0000007F ) << 25;
			
			if ( pThis->m_hWndForTestingPosting != NULL )
			{
				::PostMessage(pThis->m_hWndForTestingPosting,
							  UWM_TESTING_NOTIFICATION,
							  2L,
							  (LPARAM)code );
			}
		}
	}


	// post the total MSE of tested patterns to the hwnd

	double divisor = (double)( (iPatternsProcessed>1) ? iPatternsProcessed : 1 );
	dTotalMSE /= divisor;
	// arbitrary large pre-agreed upon scale factor; taking sqrt is simply to improve the scaling
	scaledMSE = (UINT)( sqrt( dTotalMSE ) * 2.0e8 );
		
	if ( pThis->m_hWndForTestingPosting != NULL )
	{
		::PostMessage(pThis->m_hWndForTestingPosting,
					  UWM_TESTING_NOTIFICATION,
					  4L,
					  (LPARAM)scaledMSE );
	}
	
	return 0L;
}


#undef UNIFORM_ZERO_THRU_ONE
