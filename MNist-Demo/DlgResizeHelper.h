////////////////////////////////////////////////////////////////////////
// CResizeHelper_Dlg
//	
// Author: Stephan Keil (Stephan.Keil@gmx.de)
// Date:   2000-06-26
//
// 크기변경에 대한 다이알로그 배치의 유지를 도와준다
// 이것은 자동으로 Init()(또한 직접적으로 Add()를 호출하여 다른창을 추가할 수
// 있다)를 호출하여 모든 자식창을 수집하고 OnResize()에서 크기를 변경한다.
// 기본 크기변경은 부모창에 비례한다 하지만 다양한 Fix() 멤버를 호출하여 일부
// 또는 모든 자식 창의 동작을 변경할 수 있다.

// original article can be found at 
// http://www.codeguru.com/Cpp/W-D/dislog/resizabledialogs/article.php/c1913/


#ifndef CRESIZEHELPER_DLG_H_
#define CRESIZEHELPER_DLG_H_

//#pragma warning (disable: 4786)
#include <list>

//==============================================================================
// 다이알로그 크기변경 도우미
class CResizeHelper_Dlg
{
public:

	// 수평 크기/위치를 정한다 fix horizontal dimension/position
	enum EHFix
	{
		kNoHFix     = 0,
		kWidth      = 1,
		kLeft       = 2,
		kRight      = 4,
		kWidthLeft  = 3,
		kWidthRight = 5,
		kLeftRight  = 6
	};

	// 수직 크기/위치를 정한다 fix vertical dimension/position
	enum EVFix
	{
		kNoVFix       = 0,
		kHeight       = 1,
		kTop          = 2,
		kBottom       = 4,
		kHeightTop    = 3,
		kHeightBottom = 5,
		kTopBottom    = 6
	};

	// 부모창 초기화, 모든 자식창은 이미 원래의 위치/크기를 가지고 있어야 한다.
	void Init(HWND a_hParent);

	// 직접 자식창의 목록에 창을 추가한다 (예, 형제 창)
	// 참고: 창을 추가하기 전에 Init()를 호출할 수 있다
	void Add(HWND a_hWnd);

	// fix position/dimension for a child window, determine child by...
	// 자식창을 위하여 위치/크기를 정한다, ~로 자식 결정
	// ...HWND...
	BOOL Fix(HWND a_hCtrl, EHFix a_hFix, EVFix a_vFix);
	// ...item ID (if it's a dialog item)...
	BOOL Fix(int a_itemId, EHFix a_hFix, EVFix a_vFix);
	// ...all child windows with a common class name (e.g. "Edit")
	UINT Fix(LPCTSTR a_pszClassName, EHFix a_hFix, EVFix a_vFix);
	// ...or all registered windows
	BOOL Fix(EHFix a_hFix, EVFix a_vFix);

	// 부모창의 변경에 따라 자식창의 크기를 변경하고 속성을 수정한다
	void OnSize();

private:
	struct CtrlSize
	{
		CRect	m_origSize;
		HWND	m_hCtrl;
		EHFix	m_hFix;
		EVFix	m_vFix;
		CtrlSize() : m_hFix(kNoHFix), m_vFix(kNoVFix) {}
	};

	typedef std::list<CtrlSize> CtrlCont_t;
	CtrlCont_t	m_ctrls;
	HWND		m_hParent;
	CRect		m_origParentSize;
};

#endif // CResizeHelper_Dlg_H_
