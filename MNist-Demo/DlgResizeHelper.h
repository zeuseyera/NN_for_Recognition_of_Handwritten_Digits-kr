////////////////////////////////////////////////////////////////////////
// CResizeHelper_Dlg
//	
// Author: Stephan Keil (Stephan.Keil@gmx.de)
// Date:   2000-06-26
//
// ũ�⺯�濡 ���� ���̾˷α� ��ġ�� ������ �����ش�
// �̰��� �ڵ����� Init()(���� ���������� Add()�� ȣ���Ͽ� �ٸ�â�� �߰��� ��
// �ִ�)�� ȣ���Ͽ� ��� �ڽ�â�� �����ϰ� OnResize()���� ũ�⸦ �����Ѵ�.
// �⺻ ũ�⺯���� �θ�â�� ����Ѵ� ������ �پ��� Fix() ����� ȣ���Ͽ� �Ϻ�
// �Ǵ� ��� �ڽ� â�� ������ ������ �� �ִ�.

// original article can be found at 
// http://www.codeguru.com/Cpp/W-D/dislog/resizabledialogs/article.php/c1913/


#ifndef CRESIZEHELPER_DLG_H_
#define CRESIZEHELPER_DLG_H_

//#pragma warning (disable: 4786)
#include <list>

//==============================================================================
// ���̾˷α� ũ�⺯�� �����
class CResizeHelper_Dlg
{
public:

	// ���� ũ��/��ġ�� ���Ѵ� fix horizontal dimension/position
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

	// ���� ũ��/��ġ�� ���Ѵ� fix vertical dimension/position
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

	// �θ�â �ʱ�ȭ, ��� �ڽ�â�� �̹� ������ ��ġ/ũ�⸦ ������ �־�� �Ѵ�.
	void Init(HWND a_hParent);

	// ���� �ڽ�â�� ��Ͽ� â�� �߰��Ѵ� (��, ���� â)
	// ����: â�� �߰��ϱ� ���� Init()�� ȣ���� �� �ִ�
	void Add(HWND a_hWnd);

	// fix position/dimension for a child window, determine child by...
	// �ڽ�â�� ���Ͽ� ��ġ/ũ�⸦ ���Ѵ�, ~�� �ڽ� ����
	// ...HWND...
	BOOL Fix(HWND a_hCtrl, EHFix a_hFix, EVFix a_vFix);
	// ...item ID (if it's a dialog item)...
	BOOL Fix(int a_itemId, EHFix a_hFix, EVFix a_vFix);
	// ...all child windows with a common class name (e.g. "Edit")
	UINT Fix(LPCTSTR a_pszClassName, EHFix a_hFix, EVFix a_vFix);
	// ...or all registered windows
	BOOL Fix(EHFix a_hFix, EVFix a_vFix);

	// �θ�â�� ���濡 ���� �ڽ�â�� ũ�⸦ �����ϰ� �Ӽ��� �����Ѵ�
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
