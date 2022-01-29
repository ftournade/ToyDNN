
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "ToyDNN.h"
#include "ChildView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CChildView

CChildView::CChildView()
{
}

CChildView::~CChildView()
{
}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_RBUTTONDOWN()
	ON_WM_RBUTTONUP()
	ON_WM_MOUSEMOVE()
END_MESSAGE_MAP()



// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs) 
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS, 
		::LoadCursor(nullptr, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW+1), nullptr);

	return TRUE;
}

void CChildView::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	
	CRect r;
	GetClientRect( &r );

	if( (r.Width() != m_backBuffer.GetWidth()) ||
		(r.Height() != m_backBuffer.GetHeight()) )
	{
		if( !m_backBuffer.Init( GetSafeHwnd(), r.Width(), r.Height() ) )
		{
			//TODO error handling
			ASSERT( false );
		}
	}

	CDC* pMemDC = CDC::FromHandle( m_backBuffer.GetBackBufferDC() );

	theApp.m_Example.Tick( *pMemDC );

	m_backBuffer.Blit( dc.GetSafeHdc() );

	Invalidate( FALSE );
}



void CChildView::OnLButtonDown( UINT nFlags, CPoint point )
{
	theApp.m_Example.OnLMouseButtonDown( point );

	CWnd::OnLButtonDown( nFlags, point );
}


void CChildView::OnLButtonUp( UINT nFlags, CPoint point )
{
	theApp.m_Example.OnLMouseButtonUp( point );

	CWnd::OnLButtonUp( nFlags, point );
}


void CChildView::OnRButtonDown( UINT nFlags, CPoint point )
{
	theApp.m_Example.OnRMouseButtonDown( point );

	CWnd::OnRButtonDown( nFlags, point );
}


void CChildView::OnRButtonUp( UINT nFlags, CPoint point )
{
	theApp.m_Example.OnRMouseButtonUp( point );

	CWnd::OnRButtonUp( nFlags, point );
}


void CChildView::OnMouseMove( UINT nFlags, CPoint point )
{
	theApp.m_Example.OnMouseMove( point );

	CWnd::OnMouseMove( nFlags, point );
}
