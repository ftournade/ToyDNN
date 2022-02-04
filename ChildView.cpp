
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
	Log( "Begin: OnPaint\n" );

	CPaintDC dc(this); // device context for painting

	theApp.m_pExample->PauseTraining();


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

	pMemDC->FillSolidRect( &r, RGB( 200, 200, 200 ) );
	theApp.m_pExample->Draw( *pMemDC );

	m_backBuffer.Blit( dc.GetSafeHdc() );

	theApp.m_pExample->ResumeTraining();
	Log( "End: OnPaint\n" );
}



void CChildView::OnLButtonDown( UINT nFlags, CPoint point )
{
	if( theApp.m_pExample->OnLMouseButtonDown( point ) )
		Invalidate( FALSE );

	CWnd::OnLButtonDown( nFlags, point );
}


void CChildView::OnLButtonUp( UINT nFlags, CPoint point )
{
	if( theApp.m_pExample->OnLMouseButtonUp( point ) )
		Invalidate( FALSE );

	CWnd::OnLButtonUp( nFlags, point );
}


void CChildView::OnRButtonDown( UINT nFlags, CPoint point )
{
	if( theApp.m_pExample->OnRMouseButtonDown( point ) )
		Invalidate( FALSE );

	CWnd::OnRButtonDown( nFlags, point );
}


void CChildView::OnRButtonUp( UINT nFlags, CPoint point )
{
	if( theApp.m_pExample->OnRMouseButtonUp( point ) )
		Invalidate( FALSE );

	CWnd::OnRButtonUp( nFlags, point );
}


void CChildView::OnMouseMove( UINT nFlags, CPoint point )
{
	if( theApp.m_pExample->OnMouseMove( point ) )
		Invalidate( FALSE );

	CWnd::OnMouseMove( nFlags, point );
}
