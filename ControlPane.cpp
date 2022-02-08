#include "pch.h"
#include "ControlPane.h"
#include "ToyDNN.h"
#include "MainFrm.h"
#include <thread>

#define REFRESH_INTERVAL_IN_MS 5000

void TrainingThread( HyperParameters _params )
{
	Log( "Training thread started\n" );

	theApp.m_pExample->TrainingThread( _params );

	Log( "Training thread stopped\n" );
}

IMPLEMENT_DYNAMIC( CControlPane, CPaneDialog )

BEGIN_MESSAGE_MAP( CControlPane, CPaneDialog )
	ON_MESSAGE( WM_INITDIALOG, HandleInitDialog )
	ON_CBN_SELCHANGE( IDC_COMBO_SELECT_EXAMPLE, OnExampleChanged )
	ON_BN_CLICKED( IDC_BUTTON_TRAIN, OnStartStopTraining )
	ON_BN_CLICKED( IDC_BUTTON_RESET_TRAINING, OnResetTraining )
	ON_BN_CLICKED( IDC_BUTTON_GRADIENT_CHECK, OnGradientCheck )
	ON_WM_TIMER()

	//workaround MFC bug https://social.msdn.microsoft.com/Forums/vstudio/en-US/2fd5c2a8-9e20-4d01-b02f-6be3c2f13220/button-is-disabled-by-using-cpanedialog?forum=vcgeneral
	ON_UPDATE_COMMAND_UI( IDC_BUTTON_TRAIN, OnUpdateUI )
	ON_UPDATE_COMMAND_UI( IDC_BUTTON_RESET_TRAINING, OnUpdateUI )
	ON_UPDATE_COMMAND_UI( IDC_BUTTON_GRADIENT_CHECK, OnUpdateUI )
END_MESSAGE_MAP()

LRESULT CControlPane::HandleInitDialog( WPARAM wParam, LPARAM lParam )
{
	if( !CPaneDialog::HandleInitDialog( wParam, lParam ) )
		return FALSE;

	m_SelectExample.AddString( _T( "Example1" ) );
	m_SelectExample.AddString( _T( "Example2" ) );
	m_SelectExample.AddString( _T( "Example3" ) );
	m_SelectExample.AddString( _T( "Example4" ) );
	m_SelectExample.SetCurSel( 0 );

	return TRUE;
}

void CControlPane::OnUpdateUI( CCmdUI* pCmdUI )
{
	//Workaround MFC bug

	if( pCmdUI->m_nID == IDC_BUTTON_TRAIN )
		pCmdUI->Enable(TRUE);
	else
		pCmdUI->Enable( !m_IsTraining );
	
}

void CControlPane::OnTimer( UINT_PTR _timer )
{
	assert( _timer == m_DisplayRefreshTimer );
	((CMainFrame*)theApp.GetMainWnd())->GetChildView().Invalidate( FALSE );
}


void CControlPane::DoDataExchange( CDataExchange* pDX )
{
	CPaneDialog::DoDataExchange( pDX );

	DDX_Control( pDX, IDC_COMBO_SELECT_EXAMPLE, m_SelectExample );
	DDX_Text( pDX, IDC_EDIT_LEARNING_RATE, m_LearningRate );
	DDV_MinMaxFloat( pDX, m_LearningRate, 0.0f, 1.0f );
	DDX_Text( pDX, IDC_EDIT_BATCH_SIZE, m_BatchSize );
	DDV_MinMaxUInt( pDX, m_BatchSize, 1, 1000000 );
	DDX_Text( pDX, IDC_EDIT_VALIDATION_INTERVAL, m_ValidationInterval );
	DDV_MinMaxUInt( pDX, m_ValidationInterval, 1, 1000000 );
}

void CControlPane::OnExampleChanged()
{
	int sel = m_SelectExample.GetCurSel();

	//Do something better (example registration system for example)
	switch( sel )
	{
		case 0:
			theApp.m_pExample = std::make_unique<Example1>();
			break;
		case 1:
			theApp.m_pExample = std::make_unique<Example2>();
			break;
		case 2:
			theApp.m_pExample = std::make_unique<Example3>();
			break;
		case 3:
			theApp.m_pExample = std::make_unique<Example4>();
			break;
	}

	((CMainFrame*)theApp.GetMainWnd())->GetChildView().Invalidate( FALSE );
}

void CControlPane::OnStartStopTraining()
{
	m_IsTraining = !m_IsTraining;

	BOOL bEnableControls = m_IsTraining ? FALSE : TRUE;

	GetDlgItem( IDC_EDIT_LEARNING_RATE )->EnableWindow( bEnableControls );
	GetDlgItem( IDC_EDIT_BATCH_SIZE )->EnableWindow( bEnableControls );
	GetDlgItem( IDC_EDIT_VALIDATION_INTERVAL )->EnableWindow( bEnableControls );
	GetDlgItem( IDC_BUTTON_RESET_TRAINING )->EnableWindow( bEnableControls );
	GetDlgItem( IDC_COMBO_SELECT_EXAMPLE )->EnableWindow( bEnableControls );

	GetDlgItem( IDC_BUTTON_TRAIN )->SetWindowText( m_IsTraining ? _T("Pause training") : _T( "Start training" ) );

	if( m_IsTraining )
	{
		UpdateData( TRUE );
		HyperParameters params;
		params.BatchSize = m_BatchSize;
		params.LearningRate = m_LearningRate;
		params.ValidationInterval = m_ValidationInterval;

		std::thread trainingThread( TrainingThread, params );
		trainingThread.detach();

		m_DisplayRefreshTimer = SetTimer( IDT_REFRESH_DISPLAY, REFRESH_INTERVAL_IN_MS, nullptr );
	}
	else
	{
		theApp.m_pExample->StopTraining();
	
		KillTimer( m_DisplayRefreshTimer );

		((CMainFrame*)theApp.GetMainWnd())->GetChildView().Invalidate( FALSE );
	}
}


void CControlPane::OnResetTraining()
{
	::MessageBox( NULL, _T( "Not yet implemented" ), _T( "Error" ), MB_OK );
}

void CControlPane::OnGradientCheck()
{
	theApp.m_pExample->GradientCheck();
}