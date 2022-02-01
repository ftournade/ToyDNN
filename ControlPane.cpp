#include "pch.h"
#include "ControlPane.h"
#include "ToyDNN.h"
#include "MainFrm.h"
#include <thread>

void TrainingThread( HyperParameters _params )
{
	Log( "Training started" );

	theApp.m_pExample->TrainingThread( _params );

	Log( "Training stopped" );
}

IMPLEMENT_DYNAMIC( CControlPane, CPaneDialog )

BEGIN_MESSAGE_MAP( CControlPane, CPaneDialog )
	ON_MESSAGE( WM_INITDIALOG, HandleInitDialog )
	ON_CBN_SELCHANGE( IDC_COMBO_SELECT_EXAMPLE, OnExampleChanged )
	ON_BN_CLICKED( IDC_BUTTON_TRAIN, OnStartStopTraining )
	ON_BN_CLICKED( IDC_BUTTON_RESET_TRAINING, OnResetTraining )
	ON_WM_TIMER()

	//workaround MFC bug https://social.msdn.microsoft.com/Forums/vstudio/en-US/2fd5c2a8-9e20-4d01-b02f-6be3c2f13220/button-is-disabled-by-using-cpanedialog?forum=vcgeneral
	ON_UPDATE_COMMAND_UI( IDC_BUTTON_TRAIN, OnUpdateUI )
	ON_UPDATE_COMMAND_UI( IDC_BUTTON_RESET_TRAINING, OnUpdateUI )
END_MESSAGE_MAP()

LRESULT CControlPane::HandleInitDialog( WPARAM wParam, LPARAM lParam )
{
	if( !CPaneDialog::HandleInitDialog( wParam, lParam ) )
		return FALSE;

	m_SelectExample.AddString( _T( "Example1" ) );
	m_SelectExample.AddString( _T( "Example2" ) );
	m_SelectExample.AddString( _T( "Example3" ) );
	m_SelectExample.SetCurSel( 0 );

	return TRUE;
}

void CControlPane::OnUpdateUI( CCmdUI* pCmdUI )
{
	pCmdUI->Enable();
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

	sel++;
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

		m_DisplayRefreshTimer = SetTimer( IDT_REFRESH_DISPLAY, 1000, nullptr );
	}
	else
	{
		theApp.m_pExample->StopTraining();
	
		KillTimer( m_DisplayRefreshTimer );
	}
}


void CControlPane::OnResetTraining()
{

}
