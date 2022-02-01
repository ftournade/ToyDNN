#include "pch.h"
#include "ControlPane.h"
#include "resource.h"

IMPLEMENT_DYNAMIC( CControlPane, CPaneDialog )

BEGIN_MESSAGE_MAP( CControlPane, CPaneDialog )
	ON_MESSAGE( WM_INITDIALOG, HandleInitDialog )
	ON_CBN_SELCHANGE( IDC_COMBO_SELECT_EXAMPLE, OnExampleChanged )
	ON_BN_CLICKED( IDC_BUTTON_TRAIN, OnStartStopTraining )
	ON_BN_CLICKED( IDC_BUTTON_RESET_TRAINING, OnResetTraining )
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

	}
	else
	{

	}
}


void CControlPane::OnResetTraining()
{

}
