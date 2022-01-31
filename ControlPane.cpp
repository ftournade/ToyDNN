#include "pch.h"
#include "ControlPane.h"
#include "resource.h"

IMPLEMENT_DYNAMIC( CControlPane, CPaneDialog )

BEGIN_MESSAGE_MAP( CControlPane, CPaneDialog )
	ON_BN_CLICKED( IDC_BUTTON_TRAIN, OnStartStopTraining )
	ON_BN_CLICKED( IDC_BUTTON_RESET_TRAINING, OnResetTraining )
END_MESSAGE_MAP()


void CControlPane::DoDataExchange( CDataExchange* pDX )
{
	CPaneDialog::DoDataExchange( pDX );

	DDX_Text( pDX, IDC_EDIT_LEARNING_RATE, m_LearningRate );
	DDX_Text( pDX, IDC_EDIT_BATCH_SIZE, m_BatchSize );
	DDX_Text( pDX, IDC_EDIT_VALIDATION_INTERVAL, m_ValidationInterval );
}

void CControlPane::OnStartStopTraining()
{
	m_IsTraining = !m_IsTraining;

	BOOL bEnableControls = m_IsTraining ? FALSE : TRUE;

	GetDlgItem( IDC_EDIT_LEARNING_RATE )->EnableWindow( bEnableControls );
	GetDlgItem( IDC_EDIT_BATCH_SIZE )->EnableWindow( bEnableControls );
	GetDlgItem( IDC_EDIT_VALIDATION_INTERVAL )->EnableWindow( bEnableControls );
	GetDlgItem( IDC_BUTTON_RESET_TRAINING )->EnableWindow( bEnableControls );

	GetDlgItem( IDC_BUTTON_TRAIN )->SetWindowText( m_IsTraining ? _T("Pause training") : _T( "Start training" ) );
}


void CControlPane::OnResetTraining()
{

}
