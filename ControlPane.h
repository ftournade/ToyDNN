#pragma once

class CControlPane : public CPaneDialog
{
public:
	DECLARE_DYNAMIC( CControlPane )

public:
	CControlPane() {}
	virtual ~CControlPane() {}

protected:
	DECLARE_MESSAGE_MAP()

	afx_msg LRESULT HandleInitDialog( WPARAM, LPARAM );
	afx_msg void OnExampleChanged();
	afx_msg void OnStartStopTraining();
	afx_msg void OnResetTraining();
	afx_msg void OnGradientCheck();

	afx_msg void OnTimer( UINT_PTR );

	afx_msg void OnUpdateUI( CCmdUI* pCmdUI );
	
	virtual void DoDataExchange( CDataExchange* pDX );

	CComboBox m_SelectExample;
	float m_LearningRate = 0.02f;
	UINT m_BatchSize = 32;
	UINT m_ValidationInterval = 10;

	BOOL m_IsTraining = FALSE;

	UINT_PTR m_DisplayRefreshTimer = 0;
};
