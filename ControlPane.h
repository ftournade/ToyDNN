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

	afx_msg void OnStartStopTraining();
	afx_msg void OnResetTraining();

	virtual void DoDataExchange( CDataExchange* pDX );

	float m_LearningRate = 0.001f;
	UINT m_BatchSize = 32;
	UINT m_ValidationInterval = 10;

	BOOL m_IsTraining = FALSE;
};
