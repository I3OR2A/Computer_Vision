
// Computer_Vision_Hw1.h : PROJECT_NAME ���ε{�����D�n���Y��
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�� PCH �]�t���ɮ׫e���]�t 'stdafx.h'"
#endif

#include "resource.h"		// �D�n�Ÿ�


// CComputer_Vision_Hw1App:
// �аѾ\��@�����O�� Computer_Vision_Hw1.cpp
//

class CComputer_Vision_Hw1App : public CWinApp
{
public:
	CComputer_Vision_Hw1App();

// �мg
public:
	virtual BOOL InitInstance();

// �{���X��@

	DECLARE_MESSAGE_MAP()
};

extern CComputer_Vision_Hw1App theApp;