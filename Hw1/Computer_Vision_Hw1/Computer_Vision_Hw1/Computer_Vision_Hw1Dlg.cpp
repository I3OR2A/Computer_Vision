
// Computer_Vision_Hw1Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "Computer_Vision_Hw1.h"
#include "Computer_Vision_Hw1Dlg.h"
#include "afxdialogex.h"
#include "calibration.h"
#include "stereo.h"
#include "homework.h"
#include "bmp.h"
#include <cmath>
#include <omp.h>
#define M_PI 3.14159265
#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 對 App About 使用 CAboutDlg 對話方塊

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

	// 對話方塊資料
	enum { IDD = IDD_ABOUTBOX };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

	// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CComputer_Vision_Hw1Dlg 對話方塊



CComputer_Vision_Hw1Dlg::CComputer_Vision_Hw1Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CComputer_Vision_Hw1Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CComputer_Vision_Hw1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CComputer_Vision_Hw1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CComputer_Vision_Hw1Dlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON1, &CComputer_Vision_Hw1Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON3, &CComputer_Vision_Hw1Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON6, &CComputer_Vision_Hw1Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON2, &CComputer_Vision_Hw1Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON4, &CComputer_Vision_Hw1Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CComputer_Vision_Hw1Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON7, &CComputer_Vision_Hw1Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CComputer_Vision_Hw1Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CComputer_Vision_Hw1Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &CComputer_Vision_Hw1Dlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &CComputer_Vision_Hw1Dlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON12, &CComputer_Vision_Hw1Dlg::OnBnClickedButton12)
	ON_BN_CLICKED(IDC_BUTTON13, &CComputer_Vision_Hw1Dlg::OnBnClickedButton13)
END_MESSAGE_MAP()


// CComputer_Vision_Hw1Dlg 訊息處理常式

BOOL CComputer_Vision_Hw1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	AllocConsole();
	freopen ("CONOUT$", "w", stdout );

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void CComputer_Vision_Hw1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void CComputer_Vision_Hw1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR CComputer_Vision_Hw1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CComputer_Vision_Hw1Dlg::OnBnClickedOk()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	CDialogEx::OnOK();
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_1_1("./Database//1.bmp");
}




void CComputer_Vision_Hw1Dlg::OnBnClickedButton3()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_1_3("./Database/list.txt");
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton6()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_3_1("./Database/QrCode.jpg");
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton2()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_1_2("./Database/list.txt");
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton4()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_1_4("./Database/list.txt");
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton5()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_2_1("./Database/list.txt");
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton7()
{
	// TODO: 在此加入控制項告知處理常式程式碼	
	Homework hw;
	hw.question_4_1("./Database/SceneL.bmp", "./Database/SceneR.bmp", "./Database/truedisp.bmp");
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton8()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_4_2("./Database/SceneL.bmp", "./Database/SceneR.bmp", "./Database/truedisp.bmp");
}

/*
* write the perspective transform by ourselves
* reference the singular value decomposition
*/
void CComputer_Vision_Hw1Dlg::OnBnClickedButton9()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Homework hw;
	hw.question_3_2("./Database/QrCode.jpg");
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton10()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	BMP bmp("./Database/test.bmp");
	bmp.read();
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton11()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	/*VideoCapture cap(0);
	if(!cap.isOpened()){
		cout << "camera is not opencv" << endl;
		waitKey(0);
	}
	namedWindow("video");
	for(;;){
		Mat frame;
		cap >> frame;
		imshow("video", frame);
		if(waitKey(33) >= 0) break;
	}*/
	Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);
	cout << proj1 << endl << endl;
    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    cout << proj1 << endl << endl;
	proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
	cout << proj1 << endl << endl;
}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton12()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	// initial the kernel size, here is the 3 by 3 matrix
	Mat sobel_x(3, 3, CV_32F); // float type
	Mat sobel_y(3, 3, CV_32F); // float type
	// initial the sobel filter x direction
	sobel_x.at<float>(0, 0) = -1;sobel_x.at<float>(0, 1) = 0;sobel_x.at<float>(0, 2) = 1;
	sobel_x.at<float>(1, 0) = -2;sobel_x.at<float>(1, 1) = 0;sobel_x.at<float>(1, 2) = 2;
	sobel_x.at<float>(2, 0) = -1;sobel_x.at<float>(2, 1) = 0;sobel_x.at<float>(2, 2) = 1;
	// initial the sobel filter y direction
	sobel_y.at<float>(0, 0) = -1;sobel_y.at<float>(0, 1) = -2;sobel_y.at<float>(0, 2) = -1;
	sobel_y.at<float>(1, 0) = 0;sobel_y.at<float>(1, 1) = 0;sobel_y.at<float>(1, 2) = 0;
	sobel_y.at<float>(2, 0) = 1;sobel_y.at<float>(2, 1) = 2;sobel_y.at<float>(2, 2) = 1;
	Mat img1 = imread("./Database/test.bmp", CV_LOAD_IMAGE_GRAYSCALE); // load the image as the gray scale image
	Mat imgx(img1.rows, img1.cols, CV_32F, Scalar(0.)); // the final result will be show on this image
	Mat imgy(img1.rows, img1.cols, CV_32F, Scalar(0.)); // the final result will be show on this image
	Mat imgxy(img1.rows, img1.cols, CV_32F, Scalar(0.)); // the final result will be show on this image
	Mat imgangle(img1.rows, img1.cols, CV_32F, Scalar(0.)); // the final result will be show on this image
	int kernel_size = sobel_x.rows;
	int rows = img1.rows;
	int cols = img1.cols;
	int offset = kernel_size / 2;
	// set the constant direction for the 3 by 3 kernel
	const int direction[9][2] = {
		{-1, -1},
		{-1, 0},
		{-1, 1},
		{0 , -1},
		{0, 0},
		{0, 1},
		{1, -1},
		{1, 0},
		{1, 1}
	}; 
	img1.convertTo(img1, CV_32F);
	namedWindow("x direction");
	// here the image margin will not be considered, y direction
	#pragma omp for
	for(int i = offset; i < (rows - offset) - 1; ++i){
		for(int j = offset; j < (cols - offset) - 1; ++j){
			int center = kernel_size / 2;
			imgx.at<float>(i, j) = 0.0f;
			for(int k = 0; k < 9; ++k){
				imgx.at<float>(i, j) += img1.at<float>(i + direction[k][0], j + direction[k][1]) * sobel_x.at<float>(center + direction[k][0], center + direction[k][1]);
			}
			
			imgx.at<float>(i, j) = (float)(imgx.at<float>(i, j));
		}
	}
	Mat imgxx;
	imgxx = abs(imgx);
	imgxx.convertTo(imgxx, CV_8U);
	imshow("x direction", imgxx);
	
	namedWindow("y direction");
	#pragma omp for
	for(int i = offset; i < (rows - offset); ++i){
		for(int j = offset; j < (cols - offset); ++j){
			int center = kernel_size / 2;
			imgy.at<float>(i, j) = 0.0f;
			for(int k = 0; k < 9; ++k){
				imgy.at<float>(i, j) += img1.at<float>(i + direction[k][0], j + direction[k][1]) * sobel_y.at<float>(center + direction[k][0], center + direction[k][1]);
			}
			imgy.at<float>(i, j) = (float)(imgy.at<float>(i, j));
		}
	}
	Mat imgyy;
	imgyy = abs(imgy);
	imgyy.convertTo(imgyy, CV_8U);
	imshow("y direction", imgyy);

	namedWindow("xy direction");
	#pragma omp for
	for(int i = offset; i < (rows - offset); ++i){
		for(int j = offset; j < (cols - offset); ++j){
			int center = kernel_size / 2;
			imgxy.at<float>(i, j) = 0.0f;
			for(int k = 0; k < 9; ++k){
				imgxy.at<float>(i, j) = (float)fabs(imgx.at<float>(i , j)) + (float)fabs(imgy.at<float>(i , j));
			}
		}
	}
	Mat imgxyxy;
	imgxy.convertTo(imgxyxy, CV_8U);
	imshow("xy direction", imgxyxy); // output is the magnitu
	
	namedWindow("angle direction");
	// filter out the certain direction
	// paralle process the image using the openmp
	#pragma omp for
	for(int i = offset; i < (rows - offset); ++i){
		// cout << "here " << i << endl;
		for(int j = offset; j < (cols - offset); ++j){
			// process is too slow
			imgangle.at<float>(i, j) = 0.0f;
			float  result = (float) atan2 (imgy.at<float>(i, j), imgx.at<float>(i, j)) * 180.0f / M_PI + 180.0f;
			result = result > 180.0f ? result - 180.0 : result;
			// cout << result << endl;
			// we want to filter out the 45 degree angle
			if((result >= 35 && result <= 55)){
				imgangle.at<float>(i, j) = imgxy.at<float>(i, j);
			}
		}
	}	
	cout << "finished" << endl;
	Mat imgangleangle;
	imgangle.convertTo(imgangleangle, CV_8U);
	imshow("angle direction", imgangleangle);

	Mat open_img = imread("./Database/test.bmp", CV_LOAD_IMAGE_GRAYSCALE); // load the image as the gray scale image
	Mat open_img_x, open_img_x_abs;
	namedWindow("open x direction");
	Sobel(open_img, open_img_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT); 
	convertScaleAbs(open_img_x, open_img_x_abs);
	imshow("open x direction", open_img_x_abs);

	Mat open_img_y, open_img_y_abs;
	namedWindow("open y direction");
	Sobel(open_img, open_img_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(open_img_y, open_img_y_abs);
	imshow("open y direction", open_img_y_abs);

}


void CComputer_Vision_Hw1Dlg::OnBnClickedButton13()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	double test_double = 4.76;
	int test_int = test_double;
	cout << test_int << endl;
}
