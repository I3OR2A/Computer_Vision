
// Computer_Vision_Hw2Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "Computer_Vision_Hw2.h"
#include "Computer_Vision_Hw2Dlg.h"
#include "afxdialogex.h"

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <fstream>
#include <ostream>
#include <sstream>
using namespace std;

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
using namespace cv;

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


// CComputer_Vision_Hw2Dlg 對話方塊



CComputer_Vision_Hw2Dlg::CComputer_Vision_Hw2Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CComputer_Vision_Hw2Dlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CComputer_Vision_Hw2Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CComputer_Vision_Hw2Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CComputer_Vision_Hw2Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON4, &CComputer_Vision_Hw2Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CComputer_Vision_Hw2Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CComputer_Vision_Hw2Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CComputer_Vision_Hw2Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CComputer_Vision_Hw2Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CComputer_Vision_Hw2Dlg::OnBnClickedButton9)
END_MESSAGE_MAP()


// CComputer_Vision_Hw2Dlg 訊息處理常式

BOOL CComputer_Vision_Hw2Dlg::OnInitDialog()
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
	freopen("CONOUT$", "w", stdout);

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void CComputer_Vision_Hw2Dlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CComputer_Vision_Hw2Dlg::OnPaint()
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
HCURSOR CComputer_Vision_Hw2Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


void CComputer_Vision_Hw2Dlg::OnBnClickedButton1()
{
	// TODO: 在此加入控制項告知處理常式程式碼
	Mat img_1 = imread("./database/plane1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread("./database/plane2.jpg", CV_LOAD_IMAGE_COLOR);

	Mat img_gray_1 = imread("./database/plane1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_gray_2 = imread("./database/plane2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	vector<KeyPoint> keypoints_1, keypoints_2;
	SiftFeatureDetector detector;
	detector.detect(img_gray_1, keypoints_1);
	detector.detect(img_gray_2, keypoints_2); 

	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	imshow("plane1", img_keypoints_1);
	imshow("plane2", img_keypoints_2);

	cout << "Total image 1 feature point: " << keypoints_1.size() << endl;
	cout << "Total image 2 feature point: " << keypoints_2.size() << endl;

	waitKey(0);

	return ;
}


void CComputer_Vision_Hw2Dlg::OnBnClickedButton4()
{
	VideoCapture capture("./database/bgSub_video.mp4");

	Mat img_frame;
	Mat img_mask;
	BackgroundSubtractorMOG substractor;
	for(;;){
		capture >> img_frame;
		substractor(img_frame, img_mask);
		imshow("frame", img_frame);
		imshow("mask", img_mask);
		if(waitKey(30) >= 0) break;
	}

	return ;
}


void CComputer_Vision_Hw2Dlg::OnBnClickedButton5()
{
	vector< Point > points;
	points.push_back(Point(118, 72));
	points.push_back(Point(112, 96));
	points.push_back(Point(119, 170));
	points.push_back(Point(136, 241));
	points.push_back(Point(131, 260));
	points.push_back(Point(176, 269));
	points.push_back(Point(193, 257));

	ofstream ostream;
	ostream.open ("hw2_1.txt");
	for(int i = 0; i < points.size(); ++i){
		ostream << "Point" << i + 1 << ":(" << points[i].x << "," << points[i].y << ")" << "\n";
	}
	ostream << "Window size:10" << "\n";
	ostream.close();

	Mat img_frame;
	VideoCapture capture("./database/tracking_video.mp4");
	capture >> img_frame;

	for(int i = 0; i < points.size(); ++i){
		Point& p = points[i];
		rectangle(img_frame, Point(p.x - 5, p.y - 5), Point(p.x + 5, p.y + 5), Scalar(0, 0, 255), 2, 8);
		line(img_frame, Point(p.x - 5, p.y), Point(p.x + 5, p.y), Scalar(0, 0, 255), 2, 8);
		line(img_frame, Point(p.x, p.y - 5), Point(p.x, p.y + 5), Scalar(0, 0, 255), 2, 8);
	}
	imshow("Tracking Whole Video", img_frame);

	waitKey(0);
	return ;
}


void CComputer_Vision_Hw2Dlg::OnBnClickedButton6()
{
	Mat img_frame_A;
	Mat img_frame_B;
	Mat img_frame_gray_A;
	Mat img_frame_gray_B;
	vector< Point2f > features_A;
	vector< Point2f > features_B;
	vector< uchar > features_found;
	vector< float > features_err;
	VideoCapture capture("./database/tracking_video.mp4");
	capture >> img_frame_A;
	cvtColor(img_frame_A, img_frame_gray_A, CV_BGR2GRAY);
	// img_frame_gray_A.convertTo(img_frame_gray_A, CV_8UC1);
	features_A.push_back(Point2f(118, 72));
	features_A.push_back(Point2f(112, 96));
	features_A.push_back(Point2f(119, 170));
	features_A.push_back(Point2f(136, 241));
	features_A.push_back(Point2f(131, 260));
	features_A.push_back(Point2f(176, 269));
	features_A.push_back(Point2f(193, 257));

	vector< vector < Point2f > > features;
	vector< vector < uchar > > features_founds;
	features.push_back(features_A);

	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 50, 0.0003);
    Size subPixWinSize(3, 3), winSize(10, 10);

	int cnt = 1;
	ofstream ostream;
	ostream.open ("hw2_2.txt");
	for(;;){
		capture >> img_frame_B;
		cvtColor(img_frame_B, img_frame_gray_B, CV_BGR2GRAY);
		// img_frame_gray_B.convertTo(img_frame_gray_B, CV_8UC1);
		calcOpticalFlowPyrLK(img_frame_gray_A, 
							 img_frame_gray_B, 
							 features_A, 
							 features_B, 
							 features_found, 
							 features_err, 
							 winSize,
                             3, 
							 termcrit, 
							 OPTFLOW_LK_GET_MIN_EIGENVALS,
							 0.001);

		cornerSubPix(img_frame_gray_B, features_B, subPixWinSize, Size(-1,-1), termcrit);
		features.push_back(features_B);
		features_founds.push_back(features_found);
		cnt++;

		ostream << "frame " << cnt - 1 << ":" ; 
		for(int i = 0; i < features_B.size(); ++i){
			if(i != features_B.size() - 1)
				ostream << "(" << features_B[i].x << "," << features_B[i].y << "),";
			else
				ostream << "(" << features_B[i].x << "," << features_B[i].y << ")";
		}
		ostream << "\n";

		for(size_t i = 0; i + 1 < cnt && i < cnt; ++i){
			vector< Point2f >& features_current = features[i];
			vector< Point2f >& features_next = features[i + 1];

			for(int j = 0; j < features_current.size(); ++j){
				if(features_founds[i][j]){
					line(img_frame_B, features_current[j], features_next[j], Scalar(0,0,255), 2, 8);
					// cout << i << " - " << i + 1 << endl;
				}
			}
        }

		for(int i = 0; i < features_B.size(); ++i){
			Point2f& p = features_B[i];
			rectangle(img_frame_B, Point(p.x - 5, p.y - 5), Point(p.x + 5, p.y + 5), Scalar(0, 0, 255), 2, 8);
			line(img_frame_B, Point(p.x - 5, p.y), Point(p.x + 5, p.y), Scalar(0, 0, 255), 2, 8);
			line(img_frame_B, Point(p.x, p.y - 5), Point(p.x, p.y + 5), Scalar(0, 0, 255), 2, 8);
		}

		img_frame_A = img_frame_B.clone();
		img_frame_gray_A = img_frame_gray_B.clone();
		features_A = features_B;

		imshow("Tracking Whole Video", img_frame_B);
		waitKey(30);
	
	}
	ostream.flush();
	ostream.close();

	cout << features_A[0] << endl;
	cout << features_B[0] << endl;

	return;
}


void CComputer_Vision_Hw2Dlg::OnBnClickedButton7()
{
	Ptr<FaceRecognizer> recognizer = createEigenFaceRecognizer();
	Mat img_0 = imread("./database/0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_1 = imread("./database/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread("./database/2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	int rows = img_0.rows;
	int type = img_0.type();
	int channels = img_0.channels();

	// Initial the training data
	vector< Mat > imgs;
	imgs.push_back(img_0);
	imgs.push_back(img_1);
	imgs.push_back(img_2);
	vector< int > labels;
	labels.push_back(0);
	labels.push_back(1);
	labels.push_back(2);

	// Train the model
	recognizer->train(imgs, labels);

	// Show the result of the mean face
	Mat img_mean = recognizer->getMat("mean");
	Mat img_inte;
	normalize(img_mean.reshape(channels, rows), img_inte, 0, 255, NORM_MINMAX, type);  
	imshow("mean", img_inte);  

	// Show the name of the person on the console
	Mat img_test = imread("./database/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int label = recognizer->predict(img_test);
	cout << "the image is : ";
	switch(label){
	case 0:
		cout << "Harry Potter" << endl;
		break;
	case 1:
		cout << "Hermione Granger" << endl;
		break;
	case 2:
		cout << "Ron Weasley" << endl;
		break;
	default:
		cout << "Something wrong!!!" << endl;
	}

	return ;
}


void CComputer_Vision_Hw2Dlg::OnBnClickedButton8()
{
	// Training the faces
	CascadeClassifier classifier;
	classifier.load("./haarcascade_frontalface_alt.xml");

	// Detect the faces
	Mat img_faces = imread("./database/face.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	vector< Rect > rect_faces;
	classifier.detectMultiScale(img_faces, rect_faces);

	// Output the number of faces detected on theconsole window
	cout << rect_faces.size() << " faces detect" << endl;

	// Draw a rectangle on each face, and show the result
	Mat img_result = imread("./database/face.jpg", CV_LOAD_IMAGE_COLOR);
	for(int i = 0; i < rect_faces.size(); ++i){
		rectangle(img_result, rect_faces[i], CV_RGB(255, 0, 0), 4);
		// cout << "width: " << rect_faces[i].width << " height: " << rect_faces[i].height << endl;
	}
	imshow("Detection", img_result);

	waitKey(0);
}


void CComputer_Vision_Hw2Dlg::OnBnClickedButton9()
{
	// Training the faces
	CascadeClassifier classifier;
	classifier.load("./haarcascade_frontalface_alt.xml");
	
	// Detect the faces
	Mat img_faces = imread("./database/face.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	vector< Rect > rect_faces;
	classifier.detectMultiScale(img_faces, rect_faces);
	
	Ptr<FaceRecognizer> recognizer = createEigenFaceRecognizer(10);

	int rows;
	int cols;
	int type;
	int channels;

	// Initial the training data
	vector< Mat > imgs;
	vector< int > labels;

	// create the training data
	for(int i = 0; i < 3; ++i){
		string name = "./database/" + to_string(i);
		name = name + ".jpg";
		Mat img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);

		rows = img.rows;
		cols = img.cols;
		type = img.type();
		channels = img.channels();

		// original
		imgs.push_back(img.clone());
		labels.push_back(i);

		// crop
		Rect rect(10, 10, cols - 20, rows - 20);
		Mat img_crop;
		img(rect).copyTo(img_crop);
		resize(img_crop, img_crop, Size(cols, rows), CV_INTER_LANCZOS4);
		imshow("crop: " + to_string(i), img_crop);
		imgs.push_back(img_crop.clone());
		labels.push_back(i);

		// crop equalization
		equalizeHist(img_crop, img_crop);
		imgs.push_back(img_crop.clone());
		labels.push_back(i);
		
		// flip
		Mat img_flip;
		flip(img, img_flip, 1);
		imgs.push_back(img_flip.clone());
		labels.push_back(i);

		// original equalization
		equalizeHist(img, img);
		imgs.push_back(img.clone());
		labels.push_back(i);

		// flip equalization
		flip(img, img_flip, 1);
		imgs.push_back(img_flip.clone());
		labels.push_back(i);
	}

	// Train the model by using the created training data
	recognizer->train(imgs, labels);

	// equalizeHist(img_faces, img_faces);
	// Expand the rectangle from the center
	Mat img_result = imread("./database/face.jpg", CV_LOAD_IMAGE_COLOR);
	for(int i = 0; i < rect_faces.size(); ++i){
		Rect &rect = rect_faces[i];
		rect.x -= 2;
		rect.y -= 2;
		rect.width;
		rect.height;
		Mat mat;
		img_faces(rect).copyTo(mat);
		resize(mat, mat, Size(cols, rows), CV_INTER_LANCZOS4);
		
		rectangle(img_result, rect, CV_RGB(255, 0, 0), 4);
		imshow("" + to_string(i), mat);
		imwrite(to_string(i) + ".jpg", mat);

		int predicted_label = -1;
		double predicted_confidence = 0.0;
		recognizer->predict(mat, predicted_label, predicted_confidence);
		switch(predicted_label){
		case 0:
			putText(img_result, "Harry Potter", Point(rect.x, rect.y - 8), 0, 0.6, Scalar(0, 0, 255), 2);
			// cout << "Harry Potter" << endl;
			break;
		case 1:
			putText(img_result, "Hermione Granger", Point(rect.x, rect.y - 8), 0, 0.6, Scalar(0, 0, 255), 2);
			// cout << "Hermione Granger" << endl;
			break;
		case 2:
			putText(img_result, "Ron Weasley", Point(rect.x, rect.y - 8), 0, 0.6, Scalar(0, 0, 255), 2);
			// cout << "Ron Weasley" << endl;
			break;
		default:
			putText(img_result, "Something wrong!!!", Point(rect.x, rect.y - 8), 0, 0.6, Scalar(0, 0, 255), 2);
			// cout << "Something wrong!!!" << endl;
		}
	}

	imshow("Detection + Recognition", img_result);

	waitKey(0);
}
