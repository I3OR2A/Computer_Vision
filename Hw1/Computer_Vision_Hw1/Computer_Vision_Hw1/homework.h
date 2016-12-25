#pragma once
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

const string name_window_1_1 = "question_1_1";
const string name_window_1_2 = "questino_1_2";
const string name_window_1_3 = "question_1_3";
const string name_window_1_4 = "question_1_4";
const string name_window_2_1 = "question_2_1";
const string name_window_3_1 = "question_3_1";
const string name_window_3_2 = "question_3_2";
const string name_window_4_1 = "question_4_1";
const string name_window_4_2 = "question_4_2";

class Homework
{
public:
	Homework(void);
	~Homework(void);

	void question_1_1(string path);

	void question_1_2(string path);

	void question_1_3(string path);

	void question_1_4(string path);

	void question_2_1(string path);

	static void question_3_1_on_mouse(int event, int x, int y, int flags, void* param);
	void question_3_1(string path);

	static void question_3_2_on_mouse(int event, int x, int y, int flags, void* param);
	void question_3_2(string path);

	void question_4_1(string path1, string path2, string path3);

	void question_4_2(string path1, string path2, string path3);

	static void usePerspective(Mat &src, Mat &dst, Mat &warpMatrix);

	static Mat genPerspectiveTransform2(vector<Point2f> src, vector<Point2f> dst);
private:
	// store the mouse click coordinate
	vector<Point2f> srcQuad; // for the question_3_1
	// vector<Point2d> dstQuad; // for the question_3_1
	string path;
	string path1, path2, path3; // for the question that has more than one input image
};

