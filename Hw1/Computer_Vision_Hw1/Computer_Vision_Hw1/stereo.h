#pragma once

#include <iostream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
class Stereo
{
public:
	Stereo(void);
	Stereo(int count);
	~Stereo(void);
	void print_help();
	void calibrate(vector<string> imagelist, Size boardSize, bool useCalibrated=true, bool showRectified=true);
	bool readStringList(string imagelistfn, vector<string>& imageList);
	void drawEpipolarLines(const string& title, const Mat& F, const Mat& img1, const Mat& img2, const vector<Point2f> points1, const vector<Point2f> points2, const float inlierDistance);
};

