#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class Calibration
{
public:
	Calibration(void);
	~Calibration(void);

	/*
	* run and save camera calibration parameter
	*/
	bool calibrate(char* path);
private:
	void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);
	bool runCalibration(Size& imageSize, Mat& cameraMatrix, Mat& distMatrix, vector<vector<Point2f> > imagePoints, vector<Mat> &rvecs, vector<Mat> &tvecs, vector<float> &reprojErrs, double &totalAvgErr);
	double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                         const vector<vector<Point2f> >& imagePoints,
                                         const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                         const Mat& cameraMatrix , const Mat& distCoeffs,
                                         vector<float>& perViewErrors);

	// need to decalre some global variable to store the final calibrate result
	
	// create the new method to reproject 3D points to 2D image
};

