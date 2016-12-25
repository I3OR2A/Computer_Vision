#pragma once

#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Camera
{
public:
	Camera(void);
	~Camera(void);

	void calibrate(string path);

private:
};

