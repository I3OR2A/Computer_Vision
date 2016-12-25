#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef unsigned short WWORD;
typedef unsigned int BBYTE;
typedef int DDWORD;
const double m2i = 39.3701;
struct Color {
	int R;
	int G;
	int B;
};

class BMP
{
public:
	BMP(void);
	BMP(string path);
	~BMP(void);
public:
	void read();
private:
	string path;
	int x,y;
	WWORD bmpId;
	BBYTE fileSize;
	WWORD bmpReserved1;
	WWORD bmpReserved2;
	BBYTE bmpOffset;
	BBYTE bmpInfoHeaderSize;
	DDWORD bmpWidth;
	DDWORD bmpHeight;
	WWORD bmpPlanes;
	WWORD bmpbitCount;
	DDWORD bmpCompression;
	DDWORD bmpDataSize;
	DDWORD bmpXPixelPerMeter;
	DDWORD bmpYPixelPerMeter;
	BBYTE bmpColorUsed;
	BBYTE bmpColorImportant;
};

