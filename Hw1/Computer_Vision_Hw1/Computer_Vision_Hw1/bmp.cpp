#include "stdafx.h"
#include "bmp.h"


BMP::BMP(void)
{
}


BMP::~BMP(void)
{
}

BMP::BMP(string path){
	this->path = path;
}

void BMP::read(){
	destroyAllWindows();
	// read the bmp file data information
	fstream file;
	file.open(this->path, ios::in|ios::binary);
	file.read((char*)&bmpId, sizeof(WWORD));
	file.read((char*)&fileSize, sizeof(BBYTE));
	file.read((char*)&bmpReserved1, sizeof(WWORD));
	file.read((char*)&bmpReserved2, sizeof(WWORD));
	file.read((char*)&bmpOffset, sizeof(BBYTE));
	file.read((char*)&bmpInfoHeaderSize, sizeof(BBYTE));
	file.read((char*)&bmpWidth, sizeof(DDWORD));
	file.read((char*)&bmpHeight, sizeof(DDWORD));
	file.read((char*)&bmpPlanes, sizeof(WWORD));
	file.read((char*)&bmpbitCount, sizeof(WWORD));
	file.read((char*)&bmpCompression, sizeof(DDWORD));
	file.read((char*)&bmpDataSize, sizeof(DDWORD));
	file.read((char*)&bmpXPixelPerMeter, sizeof(DDWORD));
	file.read((char*)&bmpYPixelPerMeter, sizeof(DDWORD));
	file.read((char*)&bmpColorUsed, sizeof(BBYTE));
	file.read((char*)&bmpColorImportant, sizeof(BBYTE));
	cout << "bmpXPixelPerMeter: " << bmpXPixelPerMeter << endl;
	cout << "bmpYPixelPerMeter: " << bmpYPixelPerMeter << endl;
	cout << "bmpXPixelPerInch: " << (int) ((float) bmpXPixelPerMeter / (float) m2i) << endl;
	cout << "bmpYPixelPerInch: " << (int) ((float) bmpYPixelPerMeter / (float) m2i) << endl;
	// close the file
	file.close();
}
