#include "stdafx.h"
#include "homework.h"


Homework::Homework(void)
{
}


Homework::~Homework(void)
{
}

void Homework::question_1_1(string path){
	destroyAllWindows();
	// read the image  from the path of the file
	namedWindow(name_window_1_1);
	Size boardSize(11, 8);
	Mat color_view = imread(path, CV_LOAD_IMAGE_COLOR);
	Size size = color_view.size();
	vector<Point2f> pointBuf;
	bool found = findChessboardCorners(color_view, boardSize, pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
	// if done with success
	if(found){
		Mat gray_view;
		cvtColor(color_view, gray_view, COLOR_BGR2GRAY);
		cornerSubPix(gray_view, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
		drawChessboardCorners(color_view, boardSize, Mat(pointBuf), found);
	}
	imshow(name_window_1_1, color_view);
	waitKey(0);
}

void Homework::question_1_2(string path){
	destroyAllWindows();
	// read the image from the path of the file
	vector<vector<Point2f> > imagePoints;
	vector<Mat> tvecs, rvecs;
	Mat cameraMatrix, distCoeffs;
	Size boardSize(11, 8);
	float squareSize = 1.0f;
	Size imageSize;
	fstream infile(path);
	string line;
	while(getline(infile, line)){
		Mat view = imread(line, CV_LOAD_IMAGE_COLOR);
		imageSize = view.size();
		vector<Point2f> pointBuf;
		bool found;
		found = findChessboardCorners( view, boardSize, pointBuf,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		if ( found)                // If done with success,
		{
			Mat viewGray;
			cvtColor(view, viewGray, COLOR_BGR2GRAY);
			cornerSubPix( viewGray, pointBuf, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
			imagePoints.push_back(pointBuf);
			// draw the corners.
			drawChessboardCorners( view, boardSize, Mat(pointBuf), found );
		}
	}

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	vector<vector<Point3f> > objectPoints(1);
	for( int i = 0; i < boardSize.height; ++i )
		for( int j = 0; j < boardSize.width; ++j )
			objectPoints[0].push_back(Point3f(float( j * squareSize ), float( i * squareSize ), 0));
	objectPoints.resize(imagePoints.size(),objectPoints[0]);
	// find intrinsic and extrinsic camera parameters
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
	cout << cameraMatrix << endl;
}

void Homework::question_1_3(string path){
	destroyAllWindows();
	// read the image from the path of the file
	vector<vector<Point2f> > imagePoints;
	vector<Mat> tvecs, rvecs;
	Mat cameraMatrix, distCoeffs;
	Size boardSize(11, 8);
	float squareSize = 1.0f;
	Size imageSize;
	fstream infile(path);
	string line;
	while(getline(infile, line)){
		Mat view = imread(line, CV_LOAD_IMAGE_COLOR);
		imageSize = view.size();
		vector<Point2f> pointBuf;
		bool found;
		found = findChessboardCorners( view, boardSize, pointBuf,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		if ( found)                // If done with success,
		{
			Mat viewGray;
			cvtColor(view, viewGray, COLOR_BGR2GRAY);
			cornerSubPix( viewGray, pointBuf, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
			imagePoints.push_back(pointBuf);
			// draw the corners.
			drawChessboardCorners( view, boardSize, Mat(pointBuf), found );
		}
	}

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	vector<vector<Point3f> > objectPoints(1);
	for( int i = 0; i < boardSize.height; ++i )
		for( int j = 0; j < boardSize.width; ++j )
			objectPoints[0].push_back(Point3f(float( j * squareSize ), float( i * squareSize ), 0));
	objectPoints.resize(imagePoints.size(),objectPoints[0]);
	// find intrinsic and extrinsic camera parameters
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
	// cout << cameraMatrix << endl;
	Mat rotationMatrix, translationMatrix;
	Rodrigues(rvecs[0], rotationMatrix);
	translationMatrix = Mat(tvecs[0]);
	Mat extrinsicMatrix;
	hconcat(rotationMatrix, translationMatrix, extrinsicMatrix);
	cout << extrinsicMatrix << endl;
}

void Homework::question_1_4(string path){
	destroyAllWindows();
	// read the image from the path of the file
	vector<vector<Point2f> > imagePoints;
	vector<Mat> tvecs, rvecs;
	Mat cameraMatrix, distCoeffs;
	Size boardSize(11, 8);
	float squareSize = 1.0f;
	Size imageSize;
	fstream infile(path);
	string line;
	while(getline(infile, line)){
		Mat view = imread(line, CV_LOAD_IMAGE_COLOR);
		imageSize = view.size();
		vector<Point2f> pointBuf;
		bool found;
		found = findChessboardCorners( view, boardSize, pointBuf,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		if ( found)                // If done with success,
		{
			Mat viewGray;
			cvtColor(view, viewGray, COLOR_BGR2GRAY);
			cornerSubPix( viewGray, pointBuf, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
			imagePoints.push_back(pointBuf);
			// draw the corners.
			drawChessboardCorners( view, boardSize, Mat(pointBuf), found );
		}
	}

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	vector<vector<Point3f> > objectPoints(1);
	for( int i = 0; i < boardSize.height; ++i )
		for( int j = 0; j < boardSize.width; ++j )
			objectPoints[0].push_back(Point3f(float( j * squareSize ), float( i * squareSize ), 0));
	objectPoints.resize(imagePoints.size(),objectPoints[0]);
	// find intrinsic and extrinsic camera parameters
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
	cout << distCoeffs << endl;
}

void Homework::question_2_1(string path){
	destroyAllWindows();
	// read the image from the path of the file
	vector<vector<Point2f> > imagePoints;
	vector<Mat> tvecs, rvecs;
	Mat cameraMatrix, distCoeffs;
	Size boardSize(11, 8);
	float squareSize = 1.0f;
	Size imageSize;
	fstream infile(path);
	string line;
	while(getline(infile, line)){
		Mat view = imread(line, CV_LOAD_IMAGE_COLOR);
		imageSize = view.size();
		vector<Point2f> pointBuf;
		bool found;
		found = findChessboardCorners( view, boardSize, pointBuf,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		if ( found)                // If done with success,
		{
			Mat viewGray;
			cvtColor(view, viewGray, COLOR_BGR2GRAY);
			cornerSubPix( viewGray, pointBuf, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
			imagePoints.push_back(pointBuf);
			// draw the corners.
			// drawChessboardCorners( view, boardSize, Mat(pointBuf), found );
		}
	}

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	vector<vector<Point3f> > objectPoints(1);
	for( int i = 0; i < boardSize.height; ++i )
		for( int j = 0; j < boardSize.width; ++j )
			objectPoints[0].push_back(Point3f(float( j * squareSize ), float( i * squareSize ), 0));
	objectPoints.resize(imagePoints.size(),objectPoints[0]);
	// find intrinsic and extrinsic camera parameters
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
	int idx = 0;

	vector<Point3d> modelPoints;
	modelPoints.push_back(Point3f(0.0f,0.0f,-2.0f));
	modelPoints.push_back(Point3f(1.0f,1.0f,0.0f));
	modelPoints.push_back(Point3f(1.0f,-1.0f,0.0f));
	modelPoints.push_back(Point3f(-1.0f,-1.0f,0.0f));
	modelPoints.push_back(Point3f(-1.0f,1.0f, 0.0f));
	infile.close();
	infile.open(path, ios::in); // for the input file ios::in
	int frame_cnt = 0;
	while(getline(infile, line)){
		Mat view_color; // store the image input

		view_color = imread(line.c_str(), CV_LOAD_IMAGE_COLOR);
		idx++;
		// check the whether image is read successfully
		if(!view_color.data){
			cout << "Could not open or find the image" << endl;
			if( cvWaitKey(0) == 27 ) // allow ESC to quit, let user define we to go to the next image
				exit(-1); // exit the program, may be the result is not good		
		}

		vector<Point2d> imagePointsRe;
		projectPoints(modelPoints, rvecs[idx - 1], tvecs[idx - 1], cameraMatrix, distCoeffs, imagePointsRe);
		int thinkness = 2;
		int lineType = 8;

		// need to add cv:: to avoid the string line function
		// Scalar is the color
		cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[1]), Scalar(0,0,255), thinkness, lineType);
		cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[2]), Scalar(0,0,255), thinkness, lineType);
		cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[3]), Scalar(0,0,255), thinkness, lineType);
		cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[4]), Scalar(0,0,255), thinkness, lineType);

		cv::line(view_color, Point(imagePointsRe[1]), Point(imagePointsRe[2]), Scalar(0,0,255), thinkness, lineType);
		cv::line(view_color, Point(imagePointsRe[2]), Point(imagePointsRe[3]), Scalar(0,0,255), thinkness, lineType);
		cv::line(view_color, Point(imagePointsRe[3]), Point(imagePointsRe[4]), Scalar(0,0,255), thinkness, lineType);
		cv::line(view_color, Point(imagePointsRe[4]), Point(imagePointsRe[1]), Scalar(0,0,255), thinkness, lineType);
		// show the image on the window to check the final result
		namedWindow("reproject", WINDOW_AUTOSIZE); // create a window for display
		imshow( "reproject", view_color); // show our image inside it
		frame_cnt++;
		// wait until to get user input
		if( cvWaitKey(500) == 27 ) // allow ESC to quit, let user define we to go to the next image
			exit(-1); // exit the program, may be the result is not good
		if(frame_cnt == 5)
			break;
	}
	infile.close();
	waitKey(0);
}

void Homework::question_3_1_on_mouse(int event, int x, int y, int flags, void* param){
	Homework* hw = (Homework*) param;
	if(event == CV_EVENT_LBUTTONDOWN){
		destroyWindow(name_window_3_1);
		hw->srcQuad.push_back(Point2f(x, y));
		int size = hw->srcQuad.size();
		// the number is the four, we start to do the calculation
		if(size == 4){
			// initial the window
			namedWindow(name_window_3_1);
			// initial the input ans output image
			Mat img1 = imread(hw->path, CV_LOAD_IMAGE_COLOR);
			int rows = img1.rows;
			int cols = img1.cols;
			int size = rows > cols ? cols : rows;
			Mat img2(size, size, img1.type());
			// double type variable
			vector<Point2f> dstQuad;
			// compute the warp matrix
			// initial the dstQuad
			dstQuad.push_back(Point2f(0, 0));
			dstQuad.push_back(Point2f(size, 0));
			dstQuad.push_back(Point2f(size, size));
			dstQuad.push_back(Point2f(0, size));
			// return  3x3 matrix for the correspinding four points
			Mat warpMatrix = getPerspectiveTransform(hw->srcQuad, dstQuad);
			cout << warpMatrix << endl;
			// do the perspective transform
			warpPerspective(img1, img2, warpMatrix, Size(size, size));
			imshow(name_window_3_1, img2); // show the final result in the window
			hw->srcQuad.clear();
		}
	}
}

void Homework::question_3_1(string path){
	//destroyAllWindows();
	namedWindow(name_window_3_1 + "src");
	this->path = path;
	Mat img = imread(path, CV_LOAD_IMAGE_COLOR);
	imshow(name_window_3_1 + "src", img);
	setMouseCallback(name_window_3_1 + "src", question_3_1_on_mouse, this);
	waitKey(0);
}

void Homework::question_3_2_on_mouse(int event, int x, int y, int flags, void* param){
	Homework* hw = (Homework*) param;
	if(event == CV_EVENT_LBUTTONDOWN){
		destroyWindow(name_window_3_2);
		hw->srcQuad.push_back(Point2f(x, y));
		int size = hw->srcQuad.size();
		// the number is the four, we start to do the calculation
		if(size == 4){
			// initial the window
			namedWindow(name_window_3_2);
			// initial the input ans output image
			Mat img1 = imread(hw->path, CV_LOAD_IMAGE_COLOR);
			int rows = img1.rows;
			int cols = img1.cols;
			int size = rows > cols ? cols : rows;
			Mat img2(size, size, img1.type());
			// double type variable
			vector<Point2f> dstQuad;
			// compute the warp matrix
			// initial the dstQuad
			dstQuad.push_back(Point2f(0, 0));
			dstQuad.push_back(Point2f(size, 0));
			dstQuad.push_back(Point2f(size, size));
			dstQuad.push_back(Point2f(0, size));

			// generate the homography matrix here
			// return  3x3 matrix for the correspinding four points
			// Mat warpMatrix = getPerspectiveTransform(hw->srcQuad, dstQuad);
			// do the perspective transform
			// warpPerspective(img1, img2, warpMatrix, Size(size, size));

			vector<Point2f> srcQuad = hw->srcQuad;
			// Mat warpMatrix = genPerspectiveTransform2(srcQuad, dstQuad);
			// cout << "warpMatrix" << endl;
			// cout << warpMatrix << endl;
			Mat warpMatrix = genPerspectiveTransform2(dstQuad, hw->srcQuad);
			// cout << warpMatrix << endl;
			// genPerspectiveTransform2(srcQuad, dstQuad);
			usePerspective(img1, img2, warpMatrix);
			// do the perspective transform reference
			// warpPerspective(img1, img2, warpMatrix, Size(size, size));
			imshow(name_window_3_2, img2); // show the final result in the window
			hw->srcQuad.clear();
		}
	}
}

Mat Homework::genPerspectiveTransform2(vector<Point2f> src, vector<Point2f> dst){
	Mat H(3, 3, CV_64F); // final result

	Mat A(8, 8, CV_64F);
	Mat X(8, 1, CV_64F);
	Mat B(8, 1, CV_64F);

	for(int i = 0; i < 4; ++i){
		A.at<double>(i, 0) = src[i].x;
		A.at<double>(i, 1) = src[i].y;
		A.at<double>(i, 2) =  1;
		A.at<double>(i, 3) = 0;
		A.at<double>(i, 4) = 0;
		A.at<double>(i, 5) = 0;
		A.at<double>(i, 6) = (-1.0) * src[i].x * dst[i].x;
		A.at<double>(i, 7) = (-1.0) * src[i].y * dst[i].x;

		A.at<double>(i + 4, 0) = 0;
		A.at<double>(i + 4, 1) = 0;
		A.at<double>(i + 4, 2) = 0;
		A.at<double>(i + 4, 3) = src[i].x;
		A.at<double>(i + 4, 4) =  src[i].y;
		A.at<double>(i + 4, 5) =  1;
		A.at<double>(i + 4, 6) = (-1.0) * src[i].x * dst[i].y;
		A.at<double>(i + 4, 7) = (-1.0) * src[i].y * dst[i].y;
		
		B.at<double>(i, 0) = dst[i].x;
		B.at<double>(i + 4, 0) = dst[i].y;
	}
	
	solve( A, B, X, DECOMP_SVD );

	H.at<double>(0, 0) = X.at<double>(0, 0);
	H.at<double>(0, 1) = X.at<double>(1, 0);
	H.at<double>(0, 2) = X.at<double>(2, 0);
	H.at<double>(1, 0) = X.at<double>(3, 0);
	H.at<double>(1, 1) = X.at<double>(4, 0);
	H.at<double>(1, 2) = X.at<double>(5, 0);
	H.at<double>(2, 0) = X.at<double>(6, 0);
	H.at<double>(2, 1) = X.at<double>(7, 0);
	H.at<double>(2, 2) = 1.0;
	
	return H;
}

void Homework::usePerspective(Mat &src, Mat &dst, Mat &warpMatrix){

	// calculate the inverse matrix of warpMatrix
	Mat iWarpMatrix = warpMatrix;
	
	#pragma omp for
	for(int i = 0; i < dst.rows; ++i){
		for(int j = 0; j < dst.cols; ++j){
			Mat pt(3, 1, CV_64F);
			pt.at<double>(0, 0) = j; // x
			pt.at<double>(1, 0) = i; // y
			pt.at<double>(2, 0) = 1;
			pt = iWarpMatrix * pt;

			pt.at<double>(0, 0) /= pt.at<double>(2, 0);
			pt.at<double>(1, 0) /= pt.at<double>(2, 0);

			// do the interpolation of the intensity value
			// find the 4 nearby point to do the interpolation
			int left = pt.at<double>(0, 0); // x
			int right = left + 1;; // x
			int upper = pt.at<double>(1, 0); // y
			int bottom = upper + 1; // y

			double h_c = fabs(pt.at<double>(0, 0));
			double v_c = fabs(pt.at<double>(1, 0));
			double h_d_1 = fabs(v_c - (double)left);
			double h_d_2 = fabs((double)right - v_c);
			double v_d_1 = fabs(h_c - (double)upper);
			double v_d_2 = fabs((double)bottom - h_c);

			Vec3f l_m_v = v_d_2 * Vec3f(src.at<Vec3b>(upper, left)) + v_d_1 * Vec3f(src.at<Vec3b>(bottom, left));
			Vec3f r_m_v = v_d_2 * Vec3f(src.at<Vec3b>(upper, right)) + v_d_1 * Vec3f(src.at<Vec3b>(bottom, right));
			// Vec3f u_m_v = h_d_2 * Vec3f(src.at<Vec3b>(upper, left)) + h_d_1 * Vec3f(src.at<Vec3b>(upper, right));
			// Vec3f b_m_v = h_d_2 * Vec3f(src.at<Vec3b>(bottom, left)) + h_d_1 * Vec3f(src.at<Vec3b>(bottom, right));
			
			l_m_v /= (v_d_1 + v_d_2);
			r_m_v /= (v_d_1 + v_d_2);

			Vec3f f_h_v = h_d_2 * l_m_v + h_d_1 * r_m_v;
			f_h_v /= (h_d_1 + h_d_2);

			dst.at<Vec3b>(j, i) = Vec3b(f_h_v);
		}
	}


	Mat tmp(dst.rows, dst.cols, dst.type());
	for(int i = 0; i < dst.rows; ++i){
		for(int j = 0; j < dst.cols; ++j){
			Mat pt(3, 1, CV_64F);
			pt.at<double>(0, 0) = j; // x
			pt.at<double>(1, 0) = i; // y
			pt.at<double>(2, 0) = 1;
			pt = iWarpMatrix * pt;

			pt.at<double>(0, 0) /= pt.at<double>(2, 0);
			pt.at<double>(1, 0) /= pt.at<double>(2, 0);

			tmp.at<Vec3b>(j, i) = src.at<Vec3b>((int)pt.at<double>(1,0), (int)pt.at<double>(0, 0));
		}
	}
	imshow("tmp", tmp);
}

void Homework::question_3_2(string path){
	destroyAllWindows();
	namedWindow(name_window_3_2 + "src");
	this->path = path;
	Mat img = imread(path, CV_LOAD_IMAGE_COLOR);
	imshow(name_window_3_2 + "src", img);
	setMouseCallback(name_window_3_2 + "src", question_3_2_on_mouse, this);
	waitKey(0);
}

void Homework::question_4_1(string path1, string path2, string path3){


	destroyAllWindows();
	// debug
	// namedWindow(name_window_4_1 + "left");
	// namedWindow(name_window_4_1 + "right");
	namedWindow(name_window_4_1);
	Mat img1 = imread(path1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat trut = imread(path3, CV_LOAD_IMAGE_GRAYSCALE);
	int rows = img1.rows;
	int cols = img1.cols;
	Mat img3(rows, cols, CV_16S);

	StereoBM sbm;
	// preprocessing
	// prefilter type to eliminate photometric distortions, noise ans enhance the texture
	sbm.state->preFilterSize = 9; // averaging window size: ~5x5..21x21
	sbm.state->preFilterCap = 31; // truncation value for the prefiltered image pixel
	// correspondence using sum of absolute difference(SAD)
	sbm.state->SADWindowSize = 5; // the acceptable window size ~5x5..21x21, must be odd
	sbm.state->numberOfDisparities = 32; // the range of disparity
	sbm.state->minDisparity = 0; // minimum possible disparity value can be negative
	// post-filtering
	sbm.state->textureThreshold = 10; // the disparity is only computed for pixels
	// with textured enough neighborhood
	sbm.state->uniquenessRatio = 15;
	sbm.state->speckleRange = 8; // acceptable range of variation in window
	sbm.state->speckleWindowSize = 0; // disparity variation window
	sbm.state->disp12MaxDiff = 1;
	sbm(img1, img2, img3);
	Mat disp = Mat(img3);
	Mat img38U(rows, cols, CV_8U);
	normalize(img3, img38U, 0, 255, CV_MINMAX, CV_8U); // normalize the value 
	imshow(name_window_4_1, img38U);
	int sum = 0;
	int npt = rows * cols;
	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j){
			int diff = disp.at<int>(i, j) - trut.at<int>(i, j);
			diff = diff < 0 ? (-1) * diff : diff;
			sum += diff;
		}
	}
	waitKey(0);
}


void Homework::question_4_2(string path1, string path2, string path3){
	destroyAllWindows();
	//namedWindow(name_window_4_1 + "left");
	//namedWindow(name_window_4_1 + "right");
	// namedWindow(name_window_4_2);
	Mat img1 = imread(path1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat trut = imread(path3, CV_LOAD_IMAGE_GRAYSCALE);
	int rows = img1.rows;
	int cols = img1.cols;
	Mat img3(rows, cols, CV_16S);

	StereoBM sbm;
	// prefilter type to eliminate photometric distortions, noise ans enhance the texture
	sbm.state->preFilterSize = 9; // averaging window size: ~5x5..21x21
	sbm.state->preFilterCap = 31; // truncation value for the prefiltered image pixel
	// correspondence using sum of absolute difference(SAD)
	sbm.state->SADWindowSize = 5; // the acceptable window size ~5x5..21x21, must be odd
	sbm.state->numberOfDisparities = 32; // the range of disparity
	sbm.state->minDisparity = 0; // minimum possible disparity value can be negative
	// post-filtering
	sbm.state->textureThreshold = 10; // the disparity is only computed for pixels
	// with textured enough neighborhood
	sbm.state->uniquenessRatio = 15;
	sbm.state->speckleRange = 0; // acceptable range of variation in window
	sbm.state->speckleWindowSize = 0; // disparity variation window
	sbm.state->disp12MaxDiff = 1;
	sbm(img1, img2, img3);
	// imshow(name_window_4_1, img3);
	// check its extreme value
	// double minVal; double maxVal;
	// minMaxLoc( img3, &minVal, &maxVal );
	// cout << "Min disp:" << minVal << " Max value: " <<  maxVal << endl;
	// display the image
	// third parameter is scale factor
	// fourth parameter is added to scaled values
	StereoBM bm(StereoBM::BASIC_PRESET, 32, 5);
	Mat disp = Mat(img3);
	Mat img38U(rows, cols, CV_8U);
	img3.convertTo(img38U, CV_8U);
	// normalize(img3, img38U, 0, 255, CV_MINMAX, CV_8U); // normalize the value 
	// imshow(name_window_4_2, img38U);
	// show thw sad result to the console window
	trut.convertTo(trut, CV_8U);
	float sum = 0.0;
	int npt = rows * cols;
	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j){
			int diff = img38U.at<UINT8>(i, j) - trut.at<UINT8>(i, j);
			diff = diff < 0 ? (-1) * diff : diff;
			sum += (float)diff;
		}
	}
	printf("%.1f\n", (float) sum / (float) npt);
	waitKey(0);
}
