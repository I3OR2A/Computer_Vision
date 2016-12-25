#include "stdafx.h"
#include "calibration.h"


Calibration::Calibration(void)
{
}


Calibration::~Calibration(void)
{
}

/*
* pattern type is the chessboard type
*/
void Calibration::calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f> &corners){
	// clear the corners vector
	corners.clear(); // it is the same for the each chessboard corner
	for(int i = 0; i < boardSize.height; ++i){
		for(int j = 0; j < boardSize.width; ++j){
			// z = 0, hence we are use the chessboard as the origin of world coordinate (x, y, x).
			// It is important to check the order of the point.
			corners.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0)); 
		}
	}
}

/*
* pass the parameter by reference
*/

bool Calibration::runCalibration(Size& imageSize, Mat& cameraMatrix, Mat& distMatrix, vector<vector<Point2f> > imagePoints, vector<Mat> &rvecs, vector<Mat> &tvecs, vector<float> &reprojErrs, double &totalAvgErr){
	cameraMatrix = Mat::eye(3, 3, CV_64FC1); // 64 bits floating point

	// check whether the input file need to fix the aspect ratio
	// distortion matrix is 5 or eight elements
	// need to check what is going on here
	distMatrix = Mat::zeros(8, 1, CV_64FC1); // 64 bits floating point

	// need to initial the board size here
	Size boardSize(11, 8); // width , height of corners image
	float squareSize = 1.0f; // default value, need to manually measure the chessboards

	vector<vector<Point3f> > objectPoints(1);

	// pass by reference, the address of the fisrt element
	calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);

	// imagePoints's is the number of success detect the chessboard corner
	objectPoints.resize(imagePoints.size(), objectPoints[0]); // first is the number of the chessboard image, second is the value

	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distMatrix, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

	// final calculate the reprojection error
	cout << "Re-projection error reported by calibrateCamera: "<< rms << endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distMatrix);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distMatrix, reprojErrs);

	return ok;
}

double Calibration::computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
											  const vector<vector<Point2f> >& imagePoints,
											  const vector<Mat>& rvecs, const vector<Mat>& tvecs,
											  const Mat& cameraMatrix , const Mat& distCoeffs,
											  vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for( i = 0; i < (int)objectPoints.size(); ++i )
	{
		projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
			distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float) std::sqrt(err*err/n);
		totalErr        += err*err;
		totalPoints     += n;
	}

	return std::sqrt(totalErr/totalPoints);
}

/*
* read in the txt file and iterate through all the image path
*/
bool Calibration::calibrate(char* path){

	// read the list of the image to fina the chessboar corners
	Size imageSize;
	vector<vector<Point2f> > imagePoints; // the corner of each chessboads on the image coordinates
	Mat cameraMatrix; // intrinsic matrix
	Mat distMatrix; // distortion coefficient matrix
	Size boardSize(11, 8); // the inner corner size of the chessboards
	int nboards = 0; // number of input camera chessboard

	/*
	* detect the corner of each image in order to run the camera calibration
	*/
	ifstream infile(path);
	string line;
	while(getline(infile, line)){
		Mat view_color; // store the image input

		view_color = imread(line.c_str(), CV_LOAD_IMAGE_COLOR);

		cout << "file name: "  << line << endl;
		// check the whether image is read successfully
		if(!view_color.data){
			cout << "Could not open or find the image" << endl;
			if( cvWaitKey(0) == 27 ) // allow ESC to quit, let user define we to go to the next image
				exit(-1); // exit the program, may be the result is not good
		}

		nboards++; // increase the image number count

		vector<Point2f> pointBuf;
		imageSize = view_color.size();
		// if return nonzero means success
		bool found;
		found = findChessboardCorners( view_color, boardSize, pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

		if(found){
			cout << "corner is found good bang bang" << endl;
			// improve the fond corner's coordinate accuracy for chessboards, subpixel accuray
			Mat view_gray;
			cvtColor(view_color, view_gray, COLOR_BGR2GRAY);
			cornerSubPix(view_gray, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			// push back to the image points
			imagePoints.push_back(pointBuf);

			// draw the corners
			drawChessboardCorners(view_color, boardSize, Mat(pointBuf), found);

			// show the image on the window to check the final result
			namedWindow("corners", WINDOW_AUTOSIZE); // create a window for display
			imshow( "corners", view_color); // show our image inside it
			// wait until to get user input
			if( cvWaitKey(0) == 27 ) // allow ESC to quit, let user define we to go to the next image
				exit(-1); // exit the program, may be the result is not good
		}else{
			cout << "current image not found" << endl;
		}
	} 
	// close the file
	infile.close();

	// for the input of calibrate parameter
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(imageSize, cameraMatrix, distMatrix, imagePoints, rvecs, tvecs, reprojErrs, totalAvgErr);

	cout << "intrinsic matrix" << endl << cameraMatrix << endl;

	// choose the fisrt chessboard as output
	cout << "translation matrix(3x1)" << endl << tvecs[0] << endl;
	// choose the fisrt chessboard as output
	cout << "rotation matrix(3x1)" << endl << rvecs[0] << endl;
	// output the distortion matrix
	cout << "dist_coeff" << endl << distMatrix << endl;
	// output the extrinsic matrix
	cout << "rotation matrix(3x3)" << endl;
	Mat rotationMatrix = Mat::zeros(3, 3, CV_64FC1);
	Mat translationMatrix(tvecs[0]);
	Rodrigues(rvecs[0], rotationMatrix);
	cout << rotationMatrix << endl;
	Mat extrinsicMatrix;
	hconcat(rotationMatrix, translationMatrix, extrinsicMatrix);
	cout << "extrinsicMatrix" << endl << extrinsicMatrix << endl;

	cout << (ok ? "Calibration succeeded" : "Calibration failed") << ". avg reprojection error = " << totalAvgErr << endl;
	// save the calibration result
	if(ok);
	// saveCameraParams(imageSize, cameraMatrix, distMatrix, rvecs, tvecs, reprojErrs, totalAvgErr);

	// reproject the 3D points in world coordinate to the 2D image coordinates
	infile.open(path, ios::in); // for the input file ios::in
	// 3D points in the world coordinate system
	// (0,0,5)
	// (1,1,3) (1,-1,3) (-1,-1,3) (-1,1,3)
	// (0,0,0)
	// something is wwrong in reprojection the 3D point in the 2D image
	if(ok){
		int idx = 0;

		vector<Point3d> objectPoints;
		objectPoints.push_back(Point3f(0.0f,0.0f,-2.0f));
		objectPoints.push_back(Point3f(1.0f,1.0f,0.0f));
		objectPoints.push_back(Point3f(1.0f,-1.0f,0.0f));
		objectPoints.push_back(Point3f(-1.0f,-1.0f,0.0f));
		objectPoints.push_back(Point3f(-1.0f,1.0f, 0.0f));


		while(getline(infile, line)){
			Mat view_color; // store the image input

			view_color = imread(line.c_str(), CV_LOAD_IMAGE_COLOR);
			idx++;
			cout << "file name: "  << line << endl;
			// check the whether image is read successfully
			if(!view_color.data){
				cout << "Could not open or find the image" << endl;
				if( cvWaitKey(0) == 27 ) // allow ESC to quit, let user define we to go to the next image
					exit(-1); // exit the program, may be the result is not good		
			}

			//Mat rotVec, transVec;
			//float squareSize = 1.0f;
			//vector<Point3f> objectPointsOri;
			//objectPointsOri.clear();
			//for(int i = 0; i < boardSize.height; ++i){
			//	for(int j = 0; j < boardSize.width; ++j){
			//	// z = 0, hence we are use the chessboard as the origin of world coordinate (x, y, x).
			//	// It is important to check the order of the point.
			//		objectPointsOri.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0)); 
			//	}
			//}
			//solvePnP(objectPointsOri, imagePoints[idx-1], cameraMatrix, distMatrix, rotVec, transVec);

			vector<Point2d> imagePointsRe;
			projectPoints(objectPoints, rvecs[idx - 1], tvecs[idx - 1], cameraMatrix, distMatrix, imagePointsRe);
			// imagePoints2.push_back(imagePoints[0]);
			// projectPoints(objectPoints, rotVec, transVec, cameraMatrix, distMatrix, imagePointsRe);

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


			//cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[5]), Scalar(0,0,255), thinkness, lineType);
			
		/*	cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[1]), Scalar(0,0,255), thinkness, lineType);
			cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[2]), Scalar(0,0,255), thinkness, lineType);
			cv::line(view_color, Point(imagePointsRe[0]), Point(imagePointsRe[3]), Scalar(0,0,255), thinkness, lineType);
			*/
			// show the image on the window to check the final result
			namedWindow("reproject", WINDOW_AUTOSIZE); // create a window for display
			imshow( "reproject", view_color); // show our image inside it
			// wait until to get user input
			if( cvWaitKey(0) == 27 ) // allow ESC to quit, let user define we to go to the next image
				exit(-1); // exit the program, may be the result is not good
		}
	}
	// close the file
	infile.close();

	return ok;
}
