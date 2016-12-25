#include "stdafx.h"
#include "stereo.h"


void Stereo::print_help(){
	cout <<
		" Given a list of chessboard images, the number of corners (nx, ny)\n"
		" on the chessboards, and a flag: useCalibrated for \n"
		"   calibrated (0) or\n"
		"   uncalibrated \n"
		"     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
		"         matrix separately) stereo. \n"
		" Calibrate the cameras and display the\n"
		" rectified results along with the computed disparity images.   \n" << endl;
	cout << "Usage:\n ./stereo_calib -w board_width -h board_height [-nr /*dot not view results*/] <image list XML/YML file>\n" << endl;
}

Stereo::Stereo(void){

}

Stereo::Stereo(int count)
{
	Size boardSize;
	string imagelistfn;
	bool showrectified = true; // shoe the rectified image default is true

	imagelistfn = "Computer_Vision_Hw1_Data_Stereo/list.txt";
	boardSize = Size(10, 7); // inner corner number
	// check the image list file name is valid
	// if not set file path, set it to the default setting
	//if(imagelistfn == ""){
	//	imagelistfn = "stereo_calib.xml";
	//	boardSize = Size(9, 6); // inner corner number
	//}else if(boardSize.width <= 0 || boardSize.height <= 0){
	//	cout << "if you specified XML file with chessboards, you should also specify the the board width and height in the command line" << endl;
	//	print_help();
	//	cv::waitKey(0);
	//	exit(-1);
	//}

	vector<string> imagelist;
	bool ok = readStringList(imagelistfn, imagelist);
	cout << "constructor: " << imagelist.size() << endl;
	if(!ok || imagelist.empty()){
		cout << "cannot open " << imagelistfn << " or the string list is empty" << endl;
		print_help();
		namedWindow("wait");
		cv::waitKey(0);
		exit(-1);
	}

	// start the calibration
	calibrate(imagelist, boardSize, true, showrectified);
}


Stereo::~Stereo(void)
{
}


void Stereo::calibrate(vector<string> imagelist, Size boardSize, bool useCalibrate, bool showRectified){
	if(imagelist.size() % 2 != 0){
		cout << "Error: the image list contains odd (non-even) number of the elements" << endl;
		cv::waitKey(0);
		return ;
	}

	/* start task 1 */

	bool displayCorners = true; // show the result of findChessboardCorners result in the window
	const float squareSize = 1.f; // set this value to the actual chessboard size

	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	Size imageSize;

	// calibrate the cameras separately, will be more accurate than the stereo calibrate
	int i, j, k, nimages = (int) imagelist.size() / 2;
	int maxScale = 1;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for(i = j = 0; i < nimages; ++i){
		for( k = 0; k < 2; ++k){
			const string& filename = imagelist[i * 2 + k];
			Mat img = imread(filename, 0);
			if(img.empty()){
				break;
			}
			if(imageSize == Size()){
				imageSize = img.size();
			}else if(img.size() != imageSize){
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair" << endl;
				cv::waitKey(0);
				break;
			}
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			// no need to scale here just skip this section of code
			// nothing happen
			for(int scale = 1; scale <= maxScale; ++scale){
				Mat timg;
				if(scale == 1){
					timg = img;
				}else{
					resize(img, timg, Size(), scale, scale);
				}
				found = findChessboardCorners(timg, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
				if(found){
					if(scale > 1){
						Mat cornerMat(corners);
						cornerMat *= 1./scale;
					}
					cout << "corner is good" << endl;
					break;
				}else{
					cout << "corner is bad" << endl;
				}
			}
			if(displayCorners){
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				double sf = 640./MAX(img.rows, img.cols);
				resize(cimg, cimg1, Size(), sf, sf);
				cv::imshow("corners", cimg1);
				char c = (char) cv::waitKey(500);
				if(c == 27 || c == 'q' || c == 'Q'){
					exit(-1);
				}
			}else{
				putchar('.');
			}

			if(!found){
				break;
			}
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.01));
		}
		if(k == 2){
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected." << endl;
	cout << "goodImageList size: " << goodImageList.size() << endl;

	nimages = j;
	if(nimages < 2){
		cout << "Error: too little pairs to run the calibration" << endl;
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[0].resize(nimages);
	objectPoints.resize(nimages);

	for(i = 0; i < nimages; ++i){
		for(j = 0; j < boardSize.height; ++j){
			for(k = 0; k < boardSize.width; ++k){
				objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0.0f));
			}
		}
	}

	cout << "running camera calibration separately" << endl;

	Mat cameraMatrix[2], distCoeffs[2];
	vector<Mat> rvecs, tvecs;
	cameraMatrix[0] = Mat::eye(3, 3, CV_64FC1); // camera matrix for the fisrt matrix
	cameraMatrix[1] = Mat::eye(3, 3, CV_64FC1); // camrea matrix for the second matrix
	distCoeffs[0] = Mat::zeros(8, 1, CV_64FC1); // distortion matrix for the fisrt camera
	distCoeffs[1] = Mat::zeros(8, 1, CV_64FC1); // distortion matrix for the second camera
	Mat R, T; // rotation and translation between two camera
	Mat E; // essential matrix
	Mat F; // fundamental matrix

	double rms_camera_1 = calibrateCamera(objectPoints, imagePoints[0], imageSize, cameraMatrix[0], distCoeffs[0], rvecs, tvecs, CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

	cout << "reptojection error camera 1: " << rms_camera_1 << endl;
	double rms_camera_2 = calibrateCamera(objectPoints, imagePoints[1], imageSize, cameraMatrix[1], distCoeffs[1], rvecs, tvecs, CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
	cout << "reptojection error camera 2: " << rms_camera_2 << endl; 

	bool ok  = checkRange(cameraMatrix[0]) && checkRange(distCoeffs[0]) 
		&& checkRange(cameraMatrix[1]) && checkRange(distCoeffs[1]);

	if(!ok) {
		cout << "something wrong after single camera calibration" << endl;
		return ;
	}

	// save the calibration result to the text file or other format, this is the check point one */

	/* end task 1 */

	/* start tesk 2 */
	// stereo calibration function, guven the data calibrated from the cameraCalibrate function
	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0], // camera matrix and distortion matrix for fisrt camera
		cameraMatrix[1], distCoeffs[1], // camera matrix ans distortion matrix for second camera
		imageSize, R, T, E, F,
		TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
		CV_CALIB_FIX_ASPECT_RATIO +
		CV_CALIB_ZERO_TANGENT_DIST +
		CV_CALIB_SAME_FOCAL_LENGTH +
		CV_CALIB_RATIONAL_MODEL +
		CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
	cout << "reptojection error stereo: " << rms << endl;

	// calibration quality check
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2]; // epipolar line for the camera 1 and camera 2
	for(i = 0; i < nimages; ++i){
		int npts = (int) imagePoints[0][i].size(); // get the number od the mage points in each image
		Mat imgpt[2]; // translate the vector to mat format
		for(k = 0; k < 2; ++k){
			imgpt[k] = Mat(imagePoints[k][i]);
			// the last parameter in this function if the matrix is empty, the identity new camera matrix will be used
			// I don't know why the lat paramater is used
			// fisrt parameter is src
			// second parameter is dst
			// fifth parameter is rectification transformation, if the matrix is empty the identity matrix is used here
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]); // epipolar usually work in the the undistort image coordinate
			computeCorrespondEpilines(imgpt[k],  k + 1, F, lines[k]); // the image index is start from one
		}
		for(j = 0; j < npts; ++j){
			double errij = fabs(imagePoints[0][i][j].x * lines[1][j][0] + 
				imagePoints[0][i][j].y * lines[1][j][1] + lines[1][i][2]) +	
				fabs(imagePoints[1][i][j].x * lines[0][j][0] + 
				imagePoints[1][i][j].y * lines[0][j][1] + lines[0][i][2]);
			err += errij;
		}
		npoints += npts;
	}

	cout << "The epipolar constraint err projection: " << err/npoints << endl;

	// save the extrinsic parameters, essential matrix, and fundamental matrix in the text file, this is the check point 2 */

	/* end task 2 */

	/* start task 3 */
	// task description
	// select the pair of stereo image for this tasks
	// use undistort() undistort lens distortion of both images, manually selected 3 points of interest
	// use funfamental matrix from task 2 and computeCorrespondEpilines() to find ans draw 3 epipolar
	// lines of the selected 3 points for each image. The epipolar lines found for the left image
	// should be drawn in the right image and vice versa. Confirm that corresponding points lie on their epipolar
	// lines in the other image
	// undistort function description:
	// transform ans image to compensate for lens distortion
	// random select 3 point
	vector<Point2f> imgpt[2];
	vector<Vec3f> epilines[2];
	imgpt[0] = vector<Point2f>(imagePoints[0][0]); // the point of the left camera
	imgpt[1] = vector<Point2f>(imagePoints[1][0]); // the point of the right camera correspond to the left camera points
	cout << "check 1" << endl;
	undistortPoints(imgpt[0], imgpt[0], cameraMatrix[0], distCoeffs[0], Mat(), cameraMatrix[0]); // epipolar usually work in the the undistort image coordinate
	computeCorrespondEpilines(imgpt[0],  1, F, epilines[0]); // the image index is start from one
	cout << "check 2" << endl;
	undistortPoints(imgpt[1], imgpt[1], cameraMatrix[1], distCoeffs[1], Mat(), cameraMatrix[1]); // epipolar usually work in the the undistort image coordinate
	computeCorrespondEpilines(imgpt[1],  2, F, epilines[1]); // the image index is start from one
	cout << "check 3" << endl;
	drawEpipolarLines("epipolar", F, 
		imread(imagelist[0], CV_LOAD_IMAGE_GRAYSCALE),
		imread(imagelist[1], CV_LOAD_IMAGE_GRAYSCALE),
		imgpt[0], imgpt[1], -1
		);
	// save the final result. this is the check point 3

	/* end task 3 */

	/* start task 4 */
	// Select one pair of stereo image for this task
	// use stereoRectify for this task to rectify for both left and right images
	// confirm the images rows are aligned. please sbmit the rectified and original image
	// submit two absolute difference images (between the rectified and original image)

	/* end task 4 */

	/* start task 5 */
	// select one pair of stereo image for this task
	// manually select at least 4 3D point of interset from image pair
	//  use undistortPoints to undistort and rectify selected points
	// use perspectiveTransform() to calculate the 3D information of the selected points
	Mat R1, R2; // 
	Mat P1, P2, Q;
	Rect validRoi[2]; // show the valid region of interest
	cout << "start stereoRectify()" << endl;
	stereoRectify(cameraMatrix[0], distCoeffs[0], // camrea matrix and distCoeffs matrix for the fisrt camera
		cameraMatrix[1], distCoeffs[1], // cameraMatrix and distCoeffs matrix for the second camera
		imageSize, // size of image of the calibration data image
		R, T, // rotation and translation matrix between two camera
		R1, R2, // 3x3rectification transform  (rotation matrix) for the cameras
		P1, P2, // 3x4projection matrix in the new rectified coordinate system
		Q, // disparity to depth mapping matrix. reference the reprojectImageTo3D()
		CALIB_ZERO_DISPARITY,
		1, // alpha,  alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost)
		imageSize, // new image resolution after rectification
		&validRoi[0], // output rectangles inside the rectified images where all the pixels are valid. if alpha=0, the ROIs will cover the whole image
		&validRoi[1]);

	// save the intrinsic parameter

	// OpenCV and handle left-right
	// or up-down camera arranements
	bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3)); // the matrix representation please reference the OpenCV documentation

	cout << "start the initUndistortRectifyMap()" << endl;

	if(!showRectified)
		return;

	// Precompute maps for cv::remap()
	Mat rmap[2][2];
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	// show the rectify image in the window
	Mat canvas; 
	int h = imageSize.height;
	int w = imageSize.width;
	// no need to do the scale here
	Mat temp = imread(goodImageList[0]);
	if(!isVerticalStereo){
		canvas.create(h, w * 2, temp.type());
	}else{
		canvas.create(h * 2, w, temp.type());
	}

	cout << "show the rectified image on the window in same window" << endl;

	for(i = 0; i < nimages; ++i){
		for(k = 0; k < 2; ++k){
			Mat img = imread(goodImageList[i * 2 + k], CV_LOAD_IMAGE_GRAYSCALE), rimg, cimg;
			remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
			cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			// canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
			rectangle(cimg, validRoi[k], Scalar(0, 0, 255), 3, 5);
			if(isVerticalStereo){
				cimg.copyTo(canvas(Rect(0, h * k, w, h)));
			}else{
				cimg.copyTo(canvas(Rect(w * k, 0, w, h)));
			}
		}

		if( !isVerticalStereo )
			for( j = 0; j < canvas.rows; j += 16 )
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
		else
			for( j = 0; j < canvas.cols; j += 16 )
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
		imshow("rectified", canvas);
		char c = (char)cv::waitKey(500);
		if( c == 27 || c == 'q' || c == 'Q' )
			break;
	}
	/* end task 5 */

	/* start task 6 */
	for(i = 0; i < nimages; ++i){
		// -- 1 read the image
		// read the image
		Mat imgLeft = imread(goodImageList[i * 2 + 0], CV_LOAD_IMAGE_GRAYSCALE);
		Mat imgRight = imread(goodImageList[i * 2 + 1], CV_LOAD_IMAGE_GRAYSCALE);
		Mat imgLeftr;
		Mat imgRightr;
		cout << "load()" << endl; 
		remap(imgLeft, imgLeftr, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
		remap(imgRight, imgRightr, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
		// and create the image in which we will save our disparities
		Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
		Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8U);
		cout << "remap()" << endl;
		// -- 2 call the constructor for StereoBM
		// must be 16
		StereoBM sbm;
		// prefilter type to eliminate photometric distortions, noise ans enhance the texture
		sbm.state->preFilterSize = 9; // averaging window size: ~5x5..21x21
		sbm.state->preFilterCap = 31; // truncation value for the prefiltered image pixel
		// correspondence using sum of absolute difference(SAD)
		sbm.state->SADWindowSize = 13; // the acceptable window size ~5x5..21x21
		sbm.state->numberOfDisparities = 128;
		sbm.state->minDisparity = 0; // minimum possible disparity value can be negative
		// post-filtering
		sbm.state->textureThreshold = 10; // the disparity is only computed for pixels
											// with textured enough neighborhood
		sbm.state->uniquenessRatio = 15;
		sbm.state->speckleRange = 8; // acceptable range of variation in window
		sbm.state->speckleWindowSize = 0; // disparity variation window
		sbm.state->disp12MaxDiff = 1;
		sbm(imgLeftr, imgRightr, imgDisparity16S);
		normalize(imgDisparity16S, imgDisparity8U, 0, 255, CV_MINMAX, CV_8U); // normalize the value 
		imshow("disparity", imgDisparity8U);
		char c = (char)cv::waitKey(500);
		if( c == 27 || c == 'q' || c == 'Q' )
			break;
	}
	// save the disparity map
	/* end task 6 */

	/* start task 7 */
	// reproject the image to 3D point cloud
	// and save the final result to xyz file
	// using the given to reconstruct the 3D shape
	cout << "storing the point cloud..." << endl;

	/* end task 7 */
	
	// the above stage check each step of the work work correctly
}

/*
* the input point will be the undistorted points.
*/
void Stereo::drawEpipolarLines(const string& title, const Mat& F, const Mat& img1, const Mat& img2, const vector<Point2f> points1, const vector<Point2f> points2, const float inlierDistance){
	cout << "here" << endl;
	cout << "rows: " << img1.rows << endl;
	cout << "cols: " << img1.cols << endl;
	Mat outImg(img1.rows, img1.cols * 2, img1.type());
	Rect rect1(0, 0, img1.cols, img1.rows);
	Rect rect2(img1.cols, 0, img1.cols, img1.rows);
	/*
	* Allow color drawing
	*/
	// cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
	// cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
	img1.copyTo(outImg(rect1));
	img2.copyTo(outImg(rect2));
	cvtColor(outImg, outImg, CV_GRAY2BGR); // allow color drawing
	vector<Vec3f> epilines1, epilines2;
	computeCorrespondEpilines(points1, 1, F, epilines1);
	computeCorrespondEpilines(points2, 2, F, epilines2);

	RNG rng(0);
	int npts = points1.size();
	for(int i = 0; i < npts; ++i){
		Scalar color(rng(256), rng(256), rng(256));
		line(outImg(rect2),
			Point(0,-epilines1[i][2]/epilines1[i][1]),
			Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
			color);
		circle(outImg(rect1), points1[i], 3, color, -1, CV_AA);

		line(outImg(rect1),
			Point(0,-epilines2[i][2]/epilines2[i][1]),
			Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
			color);
		circle(outImg(rect2), points2[i], 3, color, -1, CV_AA);
	}
	cv::imshow(title, outImg);
	// wait for the user enter the key
	cv::waitKey(0);
}

bool Stereo::readStringList(string imagelistfn, vector<string>& imageList){
	imageList.resize(0);
	/*FileStorage fs(imagelistfn, FileStorage::READ);
	if( !fs.isOpened() ){
	cout << "file cannot open" << endl;
	return false;
	}
	FileNode n = fs.getFirstTopLevelNode();
	if( n.type() != FileNode::SEQ ){
	cout << "not sequence" << endl;
	return false;
	}
	FileNodeIterator it = n.begin(), it_end = n.end();
	for( ; it != it_end; ++it )
	imageList.push_back((string)*it);

	cout << "size : " <<  imageList.size() << endl;*/

	ifstream infile(imagelistfn);
	string line;
	while(getline(infile, line)){
		imageList.push_back(line);
	}
	infile.close();

	return true;
}