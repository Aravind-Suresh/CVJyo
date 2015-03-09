#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int value=0.5;

Mat img_gray_d_t_thresh_G;
Mat img_gray_d_t_G;

void dt_thresh_callback(int , void*) {
	//cout<<value<<endl;
	threshold(img_gray_d_t_G, img_gray_d_t_thresh_G, (float) (value/255), 1, THRESH_BINARY);
	imshow("dt-thresh", img_gray_d_t_thresh_G);	
}

int main(int argc, char** argv) {
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_bgr = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_gray_thresh(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));
	Mat img_gray_d_t_thresh(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));
	Mat img_gray_d_t(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));
	Mat img_gray_otsu(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));
	Mat img_gray_otsu_and_lapl(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));

	Mat img1(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));

	int kernel_size = 5;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_8UC1;

	int dilation_type[] = {0,1,2};
	int dilation_size = 1;

	int erosion_type[] = {0,1,2};
	int erosion_size = 1;

	Mat elementDilate = getStructuringElement( dilation_type[1],
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );

	Mat elementErode = getStructuringElement( erosion_type[1],
		Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		Point( erosion_size, erosion_size ) );

	threshold(img_gray, img_gray_otsu, 0, 255, THRESH_BINARY + THRESH_OTSU);
	erode(img_gray_otsu, img1, getStructuringElement(erosion_type[1], Size(3,3), Point(1,1)));
	namedWindow("otsu", WINDOW_AUTOSIZE);
	imshow("otsu", img_gray_otsu);
/*

	Laplacian(img_gray, img_gray, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);

	bitwise_and(img1, img_gray, img_gray);*/
/*

	erode(img_gray, img_gray, elementErode);
	dilate(img_gray, img_gray, elementDilate);*/

	namedWindow("lapl-img", WINDOW_AUTOSIZE);
	imshow("lapl-img", img_gray);

	GaussianBlur( img_gray, img_gray, Size(5, 5), 2, 100);

	threshold(img_gray, img_gray_thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
	bitwise_not(img_gray_thresh, img_gray_thresh);
	namedWindow("thresh", WINDOW_AUTOSIZE);
	imshow("thresh", img_gray_thresh);

	distanceTransform(img_gray_thresh, img_gray_d_t, CV_DIST_L2, 3);
	normalize(img_gray_d_t, img_gray_d_t, 0, 1, NORM_MINMAX);
	namedWindow("dt", WINDOW_AUTOSIZE);
	imshow("dt", img_gray_d_t);

	img_gray_d_t.copyTo(img_gray_d_t_G);

	threshold(img_gray_d_t_G, img_gray_d_t_thresh_G,0, 1, THRESH_BINARY);
	namedWindow("dt-thresh", WINDOW_AUTOSIZE);
	imshow("dt-thresh", img_gray_d_t_thresh_G);
	createTrackbar("Threshold value","dt-thresh",&value,255,dt_thresh_callback);

	vector<Vec4i> lines;

	img_gray_d_t_G.convertTo(img_gray_d_t_G, CV_8UC1);

	/*HoughLinesP(img_gray_d_t_G, lines, 1, CV_PI/180, 25, 15, 20);

	for(size_t i=0;i<lines.size();i++) {
		cout<<"Line from : ("<<lines[i][0]<<","<<lines[i][1]<<") to ("<<lines[i][2]<<","<<lines[i][3]<<")"<<endl;
		line(img_bgr, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8);
	}*/

	namedWindow("lines", WINDOW_AUTOSIZE);
	imshow("lines", img_bgr);

	waitKey(0);

	destroyAllWindows();

}