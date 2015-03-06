#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int value=0;

Mat img_gray_d_t_thresh_G;
Mat img_gray_d_t_G;

void dt_thresh_callback(int , void*) {
	cout<<value<<endl;
	threshold(img_gray_d_t_G, img_gray_d_t_thresh_G, value, 255, THRESH_BINARY);
	imshow("dt-thresh", img_gray_d_t_thresh_G);	
}

int main(int argc, char** argv) {
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_gray_thresh(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));
	Mat img_gray_d_t_thresh(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));
	Mat img_gray_d_t(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));

	threshold(img_gray, img_gray_thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
	bitwise_not(img_gray_thresh, img_gray_thresh);
	namedWindow("thresh", WINDOW_AUTOSIZE);
	imshow("thresh", img_gray_thresh);

	distanceTransform(img_gray_thresh, img_gray_d_t, CV_DIST_L2, 3);
	normalize(img_gray_d_t, img_gray_d_t, 0, 1, NORM_MINMAX);
	namedWindow("dt", WINDOW_AUTOSIZE);
	imshow("dt", img_gray_d_t);

	img_gray_d_t.copyTo(img_gray_d_t_G);

	threshold(img_gray_d_t_G, img_gray_d_t_thresh_G, 0.5, 1, THRESH_BINARY);
	namedWindow("dt-thresh", WINDOW_AUTOSIZE);
	imshow("dt-thresh", img_gray_d_t_thresh_G);
	//createTrackbar("Threshold value","dt-thresh",&value,255,dt_thresh_callback);

	waitKey(0);

	destroyAllWindows();

}