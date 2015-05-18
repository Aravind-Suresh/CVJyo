#include<opencv2/opencv.hpp>
#include<iostream>
#include<string.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
	//blur( src_gray, edge_gray, Size(3,3) );
	GaussianBlur( src_gray, edge_gray, Size(5,5), 2, 2 );
	Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
}

int main(int argc, char** argv) {
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_hsv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	cvtColor(img_rgb,img_hsv,CV_RGB2HSV);

	int edgeThresh = 1;
	int lowThreshold = 30;
	int const max_lowThreshold = 100;
	int ratio = 1;
	int kernel_size = 3;

	vector<Mat> imgs(25);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	for(int i=0;i<imgs.size();i++) {
		img_gray.copyTo(imgs[i]);
	}

	CannyThreshold(img_gray, imgs[0], lowThreshold, lowThreshold*ratio, kernel_size);
	findContours(imgs[0], contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	drawContours(imgs[0],contours,i, color,2, 8, hierarchy1);

}