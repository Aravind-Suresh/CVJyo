#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
	blur( src_gray, edge_gray, Size(3,3) );
	Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
}

int main(int argc, char** argv) {
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_gray_edge(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	int edgeThresh = 1;
	int lowThreshold = 30;
	int const max_lowThreshold = 100
	int ratio = 1;
	int kernel_size = 3;

	if(!img_gray.data) {
		cout<<"Error opening the file";
		return -1;
	}

	namedWindow("img_gray",WINDOW_AUTOSIZE);
	imshow("img_gray",img_gray);

	CannyThreshold(img_gray, img_gray_edge, lowThreshold, lowThreshold*ratio, kernel_size);
	namedWindow("img_gray_edge", WINDOW_AUTOSIZE);
	imshow("img_gray_edge",img_gray_edge);

	waitKey(0);

	destroyWindow("img_gray");
	destroyWindow("img_gray_edge");

	return 0;
}