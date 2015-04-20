//final edited code

#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
	//blur( src_gray, edge_gray, Size(3,3) );
	GaussianBlur( src_gray, edge_gray, Size(5,5), 2, 2 );
	Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
}

void binaryAbsDiff(Mat src1, Mat src2, Mat& res )
{   for(int i=0;i<src1.rows;i++)
	{	for(int j=0;j<src1.cols;j++)
		{
			if(src2.at<uchar> (i,j)>src1.at<uchar>(i,j))
				res.at<uchar>(i,j)=255;
			else
				res.at<uchar>(i,j)=0;
		}
	}
}



int slopeStrLine(Point a, Point b)
{
	if(a.x-b.x)
		return (a.y-b.y)/(a.x-b.x);

}

bool checkPointInRegion(Mat src, float perx, float pery,Point p)
{
	//TODO : fill the function here
	int width = src.cols;
	int height = src.rows;
	int x1 = width*perx;
	int x2 = width - x1;
	int y1 = height*pery;
	int y2 = height - y1;

	if(p.x>x1 && p.x<x2 && p.y>y1 && p.y<y2)
		return true;
	else return false;
}

bool comparatorContourAreas ( vector<Point> c1, vector<Point> c2 ) {
	double i = fabs( contourArea(Mat(c1)) );
	double j = fabs( contourArea(Mat(c2)) );
	return ( i < j );
}

bool comparatorConvexityDefectsSetDepth (Vec4i a, Vec4i b) {
	double i = static_cast<double>(a.val[3]);
	double j = static_cast<double>(b.val[3]);
	return ( i < j );
}

int main(int argc, char** argv) {
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_hsv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	cvtColor(img_rgb,img_hsv,CV_RGB2HSV);

	imshow("img_hsv", img_hsv);

	//Storage for all Mat images used during this program execution
	vector<Mat> imgs(10);

	/*
	 *	0 -
	 *	1 - 
	 *	2 - 
	 *	3 - 
	 *	4 - 
	 *
	 *
	 *
	 *
	 *
	 *
	 *
	 *
	 */

	for(int i=0;i<imgs.size();i++) {
		img_gray.copyTo(imgs[i]);
	}

	int edgeThresh = 1;
	int lowThreshold = 30;
	int const max_lowThreshold = 100;
	int ratio = 1;
	int kernel_size = 3;

	//Canny followed by Laplacian
	GaussianBlur( img_gray, imgs[0], Size(5,5), 2, 2 );
	CannyThreshold(img_gray, imgs[0], lowThreshold, lowThreshold*ratio, kernel_size);
	Laplacian(imgs[0], imgs[1], CV_8UC1, 3);

	imshow("img0", imgs[0]);
	imshow("img1", imgs[1]);

	waitKey(0);
	destroyAllWindows();

	return 0;
}