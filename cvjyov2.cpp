//final edited code

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

void showImages(int l, int h, vector<Mat> imgs) {
	for(int i=l;i<=h;i++) {
		char str[2];
		str[0] = (char)(i+49);
		str[1] = '\0';
		imshow(str, imgs[i]);
	}
}

int main(int argc, char** argv) {

	RNG rng(12345);

	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_hsv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	cvtColor(img_rgb,img_hsv,CV_RGB2HSV);

	imshow("img_hsv", img_hsv);

	//Storage for all Mat images used during this program execution
	vector<Mat> imgs(25);

	/*
	 *	0 - blur
	 *	1 - + canny
	 *	2 - + laplacian
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

	 vector<vector<Point> > contours;
	 vector<Vec4i> hierarchy;

	//Canny followed by Laplacian
	 GaussianBlur( img_gray, imgs[0], Size(3,3), 2, 2 );
	 CannyThreshold(img_gray, imgs[0], lowThreshold, lowThreshold*ratio, kernel_size);

	 //median blur on canny to remove salt and pepper noise

	 //medianBlur(imgs[0], imgs[0], 1);

	 Laplacian(imgs[0], imgs[1], CV_8UC1, 3);
	 GaussianBlur( imgs[1], imgs[2], Size(5,5), 2, 2 );
	 imshow("temp-2", imgs[2]);

	 vector<Vec4i> convexityDefectsSet;

	 Mat imgtemp;
	 imgs[2].copyTo(imgtemp);
	 findContours(imgtemp, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	 sort(contours.begin(),contours.end(),comparatorContourAreas);
	 int size1 = contours.size();

	 vector<int> hull;

	 convexHull(Mat(contours[size1-1]), hull,false);

	 Scalar color( 0,0,0 );
	 drawContours(img_gray,contours,contours.size()-1, color,2, 8, hierarchy);
	 convexityDefects(contours[size1-1], hull, convexityDefectsSet);

	 int filterThreshDepth = 10;
	 vector<Point> defectsPoints;

	 sort(convexityDefectsSet.begin(), convexityDefectsSet.end(), comparatorConvexityDefectsSetDepth);
	 int minposx=0;

	 for(int k=0;k<convexityDefectsSet.size();k++) {
	 	int startIdx = convexityDefectsSet[k].val[0];
	 	int endIdx = convexityDefectsSet[k].val[1];
	 	int defectPtIdx = convexityDefectsSet[k].val[2];
	 	double depth = static_cast<double>(convexityDefectsSet[k].val[3]) / 256.0;

	 	cout<<endl<<k<<"  ";

	 	cout << startIdx << ' ' << endIdx << ' ' << defectPtIdx << ' ' << depth << endl;

	 	Scalar color = Scalar( 0,0,0 );
	 	float toleranceFractionPointX = 0.05;
	 	float toleranceFractionPointY = 0.05;

	 	Point p = contours[size1-1][defectPtIdx];
	 	if(k>=filterThreshDepth && checkPointInRegion(img_gray, toleranceFractionPointX, toleranceFractionPointY, p)) {
	 		defectsPoints.push_back(p);
	 		circle(imgs[3], contours[size1-1][defectPtIdx] , 10, color, 2, 8, 0 );
	 	}
	 	color = Scalar(0,0,0);
	 	circle(imgs[4], contours[size1-1][defectPtIdx] , 10, color, 2, 8, 0 );
	 	circle(imgs[5], contours[size1-1][startIdx] , 10, color, 2, 8, 0 );
	 	circle(imgs[6], contours[size1-1][endIdx] , 10, color, 2, 8, 0 );
	 }

	 Point2f minEncCirCenter;
	 float minEncRad;

	 minEnclosingCircle(defectsPoints, minEncCirCenter, minEncRad);
	 imgs[7] = Scalar::all(0);
	 circle(imgs[7], minEncCirCenter, minEncRad, Scalar(255,255,255), -1, 8, 0);

	 bitwise_and(imgs[7], imgs[1], imgs[8]);

	 vector<vector<Point> > contours1;
	 vector<Vec4i> hierarchy1;

	 imgs[8].copyTo(imgtemp);
	 imgs[8].copyTo(imgs[9]);
	 findContours(imgtemp, contours1, hierarchy1, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	 for(int i=0;i<contours1.size();i++) {
	 	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	 	drawContours(imgs[9],contours1,i, color,2, 8, hierarchy1);
	 }

	 threshold( imgs[9], imgs[10], 1, 255, 0 );		
	 //threshold binary - 255 for all I(x,y)>1 , 0 for all I(x,y) = 0
	 bitwise_not(imgs[10],imgs[11]);
	 threshold(imgs[13], imgs[12], 0, 255, THRESH_BINARY + THRESH_OTSU);

	 //imshow("otsu", imgs[12]);
	 bitwise_and(imgs[11],imgs[12],imgs[13]);

	 Mat img_otsu_32(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(1));
	 imgs[12].convertTo(img_otsu_32, CV_32F);

	 //imshow("otsu_32", img_otsu_32);

	 Mat img_ones_init(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0.8));
	 Mat img_ones(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0.8));

	 bitwise_and(img_otsu_32,img_ones_init,img_ones);

	 Mat img_dt(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));

	 distanceTransform(imgs[12], img_dt, CV_DIST_L2, 3);
	 normalize(img_dt, img_dt, 0, 0.2, NORM_MINMAX);

	 //convert

	 Mat img_out(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));	 

	 add(img_ones, img_dt, img_out, noArray(), -1);

	 imshow("img_ones", img_ones);
	 imshow("img_out", img_out);
	 imshow("img_dt", img_dt);

	 //showImages(0, 14, imgs);

	 waitKey(0);
	 destroyAllWindows();

	 return 0;
	}