#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

bool comparatorContourAreas ( vector<Point> c1, vector<Point> c2 ) //comparator function to sort the contours in increasing order of area
 {
	double i = fabs( contourArea(Mat(c1)) );
	double j = fabs( contourArea(Mat(c2)) );
	return ( i < j );
}

void cropHand(Mat src, Mat* result, int max_x, int min_x, int max_y, int min_y)// crophand function to crop the hand alone to the destination image
{
	cout<<"cropHand called";
for (int i=min_y;i<max_y;i++)
	for(int j=min_x;j<max_x;j++)
(*result).at<uchar>(i-min_y,j-min_x)=src.at<uchar>(i,j);

}

void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
	//blur( src_gray, edge_gray, Size(3,3) );
	GaussianBlur( src_gray, edge_gray, Size(5,5), 2, 2 );
	Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
}

int main(int argc, char** argv) 
{
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_gray_otsu(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_otsu_temp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_otsu_dil(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	threshold(img_gray, img_gray_otsu, 0, 255, THRESH_BINARY + THRESH_OTSU);

	namedWindow("img_gray_otsu", WINDOW_AUTOSIZE);
	imshow("img_gray_otsu",img_gray_otsu);

	int dilation_type[] = {0,1,2};
	int dilation_size = 2;

	int edgeThresh = 1;
		int lowThreshold = 30;
		int const max_lowThreshold = 100;
		int ratio = 1;
		int kernel_size = 3;

	Mat element = getStructuringElement( dilation_type[0],
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );

	dilate( img_gray_otsu, img_gray_otsu_dil, element );
	//dilate( img_gray_bit_and_morph1, img_gray_bit_and_morph1_dil, element );

	namedWindow("img_gray_otsu_dil", WINDOW_AUTOSIZE);
	imshow("img_gray_otsu_dil",img_gray_otsu);

	vector<Vec4i> hierarchy_minMax;
	vector<vector<Point> > contours_minMax;
	img_gray_otsu_dil.copyTo(img_gray_otsu_temp);	
	findContours(img_gray_otsu_temp, contours_minMax, hierarchy_minMax, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    sort(contours_minMax.begin(),contours_minMax.end(),comparatorContourAreas); //sorting the contours to get the maximum area contour i.e the hand
    // cout<<"\ncontours_minMax size"<<contours_minMax.size();
	drawContours(img_gray_otsu_dil,contours_minMax,contours_minMax.size()-1,Scalar(125,125,125),3,8,hierarchy_minMax);

	Rect bRect(boundingRect(contours_minMax[contours_minMax.size()-1])); //finding the bounding rectange for the maximum contour(ie the hand)
	
	Mat img_cropped(bRect.height,bRect.width, CV_8UC1, Scalar::all(0)); //cropping the image to the size of the hand alone

	cropHand(img_gray,&img_cropped, bRect.x+bRect.width, bRect.x, bRect.y+bRect.height,bRect.y);

	namedWindow("img_cropped", WINDOW_AUTOSIZE);
	imshow("img_cropped",img_cropped);

	Mat img_cropped_edge(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));

	CannyThreshold(img_cropped, img_cropped_edge, lowThreshold, lowThreshold*ratio, kernel_size);
	
	namedWindow("img_cropped_edge", WINDOW_AUTOSIZE);
	imshow("img_cropped_edge",img_cropped_edge);

		

waitKey(0);
}