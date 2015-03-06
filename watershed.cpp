#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

void binaryAbsDiff(Mat src1, Mat src2, Mat& res )
{   for(int i=0;i<src1.rows;i++)
	{	for(int j=0;j<src1.cols;j++)
		{
			if(src2.at<uchar> (i,j)>src1.at<uchar>(i,j))
				res.at<uchar>(i,j)=0;
			else
				res.at<uchar>(i,j)=255;
		}
	}
}

int main(int argc, char** argv) {
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_gray_thresh(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_thresh_dilate(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_thresh_erode(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_thresh_erode_not(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_TBD_DE(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	int dilation_type[] = {0,1,2};
	int dilation_size = 2;

	int erosion_type[] = {0,1,2};
	int erosion_size = 2;

	namedWindow("gray", WINDOW_AUTOSIZE);
	imshow("gray", img_gray);

	threshold(img_gray, img_gray_thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
	namedWindow("thresh", WINDOW_AUTOSIZE);
	imshow("thresh", img_gray_thresh);

	Mat elementDilate = getStructuringElement( dilation_type[0],
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );
	dilate( img_gray_thresh, img_gray_thresh_dilate, elementDilate );

	namedWindow("dilate", WINDOW_AUTOSIZE);
	imshow("dilate", img_gray_thresh_dilate);

	Mat elementErode = getStructuringElement( erosion_type[0],
		Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		Point( erosion_size, erosion_size ) );

	erode( img_gray_thresh, img_gray_thresh_erode, elementErode );
	namedWindow("erode", WINDOW_AUTOSIZE);
	imshow("erode", img_gray_thresh_erode);

	bitwise_not(img_gray_thresh_erode, img_gray_thresh_erode_not);
	namedWindow("not_erode", WINDOW_AUTOSIZE);
	imshow("not_erode", img_gray_thresh_erode_not);

	binaryAbsDiff(img_gray_thresh_dilate,img_gray_thresh_erode_not,img_gray_TBD_DE);
	namedWindow("diffD_E", WINDOW_AUTOSIZE);
	imshow("diffD_E", img_gray_TBD_DE);

	waitKey(0);

	destroyAllWindows();

	return 0;
}