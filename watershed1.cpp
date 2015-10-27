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
	imshow("otsu", img_gray_otsu);

	threshold(img_gray, img_gray_thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);
	//img_gray_thresh = ~img_gray_thresh;
	//bitwise_not(img_gray_thresh, img_gray_thresh);
	//namedWindow("thresh", WINDOW_AUTOSIZE);
	dilate(img_gray_thresh, img_gray_thresh, elementDilate);
	erode(img_gray_thresh, img_gray_thresh, elementErode);
	imshow("thresh", img_gray_thresh);

	distanceTransform(img_gray_thresh, img_gray_d_t, CV_DIST_L2, 3);
	normalize(img_gray_d_t, img_gray_d_t, 0, 1, NORM_MINMAX);
	namedWindow("dt", WINDOW_AUTOSIZE);
	imshow("dt", img_gray_d_t);

	img_gray_d_t.copyTo(img_gray_d_t_G);

	threshold(img_gray_d_t_G, img_gray_d_t_thresh_G, 0.3, 1, THRESH_BINARY);
	namedWindow("dt-thresh", WINDOW_AUTOSIZE);
	imshow("dt-thresh", img_gray_d_t_thresh_G);

	waitKey(0);

	destroyAllWindows();

}