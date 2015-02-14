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
	Mat img_gray_edge_inv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_skinmask(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_and(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_and_inv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_and_inv_temp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_and_inv_open(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	vector<vector<Point> > contours;
	vector<int> arcl;
	int pos,a;
	vector<Vec4i> hierarchy;
	vector<int> row(5,1);
	vector<vector<int> > kernelOpen(5,row);

	/*
		dilation_type

		MORPH_RECT = 0,
		MORPH_CROSS = 1,
		MORPH_ELLIPSE = 2
	*/
		int dilation_type[] = {0,1,2};
		int dilation_size = 2;

		int edgeThresh = 1;
		int lowThreshold = 30;
		int const max_lowThreshold = 100;
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

		bitwise_not(img_gray_edge, img_gray_edge_inv);
		namedWindow("img_gray_edge_inv", WINDOW_AUTOSIZE);
		imshow("img_gray_edge_inv",img_gray_edge_inv);

	inRange(img_gray, 110, 255, img_skinmask);				//TODO : Make it adaptive

	bitwise_and(img_gray, img_gray, img_gray_bit_and, img_skinmask);
	namedWindow("img_gray_bit_and", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and",img_gray_bit_and);

	//Some morphological operations -- dilate on(binary+otsu on img_gray_bit_and)

	threshold(img_gray_bit_and, img_gray_bit_and_morph1, 0, 255, THRESH_BINARY + THRESH_OTSU);

	Mat element = getStructuringElement( dilation_type[0],
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );
	dilate( img_gray_bit_and_morph1, img_gray_bit_and_morph1, element );
	namedWindow("img_gray_bit_and_morph1", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1",img_gray_bit_and_morph1);

	bitwise_and(img_gray_bit_and_morph1, img_gray_edge_inv, img_gray_bit_and_morph1_bit_and);
	namedWindow("img_gray_bit_and_morph1_bit_and", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1_bit_and",img_gray_bit_and_morph1_bit_and);

	bitwise_not(img_gray_bit_and_morph1_bit_and, img_gray_bit_and_morph1_bit_and_inv);

	img_gray_bit_and_morph1_bit_and_inv.copyTo(img_gray_bit_and_morph1_bit_and_inv_temp);
	namedWindow("img_gray_bit_and_morph1_bit_and_inv_temp", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1_bit_and_inv_temp",img_gray_bit_and_morph1_bit_and_inv_temp);

	findContours(img_gray_bit_and_morph1_bit_and_inv, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Scalar color( 255, 0, 0 );
	/*int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( img_gray_bit_and_morph1_bit_and_inv, contours, idx, color, CV_FILLED, 8, hierarchy );
    }*/

        /*drawContours( img_gray_bit_and_morph1_bit_and_inv, contours, 3, color, CV_FILLED, 8, hierarchy );
        namedWindow("img_gray_bit_and_morph1_bit_and_inv", WINDOW_AUTOSIZE);
        imshow("img_gray_bit_and_morph1_bit_and_inv",img_gray_bit_and_morph1_bit_and_inv);*/
        for(int i = 0; i < contours.size(); i++) {
        	for(int j = 0; j < contours[i].size(); j++)
        		cout<<contours[i][j]<<endl;
        	cout<<endl;
        	arcl.push_back(arcLength(contours[i], true));
        	if(i==0) {
        		a=arcl[0];
        		pos = 0;
        	}
        	if(a<arcl[i]) {
        		a = arcl[i];
        		pos = i;
        	}
        	cout<<a<<endl<<pos;
        }
        vector<int> hull(contours[pos].size());
        vector<Vec4i> convexityDefectsSet;
        convexHull(contours[pos], hull, false, false );
	/*for(int k=0;k<contours[pos].size();k++) {
		cout<<hull[k]<<endl;
	}*/
	//drawContours( drawing, hull, s, Scalar(0,255,255), 1, 8, vector<Vec4i>(), 0, Point() );
		convexityDefects(contours[pos], hull, convexityDefectsSet);
		for(int k=0;k<convexityDefectsSet.size();k++) {
			int startIdx = convexityDefectsSet[k].val[0];
			int endIdx = convexityDefectsSet[k].val[1];
			int defectPtIdx = convexityDefectsSet[k].val[2];
			double depth = static_cast<double>(convexityDefectsSet[k].val[3]) / 256.0;

			cout << startIdx << ' ' << endIdx << ' ' << defectPtIdx << ' ' << depth << endl;

			Scalar color = Scalar( 255,0,0 );
			Point2f p(defectPtIdx, defectPtIdx);
			circle(img_gray_bit_and_morph1, p , 10, color, 2, 8, 0 );
			namedWindow("img_hull", WINDOW_AUTOSIZE);
			imshow("img_hull",img_gray_bit_and_morph1);
		}

	//morphologyEx(img_gray_bit_and_morph1_bit_and_inv, img_gray_bit_and_morph1_bit_and_inv_open, MORPH_OPEN, kernelOpen, Point(-1,-1), 1, BORDER_CONSTANT);
	//namedWindow("img_gray_bit_and_morph1_bit_and_inv_open", WINDOW_AUTOSIZE);
	//imshow("img_gray_bit_and_morph1_bit_and_inv_open",img_gray_bit_and_morph1_bit_and_inv_open);

		waitKey(0);

		destroyWindow("img_gray");
		destroyWindow("img_gray_edge");
		destroyWindow("img_gray_edge_inv");
		destroyWindow("img_gray_bit_and");
		destroyWindow("img_gray_bit_and_morph1");
		destroyWindow("img_gray_bit_and_morph1_bit_and");
	//destroyWindow("img_gray_bit_and_morph1_bit_and_inv");
	//destroyWindow("img_gray_bit_and_morph1_bit_and_inv_open");

		return 0;
	}