//cvjyo ver 2.0

#include<opencv2/opencv.hpp>
#include<iostream>

#define NL cout<<endl

using namespace cv;
using namespace std;


void distTransform(Mat input,Mat &result)
{

	for(int i=0;i<input.rows;i++)
		for(int j=0;j<input.cols;j++)
			if(input.at<uchar>(i,j)==255)
				
}

void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
	//blur( src_gray, edge_gray, Size(3,3) );
	GaussianBlur( src_gray, edge_gray, Size(5,5), 2, 2 );
	Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
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

bool checkPointInRegion(Mat src, float perx, float pery,Point p)
{
	//TODO : fill the function here
	int width = src.cols;
	int height = src.rows;
	int x1 = width*perx;
	int x2 = width - x1;
	int y1 = height*pery;
	int y2 = height - y1;

	if(p.x>x1 && p.x<x2 && p.y>y1 && p.y<	y2)
		return true;
	else return false;
}

int main(int argc, char** argv) {
	//Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	//Mat img_hsv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	//cvtColor(img_rgb,img_hsv,CV_RGB2HSV);

	int edgeThresh = 1;
	int lowThreshold = 40;
	int const max_lowThreshold = 100;
	int maxThreshold = 40;
	int kernel_size = 5;

	vector<vector<Point> > contours1;
	vector<vector<Point> > contours1final;
	vector<int> arcLengths1;
	vector<Vec4i> hierarchy1;
	vector<Vec4i> hierarchy1final;

	VideoCapture cap(0);

	while(1) {

		Mat img_rgb;
		cap>>img_rgb;

		Mat img_gray(img_rgb.rows, img_rgb.cols, CV_8UC1, Scalar::all(0));
		cvtColor(img_rgb, img_gray, CV_RGB2GRAY);

		Mat img1(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img2(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img3(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img4(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img5(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));	
		Mat img6(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img7(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img8(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img9(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img10(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

		Mat img_gray_temp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_gray_sharp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_hull_black(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_hull_3(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_hull(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_gray_temp3(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_defects_4(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_defects_5(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_defects_3_bin_inv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));


		Mat img_defects_1(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_defects_2(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
		Mat img_defects_3_bin(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

		img_gray.copyTo(img1);
		img_gray.copyTo(img2);
		img_gray.copyTo(img_hull_black);
		img_gray.copyTo(img_hull_3);
		img_gray.copyTo(img_gray_temp);

		CannyThreshold(img_gray, img1, lowThreshold, maxThreshold, kernel_size);
		imshow("canny", img1);

		findContours(img1, contours1, hierarchy1, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

		vector<Vec4i> convexityDefectsSet1;

		sort(contours1.begin(),contours1.end(),comparatorContourAreas);
		int size1 = contours1.size();
		double areaTemp;
		double sizeImg = img_gray.rows * img_gray.cols;

		cout<<sizeImg<<endl<<endl;
		cout<<"area"<<endl;

		for(int i=0;i<size1;i++) {
			areaTemp = fabs(contourArea(Mat(contours1[i])));
			cout<<areaTemp<<"\t";
			if(areaTemp>(double)(sizeImg*0.0) && areaTemp<(double)(sizeImg*0.90)) {
				contours1final.push_back(contours1[i]);
				hierarchy1final.push_back(hierarchy1[i]);
			}
		}

		Scalar color(255,255,255);
		cout<<contours1.size()<<endl;
		cout<<contours1final.size();
		drawContours(img1,contours1, - 1, color,2, 8, hierarchy1);

		imshow("im1", img1);

		drawContours(img2,contours1final, - 1, color,2, 8, hierarchy1final);

		imshow("im2", img2);

		vector<Vec4i> convexityDefectsSet;
		sort(contours1.begin(),contours1.end(),comparatorContourAreas);
		//int size1 = contours.size();

		vector<vector<int> >hulls( contours1.size() );

		convexHull(Mat(contours1[size1-1]), hulls[1],false);
		convexityDefects(contours1[size1-1], hulls[1], convexityDefectsSet);

		Mat drawing = Mat::zeros( img_gray.size(), CV_8UC3 );
		cout<<hulls[1].size()<<endl;

		for(int kk=0;kk<hulls[1].size();kk++) {
			cout<<hulls[1][kk]<<endl;

		}

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

			Point p = contours1[size1-2][defectPtIdx];
			if(k>=filterThreshDepth && checkPointInRegion(img_gray, toleranceFractionPointX, toleranceFractionPointY, p)) {
				//cout<<"Defect point : "<<k<<" ("<<p.x<<","<<p.y<<")";
				defectsPoints.push_back(p);
				color = Scalar(255,255,255);
			}


			circle(img_gray_temp, contours1[size1-2][defectPtIdx] , 10, color, 2, 8, 0 );
			circle(img_hull_black, contours1[size1-2][startIdx] , 10, color, 2, 8, 0 );
			circle(img_hull_3, contours1[size1-2][endIdx] , 10, color, 2, 8, 0 );
		}

		namedWindow("img_hull_defect",WINDOW_AUTOSIZE);
		imshow("img_hull_defect", img_gray_temp);
		namedWindow("img_hull_start", WINDOW_AUTOSIZE);
		imshow("img_hull_start",img_hull_black);
		namedWindow("img_hull_end", WINDOW_AUTOSIZE);
		imshow("img_hull_end",img_hull_3);

		Point2f minEncCirCenter;

		float minEncRad;

		/*minEnclosingCircle(defectsPoints, minEncCirCenter, minEncRad);
		circle(img_defects_2, minEncCirCenter, minEncRad, Scalar(255,255,255), -1, 8, 0);
		cout<<"defects points size : "<<defectsPoints.size()<<endl;

		for(int k2=0;k2<defectsPoints.size();k2++) {
			if(defectsPoints[k2].x<defectsPoints[minposx].x) {
				minposx = k2;
			}
		}
*/
		if(waitKey(1)==27) {
			destroyAllWindows();
			cap.release();
			break;
		}
	}

	//vector<vector<int> >hulls1( contours1final.size());

	//convexHull(Mat(contours1[size1-2]), hulls[1],false);

}
