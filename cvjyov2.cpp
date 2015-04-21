//final edited code

#include<opencv2/opencv.hpp>
#include<iostream>
#include<string.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

int threshvalue=100;
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
	 *	4 - + hull defect points
	 *	5 - + hull start points
	 *	6 - + hull end points
	 *	7 - defect points min enclosing circle
	 *	8 - minenclosing circle AND canny
	 *	9 - 
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

	 int dilation_type[] = {0,1,2};
	 int dilation_size = 2;

	Mat element = getStructuringElement( dilation_type[0],	//getting structuring element for dilating
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );

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

		//cout<<endl<<k<<"  ";

		//cout << startIdx << ' ' << endIdx << ' ' << defectPtIdx << ' ' << depth << endl;

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

	bitwise_and(imgs[11],imgs[12],imgs[13]);

	Mat img_ones(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(200));
	 //first, it is an CV_8UC1 image - then converted to CV_32F image

	imshow("img_ones_init", img_ones);

	bitwise_and(imgs[13],img_ones,img_ones);
	imshow("img_ones", img_ones);

	Mat img_ones_32(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));
	img_ones.convertTo(img_ones_32, CV_32F, (1/255.0));

	double max,min;
	minMaxIdx(img_ones_32, &min, &max);

	cout<<endl<<min<<endl<<max<<endl;

	Mat img_dt(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));

	distanceTransform(imgs[12], img_dt, CV_DIST_L2, 3);
	normalize(img_dt, img_dt, 0, 0.1, NORM_MINMAX);

	Mat img_out(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));	 

	add(img_ones_32, img_dt, img_out, noArray(), CV_32F);

	imshow("img_out", img_out);

	threshold(img_out, img_out, (210/255.0), 255, 1);
	imshow("img_out_final", img_out);

	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy2;
	Mat img_out_temp(img_gray.rows, img_gray.cols, CV_32F, Scalar::all(0));

	img_out.convertTo(img_out_temp,CV_8UC1);

	findContours(img_out_temp, contours2, hierarchy2, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);


	for(int i=0;i<contours2.size();i++) {
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours(imgs[15],contours2,i, color,2, 8, hierarchy2);
	}
	imshow("img_out_ctr", imgs[15]);

	Mat img_out_edge(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_out_edge_dil(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	img_out.convertTo(img_out, CV_8UC1);

	CannyThreshold(img_out,img_out_edge,lowThreshold, lowThreshold*ratio, kernel_size);

	dilate( img_out_edge, img_out_edge_dil, element ); 
	//img_out_edge.copyTo(img_out_edge_dil);

	imshow("edge_dil", img_out_edge_dil);

	vector<vector<Point> > contours3;
	vector<Vec4i> hierarchy3;

	int erosion_type[] = {0,1,2};
	int erosion_size = 8;

	Mat elementErode = getStructuringElement( erosion_type[1],
		Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		Point( erosion_size, erosion_size ) );

	findContours(img_out_edge_dil, contours3, hierarchy3, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	sort(contours3.begin(),contours3.end(),comparatorContourAreas);
	imgs[15] = Scalar::all(0);
	drawContours(imgs[15],contours3,contours3.size()-1, Scalar(255,255,255),-1, 8, hierarchy3);

	erode(imgs[15], imgs[15], elementErode);

	imshow("img_out_largestctr",imgs[15]);

	bitwise_and(img_out, imgs[15], imgs[16]);

	imshow("img16", imgs[16]);

	vector<vector<Point> > contours4;
	vector<Vec4i> hierarchy4;

	float arcLengthMaxThreshold = 100;

	//eliminating small salt and pepper noise like dots - can be reduced using median blur also
	//can give median blur a toleranceFractionPointY

	imgs[16].copyTo(imgs[17]);
	findContours(imgs[17], contours4, hierarchy4, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	cout<<endl<<"arc length :"<<endl;

	for(int i=0;i<contours4.size();i++) {
		float arclen = arcLength(contours4[i], false);
		cout<<i<<"  "<<arclen<<endl;
		if(arclen<arcLengthMaxThreshold) {
			drawContours(imgs[17],contours4,i, Scalar(0,0,0),-1, 8, hierarchy4);
		}
	}

	dilate(imgs[17], imgs[17], element);	

	imshow("img17", imgs[17]);

	vector<Point> whitePoints;

	for(int i=0;i<imgs[17].rows;i++) {
		for(int j=0;j<imgs[17].cols;j++) {
			if(imgs[17].at<uchar>(j,i) !=0) {
				imgs[17].at<uchar>(j,i) = 255;
				whitePoints.push_back(Point(j,i));
			}
		}
	}

	cout<<endl<<"whitePoints size : "<< whitePoints.size()<<endl;

	//vector<int> counters1(contours.size());

	vector<int> totalPoints(contours.size());

	for(int i=0;i<contours.size();i++) {
		for(int j=0;j<imgs[0].rows;j++) {
			for(int k=0;k<imgs[0].cols;k++) {
				if(pointPolygonTest(contours[i], Point(k,j) , false) != -1) {
					totalPoints[i]++;
				}
			}
		}
	}

	//actually the handprint wasn't recognised as a closed contour at all.. thats y it cups.. otherwise this idea will work fine.

	float fractionOfWhitePoints = 0.9;

	for(int i=0;i<contours.size();i++) {
		int ctr = 0;
		for(int j=0;j<whitePoints.size();j++) {
			if(pointPolygonTest(contours[i], whitePoints[j], false) != -1) {
				ctr++;
			}
		}
		if(ctr!=0) {
			cout<<i<<"  "<<ctr<<"  "<<totalPoints[i]<<endl;
		}
		if(ctr>=fractionOfWhitePoints*totalPoints[i]) {
			cout<<"contour : "<<i<<endl;
			drawContours(imgs[18],contours,i, Scalar(0,0,0),1, 8, hierarchy);
		}
	}

	imshow("img18", imgs[18]);

	showImages(0, 18, imgs);

	waitKey(0);
	destroyAllWindows();

	return 0;
}