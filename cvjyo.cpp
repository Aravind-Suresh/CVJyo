#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;


Point G_clockwiseRef;

/*vector<Scalar> colors;
colors.push_back(Scalar(255,0,0));
colors.push_back(Scalar(255,255,0));
colors.push_back(Scalar(255,0,255));
colors.push_back(Scalar(255,255,255));
colors.push_back(Scalar(0,0,0));
colors.push_back(Scalar(0,255,0));
colors.push_back(Scalar(0,0,255));

*/
void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
	blur( src_gray, edge_gray, Size(3,3) );
	//GaussianBlur( src_gray, edge_gray, Size(5,5), 2, 2 );
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

bool comparatorClockwise(Point a,Point b)
{
	if(slopeStrLine(a,G_clockwiseRef)>slopeStrLine(b,G_clockwiseRef))
		return true;
	else return false;

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
    namedWindow("win1", CV_WINDOW_AUTOSIZE);
    imshow("win1", img_hsv);
	Mat img_gray_edge(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_contours(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_edge_inv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_skinmask(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_and(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_and_inv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_and_inv_temp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_bit_andbit_and_inv_open(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
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
	Mat saturated(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	img_gray.copyTo(img_gray_temp);
	img_gray.copyTo(img_gray_temp3);
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
        double saturation = 10;
double scale = 1;

// what it does here is dst = (uchar) ((double)src*scale+saturation); 
img_gray.convertTo(saturated, CV_8UC1, scale, saturation); 
namedWindow("img_gray_sat", WINDOW_AUTOSIZE);
		imshow("img_gray_sat", img_gray_sharp);
	inRange(img_gray, 110, 255, img_skinmask);				//TODO : Make it adaptive
	for(int x=0;x<img_skinmask.cols;x++) {
		for(int y=0;y<10;y++) {
			img_skinmask.at<uchar>(y,x) = 0;
		}
	}

	bitwise_and(img_gray, img_gray, img_gray_bit_and, img_skinmask);
	namedWindow("img_gray_bit_and", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and",img_gray_bit_and);

	//Some morphological operations -- dilate on(binary+otsu on img_gray_bit_and)

	threshold(img_gray_bit_and, img_gray_bit_and_morph1, 0, 255, THRESH_BINARY + THRESH_OTSU);

	Mat element = getStructuringElement( dilation_type[0],
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );
	dilate( img_gray_bit_and_morph1, img_gray_bit_and_morph1, element );
	//dilate( img_gray_bit_and_morph1, img_gray_bit_and_morph1, element );
	namedWindow("img_gray_bit_and_morph1", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1",img_gray_bit_and_morph1);

	bitwise_and(img_gray_bit_and_morph1, img_gray_edge_inv, img_gray_bit_and_morph1_bit_and);
	namedWindow("img_gray_bit_and_morph1_bit_and", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1_bit_and",img_gray_bit_and_morph1_bit_and);

	bitwise_not(img_gray_bit_and_morph1_bit_and, img_gray_bit_and_morph1_bit_and_inv);

	img_gray_bit_and_morph1_bit_and_inv.copyTo(img_gray_bit_and_morph1_bit_and_inv_temp);
	namedWindow("img_gray_bit_and_morph1_bit_and_inv_temp", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1_bit_and_inv_temp",img_gray_bit_and_morph1_bit_and_inv_temp);

	findContours(img_gray_bit_and_morph1_bit_and_inv, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
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

        for(int i = 2; i < contours.size(); i++) {
        	//for(int j = 0; j < contours[i].size(); j++)
        		//cout<<contours[i][j]<<endl;
        	//cout<<endl;
        	arcl.push_back(arcLength(contours[i], true));
        	if(i==0) {
        		a=arcl[0];
        		pos = 0;
        	}
        	if(a<arcl[i]) {
        		a = arcl[i];
        		pos = i;
        	}
        	//cout<<a<<endl<<pos;
        }
        
        vector<Vec4i> convexityDefectsSet;
        sort(contours.begin(),contours.end(),comparatorContourAreas);
        int size1 = contours.size();
        
        
        vector<vector<int> >hulls( contours.size() );

        convexHull(Mat(contours[size1-2]), hulls[1],false);
        cout<<"PODA MOKKA NAAYE"<<endl;
        //Mat drawing(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
        Mat drawing = Mat::zeros( img_gray.size(), CV_8UC3 );
        cout<<hulls[1].size()<<endl;
        /*drawContours( drawing, hulls, -1, color, 1, 8);
        namedWindow( "Hull demo", WINDOW_AUTOSIZE );
        imshow( "Hull demo", drawing );
*/
        for(int kk=0;kk<hulls[1].size();kk++) {
        	cout<<hulls[1][kk]<<endl;
        	
        }

       /* for(int id=0;id<5;id++) {
        	drawContours(img_gray,contours,contours.size()-id-1, Scalar(50*id,50*id,50*id),CV_FILLED, 8);
        	cout<<"contour drawn "<<id<<endl;
        }*/
        	cout<<"\nHERE SIZE"<<contours.size()<<endl;
        	//for(int jk=contours.size()-3;jk>0;jk--)
        	{
        		Scalar color( 0,0,0 );
        		drawContours(img_gray,contours,contours.size()-2, color,2, 8, hierarchy);

        	}
        	
        	namedWindow("img_gray_contour",WINDOW_AUTOSIZE);
        	imshow("img_gray_contour",img_gray);

	/*for(int k=0;k<contours[pos].size();k++) {
		cout<<hull[k]<<endl;

	}*/

		img_gray.copyTo(img_hull);

		cout<<"print hull points ";
		for(int kk=0;kk<hulls[1].size();kk++)
		{
			Point center(contours[size1-2][hulls[1][kk]]);
			int radius =3;
			circle( img_hull, center, radius, Scalar(255,255,255), 3, 8, 0 );
			cout<<kk<<" ("<<center.x<<","<<center.y<<")"<<endl;
		}

		namedWindow("img_hull",WINDOW_AUTOSIZE);
		imshow("img_hull",img_hull);


		img_gray.copyTo(img_hull_black);
		img_gray.copyTo(img_hull_3);
		img_gray.copyTo(img_defects_1);
		//img_gray.copyTo(img_defects_2);


		convexityDefects(contours[size1-2], hulls[1], convexityDefectsSet);

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

			Point p = contours[size1-2][defectPtIdx];
			if(k>=filterThreshDepth && checkPointInRegion(img_gray, toleranceFractionPointX, toleranceFractionPointY, p)) {
				//cout<<"Defect point : "<<k<<" ("<<p.x<<","<<p.y<<")";
				defectsPoints.push_back(p);
				color = Scalar(255,255,255);
			}


			circle(img_gray_temp, contours[size1-2][defectPtIdx] , 10, color, 2, 8, 0 );
			circle(img_hull_black, contours[size1-2][startIdx] , 10, color, 2, 8, 0 );
			circle(img_hull_3, contours[size1-2][endIdx] , 10, color, 2, 8, 0 );

			
		}
		namedWindow("img_hull_defect",WINDOW_AUTOSIZE);
		imshow("img_hull_defect", img_gray_temp);
		namedWindow("img_hull_start", WINDOW_AUTOSIZE);
		imshow("img_hull_start",img_hull_black);
		namedWindow("img_hull_end", WINDOW_AUTOSIZE);
		imshow("img_hull_end",img_hull_3);

		Point2f minEncCirCenter;

		float minEncRad;

		minEnclosingCircle(defectsPoints, minEncCirCenter, minEncRad);
		circle(img_defects_2, minEncCirCenter, minEncRad, Scalar(255,255,255), -1, 8, 0);
		cout<<"defects points size : "<<defectsPoints.size()<<endl;
		
		for(int k2=0;k2<defectsPoints.size();k2++) {
			if(defectsPoints[k2].x<defectsPoints[minposx].x) {
				minposx = k2;
			}
		}

		namedWindow("img_defects_2", WINDOW_AUTOSIZE);
		imshow("img_defects_2", img_defects_2);

		G_clockwiseRef=defectsPoints[minposx];
		cout<<"Defect point ref : "<<" ("<<G_clockwiseRef.x<<","<<G_clockwiseRef.y<<")"<<endl;

		sort(defectsPoints.begin(),defectsPoints.end(),comparatorClockwise);

		fillConvexPoly(img_defects_1, &defectsPoints[0], defectsPoints.size(), Scalar(255,255,255), 8);

		for(int k1=0;k1<defectsPoints.size();k1++) {
			Point p1 = defectsPoints[k1];
			cout<<"Defect point : "<<k1<<" ("<<p1.x<<","<<p1.y<<")"<<endl;
			cout<<"Slope : "<<slopeStrLine(p1,G_clockwiseRef);
			line(img_defects_1, G_clockwiseRef, p1, Scalar(0,0,0), 1, 8, 0);
		}

		/*vector<vector<Point> > defectsPointsArray(1);
		defectsPointsArray[0]=defectsPoints;

		fillPoly(img_defects_1, defectsPointsArray, defectsPoints.size(), Scalar(255,255,255), 8);
*/
		namedWindow("img_defects_1", WINDOW_AUTOSIZE);
		imshow("img_defects_1", img_defects_1);

	//morphologyEx(img_gray_bit_and_morph1_bit_and_inv, img_gray_bit_and_morph1_bit_and_inv_open, MORPH_OPEN, kernelOpen, Point(-1,-1), 1, BORDER_CONSTANT);
	//namedWindow("img_gray_bit_and_morph1_bit_and_inv_open", WINDOW_AUTOSIZE);
	//imshow("img_gray_bit_and_morph1_bit_and_inv_open",img_gray_bit_and_morph1_bit_and_inv_open);


		add(img_gray_edge, img_gray, img_gray_sharp, noArray(), -1);
		namedWindow("img_gray_sharp", WINDOW_AUTOSIZE);
		imshow("img_gray_sharp", img_gray_sharp);
        binaryAbsDiff(img_gray_bit_and_morph1,img_defects_2,img_defects_3_bin);
		
		bitwise_not(img_defects_3_bin, img_defects_3_bin_inv);
		bitwise_and(img_defects_3_bin_inv, img_defects_2, img_defects_4);
		bitwise_and(img_defects_4, img_gray_temp3, img_gray_temp3);

		namedWindow("img_defects_3_bin",WINDOW_AUTOSIZE);
		imshow("img_defects_3_bin",img_defects_3_bin_inv);

		namedWindow("final-filter-1", WINDOW_AUTOSIZE);
		imshow("final-filter-1", img_gray_temp3);

/*      
		GaussianBlur( img_gray_temp, img_gray_temp, Size(5, 5), 2, 1000 );
		vector<Vec3f> circles;
		for( size_t i = 0; i < circles.size(); i++ )
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
      // circle center
			circle( img_gray_temp, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
			circle( img_gray_temp, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}
		namedWindow("img_gray_circles",WINDOW_AUTOSIZE);
		imshow("img_gray_circles",img_gray_temp);

*/
		
		waitKey(0);

		destroyWindow("img_gray");
		destroyWindow("img_gray_edge");
		destroyWindow("img_gray_edge_inv");
		destroyWindow("img_gray_bit_and");
		destroyWindow("img_gray_bit_and_morph1");
		destroyWindow("img_gray_bit_and_morph1_bit_and");
		destroyWindow("img_gray_contour");
		//destroyWindow("img_gray_circles");
		destroyWindow("img_gray_sharp");
		destroyWindow("Hull demo");
		destroyWindow("img_gray_sat");
		destroyWindow("img_hull_start");
		destroyWindow("img_hull_end");
		destroyWindow("img_hull");
		destroyWindow("img_hull_defect");
		destroyWindow("img_defects_1");

		destroyWindow("img_defects_2");
		destroyWindow("img_defects_3_bin");
		destroyWindow("final-filter-1");
	//destroyWindow("img_gray_bit_and_morph1_bit_and_inv");
	//destroyWindow("img_gray_bit_and_morph1_bit_and_inv_open");

		return 0;
	}
