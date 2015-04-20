#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;
float toleranceFractionPointX = 0.2;
float toleranceFractionPointY = 0.2;
int filterThreshDepth = 10;

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

void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)//Blurring and canny of image
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
	Mat img_gray_edge(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_edge_dil(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_edge_temp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));



	int dilation_type[] = {0,1,2};
	int dilation_size = 2;

	int edgeThresh = 1;
		int lowThreshold = 30;
		int const max_lowThreshold = 100;
		int ratio = 1;
		int kernel_size = 3;


	CannyThreshold(img_gray, img_gray_edge, lowThreshold, lowThreshold*ratio, kernel_size);//finding canny edges of the gray image

	threshold(img_gray, img_gray_otsu, 0, 255, THRESH_BINARY + THRESH_OTSU);//finding otsu threshold of gray image

	namedWindow("img_gray_otsu", WINDOW_AUTOSIZE);
	imshow("img_gray_otsu",img_gray_otsu);


	Mat element = getStructuringElement( dilation_type[0],	//getting structuring element for dilating and eroding
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );

	Mat element2 = getStructuringElement( dilation_type[0],
		Size( 7, 7 ),
		Point( 3, 3 ) );

	dilate( img_gray_edge, img_gray_edge_dil, element );  //dilating the edge image of the gray image
	//dilate( img_gray_otsu, img_gray_otsu_dil, element );
	//dilate( img_gray_bit_and_morph1, img_gray_bit_and_morph1_dil, element );

	namedWindow("img_gray_edge_dil", WINDOW_AUTOSIZE);
	imshow("img_gray_edge_dil",img_gray_edge_dil);

	vector<Vec4i> hierarchy_minMax;
	vector<vector<Point> > contours_minMax;
	//img_gray_otsu_dil.copyTo(img_gray_otsu_temp);	
	//findContours(img_gray_otsu_temp, contours_minMax, hierarchy_minMax, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	
	img_gray_edge_dil.copyTo(img_gray_edge_temp);	//copying dilated image to a temporary image to fidnd contours 
	findContours(img_gray_edge_temp, contours_minMax, hierarchy_minMax, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE); //finding contours in edge image
 
    sort(contours_minMax.begin(),contours_minMax.end(),comparatorContourAreas); //sorting the contours to get the maximum area contour i.e the hand
    cout<<"\ncontours_minMax size"<<contours_minMax.size();
	drawContours(img_gray_edge_dil,contours_minMax,contours_minMax.size()-1,Scalar(125,125,125),3,8,hierarchy_minMax); //drawing the maximum area contour

	namedWindow("img_gray_edge_dil", WINDOW_AUTOSIZE);
	imshow("img_gray_edge_dil",img_gray_edge_dil);

	Rect bRect(boundingRect(contours_minMax[contours_minMax.size()-1])); //finding the bounding rectange for the maximum contour(ie the hand)
	




	Mat img_cropped(bRect.height,bRect.width, CV_8UC1, Scalar::all(0)); //cropping the image to the size of the hand alone
	Mat img_cropped_otsu(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_circle(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_skin_circle(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_skin_circle_laplacian(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_prints(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_otsu_dil(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_edge(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_hull(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_otsu_and_c(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_2(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_2_inv(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_3(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_prints_and_otsu(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_prints_inv(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));



	vector<vector<Point> > contours;
	vector<int> arcl;
	int pos,a;
	vector<Vec4i> hierarchy;
	vector<int> row(5,1);
	vector<vector<int> > kernelOpen(5,row);


	cropHand(img_gray,&img_cropped, bRect.x+bRect.width, bRect.x, bRect.y+bRect.height,bRect.y); //storing the cropped hand in an image

	namedWindow("img_cropped", WINDOW_AUTOSIZE);
	imshow("img_cropped",img_cropped);

	threshold(img_cropped, img_cropped_otsu, 0, 255, THRESH_BINARY + THRESH_OTSU); //otsu thresholding the cropped image again
	dilate( img_cropped_otsu, img_cropped_otsu_dil, element );

	bitwise_and(img_cropped,img_cropped_otsu,img_cropped_otsu_and_c); // cropped image AND otsu of cropped image --> img_cropped_otsu_and_c

	CannyThreshold(img_cropped_otsu_and_c, img_cropped_prints, lowThreshold, lowThreshold*ratio, kernel_size);
	
	bitwise_not(img_cropped_prints,img_cropped_prints_inv);
	bitwise_and(img_cropped_prints_inv,img_cropped_otsu,img_cropped_prints_and_otsu);
	//erode(img_cropped_prints, img_cropped_prints,element );
	dilate(img_cropped_prints, img_cropped_prints, element );
	
	
	namedWindow("img_cropped_otsu", WINDOW_AUTOSIZE);
	imshow("img_cropped_otsu",img_cropped_otsu);

	namedWindow("img_cropped_prints", WINDOW_AUTOSIZE);
	imshow("img_cropped_prints",img_cropped_prints);

	findContours(img_cropped_prints, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    vector<Vec4i> convexityDefectsSet;
    sort(contours.begin(),contours.end(),comparatorContourAreas);
    int size1 = contours.size();
   
    cout<<"\n"<<contours.size();
   
    vector<vector<int> >hulls( contours.size() );

	drawContours(img_cropped,contours,size1-1, (0,0,0),2, 8, hierarchy);

    convexHull(Mat(contours[size1-1]), hulls[1],false);

for(int kk=0;kk<hulls[1].size();kk++) 
{        
    cout<<hulls[1][kk]<<endl;
}

cout<<"\nprint hull points ";


img_cropped.copyTo(img_hull);

for(int kk=0;kk<hulls[1].size();kk++)
	{
			Point center(contours[size1-1][hulls[1][kk]]);
			int radius =3;
			circle( img_hull, center, radius, Scalar(255,255,255), 3, 8, 0 );
			cout<<kk<<" ("<<center.x<<","<<center.y<<")"<<endl;
	}


   
convexityDefects(contours[size1-1], hulls[1], convexityDefectsSet);

vector<Point> defectsPoints;

for(int k=0;k<convexityDefectsSet.size();k++)
	{
 			int startIdx = convexityDefectsSet[k].val[0];
			int endIdx = convexityDefectsSet[k].val[1];
			int defectPtIdx = convexityDefectsSet[k].val[2];
			double depth = static_cast<double>(convexityDefectsSet[k].val[3]) / 256.0;

			// cout<<endl<<k<<"  ";

			// cout << startIdx << ' ' << endIdx << ' ' << defectPtIdx << ' ' << depth << endl;

			Scalar color = Scalar( 0,0,0 );

			Point p = contours[size1-1][defectPtIdx];
			if(/*k>=filterThreshDepth && */checkPointInRegion(img_cropped, toleranceFractionPointX, toleranceFractionPointY, p))
			{
				//cout<<"Defect point : "<<k<<" ("<<p.x<<","<<p.y<<")";
				defectsPoints.push_back(p);
				color = Scalar(255,255,255);
			}


			circle(img_hull, contours[size1-1][defectPtIdx] , 10, (0,0,0), 2, 8, 0 );
	
	}
	
	namedWindow("img_hull", WINDOW_AUTOSIZE);
	imshow("img_hull",img_hull);

		Point2f minEncCirCenter;

		float minEncRad;
		cout<<"\ndefects points size : "<<defectsPoints.size()<<endl;

		minEnclosingCircle(defectsPoints, minEncCirCenter, minEncRad);
		circle(img_defects, minEncCirCenter, minEncRad, Scalar(255,255,255), -1, 8, 0);
		cout<<"\ndefects points size : "<<defectsPoints.size()<<endl;
			
binaryAbsDiff(img_cropped_otsu,img_defects,img_defects_2);


    namedWindow("img_defects_2", WINDOW_AUTOSIZE);
	imshow("img_defects_2",img_defects_2);

	bitwise_not(img_defects_2, img_defects_2_inv);
	bitwise_and(img_defects_2_inv, img_defects, img_defects_3);
	bitwise_and(img_defects_3, img_cropped_prints_and_otsu, img_cropped_circle);

	bitwise_and(img_defects_3, img_cropped, img_cropped_skin_circle);

	//dilate( img_cropped_circle, img_cropped_circle, element2 );
	//erode(img_cropped_circle, img_cropped_circle,element2 );

    namedWindow("img_cropped_circle", WINDOW_AUTOSIZE);
	imshow("img_cropped_circle",img_cropped_circle);

	/*Laplacian( img_cropped_skin_circle,img_cropped_skin_circle_laplacian,0,1,1,0,BORDER_DEFAULT );

    namedWindow("img_cropped_skin_circle_laplacian", WINDOW_AUTOSIZE);
	imshow("img_cropped_skin_circle_laplacian",img_cropped_skin_circle_laplacian);
*/
    namedWindow("img_cropped", WINDOW_AUTOSIZE);
	imshow("img_cropped",img_cropped);


waitKey(0);
destroyAllWindows();

}