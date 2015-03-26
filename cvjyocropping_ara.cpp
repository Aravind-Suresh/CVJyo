#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;


Point G_clockwiseRef;

float toleranceFractionPointX = 0.2;
float toleranceFractionPointY = 0.2;
int filterThreshDepth = 10;

int h=0,s=0,v=0;


/*vector<Scalar> colors;
colors.push_back(Scalar(255,0,0));
colors.push_back(Scalar(255,255,0));
colors.push_back(Scalar(255,0,255));
colors.push_back(Scalar(255,255,255));
colors.push_back(Scalar(0,0,0));
colors.push_back(Scalar(0,255,0));
colors.push_back(Scalar(0,0,255));
*/

void disp_hist_hsv_2D(Mat hsv)
{
	// Quantize the hue to 30 levels
	// and the saturation to 32 levels
	int hbins = 30, sbins = 32;
	int histSize[] = {hbins, sbins};
	// hue varies from 0 to 179, see cvtColor
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 256 };
	const float * ranges[] = { hranges, sranges };
	MatND hist;
	// we compute the histogram from the 0-th and 1-st channels
	int channels[] = {0, 1};

	calcHist( &hsv, 1, channels, Mat(), // do not use mask
	hist, 2, histSize, ranges,
	true, // the histogram is uniform
	false );

	double maxVal=0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	int scale = 10;
	Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
	for( int h = 0; h < hbins; h++ )
	for( int s = 0; s < sbins; s++ )
	{
		float binVal = hist.at<float>(h, s);
		int intensity = cvRound(binVal*255/maxVal);
		rectangle( histImg, Point(h*scale, s*scale),Point( (h+1)*scale - 1, (s+1)*scale - 1),Scalar::all(intensity),CV_FILLED );
	}

	namedWindow( "H-S Histogram", 1 );
	imshow( "H-S Histogram", histImg );
}


void disp_hist_hsv_1D(Mat src)
{
/// Separate the image in 3 places ( B, G and R )
  vector<Mat> hsv_planes;
  split( src, hsv_planes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat h_hist, s_hist, v_hist;

  /// Compute the histograms:
  calcHist( &hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate );

  // Draw the histograms for h, s and v
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  /// Normalize the result to [ 0, histImage.rows ]
  normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  /// Draw for each channel
  for( int i = 1; i < histSize; i++ )
  {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(h_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(h_hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(s_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(s_hist.at<float>(i)) ),
                       Scalar( 0, 255, 0), 2, 8, 0  );
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(v_hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(v_hist.at<float>(i)) ),
                       Scalar( 0, 0, 255), 2, 8, 0  );
  }


  double min_value,max_value;
  int minIdx,maxIdx;
/*
	cvGetMinMaxHistValue( h_hist, &min_value, &max_value);
	cout<<"Minimum hue value  "<<min_value<<"   Maximum hue value  "<<max_value;
	cvGetMinMaxHistValue( s_hist, min_value, max_value);
	cout<<"Minimum saturation value  "<<min_value<<"   Maximum saturation value  "<<max_value;

*/
	minMaxLoc(h_hist,&min_value,&max_value,&minIdx,&maxIdx);
	cout<<"Minimum hue value  "<<minIdx<<"   Maximum hue value  "<<maxIdx;
	minMaxLoc(s_hist,&min_value,&max_value, &minIdx,&maxIdx);
	cout<<"\nMinimum saturation value  "<<min_value/2<<"   Maximum saturation value  "<<max_value/2;
	minMaxLoc(v_hist,&min_value,&max_value, 0, 0);
	cout<<"\nMinimum value value  "<<min_value/2<<"   Maximum value value  "<<max_value/2;



  /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );
}

void hsv2gray(Mat skin_mask,Mat* gray)
{
for(int i=0;i<skin_mask.rows;i++)
	{	for(int j=0;j<skin_mask.cols;j++)
		{	Vec3b hsv=skin_mask.at<Vec3b>(i,j);
			if(hsv.val[2]==0)
				(*gray).at<uchar>(i,j)=0;
		}
	}
}
/*
void skinmask(int,void*)
{
	inRange(*userdata,Scalar(0,30,60,0),Scalar(20,100,255,0), img_skinmask_hsv);
}*/

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


void cropHand(Mat src, Mat* result, int max_x, int min_x, int max_y, int min_y)
{cout<<"cropHand called";
for (int i=min_y;i<max_y;i++)
	for(int j=min_x;j<max_x;j++)
		(*result).at<uchar>(i-min_y,j-min_x)=src.at<uchar>(i,j);

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
	Mat img_hsv(img_gray.rows, img_gray.cols, CV_8UC3, Scalar::all(0));
	cvtColor(img_rgb,img_hsv,CV_RGB2HSV);
	namedWindow("win1", CV_WINDOW_AUTOSIZE);
	imshow("win1", img_hsv);
	Mat img_skinmask(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_skinmask_hsv(img_gray.rows, img_gray.cols, CV_32F,Scalar::all(0));
	Mat img_skinmask_hsv2rgb(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_skinmask_hsv2rgb2gray(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_dil(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_morph1_dil_temp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_bit_and_2(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat saturated(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
	Mat img_gray_sharp(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	Mat hsvchannels[3];
	split(img_hsv,hsvchannels);

	
	Mat hue_img( img_gray.rows, img_gray.cols, CV_8UC3 );
	Mat sat_img( img_gray.rows, img_gray.cols, CV_8UC3 );
	Mat value_img( img_gray.rows, img_gray.cols, CV_8UC3 );
	// forming an array of matrices is a quite efficient operation,
	// because the matrix data is not copied, only the headers
	Mat out[] = { hue_img, sat_img,value_img};
	int from_to[] = { 0,0, 1,1, 2,2,};
	mixChannels( &img_hsv, 1, out, 3, from_to, 3);
/*
	namedWindow("hue_img",WINDOW_AUTOSIZE);
	imshow("hue_img",hue_img);
	namedWindow("sat_img",WINDOW_AUTOSIZE);
	imshow("sat_img",sat_img);
	namedWindow("value_img",WINDOW_AUTOSIZE);
	imshow("value_img",value_img);
*/

	disp_hist_hsv_2D(img_hsv);
	disp_hist_hsv_1D(img_hsv);


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

		double saturation = 10;
		double scale = 1;

// what it does here is dst = (uchar) ((double)src*scale+saturation); 
		img_gray.convertTo(saturated, CV_8UC1, scale, saturation); 
		namedWindow("img_gray_sat", WINDOW_AUTOSIZE);
		imshow("img_gray_sat", img_gray_sharp);

		img_gray.copyTo(img_skinmask_hsv2rgb2gray);

//H-0 and 50, in the channel S from 0.23 to 0.68
	//inRange(img_hsv,Scalar(0,0.23,100),Scalar(50,0.68,200), img_skinmask_hsv);


	inRange(img_hsv,Scalar(390,390,0),Scalar(410,410,255), img_skinmask_hsv);

	//hsv2gray(img_skinmask_hsv,&img_skinmask_hsv2rgb2gray);
	//cvtColor(imgw_skinmask_hsv,img_skinmask_hsv2rgb,CV_HSV2RGB);
	// cvtColor(img_skinmask_hsv2rgb,img_skinmask_hsv2rgb2gray,CV_RGB2GRAY);


	inRange(img_gray, 110,255, img_skinmask);				//TODO : Make it adaptive
	for(int x=0;x<img_skinmask.cols;x++) {
		for(int y=0;y<10;y++) {
			img_skinmask.at<uchar>(y,x) = 0;
		}
	}

	bitwise_and(img_gray, img_skinmask,img_gray_bit_and);
	namedWindow("img_gray_bit_and", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and",img_gray_bit_and);

	

	bitwise_and(img_gray, img_skinmask_hsv2rgb2gray,img_gray_bit_and_2);
	namedWindow("img_gray_bit_and_2", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_2",img_gray_bit_and_2);

	namedWindow("img_skinmask_hsv", WINDOW_AUTOSIZE);
	imshow("img_skinmask_hsv",img_skinmask_hsv);

	//createTrackbar( "hsv","img_skinmask_hsv",&h,255,&hsv,userdata=&img_skinmask_hsv );


	//Some morphological operations -- dilate on(binary+otsu on img_gray_bit_and)

	threshold(img_gray_bit_and, img_gray_bit_and_morph1, 0, 255, THRESH_BINARY + THRESH_OTSU);
	//ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C
	//adaptiveThreshold(img_gray_bit_and, img_gray_bit_and_morph1, 255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 5,0);



	Mat element = getStructuringElement( dilation_type[0],
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );
	dilate( img_gray_bit_and_morph1, img_gray_bit_and_morph1, element );
	dilate( img_gray_bit_and_morph1, img_gray_bit_and_morph1_dil, element );

	namedWindow("img_gray_bit_and_morph1", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1",img_gray_bit_and_morph1);
	
	//finding the largest contour in the dilated image and cropping the source image

	vector<Vec4i> hierarchy_minMax;
	vector<vector<Point> > contours_minMax;
	img_gray_bit_and_morph1_dil.copyTo(img_gray_bit_and_morph1_dil_temp);	
	findContours(img_gray_bit_and_morph1_dil_temp, contours_minMax, hierarchy_minMax, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	sort(contours_minMax.begin(),contours_minMax.end(),comparatorContourAreas);
    // cout<<"\ncontours_minMax size"<<contours_minMax.size();
	drawContours(img_gray_bit_and_morph1_dil,contours_minMax,contours_minMax.size()-1,Scalar(125,125,125),3,8,hierarchy_minMax);

	Rect bRect(boundingRect(contours_minMax[contours_minMax.size()-1]));
	cout<<"\nbounding rectangle "<<bRect.x<<" "<<bRect.y<<" "<<bRect.width<<" "<<bRect.height;


	Mat img_cropped(bRect.height,bRect.width, CV_8UC1, Scalar::all(0));

	cropHand(img_gray,&img_cropped, bRect.x+bRect.width, bRect.x, bRect.y+bRect.height,bRect.y);

	//resize(img_gray,img_cropped, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )

	namedWindow("img_cropped", WINDOW_AUTOSIZE);
	imshow("img_cropped",img_cropped);
	//

	
	namedWindow("img_gray_bit_and_morph1_dil", WINDOW_AUTOSIZE);
	imshow("img_gray_bit_and_morph1_dil",img_gray_bit_and_morph1_dil);


	Mat img_cropped_edge(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_contours(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_edge_inv(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_skinmask_c(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c_morph1_c(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c_morph1_c_bit_and(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c_morph1_c_bit_and_inv(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c_morph1_c_bit_and_inv_temp(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c_morph1_c_bit_andbit_and_inv_open(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_temp(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_sharp(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_hull_black(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_hull_3(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_hull(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_temp3(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_4(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_5(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_3_bin_inv(img_cropped.rows,img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c_morph1_c_dil(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_bit_and_c_morph1_c_dil_temp(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_cropped_temp3_open(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));



	Mat img_defects_1(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_2(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	Mat img_defects_3_bin(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
//Mat saturated(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
	img_cropped.copyTo(img_cropped_temp);
	img_cropped.copyTo(img_cropped_temp3);
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
		
		CannyThreshold(img_cropped, img_cropped_edge, lowThreshold, lowThreshold*ratio, kernel_size);
		namedWindow("img_cropped_edge", WINDOW_AUTOSIZE);
		imshow("img_cropped_edge",img_cropped_edge);

		bitwise_not(img_cropped_edge, img_cropped_edge_inv);
		namedWindow("img_cropped_edge_inv", WINDOW_AUTOSIZE);
		imshow("img_cropped_edge_inv",img_cropped_edge_inv);


/*// what it does here is dst = (uchar) ((double)src*scale+saturation); 
		img_cropped.convertTo(saturated, CV_8UC1, scale, saturation); 
		namedWindow("img_cropped_sat", WINDOW_AUTOSIZE);
		imshow("img_cropped_sat", img_cropped_sharp);*/

	inRange(img_cropped, 110, 255, img_skinmask_c);				//TODO : Make it adaptive
	for(int x=0;x<img_skinmask_c.cols;x++) {
		for(int y=0;y<10;y++) {
			img_skinmask_c.at<uchar>(y,x) = 0;
		}
	}

	bitwise_and(img_cropped, img_cropped, img_cropped_bit_and_c, img_skinmask_c);
	namedWindow("img_cropped_bit_and_c", WINDOW_AUTOSIZE);
	imshow("img_cropped_bit_and_c",img_cropped_bit_and_c);

	//Some morphological operations -- dilate on(binary+otsu on img_cropped_bit_and_c)

	threshold(img_cropped_bit_and_c, img_cropped_bit_and_c_morph1_c, 0, 255, THRESH_BINARY + THRESH_OTSU);
	//adaptiveThreshold(img_cropped_bit_and_c, img_cropped_bit_and_c_morph1_c, 255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,5,0);



	dilate( img_cropped_bit_and_c_morph1_c, img_cropped_bit_and_c_morph1_c, element );
	dilate( img_cropped_bit_and_c_morph1_c, img_cropped_bit_and_c_morph1_c_dil, element );

	namedWindow("img_cropped_bit_and_c_morph1_c", WINDOW_AUTOSIZE);
	imshow("img_cropped_bit_and_c_morph1_c",img_cropped_bit_and_c_morph1_c);

	
	namedWindow("img_cropped_bit_and_c_morph1_c_dil", WINDOW_AUTOSIZE);
	imshow("img_cropped_bit_and_c_morph1_c_dil",img_cropped_bit_and_c_morph1_c_dil);


	bitwise_and(img_cropped_bit_and_c_morph1_c, img_cropped_edge_inv, img_cropped_bit_and_c_morph1_c_bit_and);
	namedWindow("img_cropped_bit_and_c_morph1_c_bit_and", WINDOW_AUTOSIZE);
	imshow("img_cropped_bit_and_c_morph1_c_bit_and",img_cropped_bit_and_c_morph1_c_bit_and);

	bitwise_not(img_cropped_bit_and_c_morph1_c_bit_and, img_cropped_bit_and_c_morph1_c_bit_and_inv);

	img_cropped_bit_and_c_morph1_c_bit_and_inv.copyTo(img_cropped_bit_and_c_morph1_c_bit_and_inv_temp);
	namedWindow("img_cropped_bit_and_c_morph1_c_bit_and_inv_temp", WINDOW_AUTOSIZE);
	imshow("img_cropped_bit_and_c_morph1_c_bit_and_inv_temp",img_cropped_bit_and_c_morph1_c_bit_and_inv_temp);

	findContours(img_cropped_bit_and_c_morph1_c_bit_and_inv, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	Scalar color( 255, 0, 0 );
	/*int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( img_cropped_bit_and_c_morph1_c_bit_and_inv, contours, idx, color, CV_FILLED, 8, hierarchy );
    }*/

        /*drawContours( img_cropped_bit_and_c_morph1_c_bit_and_inv, contours, 3, color, CV_FILLED, 8, hierarchy );
        namedWindow("img_cropped_bit_and_c_morph1_c_bit_and_inv", WINDOW_AUTOSIZE);
        imshow("img_cropped_bit_and_c_morph1_c_bit_and_inv",img_cropped_bit_and_c_morph1_c_bit_and_inv);*/

 /*       for(int i = 2; i < contours.size(); i++) {
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
        }*/

        	vector<Vec4i> convexityDefectsSet;
        	sort(contours.begin(),contours.end(),comparatorContourAreas);
        	int size1 = contours.size();


        	vector<vector<int> >hulls( contours.size() );

        	convexHull(Mat(contours[size1-2]), hulls[1],false);
        //cout<<"PODA MOKKA NAAYE"<<endl;
        //Mat drawing(img_cropped.rows, img_cropped.cols, CV_8UC1, Scalar::all(0));
        	Mat drawing = Mat::zeros( img_cropped.size(), CV_8UC3 );
        	cout<<hulls[1].size()<<endl;
        /*drawContours( drawing, hulls, -1, color, 1, 8);
        namedWindow( "Hull demo", WINDOW_AUTOSIZE );
        imshow( "Hull demo", drawing );
*/
        for(int kk=0;kk<hulls[1].size();kk++) {
        	cout<<hulls[1][kk]<<endl;
        	
        }

       /* for(int id=0;id<5;id++) {
        	drawContours(img_cropped,contours,contours.size()-id-1, Scalar(50*id,50*id,50*id),CV_FILLED, 8);
        	cout<<"contour drawn "<<id<<endl;
        }*/
        	cout<<"\nHERE SIZE"<<contours.size()<<endl;
        	//for(int jk=contours.size()-3;jk>0;jk--)
        	{
        		Scalar color( 0,0,0 );
        		drawContours(img_cropped,contours,contours.size()-2, color,2, 8, hierarchy);

        	}
        	
        	namedWindow("img_cropped_contour",WINDOW_AUTOSIZE);
        	imshow("img_cropped_contour",img_cropped);

	/*for(int k=0;k<contours[pos].size();k++) {
		cout<<hull[k]<<endl;

	}*/

		img_cropped.copyTo(img_hull);

		cout<<"\nprint hull points ";
		for(int kk=0;kk<hulls[1].size();kk++)
		{
			Point center(contours[size1-2][hulls[1][kk]]);
			int radius =3;
			circle( img_hull, center, radius, Scalar(255,255,255), 3, 8, 0 );
			cout<<kk<<" ("<<center.x<<","<<center.y<<")"<<endl;
		}

		namedWindow("img_hull",WINDOW_AUTOSIZE);
		imshow("img_hull",img_hull);


		img_cropped.copyTo(img_hull_black);
		img_cropped.copyTo(img_hull_3);
		img_cropped.copyTo(img_defects_1);
		//img_cropped.copyTo(img_defects_2);


		convexityDefects(contours[size1-2], hulls[1], convexityDefectsSet);

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

			Point p = contours[size1-2][defectPtIdx];
			if(/*k>=filterThreshDepth && */checkPointInRegion(img_cropped, toleranceFractionPointX, toleranceFractionPointY, p)) {
				//cout<<"Defect point : "<<k<<" ("<<p.x<<","<<p.y<<")";
				defectsPoints.push_back(p);
				color = Scalar(255,255,255);
			}


			circle(img_cropped_temp, contours[size1-2][defectPtIdx] , 10, color, 2, 8, 0 );
			circle(img_hull_black, contours[size1-2][startIdx] , 10, color, 2, 8, 0 );
			circle(img_hull_3, contours[size1-2][endIdx] , 10, color, 2, 8, 0 );

			
		}
		namedWindow("img_hull_defect",WINDOW_AUTOSIZE);
		imshow("img_hull_defect", img_cropped_temp);
		namedWindow("img_hull_start", WINDOW_AUTOSIZE);
		imshow("img_hull_start",img_hull_black);
		namedWindow("img_hull_end", WINDOW_AUTOSIZE);
		imshow("img_hull_end",img_hull_3);

		Point2f minEncCirCenter;

		float minEncRad;
		cout<<"defects points size : "<<defectsPoints.size()<<endl;

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

		for(int k1=0;k1<defectsPoints.size();k1++) 
		{
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

	//morphologyEx(img_cropped_bit_and_c_morph1_c_bit_and_inv, img_cropped_bit_and_c_morph1_c_bit_and_inv_open, MORPH_OPEN, kernelOpen, Point(-1,-1), 1, BORDER_CONSTANT);
	//namedWindow("img_cropped_bit_and_c_morph1_c_bit_and_inv_open", WINDOW_AUTOSIZE);
	//imshow("img_cropped_bit_and_c_morph1_c_bit_and_inv_open",img_cropped_bit_and_c_morph1_c_bit_and_inv_open);


		add(img_cropped_edge, img_cropped, img_cropped_sharp, noArray(), -1);
		namedWindow("img_cropped_sharp", WINDOW_AUTOSIZE);
		imshow("img_cropped_sharp", img_cropped_sharp);


		binaryAbsDiff(img_cropped_bit_and_c_morph1_c,img_defects_2,img_defects_3_bin);
		
		Mat element1 = getStructuringElement( dilation_type[0],
			Size( 2*dilation_size + 1, 2*dilation_size+1 ),
			Point( dilation_size, dilation_size ) );
		dilate( img_defects_3_bin, img_defects_3_bin, element );

		bitwise_not(img_defects_3_bin, img_defects_3_bin_inv);
		bitwise_and(img_defects_3_bin_inv, img_defects_2, img_defects_4);
		//bitwise_and(img_defects_4, img_cropped_temp3, img_cropped_temp3);
		bitwise_and(img_defects_4, img_cropped_bit_and_c_morph1_c_bit_and, img_cropped_temp3);
		/*erode( img_cropped_temp3, img_cropped_temp3 , element );
		dilate( img_cropped_temp3, img_cropped_temp3 , element );*/
		//morphologyEx(img_cropped_temp3, img_cropped_temp3_open, OPEN, kernelOpen, Point(-1,-1), 1, BORDER_CONSTANT);
		//Laplacian( const oclMat& src, oclMat& dst, int ddepth, int ksize=1, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )


		namedWindow("img_defects_3_bin",WINDOW_AUTOSIZE);
		imshow("img_defects_3_bin",img_defects_3_bin);

		namedWindow("final-filter-1", WINDOW_AUTOSIZE);
		imshow("final-filter-1", img_cropped_temp3);

		namedWindow("final-filter-1_open", WINDOW_AUTOSIZE);
		imshow("final-filter-1_open", img_cropped_temp3_open);

/*      
		GaussianBlur( img_cropped_temp, img_cropped_temp, Size(5, 5), 2, 1000 );
		vector<Vec3f> circles;
		for( size_t i = 0; i < circles.size(); i++ )
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
      // circle center
			circle( img_cropped_temp, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
			circle( img_cropped_temp, center, radius, Scalar(0,0,255), 3, 8, 0 );
		}
		namedWindow("img_cropped_circles",WINDOW_AUTOSIZE);
		imshow("img_cropped_circles",img_cropped_temp);

*/
		//createTrackbar("filterThreshDepth","",&filterThreshDepth,);


		int lap_kernel_size = 5;
		int lap_scale = 1;
		int lap_delta = 0;
		int lap_ddepth = CV_8UC1;

		//GaussianBlur( img_cropped_temp3, img_cropped_temp3, Size(5, 5), 2, 1000 );

		Laplacian(img_cropped_temp3, img_cropped_temp3, lap_ddepth, lap_kernel_size, lap_scale, lap_delta, BORDER_DEFAULT);

		GaussianBlur( img_cropped_temp3, img_cropped_temp3, Size(5, 5), 2, 100 );

		//findContours(img_cropped_temp3, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		//drawContours(img_cropped_temp3,contours,-1,Scalar(125,125,125),3,8,hierarchy);

		namedWindow("final-filter-1-lap", WINDOW_AUTOSIZE);
		imshow("final-filter-1-lap", img_cropped_temp3);

		
		waitKey(0);
/*
		destroyWindow("img_cropped");
		destroyWindow("img_cropped_edge");
		destroyWindow("img_cropped_edge_inv");
		destroyWindow("img_cropped_bit_and_c");
		destroyWindow("img_cropped_bit_and_c_morph1_c");
		destroyWindow("img_cropped_bit_and_c_morph1_c_dil");
		destroyWindow("img_cropped_bit_and_c_morph1_c_bit_and");
		destroyWindow("img_cropped_contour");
		//destroyWindow("img_cropped_circles");
		destroyWindow("img_cropped_sharp");
		destroyWindow("Hull demo");
		destroyWindow("img_cropped_sat");
		destroyWindow("img_hull_start");
		destroyWindow("img_hull_end");
		destroyWindow("img_hull");
		destroyWindow("img_hull_defect");
		destroyWindow("img_defects_1");

		destroyWindow("img_defects_2");
		destroyWindow("img_defects_3_bin");
		destroyWindow("final-filter-1");
		destroyWindow("img_cropped");*/
		destroyAllWindows();
	//destroyWindow("img_cropped_bit_and_c_morph1_c_bit_and_inv");
	//destroyWindow("img_cropped_bit_and_c_morph1_c_bit_and_inv_open");

		return 0;
	}
