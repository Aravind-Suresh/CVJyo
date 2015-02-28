#include<opencv2/opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

    Mat img_gray,
img_gray_edge,
img_gray_contours,
img_gray_edge_inv,
img_skinmask,
img_gray_bit_and,
img_gray_bit_and_morph1,
img_gray_bit_and_morph1_bit_and,
img_gray_bit_and_morph1_bit_and_inv,
img_gray_bit_and_morph1_bit_and_inv_temp,
img_gray_bit_and_morph1_bit_and_inv_open;

    
    vector<vector<Point> > contours1;
    vector<vector<Point> > contours2;

    vector<int> arcl;
    int pos,a;
    vector<Vec4i> hierarchy;
    vector<int> row(5,1);
    vector<vector<int> > kernelOpen(5,row);


        int dilation_type[] = {0,1,2};
        int dilation_size = 2;

        int edgeThresh = 1;
        int lowThreshold = 30;
        int const max_lowThreshold = 100;
        int ratio = 1;
        int kernel_size = 3;
        int upperThreshold=30;
        int value = 0;

void CannyThreshold(int,void*)
    {
    
    Canny( img_gray, img_gray_edge, lowThreshold, upperThreshold, kernel_size );
    imshow("img_gray_edge",img_gray_edge);
    }

void value_img_gray(int,void*)
{
    if (value) 
    cout<<"Poda mokka naaye";
else
    cout<<"Poda  loosu naaye";
}

int main(int argc, char** argv) {
   
    /*
        dilation_type

        MORPH_RECT = 0,
        MORPH_CROSS = 1,
        MORPH_ELLIPSE = 2
    */
        img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        if(!img_gray.data) {
            cout<<"Error opening the file";
            return -1;
        }
    img_gray_edge=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_contours=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_edge_inv=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_skinmask=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_bit_and=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_bit_and_morph1=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_bit_and_morph1_bit_and=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_bit_and_morph1_bit_and_inv=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_bit_and_morph1_bit_and_inv_temp=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));
    img_gray_bit_and_morph1_bit_and_inv_open=Mat(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

        

        namedWindow("img_gray",WINDOW_AUTOSIZE);
        imshow("img_gray",img_gray);
        createTrackbar("Value","img_gray",&value,255,value_img_gray);

        blur( img_gray, img_gray_edge, Size(3,3) );
        namedWindow("img_gray_edge", WINDOW_AUTOSIZE);

        
        createTrackbar("lowerThreshold","img_gray_edge",&lowThreshold,250,CannyThreshold);
        createTrackbar("upperThreshold","img_gray_edge",&upperThreshold,250,CannyThreshold);


        CannyThreshold(max_lowThreshold,0);
        

        bitwise_not(img_gray_edge, img_gray_edge_inv);
        namedWindow("img_gray_edge_inv", WINDOW_AUTOSIZE);
        imshow("img_gray_edge_inv",img_gray_edge_inv);

    inRange(img_gray, 110, 255, img_skinmask);              //TODO : Make it adaptive
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
    namedWindow("img_gray_bit_and_morph1", WINDOW_AUTOSIZE);
    imshow("img_gray_bit_and_morph1",img_gray_bit_and_morph1);

    bitwise_and(img_gray_bit_and_morph1, img_gray_edge_inv, img_gray_bit_and_morph1_bit_and);
    namedWindow("img_gray_bit_and_morph1_bit_and", WINDOW_AUTOSIZE);
    imshow("img_gray_bit_and_morph1_bit_and",img_gray_bit_and_morph1_bit_and);

    bitwise_not(img_gray_bit_and_morph1_bit_and, img_gray_bit_and_morph1_bit_and_inv);

    img_gray_bit_and_morph1_bit_and_inv.copyTo(img_gray_bit_and_morph1_bit_and_inv_temp);
    namedWindow("img_gray_bit_and_morph1_bit_and_inv_temp", WINDOW_AUTOSIZE);
    imshow("img_gray_bit_and_morph1_bit_and_inv_temp",img_gray_bit_and_morph1_bit_and_inv_temp);

    findContours(img_gray_bit_and_morph1_bit_and_inv, contours1, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
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
        //findContours(img_gray, contours2, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        drawContours(img_gray_bit_and,contours1,contours1.size()-2,color,CV_FILLED,8,hierarchy);

/*
        for(int i = 0; i < contours1.size(); i++) {
            //for(int j = 0; j < contours[i].size(); j++)
                //cout<<contours[i][j]<<endl;
            //cout<<endl;
            arcl.push_back(arcLength(contours1[i], true));
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
        vector<int> hull(contours1[pos].size());
        vector<Vec4i> convexityDefectsSet;
        convexHull(contours1[pos], hull, false, false );
        cout<<contours1.size();
        drawContours(img_gray,contours,-1, Scalar(255,0,0),CV_FILLED, 8);
        namedWindow("img_gray_contour",WINDOW_AUTOSIZE);
        imshow("img_gray_contour",img_gray);
*/
    /*for(int k=0;k<contours[pos].size();k++) {
        cout<<hull[k]<<endl;
    }*/
    //drawContours( drawing, hull, s, Scalar(0,255,255), 1, 8, vector<Vec4i>(), 0, Point() );
       /*  
      convexityDefects(contours[pos], hull, convexityDefectsSet);
        for(int k=0;k<convexityDefectsSet.size();k++) {
            int startIdx = convexityDefectsSet[k].val[0];
            int endIdx = convexityDefectsSet[k].val[1];
            int defectPtIdx = convexityDefectsSet[k].val[2];
            double depth = static_cast<double>(convexityDefectsSet[k].val[3]) / 256.0;

            cout << startIdx << ' ' << endIdx << ' ' << defectPtIdx << ' ' << depth << endl;

            Scalar color = Scalar( 255,0,0 );
            circle(img_gray_bit_and_morph1, contours[pos][startIdx] , 10, color, 2, 8, 0 );
            namedWindow("img_hull", WINDOW_AUTOSIZE);
            imshow("img_hull",img_gray_bit_and_morph1);
        }
        */

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
        destroyWindow("img_gray_contour");
    //destroyWindow("img_gray_bit_and_morph1_bit_and_inv");
    //destroyWindow("img_gray_bit_and_morph1_bit_and_inv_open");

            return 0;
        }
