

#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;


int main(int argc, char ** argv)
{
VideoCapture cap(0);

if(!cap.isOpened()) return -1;
Mat frame,blur,edges,sharp;

namedWindow("orginal",1);
namedWindow("sharp",1);
frame = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

for(;;)
{
//cap >> frame;

GaussianBlur(frame, blur, cv::Size(0, 0), 3);
addWeighted(frame, 1.5, blur, -0.5, 0, sharp);

imshow("orginal", frame);
imshow("sharp", sharp);

if(waitKey(2)==27) break;

}

return 0;
}