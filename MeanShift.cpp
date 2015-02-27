
// header files
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

// library files
/*
#pragma comment(lib, "opencv_core249d.lib")
#pragma comment(lib, "opencv_highgui249d.lib")
#pragma comment(lib, "opencv_imgproc249d.lib")
*/
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")

void cvArrow(CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness=1, int lineType=8, int shift=0)
{
    cvLine(img,pt1,pt2,color,thickness,lineType,shift);
    float vx = (float)(pt2.x - pt1.x);
    float vy = (float)(pt2.y - pt1.y);
    float v = sqrt( vx*vx + vy*vy );
    float ux = vx / v;
    float uy = vy / v;
    //ñÓàÛÇÃïùÇÃïîï™
    float w=5,h=10;
    CvPoint ptl,ptr;
    ptl.x = (int)((float)pt2.x - uy*w - ux*h);
    ptl.y = (int)((float)pt2.y + ux*w - uy*h);
    ptr.x = (int)((float)pt2.x + uy*w - ux*h);
    ptr.y = (int)((float)pt2.y - ux*w - uy*h);
    //ñÓàÛÇÃêÊí[Çï`âÊÇ∑ÇÈ
    cvLine(img,pt2,ptl,color,thickness,lineType,shift);
    cvLine(img,pt2,ptr,color,thickness,lineType,shift);
}

double epanechnikov(double r) {
	if (1-r<0) {
		return 0;
	} else {
		return 1-r;
	}
}

void meanShiftRGBXY(int argc, char** argv) {
	IplImage* inputImage;
	if (argc >= 2) {
		inputImage = cvLoadImage(argv[1]);
	} else {
		inputImage = cvLoadImage("D:\\opencv\\image\\hand2.png");
	}
	IplImage* outputImage = cvCloneImage(inputImage);
	//cv::Mat inputMat = cv::cvarrToMat(inputImage);
	//cv::Mat labImage;
	//cv::cvtColor(inputMat, labImage, CV_BGR2Lab);
	//*inputImage = labImage;
	
	const int LENGTH = inputImage->imageSize;
	const double THRESHOLD = 0.01;
	const int HR = 19;
	const int HS = 16;
	const int HR2 = HR*HR;
	const int HS2 = HS*HS;
	int **points = new int*[LENGTH];
	int **convergents = new int*[LENGTH];
	for (int i=0; i<LENGTH; i++) {
		points[i] = new int[5];
		convergents[i] = new int[5];
	}


	// load rgb
	for (int y=0; y<inputImage->height; y++) {
		uchar* ptr = (uchar*)(inputImage->imageData + y*inputImage->widthStep);
		for (int x=0; x<inputImage->width; x++) {
			int i = y*inputImage->width + x;
			points[i][0] = *(ptr+2);	// r
			points[i][1] = *(ptr+1);	// g
			points[i][2] = *(ptr+0);	// b
			points[i][3] = x;
			points[i][4] = y;

			ptr+=3;
		}
	}


	// mean shift
	
	for (int i=0; i<LENGTH; i++) {

		double meanR = points[i][0];
		double meanG = points[i][1];
		double meanB = points[i][2];
		double meanX = points[i][3];
		double meanY = points[i][4];
		double nextMeanR = 0;
		double nextMeanG = 0;
		double nextMeanB = 0;
		double nextMeanX = 0;
		double nextMeanY = 0;

		while(true) {

			// calcurate mean
			double numeratorR = 0;
			double numeratorG = 0;
			double numeratorB = 0;
			double numeratorX = 0;
			double numeratorY = 0;
			double denominator = 0;
			double pR, pG, pB, pX, pY, dR, dG, dB, dX, dY;
			double g;
			for (int j=0; j<LENGTH; j++) {
				pR = points[j][0];
				pG = points[j][1];
				pB = points[j][2];
				pX = points[j][3];
				pY = points[j][4];
				dR = meanR-pR;
				dG = meanG-pG;
				dB = meanB-pB;
				dX = meanX-pX;
				dY = meanY-pY;
				if (dR > HR || dG > HR || dB > HR || dX > HS || dY > HS) {
					continue;
				}
				g = epanechnikov((dR*dR + dG*dG + dB*dB)/(double)(HR2))
					* epanechnikov((dX*dX + dY*dY)/(double)(HS2));
				numeratorR += g*pR;
				numeratorG += g*pG;
				numeratorB += g*pB;
				numeratorX += g*pX;
				numeratorY += g*pY;
				denominator += g;
			}

			nextMeanR = numeratorR/denominator;
			nextMeanG = numeratorG/denominator;
			nextMeanB = numeratorB/denominator;
			nextMeanX = numeratorX/denominator;
			nextMeanY = numeratorY/denominator;

			double d = sqrt( (nextMeanR-meanR)*(nextMeanR-meanR) + 
				(nextMeanG-meanG)*(nextMeanG-meanG) + 
				(nextMeanB-meanB)*(nextMeanB-meanB) +
				(nextMeanX-meanX)*(nextMeanX-meanX) +
				(nextMeanY-meanY)*(nextMeanY-meanY) );
			//std::cout<<d<<std::endl;
			if (d < THRESHOLD) {
				break;
			}
			meanR = nextMeanR;
			meanG = nextMeanG;
			meanB = nextMeanB;
			meanX = nextMeanX;
			meanY = nextMeanY;
		}

		convergents[i][0] = nextMeanR;
		convergents[i][1] = nextMeanG;
		convergents[i][2] = nextMeanB;
		convergents[i][3] = nextMeanX;
		convergents[i][4] = nextMeanY;
	}

	for (int y=0; y<outputImage->height; y++) {
		uchar* ptr = (uchar*)(outputImage->imageData + y*outputImage->widthStep);
		for (int x=0; x<outputImage->width; x++) {
			int i = y*outputImage->width + x;
			*(ptr+2) = convergents[i][0];	// r
			*(ptr+1) = convergents[i][1];	// g
			*(ptr+0) = convergents[i][2];	// b

			ptr+=3;
		}
	}
	
	/*
	// output
	const int IMAGE_WIDTH = 800;
	const int IMAGE_HEIGHT = 800;
	const int BASE_HEIGHT = 800;

	CvMat* image = cvCreateMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	cvRectangle(image, cvPoint(0,0), cvPoint(image->width-1, image->height-1), cvScalar(255, 255, 255, 0), CV_FILLED, CV_AA, 0);
	//cvLine(image, cvPoint(0, BASE_HEIGHT), cvPoint(image->width-1, BASE_HEIGHT), cvScalar(0, 0, 0, 0), 1, CV_AA, 0);

	for (int i=0; i<LENGTH; i++) {
		int pX = points[i][0]*7+50;
		int pY = points[i][1]*7+50;
		int meanX = convergents[i][0]*7+50;
		int meanY = convergents[i][1]*7+50;
		
		cvCircle(image, cvPoint(pX, pY), 5, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvCircle(image, cvPoint(meanX, meanY), 3, cvScalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		if ((pX!=meanX) || (pY!=meanY)) {
			cvArrow(image, cvPoint(pX, pY), cvPoint(meanX, meanY), cvScalar(0, 255, 0, 0), 1, CV_AA, 0);
		} else {
		}
	}
	*/

	cvNamedWindow("input", 1);
	cvNamedWindow("output", 1);
	cvShowImage("input", inputImage);
	cvShowImage("output", outputImage);

	if (argc >= 3) {
		cvSaveImage(argv[2], outputImage);
	} else {
		cvSaveImage("output.png", outputImage);
	}

	//cvWaitKey(0);

	
	
	for (int i=0; i<LENGTH; i++) {
		delete[] points[i];
		delete[] convergents[i];
	}
	delete[] points;
	delete[] convergents;

	return;
}

void meanShiftRGB() {
	IplImage* inputImage = cvLoadImage("D:\\opencv\\image\\hand2.png");
	IplImage* outputImage = cvCloneImage(inputImage);
	//cv::Mat inputMat = cv::cvarrToMat(inputImage);
	//cv::Mat labImage;
	//cv::cvtColor(inputMat, labImage, CV_BGR2Lab);
	//*inputImage = labImage;
	
	const int LENGTH = inputImage->imageSize;
	const double THRESHOLD = 0.01;
	const int H = 30;
	const int H2 = H*H;
	int **points = new int*[LENGTH];
	int **convergents = new int*[LENGTH];
	for (int i=0; i<LENGTH; i++) {
		points[i] = new int[3];
		convergents[i] = new int[3];
	}


	// load rgb
	for (int y=0; y<inputImage->height; y++) {
		uchar* ptr = (uchar*)(inputImage->imageData + y*inputImage->widthStep);
		for (int x=0; x<inputImage->width; x++) {
			int i = y*inputImage->width + x;
			points[i][0] = *(ptr+2);	// r
			points[i][1] = *(ptr+1);	// g
			points[i][2] = *(ptr+0);	// b

			ptr+=3;
		}
	}


	// mean shift
	
	for (int i=0; i<LENGTH; i++) {

		double meanR = points[i][0];
		double meanG = points[i][1];
		double meanB = points[i][2];
		double nextMeanR = 0;
		double nextMeanG = 0;
		double nextMeanB = 0;

		while(true) {

			// calcurate mean
			double numeratorR = 0;
			double numeratorG = 0;
			double numeratorB = 0;
			double denominator = 0;
			double pR, pG, pB, dR, dG, dB;
			double g;
			for (int j=0; j<LENGTH; j++) {
				pR = points[j][0];
				pG = points[j][1];
				pB = points[j][2];
				dR = meanR-pR;
				dG = meanG-pG;
				dB = meanB-pB;
				g = epanechnikov((dR*dR + dG*dG + dB*dB)/(double)(H2));
				numeratorR += g*pR;
				numeratorG += g*pG;
				numeratorB += g*pB;
				denominator += g;
			}

			nextMeanR = numeratorR/denominator;
			nextMeanG = numeratorG/denominator;
			nextMeanB = numeratorB/denominator;
			double d = sqrt((nextMeanR-meanR)*(nextMeanR-meanR) + (nextMeanG-meanG)*(nextMeanG-meanG) + (nextMeanB-meanB)*(nextMeanB-meanB));
			//std::cout<<d<<std::endl;
			if (d < THRESHOLD) {
				break;
			}
			meanR = nextMeanR;
			meanG = nextMeanG;
			meanB = nextMeanB;
		}

		convergents[i][0] = nextMeanR;
		convergents[i][1] = nextMeanG;
		convergents[i][2] = nextMeanB;
	}

	for (int y=0; y<outputImage->height; y++) {
		uchar* ptr = (uchar*)(outputImage->imageData + y*outputImage->widthStep);
		for (int x=0; x<outputImage->width; x++) {
			int i = y*outputImage->width + x;
			*(ptr+2) = convergents[i][0];	// r
			*(ptr+1) = convergents[i][1];	// g
			*(ptr+0) = convergents[i][2];	// b

			ptr+=3;
		}
	}
	
	/*
	// output
	const int IMAGE_WIDTH = 800;
	const int IMAGE_HEIGHT = 800;
	const int BASE_HEIGHT = 800;

	CvMat* image = cvCreateMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	cvRectangle(image, cvPoint(0,0), cvPoint(image->width-1, image->height-1), cvScalar(255, 255, 255, 0), CV_FILLED, CV_AA, 0);
	//cvLine(image, cvPoint(0, BASE_HEIGHT), cvPoint(image->width-1, BASE_HEIGHT), cvScalar(0, 0, 0, 0), 1, CV_AA, 0);

	for (int i=0; i<LENGTH; i++) {
		int pX = points[i][0]*7+50;
		int pY = points[i][1]*7+50;
		int meanX = convergents[i][0]*7+50;
		int meanY = convergents[i][1]*7+50;
		
		cvCircle(image, cvPoint(pX, pY), 5, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvCircle(image, cvPoint(meanX, meanY), 3, cvScalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		if ((pX!=meanX) || (pY!=meanY)) {
			cvArrow(image, cvPoint(pX, pY), cvPoint(meanX, meanY), cvScalar(0, 255, 0, 0), 1, CV_AA, 0);
		} else {
		}
	}
	*/

	cvNamedWindow("input", 1);
	cvNamedWindow("output", 1);
	cvShowImage("input", inputImage);
	cvShowImage("output", outputImage);
	cvWaitKey(0);
	
	for (int i=0; i<LENGTH; i++) {
		delete[] points[i];
		delete[] convergents[i];
	}
	delete[] points;
	delete[] convergents;

	return;
}

void meanShift2D() {
	const int LENGTH = 100;
	const double THRESHOLD = 0.01;
	const int H = 12;
	const int H2 = H*H;
	int points[LENGTH][2];
	int convergents[LENGTH][2];

	// create points
	srand(time(0));
	for (int i=0; i<LENGTH; i++) {
		points[i][0] = 0;
		points[i][1] = 0;
		for (int j=0; j<3; j++) {
			points[i][0] += rand()%34;
			points[i][1] += rand()%34;
		}
	} 


	// mean shift
	
	for (int i=0; i<LENGTH; i++) {

		double meanX = points[i][0];
		double meanY = points[i][1];
		double nextMeanX = 0;
		double nextMeanY = 0;

		while(true) {

			// calcurate mean
			double numeratorX = 0;
			double numeratorY = 0;
			double denominator = 0;
			for (int j=0; j<LENGTH; j++) {
				double pX = (double)points[j][0];
				double pY = (double)points[j][1];
				double dX = meanX-pX;
				double dY = meanY-pY;
				double g = epanechnikov((double)(dX*dX+dY*dY)/(double)(H2));
				numeratorX += g*pX;
				numeratorY += g*pY;
				denominator += g;
			}

			nextMeanX = numeratorX/denominator;
			nextMeanY = numeratorY/denominator;
			double d = sqrt((nextMeanX-meanX)*(nextMeanX-meanX) + (nextMeanY-meanY)*(nextMeanY-meanY));
			//std::cout<<d<<std::endl;
			if (d < THRESHOLD) {
				break;
			}
			meanX = nextMeanX;
			meanY = nextMeanY;
		}

		convergents[i][0] = (int)nextMeanX;
		convergents[i][1] = (int)nextMeanY;
	}
	
	
	

	// output
	const int IMAGE_WIDTH = 800;
	const int IMAGE_HEIGHT = 800;
	const int BASE_HEIGHT = 800;

	CvMat* image = cvCreateMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	cvRectangle(image, cvPoint(0,0), cvPoint(image->width-1, image->height-1), cvScalar(255, 255, 255, 0), CV_FILLED, CV_AA, 0);
	//cvLine(image, cvPoint(0, BASE_HEIGHT), cvPoint(image->width-1, BASE_HEIGHT), cvScalar(0, 0, 0, 0), 1, CV_AA, 0);

	for (int i=0; i<LENGTH; i++) {
		int pX = points[i][0]*7+50;
		int pY = points[i][1]*7+50;
		int meanX = convergents[i][0]*7+50;
		int meanY = convergents[i][1]*7+50;
		
		cvCircle(image, cvPoint(pX, pY), 5, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvCircle(image, cvPoint(meanX, meanY), 3, cvScalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		if ((pX!=meanX) || (pY!=meanY)) {
			cvArrow(image, cvPoint(pX, pY), cvPoint(meanX, meanY), cvScalar(0, 255, 0, 0), 1, CV_AA, 0);
		} else {
		}
	}

	cvNamedWindow("MeanShift", 1);
	cvShowImage("MeanShift", image);
	cvWaitKey(0);

	return;
}

void meanShift1D() {
	const int LENGTH = 100;
	const int THRESHOLD = 1;
	const int H = 10;
	const int H2 = H*H;
	int points[LENGTH];
	int convergents[LENGTH];

	// create points
	srand(time(0));
	for (int i=0; i<LENGTH; i++) {
		points[i] = 0;
		for (int j=0; j<10; j++) {
			points[i] += rand()%11;
		}
	} 

	// mean shift
	
	for (int i=0; i<LENGTH; i++) {

		double mean = points[i];
		double nextMean = 0;

		while(true) {

			// calcurate mean
			double numerator = 0;
			double denominator = 0;
			for (int j=0; j<LENGTH; j++) {
				double p = (double)points[j];
				double d = mean-p;
				double g = epanechnikov((double)(d*d)/(double)(H2));
				numerator += g*p;
				denominator += g;
			}

			nextMean = numerator/denominator;
			double d = sqrt((nextMean-mean)*(nextMean-mean));
			//std::cout<<d<<std::endl;
			if (d < THRESHOLD) {
				break;
			}
			mean = nextMean;
		}

		convergents[i] = (int)nextMean;
	}
	
	
	

	// output
	const int IMAGE_WIDTH = 1200;
	const int IMAGE_HEIGHT = 800;
	const int BASE_HEIGHT = 800;

	CvMat* image = cvCreateMat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
	cvRectangle(image, cvPoint(0,0), cvPoint(image->width-1, image->height-1), cvScalar(255, 255, 255, 0), CV_FILLED, CV_AA, 0);
	cvLine(image, cvPoint(0, BASE_HEIGHT), cvPoint(image->width-1, BASE_HEIGHT), cvScalar(0, 0, 0, 0), 1, CV_AA, 0);

	for (int i=0; i<LENGTH; i++) {
		int x = i*10+100;
		int y = BASE_HEIGHT - points[i]*8;
		int mean = BASE_HEIGHT - convergents[i]*8;
		cvLine(image, cvPoint(x, 0), cvPoint(x, image->height-1), cvScalar(192, 192, 192, 0), 1, CV_AA, 0);
		
		cvCircle(image, cvPoint(x, y), 5, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvCircle(image, cvPoint(x, mean), 3, cvScalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		if (y!=mean) {
			cvArrow(image, cvPoint(x, y), cvPoint(x, mean), cvScalar(0, 255, 0, 0), 1, CV_AA, 0);
		} else {
		}
	}

	
	cvNamedWindow("MeanShift", 1);
	cvShowImage("MeanShift", image);
	cvWaitKey(0);

	return;
}

int main(int argc, char** argv) {
	//meanShift1D();
	//meanShift2D();
	meanShiftRGBXY(argc, argv);

	return 0;
}

