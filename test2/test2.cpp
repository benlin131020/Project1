// test2.cpp: 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include "functions.h"
using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
	int index = 0;
	VideoCapture video;
	while (true) {
		cout << "1.Video1 2.Video2 0.Exit:";
		cin >> index;
		if (index == 1)	video.open("test_video1.mp4");
		else if (index == 2) video.open("GoPro_shoulder_003.avi");
		else break;

		Mat original_img, processed_img, skin_img, result_img;
		vector<Rect> rois;
		vector<Mat> rois_img;
		//Set up HOG
		cv::HOGDescriptor hog(
			cv::Size(SVM_SIZE, SVM_SIZE), //winSize
			cv::Size(16, 16), //blocksize
			cv::Size(8, 8), //blockStride,
			cv::Size(8, 8), //cellSize,
			9 //nbins,
		);
		//Set up SVM 
		cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>("testNT.yml");
		//fps
		long frameCounter = 0;
		time_t timeBegin = time(0);
		int tick = 0;
		int counter = 0;


		while (true) {
			video >> original_img;
			if (original_img.empty() || cv::waitKey(1) == 27) {
				break;
			}
			cv::Point tl(0, original_img.rows * 2 / 5);
			cv::Point br(original_img.cols, original_img.rows * 3 / 4);

			Preprocess(original_img, processed_img, tl, br);
			Skin_Det(processed_img, skin_img);
			result_img = ROI(processed_img.clone(), skin_img, hog, svm, rois, rois_img);
			imshow("original", original_img);
			imshow("preprocess", processed_img);
			//fps
			frameCounter++;
			time_t timeNow = time(0) - timeBegin;
			if (timeNow - tick >= 1) {
				tick++;
				cout << "Frames per second: " << frameCounter << endl;
				frameCounter = 0;
			}
			////save
			//if (counter % 3 == 0) {
			//	imwrite("result_GoPro_shoulder_003/result" + to_string(counter)+".jpg", result_img);
			//}
			//counter++;
		}

		video.release();
		
	}
	
	
	
    return 0;
}