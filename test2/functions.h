#pragma once
#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include<opencv2/opencv.hpp>

#define SVM_SIZE 32

using namespace std;
using namespace cv;
using namespace cv::ml;

void Preprocess(cv::Mat input_img, cv::Mat &output_img, cv::Point tl, cv::Point br);
void Skin_Det(cv::Mat input_img, cv::Mat &skin_img);
void ROI(cv::Mat input_img, cv::Mat skin_img, cv::HOGDescriptor hog, cv::Ptr<cv::ml::SVM> svm, vector<Rect> &rois, vector<Mat> &rois_img);

#endif