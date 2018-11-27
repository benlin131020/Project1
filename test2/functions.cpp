#include<opencv2/opencv.hpp>
#include"functions.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

void Preprocess(cv::Mat input_img, cv::Mat &output_img, cv::Point tl, cv::Point br) {
	//ÁY´î¨ú¼Ë
	output_img = input_img(cv::Rect(tl, br));
	//Convert processed_img to YCbCr
	cv::cvtColor(output_img, output_img, CV_BGR2YCrCb);
	//splits a multi-channel array into separate single-channel arrays
	std::vector<cv::Mat> channels;
	cv::split(output_img, channels);
	//CLAHE (Contrast Limited Adaptive Histogram Equalization)
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(2);
	clahe->apply(channels[0], channels[0]);
	//Creates one multi-channel array out of several single-channel ones. 
	cv::merge(channels, output_img);
	//Convert processed_img to BGR
	cv::cvtColor(output_img, output_img, CV_YCrCb2BGR);
}

bool rule_bgr(int R, int G, int B) {
	bool e1 = (R > 95) && (G > 40) && (B > 20) && ((max(R, max(G, B)) - min(R, min(G, B))) > 15) && (abs(R - G) > 15) && (R > G) && (R > B);
	bool e2 = (R > 220) && (G > 210) && (B > 170) && (abs(R - G) <= 15) && (R > B) && (G > B);
	return (e1 || e2);
}

bool rule_ycrcb(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862*Cb + 20;
	bool e4 = Cr >= 0.3448*Cb + 76.2069;
	bool e5 = Cr >= -4.5652*Cb + 234.5652;
	bool e6 = Cr <= -1.15*Cb + 301.75;
	bool e7 = Cr <= -2.2857*Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool rule_hsv(float H, float S, float V) {
	return (H < 25) || (H > 230);
}

void Skin_Det(cv::Mat input_img, cv::Mat &skin_img) {
	cv::Mat ycrcb_img, hsv_img, canny_img, gb_img;
	//get skin color
	cv::cvtColor(input_img, ycrcb_img, CV_BGR2YCrCb);
	cv::cvtColor(input_img, hsv_img, CV_BGR2HSV);
	skin_img = input_img.clone();
	cvtColor(skin_img, skin_img, CV_BGR2GRAY);
	for (int i = 0; i < input_img.rows; i++) {
		for (int j = 0; j < input_img.cols; j++) {
			Vec3b ycrcb_pixel = ycrcb_img.at<Vec3b>(i, j);
			int y = ycrcb_pixel[0];
			int cr = ycrcb_pixel[1];
			int cb = ycrcb_pixel[2];
			bool ycrcb_bool = rule_ycrcb(y, cr, cb);
			Vec3b hsv_pixel = hsv_img.at<Vec3b>(i, j);
			int h = hsv_pixel[0];
			int s = hsv_pixel[1];
			int v = hsv_pixel[2];
			bool hsv_bool = rule_hsv(h, s, v);
			Vec3b bgr_pixel = input_img.at<Vec3b>(i, j);
			int b = bgr_pixel[0];
			int g = bgr_pixel[1];
			int r = bgr_pixel[2];
			bool bgr_bool = rule_bgr(r, g, b);
			if (!(ycrcb_bool&&hsv_bool&&bgr_bool)) skin_img.at<uchar>(i, j) = 0;
			else skin_img.at<uchar>(i, j) = 255;
		}
	}
	imshow("skin_color", skin_img);
	//Opening
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::erode(skin_img, skin_img, element);
	cv::dilate(skin_img, skin_img, element);
	imshow("opening", skin_img);
}

bool templateMatching(Mat src, Mat roiImg, Point &roi_tl, Point &roi_br) {
	Mat result;
	Mat rotate1, rotate2;
	//tl
	matchTemplate(src, roiImg, result, CV_TM_SQDIFF_NORMED);
	double minVal1;
	minMaxLoc(result, &minVal1, 0, &roi_tl, 0);
	//rotate image
	Point2f pt1(src.cols / 2., src.rows / 2.);
	Mat r1 = getRotationMatrix2D(pt1, 180, 1.0);
	warpAffine(src, rotate1, r1, Size(src.cols, src.rows));
	Point2f pt2(roiImg.cols / 2., roiImg.rows / 2.);
	Mat r2 = getRotationMatrix2D(pt2, 180, 1.0);
	warpAffine(roiImg, rotate2, r2, Size(roiImg.cols, roiImg.rows));
	//br
	matchTemplate(rotate1, rotate2, result, CV_TM_SQDIFF_NORMED);
	double minVal2;
	Point minLoc;
	minMaxLoc(result, &minVal2, 0, &minLoc, 0);
	//rotate point
	cv::Point2f trans_pt = Point2f(minLoc) - pt1;
	float x = std::cos(CV_PI) * trans_pt.x - std::sin(CV_PI) * trans_pt.y;
	float y = std::sin(CV_PI) * trans_pt.x + std::cos(CV_PI) * trans_pt.y;
	cv::Point2f rot_pt(x, y);
	roi_br = rot_pt + pt1;
	//match_success
	if ((minVal1 + minVal2) / 2 > 0.1) return false;
	else true;
}

void tracking(cv::Mat input_img, vector<Rect> &rois, vector<Mat> &rois_img, cv::Mat &display) {
	for (size_t i = 0; i < rois.size(); i++) {
		//delete if roi out of image's edge
		if (rois[i].x - rois[i].width / 2 < 0 || rois[i].y - rois[i].height / 2 < 0 
			|| rois[i].x - rois[i].width / 2 + rois[i].width * 2 > input_img.cols
			|| rois[i].y - rois[i].height / 2 + rois[i].height * 2 > input_img.rows) {
			rois.erase(rois.begin() + i);
			rois_img.erase(rois_img.begin() + i);
			continue;
		}
		//template matching
		Mat src = input_img(Rect(rois[i].x - rois[i].width / 2, rois[i].y - rois[i].height / 2, rois[i].width * 2, rois[i].height * 2));
		Point roi_tl, roi_br;
		bool match_success;
		match_success = templateMatching(src, rois_img[i], roi_tl, roi_br);
		//delete if match fail
		if (!match_success) {
			rois.erase(rois.begin() + i);
			rois_img.erase(rois_img.begin() + i);
			continue;
		}
		//draw and update rois
		rectangle(display, Point(roi_tl.x + rois[i].x - rois[i].width / 2, roi_tl.y + rois[i].y - rois[i].height / 2), Point(roi_br.x + rois[i].x - rois[i].width / 2, roi_br.y + rois[i].y - rois[i].height / 2), Scalar(0, 255, 0), 2);
		rois_img[i] = input_img(Rect(Point(roi_tl.x + rois[i].x - rois[i].width / 2, roi_tl.y + rois[i].y - rois[i].height / 2), Point(roi_br.x + rois[i].x - rois[i].width / 2, roi_br.y + rois[i].y - rois[i].height / 2)));
		rois[i] = Rect(Point(roi_tl.x + rois[i].x - rois[i].width / 2, roi_tl.y + rois[i].y - rois[i].height / 2), Point(roi_br.x + rois[i].x - rois[i].width / 2, roi_br.y + rois[i].y - rois[i].height / 2));
	}
}

Mat ROI(cv::Mat input_img, cv::Mat skin_img, cv::HOGDescriptor hog, cv::Ptr<cv::ml::SVM> svm, vector<Rect> &rois, vector<Mat> &rois_img) {
	Mat display = input_img.clone();
	Mat roi_display = input_img.clone();
	//Find contourst
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(skin_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	std::vector<cv::Rect> boundRect(contours.size());
	cv::cvtColor(skin_img, skin_img, CV_GRAY2BGR);
	//track
	tracking(input_img, rois, rois_img, display);
	//Draw Bounding rects
	for (int i = 0; i < contours.size(); i++) {
		boundRect[i] = cv::boundingRect(contours[i]);
		if (//condition of rect's area
			//boundRect[i].area() < 5000 &&
			boundRect[i].area() > 256 &&
			//condition of rect's ratio
			//((boundRect[i].width / boundRect[i].height) <= 0.4 || (boundRect[i].height / boundRect[i].width) <= 0.7)
			MAX(boundRect[i].height, boundRect[i].width) / MIN(boundRect[i].height, boundRect[i].width) < 1.5) {
			rectangle(roi_display, boundRect[i], cv::Scalar(0, 0, 255), 2, 8, 0);
			//normalization
			Rect roi_rect(boundRect[i].x, boundRect[i].y, MIN(boundRect[i].width, boundRect[i].height), MIN(boundRect[i].width, boundRect[i].height));
			//delete overlapped area
			bool overlapped = false;
			for (size_t i = 0; i < rois.size(); i++) {
				if ((roi_rect & rois[i]).area() >= MIN(roi_rect.area(), rois[i].area()) / 2) {
					overlapped = true;
					break;
				}
			}
			if (overlapped)continue;
			//svm
			std::vector<float> descriptors;
			cv::Mat src = input_img(roi_rect);
			cv::resize(src, src, cv::Size(SVM_SIZE, SVM_SIZE));
			hog.compute(src, descriptors);
			float response = svm->predict(descriptors);
			if (response == 1) {
				rectangle(display, roi_rect, cv::Scalar(255, 0, 0), 2, 8, 0);
				rois.push_back(roi_rect);
				rois_img.push_back(input_img(rois.back()));
			}
			else rectangle(display, roi_rect, cv::Scalar(0, 0, 255), 2, 8, 0);
		}
		//drawContours(skin_img, contours, i, cv::Scalar(255, 0, 0), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
		//rectangle(skin_img, boundRect[i], cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	cv::imshow("result", display);
	imshow("ROI", roi_display);
	return display;
}