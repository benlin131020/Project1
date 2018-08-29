#include<opencv2/opencv.hpp>

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
	//cv::equalizeHist(channels[0], channels[0]);
	//Creates one multi-channel array out of several single-channel ones. 
	cv::merge(channels, output_img);
	//Convert processed_img to BGR
	cv::cvtColor(output_img, output_img, CV_YCrCb2BGR);
}

void Skin_Det(cv::Mat input_img, cv::Mat &skin_img) {
	cv::Mat ycbcr_img, canny_img, gb_img;
	//gaussianblur
	cv::GaussianBlur(input_img, gb_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	//Use YCbCr to get skin color
	cv::cvtColor(gb_img, ycbcr_img, CV_BGR2YCrCb);
	cv::inRange(ycbcr_img, cv::Scalar(80, 135, 85), cv::Scalar(255, 180, 135), skin_img);
	/*
	//Get Canny of image
	cv::Canny(gb_img, canny_img, 50, 150, 3);
	cv::bitwise_not(canny_img, canny_img);
	//ycbcr_img bitwise_and canny_img
	cv::bitwise_and(skin_img, canny_img, skin_img);
	*/
	// Floodfill
	cv::Mat im_floodfill = skin_img.clone();
	rectangle(im_floodfill, cv::Point(0, 0), cv::Point(im_floodfill.cols, im_floodfill.rows), cv::Scalar(0, 0, 0), 2, 8, 0);
	cv::floodFill(im_floodfill, cv::Point(0, 0), cv::Scalar(255));
	cv::bitwise_not(im_floodfill, im_floodfill);
	skin_img = (skin_img | im_floodfill);
	/*
	//Opening
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::erode(skin_img, skin_img, element);
	cv::dilate(skin_img, skin_img, element);
	
	imshow("Skin Color", skin_img);
	imshow("Canny", canny_img);
	imshow("and", skin_img);
	imshow("fill", skin_img);
	imshow("opening", skin_img);
	*/
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

void ROI(cv::Mat input_img, cv::Mat skin_img, cv::HOGDescriptor hog, cv::Ptr<cv::ml::SVM> svm, vector<Rect> &rois, vector<Mat> &rois_img) {
	Mat display = input_img.clone();
	//Find contours
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
			MAX(boundRect[i].height, boundRect[i].width) / MIN(boundRect[i].height, boundRect[i].width) < 2) {
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
				cv::resize(src, src, cv::Size(32, 32));
				hog.compute(src, descriptors);
				float response = svm->predict(descriptors);
				if (response == 1) {
					rectangle(display, roi_rect, cv::Scalar(255, 0, 0), 2, 8, 0);
					rois.push_back(roi_rect);
					rois_img.push_back(input_img(rois.back()));
				}
				//else rectangle(display, roi_rect, cv::Scalar(0, 0, 255), 2, 8, 0);
		}
		//drawContours(skin_img, contours, i, cv::Scalar(255, 0, 0), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
		//rectangle(skin_img, boundRect[i], cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	cv::imshow("ROI_binary", skin_img);
	cv::imshow("ROI", display);
}