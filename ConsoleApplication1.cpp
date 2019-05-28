// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

RNG random;
const int candidate_limit = 20;
const int ransac_iterations = 1e3;
Mat random_indices(ransac_iterations, 2, CV_8UC1);

Mat glasses;

int display(Mat img) {
	Mat img_show;
	resize(img, img_show, Size(), 0.5, 0.5);

	imshow("wdw", img_show);
	return(waitKey(1));
}

//Input: An inputpoint and a vector of points
//Output: A pair containing the index of the point in the vector that is closest to the input-point and the respective distance.
pair<int, float> calc_closest_index_and_distance(Point2f point, const vector<Point2f> candidates) {
	vector<float> distances(candidates.size());
	for (int i = 0; i < candidates.size(); i++) {
		distances[i] = (point.x - candidates[i].x)*(point.x - candidates[i].x)
			+ (point.y - candidates[i].y)*(point.y - candidates[i].y);
	}

	int minimum_index = std::min_element(distances.begin(), distances.end()) - distances.begin();

	return pair<int, float>(minimum_index, distances[minimum_index]);
}

//Input: a vector of detected candidates
//Output: a vector containing the detected points of the Physical Sample Frame. Return is empty if RANSAC failed to detect a viable model.
vector<Point2f> ransac_candidates(vector<Point2f> candidates, Mat debug_img = Mat()) {
	vector<Point2f> anchors;

	//Limit the search to the first <candidate_limit> candidates.
	while (candidates.size() > candidate_limit)
		candidates.pop_back();

	//At least 5 candidates are needed to fit at model.
	if (candidates.size() < 5) {
		printf("ransac_candidates(): not enough candidates.");
		return anchors;
	}

	//Try to guess the top-left and top-right points of the glasses.
	random.fill(random_indices, RNG::UNIFORM, 0, candidates.size());

	std::vector<int> best_model_indices;
	double best_model_error = 1e12;

	for (int iteration = 0; iteration < ransac_iterations; iteration++) {
		int index0 = random_indices.at<unsigned char>(iteration, 0);
		int index1 = random_indices.at<unsigned char>(iteration, 1);

		if (index0 == index1)continue;

		Point2f c0 = candidates[index0];
		Point2f c1 = candidates[index1];
		Point2f vec = c1 - c0;
		Point2f vec_orth;
		if (vec.x >= 0)
			vec_orth = Point2f(-vec.y, vec.x);
		else
			vec_orth = Point2f(vec.y, -vec.x);

		//Estimate the expected locations of the 3 remaining points based on our guesses.
		Point2i p0 = c0 + 0.5*vec + 0.025*vec_orth;
		Point2i p1 = c0 + 0.2*vec + 0.3*vec_orth;
		Point2i p2 = c0 + 0.8*vec + 0.3*vec_orth;

		if (!debug_img.empty()) {
			Mat img_show;
			debug_img.copyTo(img_show);
			circle(img_show, c0, 15, Scalar(0, 0, 255), 5);
			circle(img_show, c1, 15, Scalar(0, 0, 255), 5);
			circle(img_show, p0, 15, Scalar(0, 255, 255), 5);
			circle(img_show, p1, 15, Scalar(0, 255, 255), 5);
			circle(img_show, p2, 15, Scalar(0, 255, 255), 5);
	//		display(img_show);
			waitKey(100);
		}

		//Find the candidates in the image that are closest to the estimated positions.
		pair<int, float> index_dist_p0 = calc_closest_index_and_distance(p0, candidates);
		pair<int, float> index_dist_p1 = calc_closest_index_and_distance(p1, candidates);
		pair<int, float> index_dist_p2 = calc_closest_index_and_distance(p2, candidates);

		//Make sure that those closest candidates are unique; a candidate can not account for multiple points on the PSF.
		vector<int> anchor_indices{ index0, index1, index_dist_p0.first, index_dist_p1.first, index_dist_p2.first };
		vector<int>::iterator it = unique(anchor_indices.begin(), anchor_indices.end());
		if (distance(anchor_indices.begin(), it) != 5) {
			//printf("Points not unique, rejecting model.");
			continue;
		}

		//Update the model and error if a better model has been found.
		float model_error = index_dist_p0.second + index_dist_p1.second + index_dist_p2.second;
		if (model_error < best_model_error) {
			best_model_indices = { index0,index1,index_dist_p0.first,index_dist_p1.first,index_dist_p2.first };
			best_model_error = model_error;
		}

	}

	//Make sure the rotation is always correct
	if (!best_model_indices.empty() && candidates[best_model_indices[0]].x > candidates[best_model_indices[1]].x) {
		best_model_indices = { best_model_indices[1],best_model_indices[0],best_model_indices[2],best_model_indices[4],best_model_indices[3] };
	}
	//int count = 0;
	if (!best_model_indices.empty()) {
		cout << "best model err:" << best_model_error << endl;
		anchors.clear();
		for (int idx : best_model_indices) {
			anchors.push_back(candidates[idx]);
		//	count++;
		}
	}
	
	return anchors;
}

//Input: image with green dots painted on the PSF. (Real world frame, real world dots.)
//Output: binary mask image containing the detected regions
Mat threshColor_greenBlobs(Mat img) {
	Mat img_hsv;
	cvtColor(img, img_hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(img_hsv, Scalar(30, 10, 80), Scalar(80, 255, 255), mask);
//	display(mask);
	return mask;
}

//Input: binary mask of detected regions (output of threshColor_greenBlobs())
//Output: vector containing the centers of the detected region-blobs.
vector<Point2f> detectCandidates_blobs(Mat mask) {
	Mat labels, stats, centroids;
	connectedComponentsWithStats(mask, labels, stats, centroids);

	vector<Point2f> candidates;
	Mat img_return(mask.size(), CV_8UC3);
	for (int i = 0; i < stats.rows; i++) {
		int blob_area = stats.at<int>(i, CC_STAT_AREA);
		if (blob_area > 1e2 && blob_area < 1e6) {
			candidates.push_back(Point2f(centroids.at<double>(i, 0), centroids.at<double>(i, 1)));
			cout << centroids.at<double>(i, 0) << "++" << centroids.at<double>(i, 1)<<endl;
		}
	}
	
	return candidates;
}

//Input: image to paint the glasses on, vector with anchors points (output of ransac_candidates())
//Output: input image with glasses painted on top
Mat paint_glasses(Mat img, vector<Point2f> anchors) {
	Mat glasses_rgb(glasses.size(), CV_8UC3);
	Mat glasses_alpha(glasses.size(), CV_8UC1);

	int from_to[]{ 3,0 };
	mixChannels(&glasses, 1, &glasses_alpha, 1, from_to, 1);
	int from_to2[]{ 0,0,1,1,2,2 };
	mixChannels(&glasses, 1, &glasses_rgb, 1, from_to2, 3);

	vector<Point2f> anchors_glasses{
		Point2f(35,100),
		Point2f(705,110),
		Point2f(185,280),
		Point2f(555,280),
	};
	vector<Point2f> anchors_image;
	anchors_image.push_back(anchors[0]);
	anchors_image.push_back(anchors[1]);
	anchors_image.push_back(anchors[3]);
	anchors_image.push_back(anchors[4]);
	cout << "GPT..." << endl;
	for (auto a : anchors_image)
		cout << a << endl;

	Mat M = getPerspectiveTransform(anchors_glasses, anchors_image);
	cout << "M" << M << endl;

	Mat alpha_warped, rgb_warped;
	warpPerspective(glasses_rgb, rgb_warped, M, img.size());
	warpPerspective(glasses_alpha, alpha_warped, M, img.size());

	Mat img_return(img.size(), CV_8UC3);
	for (int i = 0; i < img_return.total(); i++) {
		int alpha = alpha_warped.at<unsigned char>(i) / 255;
		img_return.at<Vec3b>(i) = alpha * rgb_warped.at<Vec3b>(i) + (1 - alpha)*img.at<Vec3b>(i);  
		/*
		glasses + (glasses & man)
		*/
	}
	//display(img_return);
	return img_return;
}

void process_video(string path_in, string path_out) {
	Mat img;
	VideoCapture cap(path_in);
	cap.read(img);
	cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
	VideoWriter writer = VideoWriter(path_out, VideoWriter::fourcc('H', '2', '6', '4'), 30.0, img.size());
	int cnt = 0;
	while (cap.read(img)) {
		cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);

		Mat mask = threshColor_greenBlobs(img);
		vector<Point2f> candidates = detectCandidates_blobs(mask);
		vector<Point2f> anchors = ransac_candidates(candidates, mask);

		if (!anchors.empty()) {
			Mat img_with_glasses = paint_glasses(img, anchors);
			display(img_with_glasses);
			writer.write(img_with_glasses);
		}
		else {
			printf("detection failed\n");
			writer.write(img);
		}


		printf("count %i\n", cnt++);
	}
	cap.release();
	writer.release();
}

int main()
{
	glasses = imread("glass1.png", IMREAD_UNCHANGED);
	//vector<Point2f>xx{
	//	Point2f(100,100),
	//	Point2f(200,100),
	//	Point2f(100,200),
	//	Point2f(200,200),
	//};
	//paint_glasses(Mat::zeros(500,1200,CV_8UC3), xx);
	//return 0;

	process_video("marker_B_1.mp4", "out_CPP.mp4");


	return 0;
}
