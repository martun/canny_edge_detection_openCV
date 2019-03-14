#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

Mat make_greyscale(Mat& img) {
	Mat result(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			Vec3b intensity = img.at<Vec3b>(i, j);
			int avg = (intensity[0] + intensity[1] + intensity[2]) / 3;
			result.at<uint8_t>(i, j) = avg;
		}
	return result;
}

Mat apply_gausian_filter(const Mat& img) {
	// Apply 5x5 Gausian filter, do not touch pixels near corners. 
	Mat filter = (Mat_<int>(5, 5) << 2, 4, 5, 4, 2,
		4, 9, 12, 9, 4,
		5, 12, 15, 12, 5,
		4, 9, 12, 9, 4,
		2, 4, 5, 4, 2);
	filter.convertTo(filter, CV_32SC1);

	// Deep copy the input matrix.
	Mat result = img.clone();
	for (int i = 2; i < img.rows - 2; ++i)
		for (int j = 2; j < img.cols - 2; ++j)
		{
			// On the next line width comes first, so j-2, i-2, not the other way around. 
			Mat part = img(Rect(j - 2, i - 2, 5, 5));
			// Change this matrix to a matrix of 32 bit integers, otherwise 
			// when we multiply, the values are moved into the range [0..255], 
			// which is NOT what we want here.
			part.convertTo(part, CV_32SC1);

			Mat M = part.mul(filter);
			int sum = cv::sum(M).val[0];
			result.at<uint8_t>(i, j) = (uint8_t)(sum / 159);
		}
	return result;
}

Mat get_edge_gradient_values_and_suppress_non_maximums(const Mat& img) {
	// Apply 3x3 Sobel operators to gind the gradients. 
	Mat soble_x = (Mat_<int>(3, 3) << -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	Mat soble_y = (Mat_<int>(3, 3) << -1, -2, -1,
		0, 0, 0,
		+1, +2, +1);

	soble_x.convertTo(soble_x, CV_32SC1);
	soble_y.convertTo(soble_y, CV_32SC1);

	// CV_32SC1 stands for 32 bit signed integer number martix type.
	Mat gradient_img = Mat::zeros(img.size(), CV_32SC1);

	// CV_32SC1 stands for 32 bit signed integer number martix type.
	// We store direction as integer degrees, it's easier to write code that way.
	Mat gradient_direction = Mat::zeros(img.size(), CV_32SC1);

	int32_t min_gradient_value = INT_MAX;
	int32_t max_gradient_value = INT_MIN;

	for (int i = 1; i < img.rows - 1; ++i)
		for (int j = 1; j < img.cols - 1; ++j)
		{
			// On the next line width comes first, so j-1, i-1, not the other way around. 
			Mat part = img(Rect(j - 1, i - 1, 3, 3));
			// Change this matrix to a matrix of 32 bit integers, otherwise 
			// when we multiply, the values are moved into the range [0..255], 
			// which is NOT what we want here.
			part.convertTo(part, CV_32SC1);

			Mat M_Gx = part.mul(soble_x);
			int Gx = cv::sum(M_Gx).val[0];

			Mat M_Gy = part.mul(soble_y);
			int Gy = cv::sum(M_Gy).val[0];

			gradient_img.at<int32_t>(i, j) = (int32_t)(sqrt(Gx * Gx + Gy * Gy));
			gradient_direction.at<int32_t>(i, j) = atan2(Gy, Gx) * 180 / 3.141592653589793;

			// Compute min and max values, to change to [0..255] range later.
			min_gradient_value = std::min(min_gradient_value, gradient_img.at<int32_t>(i, j));
			max_gradient_value = std::max(max_gradient_value, gradient_img.at<int32_t>(i, j));
		}

	// Now we need to run over the pixels and suppress those gradient values which are not
	// greater than 2 other gradient values in the direction of the gradient.
	for (int i = 1; i < img.rows - 1; ++i)
		for (int j = 1; j < img.cols - 1; ++j)
		{
			int other_gradient_value;
			int grad_dir = gradient_direction.at<int32_t>(i, j);
			// Gradient near horizontal.
			if ((grad_dir > -22 && grad_dir < 22) || (grad_dir > (180-22)) || (grad_dir < (-180 + 22))) {
				other_gradient_value = std::max(gradient_img.at<int32_t>(i, j - 1),
					gradient_img.at<int32_t>(i, j + 1));
			}
			// Gradient vertical.
			else if ((grad_dir > (90-22) && grad_dir < (90 + 22)) || (grad_dir > (-90 - 22) && grad_dir < (-90 + 22))) {
				other_gradient_value = std::max(gradient_img.at<int32_t>(i - 1, j), 
					gradient_img.at<int32_t>(i - 1, j));
			}
			// Gradient diagonal /.
			else if ((grad_dir > 22 && grad_dir < (45 + 22)) || (grad_dir > (-90 - 22)) || (grad_dir < (-180 + 22))) {
				other_gradient_value = std::max(gradient_img.at<int32_t>(i + 1, j + 1), 
					gradient_img.at<int32_t>(i - 1, j - 1));
			}
			// Gradient diagonal \.
			else {
				other_gradient_value = std::max(gradient_img.at<int32_t>(i + 1, j - 1), 
					gradient_img.at<int32_t>(i - 1, j + 1));
			}

			// Clear the edge gradient value if it's not the maximum compared to 2 other values
			// in the direction  of it's gradient.
			if (gradient_img.at<int32_t>(i, j) < other_gradient_value) {
				gradient_img.at<int32_t>(i, j) = 0;
			}
		}

	// Deep copy the input matrix.
	Mat result = img.clone();

	// Not we have gradient values in the range [min_gradient_value, max_gradient_value],
	// bring them to the range [0..255] so we can draw the edges.
	for (int i = 1; i < img.rows - 1; ++i)
		for (int j = 1; j < img.cols - 1; ++j)
		{
			result.at<uint8_t>(i, j) = (uint8_t)((gradient_img.at<int32_t>(i, j) - min_gradient_value) * 255
				/ (max_gradient_value - min_gradient_value));
		}
	return result;
}

Mat make_edges_weak_or_strong_thresholding(const Mat& img) {
	// Deep copy the input matrix.
	Mat result = img.clone();

	// Some internet page suggested to find the mean value of 
	// edges and choose +-25% as thresholds. 
	// I tried that, but it did not work well for the images I had, because at this
	// step most of the image was black, and the mean value was very close to 0.
	// So I decided to choose the mean value of pixels which are not completely black,
	// I.E. are larger than 5.
	int mean = 0;
	int count = 0;
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (result.at<uint8_t>(i, j) > 5) {
				mean += result.at<uint8_t>(i, j);
				++count;
			}
		}
	mean /= count;
	int threshold_weak = mean * 3 / 4;
	int threshold_strong = mean * 5 / 4;

	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			int pixel = result.at<uint8_t>(i, j);
			if (pixel < threshold_weak) {
				result.at<uint8_t>(i, j) = 0;
			}
			else if (pixel > threshold_strong) {
				result.at<uint8_t>(i, j) = 255;
			}
			else {
				result.at<uint8_t>(i, j) = 255 / 2;
			}
		}
	return result;
}

void make_strong(Mat& result, int x, int y) {
	int dirx[] = {-1, 0, 1};
	int diry[] = { -1, 0, 1 };

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			// Find all 8-connected neighbours.
			int new_x = x + dirx[i];
			int new_y = y + diry[j];
			// Don't run over the edges of the image.
			if (new_x < 0 || new_y < 0 || new_x >= result.rows || new_y >= result.cols)
				continue;
			if (result.at<uint8_t>(new_x, new_y) == 255 / 2) {
				result.at<uint8_t>(new_x, new_y) = 255;
				make_strong(result, new_x, new_y);
			}
		}
	}
}

Mat apply_edge_tracking_by_hysteresis(const Mat& img) {
	// For each strong edge pixel, check if there is a 
	// weak edge in the 8-connected neighbourhood, and if any
	// make it strong. Continue making edges strong based on the new
	// strong edge as far as you can.
	
	// Deep copy the input matrix.
	Mat result = img.clone();

	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j) {
			if (result.at<uint8_t>(i, j) == 255) {
				make_strong(result, i, j);
			}
		}
	// Now when all the necessary edges are made strong, delete everything else.
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j) {
			if (result.at<uint8_t>(i, j) != 255) {
				result.at<uint8_t>(i, j) = 0;
			}
		}
	return result;
}

// A function to display multiple images in the same window,
// copied from the internet.
void display_multiple_images_in_one_window(string title, int nArgs, ...) {
	int size;
	int i;
	int m, n;
	int x, y;

	// w - Maximum number of images in a row
	// h - Maximum number of images in a column
	int w, h;

	// scale - How much we have to resize the image
	float scale;
	int max;

	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying
	if (nArgs <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nArgs > 14) {
		printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
		return;
	}
	// Determine the size of the image,
	// and the number of rows/cols
	// from number of arguments
	else if (nArgs == 1) {
		w = h = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		w = 2; h = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		w = 2; h = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		w = 3; h = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		w = 4; h = 2;
		size = 200;
	}
	else {
		w = 4; h = 3;
		size = 150;
	}

	// Create a new 3 channel image
	Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC1);

	// Used to get the arguments passed
	va_list args;
	va_start(args, nArgs);

	// Loop for nArgs number of arguments
	for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
		// Get the Pointer to the IplImage
		Mat img = va_arg(args, Mat);

		// Check whether it is NULL or not
		// If it is NULL, release the image, and return
		if (img.empty()) {
			printf("Invalid arguments");
			return;
		}

		// Find the width and height of the image
		x = img.cols;
		y = img.rows;

		// Find whether height or width is greater in order to resize the image
		max = (x > y) ? x : y;

		// Find the scaling factor to resize the image
		scale = (float)((float)max / size);

		// Used to Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		// Set the image ROI to display the current image
		// Resize the input image and copy the it to the Single Big Image
		Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
		Mat temp; resize(img, temp, Size(ROI.width, ROI.height));
		temp.copyTo(DispImage(ROI));
	}

	// Create a new window, and show the Single Big Image
	namedWindow(title, 1);
	imshow(title, DispImage);
	waitKey();

	// End the number of arguments
	va_end(args);
}
int main()
{
	// Let's run all the steps of Canny edge detection, 
	// show results after each step and measure performance.

	Mat img1 = imread("Valve_original.PNG");

	clock_t start = clock();
	
	Mat img2 = make_greyscale(img1);
	clock_t made_greyscale_time = clock();
	
	Mat img3 = apply_gausian_filter(img2);
	clock_t applied_gausian_blur_time = clock();
	
	// This step takes all the time, the reason is computation of atan2 for gradient
	// angle is very slow operation. We don't actually need to compute the 
	// gradient angle, since all we want is to know in which range it falls into,
	// which can be done with simple (mathematical) vector operations. 
	// Yet I was too lazy to optimize this.
	Mat img4 = get_edge_gradient_values_and_suppress_non_maximums(img3);
	clock_t gradient_computation_time = clock();
	
	Mat img5 = make_edges_weak_or_strong_thresholding(img4);
	clock_t edge_thresholding_time = clock();
	
	Mat img6 = apply_edge_tracking_by_hysteresis(img5);
	clock_t edge_tracking_with_hysteresis_time = clock();

	display_multiple_images_in_one_window(
		"Steps of Canny edge detection", 5,
		img2, img3,
		img4, img5, img6);

	std::cout << "Performance metrics are (CPU time in ms (1/1000 sec)) " <<
		"converting to greyscale " << (made_greyscale_time - start) * 1000 / CLOCKS_PER_SEC << " ms " <<
		"Gausian blur " << (applied_gausian_blur_time - made_greyscale_time) * 1000 / CLOCKS_PER_SEC << " ms " <<
		"gradient computation and suppres non-max " << (gradient_computation_time - applied_gausian_blur_time) * 1000 / CLOCKS_PER_SEC << " ms " <<
		"edge thresholding " << (edge_thresholding_time - gradient_computation_time) * 1000 / CLOCKS_PER_SEC << " ms " <<
		"edge tracking with hysteresis " << (edge_tracking_with_hysteresis_time - edge_thresholding_time) * 1000 / CLOCKS_PER_SEC << " ms " <<
		std::endl;

	waitKey(0);
	return 0;
}