/* Vision Based Rocket Tracking
 * Technical University of Denmark
 * Part of Course 30330
 * Fall & Winter 2015
 * By;
 * Christian Haaning (s092847)
 * Kenneth Chabert Nielsen (s103062)
 * Mads Bolmgren (s103728)
 * Fabian Krogsbøll Holt (s113020)
 *
 * Using OpenCV to Track a suborbital rocket
 */

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <math.h>
#include <Windows.h>

#define WIDTH_MULTIPLIER 5
#define MAX_SIZE 800
#define MAX_WIDTH 255*WIDTH_MULTIPLIER

using namespace std;
using namespace cv;
// Function declaration
Mat ConvToGray(Mat frame);
Mat MakeHistogram(Mat frame);
Mat ConvToBin(Mat frame, double threshold);
Point trackCanny(Mat frame);
Mat dilate(Mat frame);
double millisecondsNow();
void putTxt(Mat mat, String string, Point point, double font_scale, int ftype);

// Global variables
Point midpoint;

int main(int argc, char* argv[])
{
	// Variables
	Mat cam, gray_cam, canny, binary_cam;
	char key;
	int current_time, max_time, set_time = 1;
	double start, end;
	Scalar DrawC;
	string file = "";

	// Set file to track rocket in
	cout << "Filename:";
	getline(cin, file);

	// Load file
	VideoCapture cap(file);
	cap.set(CAP_PROP_POS_FRAMES, 190); // Jump to frame 190 (nothing but noise before)
	if (cap.isOpened())
	{
		while (1)
		{
			set_time++;
			start = millisecondsNow(); // Time measurement for determination of processing time.
			cap >> cam;
			if (cam.empty()) // Restart if movie ended
			{
				set_time = 190;
				cap.set(CAP_PROP_POS_FRAMES, set_time);
				cap >> cam;
			}
			current_time = cap.get(CAP_PROP_POS_MSEC);
			max_time = cap.get(CAP_PROP_FRAME_COUNT);
			
			// Convert to grayscale
			gray_cam = ConvToGray(cam); 

			// Convert to binary map
			binary_cam = ConvToBin(gray_cam, 0.4);

			// If no centroid was found in the binary map, use Canny edge detection to track
			if (midpoint.x == 0 && midpoint.y == 0)
			{
				Canny(gray_cam, canny, 0.3 * 255, 0.3 * 255);
				midpoint = trackCanny(canny);
				putTxt(cam,"Canny", Point(5, 10), 0.8, 1);
				DrawC = CV_RGB(0, 255, 0); // Green dot if Canny edge
			}
			else
			{
				putTxt(cam,"Binary", Point(5, 10), 0.8, 1);
				DrawC = CV_RGB(255, 0, 0); // Red dot if Binary mid
			}
			// Draw indicator of the center of mass of the tracked rocket
			circle(cam, midpoint, 2, DrawC,3);

			end = millisecondsNow(); // Time measurement for determination of processing time.
			imshow("Cam", cam); // Show the frame and the tracking

			
			key = waitKey(1);

			// End the tracking by hitting escape
			if (char(key) == 27) break;
			
			if (char(key) == 112){
				cout << "Type frame to look at, Frame:";
				cin >> set_time;
				cout << endl;
				if (set_time > max_time) set_time = 1;
				cap.set(CAP_PROP_POS_FRAMES, set_time);
			}
		}
	}
	destroyAllWindows();
	cap.release();
	return 0;
}


// Convert RGB to Grayscale using luminosity method
Mat ConvToGray(Mat frame)
{
	Mat output = Mat(frame.rows,frame.cols, CV_8U);
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			output.data[output.step[0] * i + output.step[1] * j + 0] = 0.25*frame.data[frame.step[0] * i + frame.step[1] * j + 2] + 0.65*frame.data[frame.step[0] * i + frame.step[1] * j + 1] + 0.1*frame.data[frame.step[0] * i + frame.step[1] * j + 0];
			
		}
	}
	return output;
}

// Make histogram
Mat MakeHistogram(Mat frame)
{
	Mat canvas = Mat(MAX_SIZE+50, MAX_WIDTH+50, CV_8UC3);
	Point P1, P2;
	Scalar C1;
	int histogram[256] = { 0 };
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			histogram[frame.data[frame.step[0] * i + frame.step[1] * j]] += 1;
		}
	}
	double max = 0;
	for (int i = 0; i < 256; i++)
	{
		if (histogram[i] > max)	max = (double)histogram[i];
	}

	P1.x = 0;
	P1.y = 0;
	P2.x = MAX_WIDTH + 50;
	P2.y = MAX_SIZE + 50;
	C1 = CV_RGB(155, 155, 155);
	// Setting the background as a gradient
	for (int i = 0; i <= 270; i++) line(canvas, Point(i*WIDTH_MULTIPLIER, 0), Point(i*WIDTH_MULTIPLIER, MAX_SIZE + 50), CV_RGB(0, 200 - i, 200 - i), WIDTH_MULTIPLIER);

	for (int i = 0; i < 256; i++){
		P1.x = i * WIDTH_MULTIPLIER + 25;
		P1.y = MAX_SIZE + 25;
		P2.x = P1.x;
		P2.y = MAX_SIZE + 25 - (int)((double)histogram[i] / max*MAX_SIZE);
		C1 = CV_RGB(i, i, i);
		line(canvas, P1, P2, C1, 2);
		//printf("%d ", P2.y);				

		String txt;
		if (i % 5 == 0)
		{
			line(canvas, Point(P1.x, MAX_SIZE + 25), Point(P1.x, MAX_SIZE + 30), CV_RGB(0, 0, 0), 2);
			line(canvas, Point(P1.x, MAX_SIZE + 25), Point(P1.x, MAX_SIZE + 30), CV_RGB(255, 255, 255), 1);
			txt = to_string(i);
			putTxt(canvas, txt, Point(P1.x - txt.length() * 3, MAX_SIZE + 40),0.3,1);
		}
	}
	return canvas;
}


// Convert Grayscale to Binary using threshold
// Determine center of the binary image and save in midpoint
Mat ConvToBin(Mat frame, double threshold)
{
	Mat output = Mat(frame.rows, frame.cols, CV_8U);
	int centerIndex = 0, midRow = 0, midCol = 0;
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			if (frame.data[frame.step[0] * i + frame.step[1] * j] >= threshold * 255)
			{
				output.data[output.step[0] * i + output.step[1] * j] = 255;
			}
			else
			{
				output.data[output.step[0] * i + output.step[1] * j] = 0;
				centerIndex += 1;
				midRow += i;
				midCol += j;
			}
		}
	}
	if (centerIndex != 0)
	{
		midpoint.x = midCol / centerIndex;
		midpoint.y = midRow / centerIndex;
	}
	else
	{
		midpoint.x = 0;
		midpoint.y = 0;

	}
	return output;
}

// Find the first (top) point from the Edge detection.
Point trackCanny(Mat frame)
{
	Point output;
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			if (frame.data[frame.step[0] * i + frame.step[1] * j] == 255)
			{
				output.x = j;
				output.y = i;
				break;
			}
		}
		if (output.y == i && output.y != 0) break;
	}
	return output;
}

// Dilate binary mapping using 3x3 kernel
Mat dilate(Mat frame)
{
	Mat output = Mat(frame.rows, frame.cols, CV_8U);
		for (int i = 0; i < frame.rows; i++)
		{
			for (int j = 0; j < frame.cols; j++)
			{
				if (frame.data[frame.step[0]*i + frame.step[1]*j] == 0)
				{
					if (i > 0 && frame.data[frame.step[0]*(i - 1) + frame.step[1]*j] == 255) output.data[output.step[0]*(i - 1) + output.step[1]*j] = 2;
					if (i > 0 && frame.data[frame.step[0] * i + frame.step[1] * (j - 1)] == 255) output.data[output.step[0] * i + output.step[1] * (j-1)] = 2;
					if (i + 1 < frame.rows && frame.data[frame.step[0] * (i + 1) + frame.step[1] * j] == 255) output.data[output.step[0] * (i + 1) + output.step[1] * j] = 2;
					if (j + 1 < frame.cols && frame.data[frame.step[0] * i + frame.step[1] * (j+1)] == 255) output.data[output.step[0] * i + output.step[1] * (j+1)] = 2;
					if (i > 0 && j > 0 && frame.data[frame.step[0] * (i - 1) + frame.step[1] * (j - 1)] == 255) output.data[output.step[0] * (i - 1) + output.step[1] * (j-1)] = 2;
					if (i + 1 < frame.rows && j + 1 < frame.cols && frame.data[frame.step[0] * (i + 1) + frame.step[1] * (j + 1)] == 255) output.data[output.step[0] * (i + 1) + output.step[1] * (j + 1)] = 2;
					if (i > 0 && j + 1 < frame.cols && frame.data[frame.step[0] * (i - 1) + frame.step[1] * (j + 1)] == 255) output.data[output.step[0] * (i - 1) + output.step[1] * (j + 1)] = 2;
					if (i + 1 < frame.rows && j > 0 && frame.data[frame.step[0] * (i + 1) + frame.step[1] * (j - 1)] == 255) output.data[output.step[0] * (i + 1) + output.step[1] * (j - 1)] = 2;

					output.data[output.step[0] * i + output.step[1] * j] = 2;
				}
			}
		}
		for (int i = 0; i < output.rows; i++)
		{
			for (int j = 0; j < output.cols; j++)
			{
				if (output.data[output.step[0] * i + output.step[1] * j] == 2)
				{
					output.data[output.step[0] * i + output.step[1] * j] = 0;
				}
				else
					output.data[output.step[0] * i + output.step[1] * j] = 255;
			}
		}
		return output;
}


// Calculate the Image moments.
double calcMoment(int c_row, int c_col, Mat frame, int p, int q)
{
	double my = 0;
	int color = 0;
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++){
			if (frame.data[frame.step[0] * i + frame.step[1] * j] == 0) color = 1;
			else color = 0;
			my += (pow((i - c_row), q) * pow((j - c_col), p))*color;
		}
	}
	return my;
}

double invMoment(int c_row, int c_col, Mat frame, int p, int q)
{
	return calcMoment(c_row, c_col, frame, p, q) / (pow(calcMoment(c_row, c_col, frame, 0, 0), (1 + (p + q) / 2)));
}

// Function for determining processing time of functions.
double millisecondsNow()
{
	LARGE_INTEGER s_frequency;
	BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
	if (s_use_qpc) {
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		return (1000.0 * now.QuadPart) / s_frequency.QuadPart;
	}
	else {
		return GetTickCount();
	}
}

// Prettify text in OpenCV frame.
void putTxt(Mat mat, String string, Point point, double font_scale, int ftype)
{
	if (ftype > 7) ftype = 1;
	putText(mat, string, point, ftype, font_scale, CV_RGB(0, 0, 0), 2);
	putText(mat, string, point, ftype, font_scale, CV_RGB(255, 255, 255));
}