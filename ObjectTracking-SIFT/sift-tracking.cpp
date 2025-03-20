#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/features2d.hpp"


using namespace cv;
using namespace std;

int x_min = 36000;
int y_min = 36000;
int x_max = 0;
int y_max = 0;


std::string video_path="../../videos/plane.mp4";

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    // Access the frame passed as userdata
    Mat* framePtr = static_cast<Mat*>(userdata);

    if (event == EVENT_LBUTTONDOWN)
    {
        cout << "Right button of the  is clicked - position (" << x << ", " << y << ")" << endl;

        // Update x_min, y_min, x_max, y_max based on mouse click position
        x_min = min(x, x_min);
        y_min = min(y, y_min);
        x_max = max(x, x_max);
        y_max = max(y, y_max);

        // Draw rectangle on the frame
        rectangle(*framePtr, Point(x_min, y_min), Point(x_max, y_max), Scalar(255, 255, 0), 5, LINE_8);

        // Display the updated frame
        imshow("Video Player", *framePtr);
    }
}


void display_points(const Mat& descriptors_1, Ptr<Feature2D>& sift)
{

    // Open video file
    VideoCapture video(video_path);


    // Loop through video frames
    Mat frame;

    BFMatcher matcher(NORM_L2);


    while (video.read(frame)) {


        Mat gray_frame;

        resize(frame, frame, Size(), 0.4, 0.4, INTER_CUBIC);

        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

        // Detect keypoints
        vector<KeyPoint> keypoints_2;
        Mat descriptors_2;


        sift->detectAndCompute(gray_frame, Mat(), keypoints_2, descriptors_2);


        // Compare descriptors with saved_descriptors

        vector<vector<DMatch>> matches;
        matcher.knnMatch(descriptors_2, descriptors_1, matches, 5);


        // Draw points for matching keypoints
        for (int i = 0; i < matches.size(); ++i) {
            if (matches[i][0].distance < 0.7 * matches[i][1].distance) {
                Point2f pt = keypoints_2[matches[i][0].queryIdx].pt;
                // Draw point on the frame
                circle(frame, pt, 5, Scalar(0, 255, 0), 2);
            }
        }


        // Display frame
        imshow("Video", frame);

        // Break if 'Esc' key is pressed
        if (waitKey(30) == 27) {
            break;
        }
    }

    // Release video capture object
    video.release();
    destroyAllWindows();

}


void take_roi(void* frame)
{

    Mat* framePtr = static_cast<Mat*>(frame);

    // Create the rectangle (x, y, width, height)
    Rect roi(x_min, y_min, x_max - x_min, y_max - y_min);

    // Extract the ROI
    Mat image_roi = (*framePtr)(roi);

    // Do something with the extracted ROI, for example, display it
    imshow("ROI", image_roi);
    waitKey(0); // Wait indefinitely until a key is pressed

    // Convert ROI to grayscale
    Mat gray_roi;
    cvtColor(image_roi, gray_roi, COLOR_BGR2GRAY);

    // Create SIFT detector
    Ptr<Feature2D> sift = SIFT::create();

    // Detect keypoints
    vector<KeyPoint> keypoints_1;
    Mat descriptors_1;
    sift->detectAndCompute(gray_roi, Mat(), keypoints_1, descriptors_1);

    display_points(descriptors_1,sift);

}



int main() {

    VideoCapture video(video_path);

    // Check if the video file was opened successfully
    if (!video.isOpened()) {
        std::cout << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    // Create a window to display the video
    namedWindow("Video Player", WINDOW_NORMAL);
    resizeWindow("Video Player", Size(video.get(CAP_PROP_FRAME_WIDTH), video.get(CAP_PROP_FRAME_HEIGHT)));

    // Create Mat object for first frame 
    Mat frame;

    // Read the first frame
    if (!video.read(frame)) {
        std::cout << "Error: Could not read the first frame." << std::endl;
        return -1;
    }

    // Set the callback function and pass frame as userdata
    setMouseCallback("Video Player", CallBackFunc, &frame);

    // Display the first frame
    imshow("Video Player", frame);

    // Wait for a key press
    waitKey(0);

    // Release the VideoCapture object and close the display window
    video.release();
    destroyAllWindows();

    take_roi(&frame);

    return 0;
}