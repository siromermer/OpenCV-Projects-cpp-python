#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Global variables for rectangle coordinates
int x_min = 36000, y_min = 36000, x_max = 0, y_max = 0;
Mat frame;

// Function to handle mouse events for selecting ROI
void coordinateChooser(int event, int x, int y, int flags, void* param) {
    if (event == EVENT_LBUTTONDOWN) {
        // Update min and max coordinates
        x_min = min(x, x_min);
        y_min = min(y, y_min);
        x_max = max(x, x_max);
        y_max = max(y, y_max);

        // Draw rectangle on the frame
        rectangle(frame, Point(x_min, y_min), Point(x_max, y_max), Scalar(0, 255, 0), 1);
    }

    if (event == EVENT_MBUTTONDOWN) {
        // Reset coordinates if middle mouse button is pressed
        cout << "Reset coordinate data" << endl;
        x_min = 36000; y_min = 36000; x_max = 0; y_max = 0;
    }
}

// Function to detect target using FAST and BRIEF
Point detectTargetFast(Mat& frame, Ptr<FastFeatureDetector>& fast, Ptr<xfeatures2d::BriefDescriptorExtractor>& brief,
                      const vector<KeyPoint>& keypoints_1, const Mat& descriptors_1, BFMatcher& bf) {
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    // Detect keypoints in the current frame
    vector<KeyPoint> keypoints_2;
    fast->detect(frame_gray, keypoints_2);

    // Compute descriptors
    Mat descriptors_2;
    brief->compute(frame_gray, keypoints_2, descriptors_2);

    int avg_x = 0, avg_y = 0;
    if (!descriptors_2.empty()) {
        vector<DMatch> matches;
        bf.match(descriptors_1, descriptors_2, matches);

        if (!matches.empty()) {
            int sum_x = 0, sum_y = 0, match_count = 0;

            for (const auto& match : matches) {
                // Get the coordinates of the matched keypoint
                Point2f pt2 = keypoints_2[match.trainIdx].pt;
                sum_x += pt2.x;
                sum_y += pt2.y;
                match_count++;
            }

            avg_x = sum_x / match_count;
            avg_y = sum_y / match_count;
        }
    }
    return Point(avg_x, avg_y);
}

int main() {
    // Load the video
    string video_path = "/home/omer/tracking-projects/opencv-projects-c++/videos/plane (1).mp4";
    VideoCapture video(video_path);

    if (!video.isOpened()) {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }

    // Read the first frame
    video >> frame;
    if (frame.empty()) {
        cerr << "Error: Could not read frame." << endl;
        return -1;
    }

    // Set up mouse callback
    namedWindow("coordinate_screen", 1);
    setMouseCallback("coordinate_screen", coordinateChooser);

    while (true) {
        imshow("coordinate_screen", frame);
        char k = waitKey(5);
        if (k == 27) { // Press ESC to break
            destroyAllWindows();
            break;
        }
    }

    // Extract ROI from the selected rectangle
    Mat roi_image = frame(Rect(x_min + 2, y_min + 2, x_max - x_min - 4, y_max - y_min - 4));
    Mat roi_gray;
    cvtColor(roi_image, roi_gray, COLOR_BGR2GRAY);

    // Initialize FAST and BRIEF
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(1);
    Ptr<xfeatures2d::BriefDescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create();

    // Detect keypoints and compute descriptors for the ROI
    vector<KeyPoint> keypoints_1;
    fast->detect(roi_gray, keypoints_1);
    Mat descriptors_1;
    brief->compute(roi_gray, keypoints_1, descriptors_1);

    // Initialize Kalman filter
    KalmanFilter kalman(4, 2, 0);
    kalman.measurementMatrix = (Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
    kalman.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
    setIdentity(kalman.processNoiseCov, Scalar::all(0.03));
    setIdentity(kalman.measurementNoiseCov, Scalar::all(0.5));

    // Start processing the video
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }

    BFMatcher bf(NORM_HAMMING);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Predict the new position using Kalman filter
        Mat prediction = kalman.predict();
        Point predicted_pt(prediction.at<float>(0), prediction.at<float>(1));

        // Detect the target in the current frame
        Point measured_pt = detectTargetFast(frame, fast, brief, keypoints_1, descriptors_1, bf);

        if (measured_pt.x != 0 && measured_pt.y != 0) {
            // Correct the Kalman filter with the measurement
            Mat_<float> measurement(2, 1);
            measurement(0) = measured_pt.x;
            measurement(1) = measured_pt.y;
            kalman.correct(measurement);

            // Draw the measured position
            circle(frame, measured_pt, 6, Scalar(0, 255, 0), 2); // Green circle for the measured position
        }

        // Draw the predicted position
        circle(frame, predicted_pt, 8, Scalar(0, 0, 255), 2); // Red circle for the predicted position

        // Display the frame
        imshow("Kalman Ball Tracking", frame);

        // Break on 'q' key press
        if (waitKey(30) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
