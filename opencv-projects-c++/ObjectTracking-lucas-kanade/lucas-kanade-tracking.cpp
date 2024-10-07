#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>

// Global variables for the ROI coordinates
int x_min = 36000, y_min = 36000, x_max = 0, y_max = 0;
bool go = false;
cv::Mat frame;

std::string video_path ="/home/omer/tracking-projects/opencv-projects-c++/videos/bird1.mp4"; // Replace with your video path

// Function to handle mouse events
void coordinat_chooser(int event, int x, int y, int flags, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Update the ROI coordinates
        x_min = std::min(x, x_min);
        y_min = std::min(y, y_min);
        x_max = std::max(x, x_max);
        y_max = std::max(y, y_max);

        // Draw the rectangle
        cv::rectangle(frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(0, 255, 0), 1);
    } else if (event == cv::EVENT_MBUTTONDOWN) {
        // Reset coordinates if the middle button is clicked
        std::cout << "Reset coordinate data" << std::endl;
        x_min = y_min = 36000;
        x_max = y_max = 0;
    }
}


void track_object()
    {
    
    // Region of Interest (ROI) after user selects it --> here frame is first frame, therefore I defined it globally so I can use it here
    cv::Mat roi_image = frame(cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min));
    cv::Mat roi_gray;
    cv::cvtColor(roi_image, roi_gray, cv::COLOR_BGR2GRAY);

    // Convert the first frame to grayscale for corner detection
    cv::Mat first_gray;
    cv::cvtColor(frame, first_gray, cv::COLOR_BGR2GRAY);

    // Parameters for goodFeaturesToTrack
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(first_gray, points, 20, 0.2, 7, cv::Mat(), 7);

    // Filter the detected points to find one within the bounding box
    cv::Point2f selected_point(-1, -1);
    for (const auto& point : points) {
        if (point.x >= x_min && point.x <= x_max && point.y >= y_min && point.y <= y_max) {
            selected_point = point;
            break;
        }
    }

    cv::VideoCapture video(video_path);

    // If a point is found, we start tracking
    if (selected_point.x != -1 && selected_point.y != -1) {
        std::vector<cv::Point2f> p0 = {selected_point};
        cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());

        cv::Mat old_gray = first_gray.clone();

        while (true) {
            cv::Mat frame, frame_gray;
            if (!video.read(frame)) break;

            cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> p1;
            std::vector<uchar> st;
            std::vector<float> err;

            if (!p0.empty()) {
                // Calculate optical flow
                cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, st, err);

                std::vector<cv::Point2f> good_new, good_old;
                for (size_t i = 0; i < p1.size(); i++) {
                    if (st[i] == 1) {
                        good_new.push_back(p1[i]);
                        good_old.push_back(p0[i]);
                    }
                }

                if (!good_new.empty()) {
                    // Draw the tracks
                    for (size_t i = 0; i < good_new.size(); i++) {
                        cv::line(mask, good_new[i], good_old[i], cv::Scalar(0, 255, 0), 2);
                        cv::circle(frame, good_new[i], 5, cv::Scalar(0, 255, 0), -1);
                    }

                    // Overlay the current frame and mask
                    cv::Mat img = frame + mask;

                  
                    // Show the frame with tracking
                    cv::imshow("Lucas-Kanade Tracking", img);
                }

                old_gray = frame_gray.clone();
                p0 = good_new;
            }

            if (cv::waitKey(30) == 27) break;
        }
    } else {
        std::cout << "No point found inside the ROI." << std::endl;
    }

    }


int main() {
    
    cv::VideoCapture video(video_path);

    if (!video.isOpened()) {
        std::cerr << "Error opening video!" << std::endl;
        return -1;
    }

    // Read the first frame
    video.read(frame);
    if (frame.empty()) {
        std::cerr << "Error reading first frame!" << std::endl;
        return -1;
    }

    // Set the mouse callback to select the ROI
    cv::namedWindow("coordinate_screen");
    cv::setMouseCallback("coordinate_screen", coordinat_chooser);

    // Show the first frame and let the user draw the rectangle
    while (true) {
        cv::imshow("coordinate_screen", frame);

        int k = cv::waitKey(5) & 0xFF;
        if (k == 27) {  // Press ESC to break
            cv::destroyAllWindows();
            break;
        }
    }

    track_object();

    
    return 0;
}
