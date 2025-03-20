#include <opencv2/opencv.hpp>
#include <iostream>

// path to video
std::string video_path="../../videos/bird.mp4";

// Create KNN background subtractor
cv::Ptr<cv::BackgroundSubtractor> KNN_subtractor = cv::createBackgroundSubtractorKNN(true);

// Create MOG2 background subtractor
cv::Ptr<cv::BackgroundSubtractor> MOG2_subtractor = cv::createBackgroundSubtractorMOG2(true);


int main() {

    // Choose your subtractor (here using MOG2)
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor = MOG2_subtractor;


    // Open the video file
    cv::VideoCapture video(video_path);
    cv::Mat frame, foreground_mask, threshold_img, dilated;

    while (true) {
        bool ret = video.read(frame);
        if (!ret) {
            std::cout << "End of video." << std::endl;
            break;
        }

        // Apply the background subtractor to get the foreground mask
        bg_subtractor->apply(frame, foreground_mask);

        // Apply threshold to create a binary image
        cv::threshold(foreground_mask, threshold_img, 120, 255, cv::THRESH_BINARY);

        // Dilate the threshold image to thicken the regions of interest
        cv::dilate(threshold_img, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)), cv::Point(-1, -1), 1);

        // Find contours in the dilated image
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw bounding boxes for contours that exceed a certain area threshold
        for (size_t i = 0; i < contours.size(); i++) {
            if (cv::contourArea(contours[i]) > 150) {
                cv::Rect bounding_box = cv::boundingRect(contours[i]);
                cv::rectangle(frame, bounding_box, cv::Scalar(255, 255, 0), 2);
            }
        }

        // Show the different outputs
        cv::imshow("Subtractor", foreground_mask);
        cv::imshow("Threshold", threshold_img);
        cv::imshow("Detection", frame);

        // Exit when 'ESC' is pressed
        if (cv::waitKey(30) == 27) break;
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}