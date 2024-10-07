#include <opencv2/opencv.hpp>
#include <iostream>
#include "opencv2/features2d.hpp"

using namespace std;

// coordinates of rectangle (user select these)
int x_min = 36000;
int y_min = 36000;
int x_max = 0;
int y_max = 0;

// center points of selected coordinates
int center_points[2];

// frame variables
cv::Mat frame, hsv_roi, mask, hsv_image;

string video_path="/home/omer/tracking-projects/opencv-projects-c++/videos/plane (1).mp4";

// create VideoCapture object
cv::VideoCapture video(video_path);


// hue dict
std::map<std::string, std::vector<std::vector<int>>> hue_dict = {
        {"red", {{0, 100, 100}, {10, 255, 255}}},
        {"orange", {{10, 100, 100}, {20, 255, 255}}},
        {"yellow", {{20, 100, 100}, {30, 255, 255}}},
        {"green", {{50, 100, 100}, {70, 255, 255}}},
        {"blue", {{110, 50, 50}, {130, 255, 255}}},
        {"violet", {{140, 50, 50}, {170, 255, 255}}}
};


// this function check mouse activity , user click screen and this func draw rectangle
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    // Access the frame passed as userdata , here static_cast creates Mat pointer 
    cv::Mat* framePtr = static_cast<cv::Mat*>(userdata);


    if (event == cv::EVENT_LBUTTONDOWN)
    {
        cout << "Right button of the  is clicked - position (" << x << ", " << y << ")" << endl;

        // Update x_min, y_min, x_max, y_max based on mouse click position
        x_min = min(x, x_min);
        y_min = min(y, y_min);
        x_max = max(x, x_max);
        y_max = max(y, y_max);

        // Draw rectangle on the frame
        rectangle(*framePtr, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar(255, 255, 0), 5, cv::LINE_8);

        // Display the updated frame
        imshow("Video Player", *framePtr);
    }
}

void track_object(const std::vector<int>& first_range, const std::vector<int>& second_range)
{


    // rectangle that user choiced
    cv::Rect track_window(x_min, y_min, x_max - x_min, y_max - y_min);

    // set up the ROI for tracking
    cv::Mat hsv_roi = hsv_image(track_window);

    // creates a binary mask where each pixel that falls within the range defined by lower_bound and upper_bound 
    inRange(hsv_roi, first_range, second_range, mask);

    float range_[] = { 0, 180 };
    const float* range[] = { range_ };
    cv::Mat roi_hist;
    int histSize[] = { 180 };
    int channels[] = { 0 };

    // calculates the histogram
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);

    // normalizes the histogram to a specified range [0,255] in this case
    normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);

    // Setup the termination criteria, either 10 iteration or move by at least 1 pt
    cv::TermCriteria term_crit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);


    cv::VideoCapture video(video_path);

    while (true)
    {
        cv::Mat hsv, dst;

        video.read(frame);

        if (frame.empty())
            break;

        // convert HSV
        cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        //  calculate the back projection of an image using a histogram. 
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

        // apply meanshift to get the new location
        meanShift(dst, track_window, term_crit);

        // Draw it on image
        rectangle(frame, track_window, 255, 2);

        imshow("img2", frame);

        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }


}



void extract_color(int& width, int& height)
{

    // convert hsv
    cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

    // take values of pixel in the center --> HSV : hue,color,value(intensity)
    cv::Vec3b pixel_value = hsv_image.at<cv::Vec3b>(height, width);

    // values of hsv , static_cast<int> convert values to integer
    std::cout << "HUE: " << static_cast<int>(pixel_value[0]) << std::endl;
    std::cout << "SATURATION: " << static_cast<int>(pixel_value[1]) << std::endl;
    std::cout << "VALUE: " << static_cast<int>(pixel_value[2]) << std::endl;

    // take hue value it is in 0 index , HSV--> h:0 
    int hue_value;
    hue_value = pixel_value[0];

    // color is decided with respect to hue value
    string color = "";
    if (hue_value < 5)
        color = "red";
    else if (hue_value < 22)
        color = "orange";
    else if (hue_value < 33)
        color = "yellow";
    else if (hue_value < 78)
        color = "green";
    else if (hue_value < 131)
        color = "blue";
    else if (hue_value < 170)
        color = "violet";
    else
        color = "red";

    cout << color << endl;


    // above variable "color" is decided color , from "hue_dict" take range values , one for lower boundary , one for upper boundary  
    vector<vector<int>> color_ranges = hue_dict[color];
    vector<int> first_range = color_ranges[0];
    vector<int> second_range = color_ranges[1];

    // send range values to the track_object function
    track_object(first_range, second_range);

}


int main() {

    // create window
    namedWindow("Video Player", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Player", cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT)));

    // Read the first frame
    if (!video.read(frame)) {
        std::cout << "Error: Could not read the first frame." << std::endl;
        return -1;
    }


    // Display the first frame
    imshow("Video Player", frame);

    // Set the callback function and pass frame as userdata
    setMouseCallback("Video Player", CallBackFunc, &frame);

    // Wait for a key press
    cv::waitKey(0);

    // Release the VideoCapture object and close the display window
    video.release();
    cv::destroyAllWindows();

    // x_min,x_max,y_min,y_max values are assigned in mouse callback function
    center_points[0] = x_min + ((x_max - x_min) / 2);
    center_points[1] = y_max + ((y_min - y_max) / 2);

    // send center_points to the extract_color function 
    extract_color(center_points[0], center_points[1]);


    return 0;
}