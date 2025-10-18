/*
 *  READ ALL COMMENTS IN THIS FILE TO UNDERSTAND THE CODE THAT HAS BEEN WRITTEN FOR YOU
 *
 *  This is a header file for armor_detector_node. It contains some useful constants and defines the subscriber node
 *  class.
 *
 *  Since this file implements a ROS node like camera_publisher_node does, you may want to refer to that file to get
 *  ideas for things you need to consider while writing your code.
 */

#include <iostream>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <vector>

// Constants
#define HUE_RANGE_LIMIT 30.0
#define SATURATION_LOWER_LIMIT 100.0
#define VALUE_LOWER_LIMIT 150.0
#define LIGHT_BAR_ANGLE_LIMIT 30.0
#define LIGHT_BAR_ASPECT_RATIO_LOWER_LIMIT 2.0
#define LIGHT_BAR_WIDTH_LOWER_LIMIT 2.0
#define LIGHT_BAR_HEIGHT_LOWER_LIMIT 5.0
#define ARMOR_ANGLE_DIFF_LIMIT 5.0
#define ARMOR_LIGHT_BAR_ASPECT_RATIO_RATIO_LIMIT 5.0
#define ARMOR_Y_DIFF_LIMIT 1.5
#define ARMOR_HEIGHT_RATIO_LIMIT 1.5
#define ARMOR_ASPECT_RATIO_LIMIT 2.5

// Define the Red HSV color range for segmentation 
cv::Scalar lowerHSV(0, 120, 70);   // Lower bound of HSV
cv::Scalar upperHSV(10, 255, 255); // Upper bound of HSV
cv::Scalar lowerHSV2(170, 120, 70);   // Lower bound of HSV
cv::Scalar upperHSV2(179, 255, 255); // Upper bound of HSV

class ArmorDetectorNode : public rclcpp::Node{
    public:
        ArmorDetectorNode();
    private:
        // This is a subscription. Since our node contains it, we consider our node to be a subscriber.
        // Don't worry too much about the types, but basically we define this subscription as one that fetches images.
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

        // We want to track the frame number for debugging purposes
        int frame_count;
        
        // This callback will be how this node actually reads images published to the topic.
        // You will learn more about that in armor_detector_node.cpp
        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

        // This method will display processed frames. How else are we going to judge what your code does?
        void show_frame(cv::Mat &frame);

        // Armor detection methods
        std::vector<cv::RotatedRect> search(cv::Mat &frame, cv::Scalar lowerHSV, cv::Scalar upperHSV, cv::Scalar lowerHSV2, cv::Scalar upperHSV2);
        bool is_light_bar(cv::RotatedRect &rect);
        bool is_armor(cv::RotatedRect &left_rect, cv::RotatedRect &right_rect);
        void draw_rotated_rect(cv::Mat &frame, cv::RotatedRect &rect);
        std::vector<cv::Point2f> rect_to_point(cv::RotatedRect &rect);
};