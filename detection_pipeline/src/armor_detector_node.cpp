// 
//  *  READ ALL COMMENTS IN THIS FILE TO UNDERSTAND THE CODE THAT HAS BEEN WRITTEN FOR YOU
//  *
//  *  This file will create a subscriber node that reads images from the topic where camera_publisher_node publishes
//  *  images. It will contain the functionality necessary for locating armor plates within a given image.
//  *
//  *  Since this file implements a ROS node like camera_publisher_node does, you may want to refer to that file to get
//  *  ideas for things you need to consider while writing your code.
//  */





#include "../include/armor_detector/armor_detector_node.hpp"
#include <algorithm>

// helpers to keep angle/size handling consistent
static inline float normalize_angle_deg(float a) {
    // Map to [0,180)
    while (a < 0.f) a += 180.f;
    while (a >= 180.f) a -= 180.f;
    return a;
}

static inline float long_side_angle_deg(const cv::RotatedRect &r) {
    // OpenCV angle refers to the rectangle's width axis vs +x.
    // We want the angle of the LONG axis.
    float a = r.angle;
    if (r.size.width < r.size.height) {
        // Height is longer, so add 90 to get the long axis angle
        a = normalize_angle_deg(a + 90.f);
    } else {
        a = normalize_angle_deg(a);
    }
    return a; // [0,180)
}

static inline float long_side(const cv::RotatedRect &r) {
    return std::max(r.size.width, r.size.height);
}

static inline float short_side(const cv::RotatedRect &r) {
    return std::min(r.size.width, r.size.height);
}



ArmorDetectorNode::ArmorDetectorNode() : rclcpp::Node("armor_detector_node"), frame_count(0)
{
    const std::string topic = this->declare_parameter<std::string>("image_topic", "camera/image_raw");
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        topic, rclcpp::QoS(10),
        std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "ArmorDetectorNode subscribed to '%s'", topic.c_str());
}



void ArmorDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    cv::Mat frame;
    
    try {
        frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    std::vector<cv::RotatedRect> armors = search(frame, lowerHSV, upperHSV, lowerHSV2, upperHSV2);
    frame_count++;

    if (armors.size() == 2)
    {
        auto p0 = rect_to_point(armors[0]);
        auto p1 = rect_to_point(armors[1]);
        std::cout << frame_count << "," << p0[0] << "," << p0[1] << "," << p1[0] << "," << p1[1] << std::endl;

        draw_rotated_rect(frame, armors[0]);
        draw_rotated_rect(frame, armors[1]);
    }
    else
    {
        std::cout << frame_count << "," << "no armor found" << std::endl;
    }

    if (frame_count % 5 == 0)
    {
        show_frame(frame);
    }
}



void ArmorDetectorNode::show_frame(cv::Mat &frame)
{
    std::vector<uchar> buf;
    cv::resize(frame, frame, cv::Size(640, 480));
    cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 20});
    cv::imshow("Detection Frame", cv::imdecode(buf, cv::IMREAD_COLOR));
    if (cv::waitKey(1) == 27)
    {
        cv::destroyAllWindows();
        rclcpp::shutdown();
    }
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}


std::vector<cv::RotatedRect> ArmorDetectorNode::search(cv::Mat& frame, cv::Scalar lowerHSV, cv::Scalar upperHSV, 
                                                        cv::Scalar lowerHSV2, cv::Scalar upperHSV2) {
    // 1) Image Preprocessing
    cv::Mat blurred, hsv;
    cv::GaussianBlur(frame, blurred, cv::Size(9,9), 0); //remove noise
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV); // Convert to HSV color space

    // 2) Color segmentation - more aggressive for bright reds
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lowerHSV, upperHSV, mask1);
    cv::inRange(hsv, lowerHSV2, upperHSV2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // Morphological operations - clean up noise but preserve shape
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,5)); // Vertical bias
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel_open);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel_close);

    // 2.5) Try finding contours directly from mask first, then edges if needed
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // If we don't get enough contours, try edge-based approach
    if (contours.size() < 2) {
        cv::Mat edges;
        cv::Canny(mask, edges, 50, 150);
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    }

    // 3) Contour Filtering - find valid light bars
    std::vector<cv::RotatedRect> light_bars;
    light_bars.reserve(contours.size());
    
    for (const auto &c : contours) {
        // Need at least 5 points for minAreaRect
        if (c.size() < 5) continue;
        
        // Check minimum area
        double area = cv::contourArea(c);
        if (area < 15.0) continue; // Filter tiny regions
        
        cv::RotatedRect rr = cv::minAreaRect(c);
        
        // Additional sanity checks before is_light_bar
        float long_len = long_side(rr);
        float short_len = short_side(rr);
        if (long_len < 5.0 || short_len < 1.0) continue;
        
        if (is_light_bar(rr)) {
            light_bars.push_back(rr);
        }
    }
    
    if (light_bars.size() < 2) return {};

    // 4) Pairing → choose best armor candidate
    std::vector<cv::RotatedRect> best_pair;
    double best_score = 1e9;

    // Sort by x for stable left/right identification
    std::sort(light_bars.begin(), light_bars.end(),
              [](const cv::RotatedRect& a, const cv::RotatedRect& b){ return a.center.x < b.center.x; });

    for (size_t i = 0; i < light_bars.size(); ++i) {
        for (size_t j = i + 1; j < light_bars.size(); ++j) {
            auto L = light_bars[i];
            auto R = light_bars[j];
            
            if (!is_armor(L, R)) continue;

            // Improved scoring: favor parallel, aligned, and properly spaced bars
            float angL = long_side_angle_deg(L);
            float angR = long_side_angle_deg(R);
            float dang = std::min(std::abs(angL - angR), 180.f - std::abs(angL - angR));
            
            float avg_height = (long_side(L) + long_side(R)) / 2.f;
            float y_diff = std::abs(L.center.y - R.center.y);
            float y_norm = y_diff / std::max(1.f, avg_height);

            float width = std::abs(R.center.x - L.center.x);
            float armor_ar = width / std::max(1.f, avg_height);
            
            // Score function: lower is better
            // Penalize: angle difference, y misalignment, deviation from ideal aspect ratio
            double score = dang * 2.0 + y_norm * 10.0 + std::abs(armor_ar - 1.8f) * 5.0;

            if (score < best_score) {
                best_score = score;
                best_pair = {L, R};
            }
        }
    }
    
    return (best_pair.size() == 2) ? best_pair : std::vector<cv::RotatedRect>{};
}



void ArmorDetectorNode::draw_rotated_rect(cv::Mat &frame, cv::RotatedRect &rect)
{
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
}



bool ArmorDetectorNode::is_light_bar(cv::RotatedRect &rect)
{
    float long_len = long_side(rect);
    float short_len = short_side(rect);
    
    // Verify minimum dimensions
    if (short_len < LIGHT_BAR_WIDTH_LOWER_LIMIT) return false;
    if (long_len < LIGHT_BAR_HEIGHT_LOWER_LIMIT) return false;
    
    // Verify aspect ratio (should be tall and thin)
    float aspect_ratio = long_len / std::max(0.1f, short_len);
    if (aspect_ratio < LIGHT_BAR_ASPECT_RATIO_LOWER_LIMIT) return false;
    
    // Verify angle - light bars should be roughly vertical
    // The long axis should point up/down, i.e., angle near 90° or 270° (which normalizes to 90°)
    float long_angle = long_side_angle_deg(rect);
    
    // Check if angle is close to 90° (vertical)
    float angle_from_vertical = std::abs(long_angle - 90.f);
    if (angle_from_vertical > 90.f) {
        angle_from_vertical = 180.f - angle_from_vertical;
    }
    
    if (angle_from_vertical > LIGHT_BAR_ANGLE_LIMIT) return false;
    
    return true;
}



bool ArmorDetectorNode::is_armor(cv::RotatedRect &left_rect, cv::RotatedRect &right_rect)
{
    // Ensure left/right are correctly ordered
    if (left_rect.center.x > right_rect.center.x) {
        std::swap(left_rect, right_rect);
    }

    float aL = long_side_angle_deg(left_rect);
    float aR = long_side_angle_deg(right_rect);

    // 1) Parallelism check
    float angle_diff = std::min(std::abs(aL - aR), 180.f - std::abs(aL - aR));
    if (angle_diff > ARMOR_ANGLE_DIFF_LIMIT) return false;

    // 2) Aspect ratio similarity
    float arL = long_side(left_rect) / std::max(1e-3f, short_side(left_rect));
    float arR = long_side(right_rect) / std::max(1e-3f, short_side(right_rect));
    float ar_ratio = std::max(arL, arR) / std::max(1e-3f, std::min(arL, arR));
    if (ar_ratio > ARMOR_LIGHT_BAR_ASPECT_RATIO_RATIO_LIMIT) return false;

    // 3) Y-alignment check
    float avg_h = (long_side(left_rect) + long_side(right_rect)) / 2.f;
    float y_norm = std::abs(left_rect.center.y - right_rect.center.y) / std::max(1e-3f, avg_h);
    if (y_norm > ARMOR_Y_DIFF_LIMIT) return false;

    // 4) Height similarity
    float hL = long_side(left_rect);
    float hR = long_side(right_rect);
    float h_ratio = std::max(hL, hR) / std::max(1e-3f, std::min(hL, hR));
    if (h_ratio > ARMOR_HEIGHT_RATIO_LIMIT) return false;

    // 5) Armor plate aspect ratio
    float width = std::abs(right_rect.center.x - left_rect.center.x);
    // Add half the bar widths to get outer edge distance
    width += (short_side(left_rect) + short_side(right_rect)) * 0.5f;
    float armor_ar = width / std::max(1e-3f, avg_h);
    
    // Armor should be wider than tall, but not too wide
    if (armor_ar < 0.8f || armor_ar > ARMOR_ASPECT_RATIO_LIMIT) return false;

    return true;
}



std::vector<cv::Point2f> ArmorDetectorNode::rect_to_point(cv::RotatedRect &rect)
{
    float rad = rect.angle < 90 ? rect.angle * M_PI / 180.f : (rect.angle - 180) * M_PI / 180.f;
    float x_offset = rect.size.height * std::sin(rad) / 2.f;
    float y_offset = rect.size.height * std::cos(rad) / 2.f;

    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(int(rect.center.x + x_offset), int(rect.center.y - y_offset)));
    points.push_back(cv::Point2f(int(rect.center.x - x_offset), int(rect.center.y + y_offset)));
    return points;
}
