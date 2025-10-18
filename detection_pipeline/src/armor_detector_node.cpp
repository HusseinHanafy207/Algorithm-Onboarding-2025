/*
 *  READ ALL COMMENTS IN THIS FILE TO UNDERSTAND THE CODE THAT HAS BEEN WRITTEN FOR YOU
 *
 *  This file will create a subscriber node that reads images from the topic where camera_publisher_node publishes
 *  images. It will contain the functionality necessary for locating armor plates within a given image.
 *
 *  Since this file implements a ROS node like camera_publisher_node does, you may want to refer to that file to get
 *  ideas for things you need to consider while writing your code.
 */

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
    float a = normalize_angle_deg(r.angle);
    if (r.size.width > r.size.height) a = normalize_angle_deg(a + 90.f);
    return a; // [0,180)
}

static inline float long_side(const cv::RotatedRect &r) {
    return std::max(r.size.width, r.size.height);
}

static inline float short_side(const cv::RotatedRect &r) {
    return std::min(r.size.width, r.size.height);
}




/*
 *  This is the constructor for our subscriber node. It initializes the node inherited from the base class and creates
 *  the subscription to the topic with messages from camera_publisher_node.
 */
ArmorDetectorNode::ArmorDetectorNode() : rclcpp::Node("armor_detector_node"), frame_count(0)
{
    const std::string topic = this->declare_parameter<std::string>("image_topic", "camera/image_raw");
    // Subscribe to the camera publisher topic
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        topic, rclcpp::QoS(10),
        std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "ArmorDetectorNode subscribed to '%s'", topic.c_str());
}

/*
 *  This is an image callback method. It fetches messages (which are images in this case) from the topic this node
 *  subscribes to. The method will also run your armor detection algorithm on the image and show the result.
 *
 *  Callback methods are how we actually read data from a topic. Notice the parameter type and compare it to that of
 *  image_sub_. Our subscription here fetches images, and this method is how you actually operate on that image. Even
 *  though your image processing logic is in a different method, this callback uses those methods as helpers. Make sure
 *  that you understand how callbacks function in a ROS node architecture. If you completed the constructor correctly,
 *  you should know how a callback connects to a subscription in code.
 */
void ArmorDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // Images are represented by cv::Mat objects. This will be useful when you write image processing logic.
    cv::Mat frame;
    
    // Read the image from the topic into our frame with the proper color space (BGR8)
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

    // Reduce the computational load and just show every 5th image
    if (frame_count % 5 == 0)
    {
        show_frame(frame);
    }
}

/*
 *  This method displays a frame to your screen.
 */
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

/*
 *  The main method activates this subscriber node.
 */
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}

/*
 *  This method will search a frame for armor plates and return the RotatedRect objects that correspond to the two light
 *  bars that exist on an armor plate.
 */
std::vector<cv::RotatedRect> ArmorDetectorNode::search(cv::Mat& frame, cv::Scalar lowerHSV, cv::Scalar upperHSV, cv::Scalar lowerHSV2, cv::Scalar upperHSV2) {
    // TODO: Complete the rest of the method. The onboarding instructions document will be very helpful.

    // 1) Image Preprocessing
    cv::Mat blurred, hsv;
    cv::GaussianBlur(frame, blurred, cv::Size(5,5), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // 2) Color segmentation
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lowerHSV, upperHSV, mask1);
    cv::inRange(hsv, lowerHSV2, upperHSV2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // Clean up noise
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {3,3}));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));

    // 2.5) Edge Detection
    cv::Mat edges;
    cv::Canny(mask, edges, 80, 160);


    // 3) Contour Detection
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    // 4) Contour Filtering
    std::vector<cv::RotatedRect> light_bars;
    light_bars.reserve(contours.size());
    for (const auto &c : contours) {
        if (c.size() < 5) continue;
        cv::RotatedRect rr = cv::minAreaRect(c);
        if (is_light_bar(rr)) light_bars.push_back(rr);
    }
    if (light_bars.size() < 2) return {};


    // 5) Pairing → choose best armor candidate
    std::vector<cv::RotatedRect> best_pair;
    double best_score = 1e9;

    // sort by x for stable "left/right"
    std::sort(light_bars.begin(), light_bars.end(),
              [](const cv::RotatedRect& a, const cv::RotatedRect& b){ return a.center.x < b.center.x; });

    for (size_t i = 0; i < light_bars.size(); ++i) {
        for (size_t j = i + 1; j < light_bars.size(); ++j) {
            auto L = light_bars[i];
            auto R = light_bars[j];
            if (!is_armor(L, R)) continue;

            // A simple score: favor parallel & aligned, with reasonable width/height
            float angL = long_side_angle_deg(L), angR = long_side_angle_deg(R);
            float dang = std::min(std::abs(angL-angR), 180.f-std::abs(angL-angR));
            float ydiff = std::abs(L.center.y - R.center.y) / ((long_side(L)+long_side(R))/2.f);

            float width = std::abs(R.center.x - L.center.x);
            float plate_h = (long_side(L) + long_side(R)) / 2.f;
            float ar = width / std::max(1.f, plate_h);
            double score = dang + ydiff + std::abs(ar - 1.6f); // 1.6 ~ typical small-armor proportion

            if (score < best_score) {
                best_score = score;
                best_pair = {L, R};
            }
        }
    }
    return (best_pair.size()==2) ? best_pair : std::vector<cv::RotatedRect>{};

}

/*
 *  This method draws a rotated rectangle onto a frame. This is used to display the results of your algorithm when you
 *  run the node.
 */
void ArmorDetectorNode::draw_rotated_rect(cv::Mat &frame, cv::RotatedRect &rect)
{
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
}

/*
 *  This method determines whether a RotatedRect object can represent a light bar based on the constants defined in the
 *  header file. It checks dimensions, angles, and ratios against our configured thresholds to do so.
 */

 /*
 *  This method determines whether a RotatedRect object can represent a light bar based on the constants defined in the
 *  header file. It checks dimensions, angles, and ratios against our configured thresholds to do so.
 */
bool ArmorDetectorNode::is_light_bar(cv::RotatedRect &rect)
{
    // Verify that the light bar width is valid
    if (rect.size.width < LIGHT_BAR_WIDTH_LOWER_LIMIT)
        return false;
    
    // Verify that the light bar height is valid
    if (rect.size.height < LIGHT_BAR_HEIGHT_LOWER_LIMIT)
        return false;
    
    // Verify that the light bar angle is valid
    // Light bars should be roughly vertical (around 90 degrees)
    // We check if angle is between (90 - 30) and (90 + 30), i.e., 60 to 120 degrees
    // OR between (-90 + 30) and 0, i.e., -60 to 0 degrees (for the other orientation)
    if (!((rect.angle >= 90.0 - LIGHT_BAR_ANGLE_LIMIT && rect.angle <= 90.0 + LIGHT_BAR_ANGLE_LIMIT) ||
          (rect.angle >= -LIGHT_BAR_ANGLE_LIMIT && rect.angle <= LIGHT_BAR_ANGLE_LIMIT) ||
          (rect.angle >= 180.0 - LIGHT_BAR_ANGLE_LIMIT)))
        return false;
    
    // Verify that the light bar aspect ratio is valid
    // Aspect ratio is height / width, should be at least 2.0 (tall and thin)
    float aspect_ratio = rect.size.height / rect.size.width;
    if (aspect_ratio < LIGHT_BAR_ASPECT_RATIO_LOWER_LIMIT)
        return false;
    
    // All checks passed - this is a valid light bar
    return true;
}

// bool ArmorDetectorNode::is_light_bar(cv::RotatedRect &rect)
// {
//     // TODO: Use the LIGHT_BAR constants defined in the header file to complete this method.
//     // You may want to read the OpenCV documentation for RotatedRect

//     // normalize dims so height is the long side
//     float h = long_side(rect);
//     float w = short_side(rect);

//     // Verify that the light bar width is valid
//     if (w < LIGHT_BAR_WIDTH_LOWER_LIMIT)  return false;

//     // Verify that the light bar height is valid
//     if (h < LIGHT_BAR_HEIGHT_LOWER_LIMIT) return false;

//     // Verify that the light bar angle is valid
//     // You will want to compare against both the limit and its supplement; think about the unit circle
//     float a = long_side_angle_deg(rect);           // 0..180 (long axis)
//     float d_to_vertical = std::abs(a - 90.f);      // distance from vertical
//     if (d_to_vertical > LIGHT_BAR_ANGLE_LIMIT) return false;

//     // Verify that the light bar aspect ratio is valid
//     // Aspect ratio refers to height / width, not width / height
//     float aspect = h / std::max(1e-3f, w);
//     if (aspect < LIGHT_BAR_ASPECT_RATIO_LOWER_LIMIT) return false;

//     return true;
// }

/*
 *  This method determines whether a pair of light bars (RotatedRect objects) can represent an armor plate based on the
 *  constants defined in the header file. It checks dimensions, angles, and ratios against our configured thresholds to
 *  do so.
 */
bool ArmorDetectorNode::is_armor(cv::RotatedRect &left_rect, cv::RotatedRect &right_rect)
{
    // TODO: Use the ARMOR constants defined in the header file to complete this method.

    // Verify that the light bars are roughly parallel by checking that their difference does not exceed the threshold
    // Again, you will want to compare against both the limit and its supplement
    // Ensure left/right are actually left/right
    if (left_rect.center.x > right_rect.center.x) std::swap(left_rect, right_rect);

    float aL = long_side_angle_deg(left_rect);
    float aR = long_side_angle_deg(right_rect);

    // Parallelism of bars
    float diff = std::min(std::abs(aL-aR), 180.f-std::abs(aL-aR));
    if (diff > ARMOR_ANGLE_DIFF_LIMIT) return false;


    // Verify that the ratio between the light bar aspect ratios (that's a mouthful) is within the threshold
    // You will want to compare both left / right and right / left against the threshold
    float arL = long_side(left_rect)  / std::max(1e-3f, short_side(left_rect));
    float arR = long_side(right_rect) / std::max(1e-3f, short_side(right_rect));
    float r1 = arL / std::max(1e-3f, arR);
    float r2 = arR / std::max(1e-3f, arL);
    if (r1 > ARMOR_LIGHT_BAR_ASPECT_RATIO_RATIO_LIMIT || r2 > ARMOR_LIGHT_BAR_ASPECT_RATIO_RATIO_LIMIT) return false;

    // Verify that the light bars are at roughly the same elevation (as in their y difference is within the threshold)
    // The way the constant was determined assumes that you normalize this difference using the average light bar height
    // What that means is that the expression you should be checking is abs(y_left - y_right) / avg_height
    float avg_h = (long_side(left_rect) + long_side(right_rect)) / 2.f;
    float y_norm = std::abs(left_rect.center.y - right_rect.center.y) / std::max(1e-3f, avg_h);
    if (y_norm > ARMOR_Y_DIFF_LIMIT) return false;



    // Verify that the ratio between light bar heights is within the threshold
    // Again, you will want to compare both left / right and right / left
    float hL = long_side(left_rect), hR = long_side(right_rect);
    float hr1 = hL / std::max(1e-3f, hR);
    float hr2 = hR / std::max(1e-3f, hL);
    if (hr1 > ARMOR_HEIGHT_RATIO_LIMIT || hr2 > ARMOR_HEIGHT_RATIO_LIMIT) return false;

    // Verify that the armor aspect ratio is within the threshold
    // For some goofy reason, the constant for this step requires that you calculate aspect ratio as width / height
    // There are multiple ways to define armor plate "height" and "width." Hopefully your idea is effective!
    float width = std::abs(right_rect.center.x - left_rect.center.x);
    // include half bar widths to approximate total plate width between outer edges
    width += (short_side(left_rect) + short_side(right_rect)) * 0.5f;
    float armor_ar = width / std::max(1e-3f, avg_h);
    if (armor_ar > ARMOR_ASPECT_RATIO_LIMIT) return false;

    return true;
}

/*
 *  This method represents a RotatedRect object as a point and returns it. It exists for debugging output while the node
 *  is being run.
 */
std::vector<cv::Point2f> ArmorDetectorNode::rect_to_point(cv::RotatedRect &rect)
{
    float rad = rect.angle < 90 ? rect.angle * M_PI / 180.f : (rect.angle - 180) * M_PI / 180.f;
    float x_offset = rect.size.height * std::sin(rad) / 2.f;
    float y_offset = rect.size.height * std::cos(rad) / 2.f;

    std::vector<cv::Point2f> points;
    points = std::vector<cv::Point2f>();
    points.push_back(cv::Point2f(int(rect.center.x + x_offset), int(rect.center.y - y_offset)));
    points.push_back(cv::Point2f(int(rect.center.x - x_offset), int(rect.center.y + y_offset)));
    return points;
}