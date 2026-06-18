// Confidential, Copyright 2024, Sony AI, All rights reserved.
/**
 * Used to playback recorded images
 */

#include <rclcpp/rclcpp.hpp>
#include <ace_interfaces/msg/image_data.hpp>
#include <ace_interfaces/msg/logger_control.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

namespace fs = std::filesystem;

/**
 * Converts from RGB/BGR representation to Bayer8 representation
 */
cv::Mat rgb2bayer(const cv::Mat& bgr) {
    cv::Mat bayer = cv::Mat::zeros(bgr.rows, bgr.cols, CV_8UC1);

    for (int i = 0; i < bgr.rows; i += 2) {
        for (int j = 0; j < bgr.cols; j += 2) {
            // RGGB pattern
            bayer.at<uint8_t>(i, j + 1) = bgr.at<cv::Vec3b>(i, j + 1)[1];      // G
            bayer.at<uint8_t>(i, j) = bgr.at<cv::Vec3b>(i, j)[2];              // R
            if (i + 1 < bgr.rows) {
                bayer.at<uint8_t>(i + 1, j + 1) = bgr.at<cv::Vec3b>(i + 1, j + 1)[0]; // B
                bayer.at<uint8_t>(i + 1, j) = bgr.at<cv::Vec3b>(i + 1, j)[1];         // G
            }
        }
    }

    return bayer;
}

/**
 * List files with a given extension in a directory
 */
std::vector<std::string> list_files(const std::string& path, const std::string& extension) {
    std::vector<std::string> file_list;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                if (file_path.size() >= extension.size() &&
                    file_path.compare(file_path.size() - extension.size(), extension.size(), extension) == 0) {
                    file_list.push_back(file_path);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error listing files: " << e.what() << std::endl;
    }

    return file_list;
}

/**
 * Holds data for a single camera/topic
 */
struct TopicData {
    std::string camera_name;
    std::shared_ptr<cv::VideoCapture> video_reader;
    int curr_frame = 0;
    std::vector<int> frames;
    rclcpp::Publisher<ace_interfaces::msg::ImageData>::SharedPtr publisher;
    std::string topic;
    ace_interfaces::msg::ImageData message;

    // Threading support
    std::deque<cv::Mat> frame_cache;
    std::mutex cache_lock;
    std::unique_ptr<std::thread> reader_thread;
    std::atomic<bool> stop_reading{false};
    std::atomic<bool> cache_ready{false};
    std::atomic<bool> eof{false};
};

/**
 * Background thread to read frames from video and cache them
 */
class FrameReaderThread {
public:
    FrameReaderThread(std::shared_ptr<TopicData> topic_data, int cache_size = 10)
        : topic_data_(topic_data), cache_size_(cache_size) {}

    void operator()() {
        while (!topic_data_->stop_reading) {
            {
                std::lock_guard<std::mutex> lock(topic_data_->cache_lock);

                // Only cache if we have space
                if (topic_data_->frame_cache.size() < static_cast<size_t>(cache_size_)) {
                    cv::Mat frame;
                    bool ret = topic_data_->video_reader->read(frame);

                    if (ret && !frame.empty()) {
                        cv::Mat image_data = rgb2bayer(frame);
                        topic_data_->frame_cache.push_back(image_data);
                        topic_data_->cache_ready = true;
                    } else {
                        // End of video reached
                        topic_data_->eof = true;
                        break;
                    }
                }
            }

            // Small sleep to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        topic_data_->eof = true;
    }

private:
    std::shared_ptr<TopicData> topic_data_;
    int cache_size_;
};

/**
 * Main context class managing the video playback system
 */
class Context {
public:
    Context(const std::string& path, rclcpp::Node::SharedPtr node)
        : path_(path), node_(node), current_frame_(0), update_count_(0) {

        // Set up clusters
        clusters_["_a"] = {"aps20501495", "aps21107993", "aps21115383", "aps22071845", "aps22240292"};
        clusters_["_b"] = {"aps20370530", "aps21107992", "aps22087079", "aps22106662", "aps22240282"};

        // Copy config file
        std::string config_path = fs::path(path) / "tyo01.yaml";
        if (!fs::exists(config_path)) {
            throw std::runtime_error("Config file not found: " + config_path);
        }

        std::string target_location = fs::path(ament_index_cpp::get_package_share_directory("calibration")) /
                                      "parameters/camera_calibration/tyo01.yaml";
        fs::copy_file(config_path, target_location, fs::copy_options::overwrite_existing);

        // Set up logger control publisher
        auto qos = rclcpp::QoS(1);
        qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

        loggercontrol_pub_ = node_->create_publisher<ace_interfaces::msg::LoggerControl>(
            "/logger/control", qos);
    }

    ~Context() {
        destroy();
    }

    pid_t launch_process(const std::string& cmd) {
        RCLCPP_INFO(node_->get_logger(), "Launching: %s", cmd.c_str());

        printf("Launching command: %s\n", cmd.c_str());

        pid_t pid = fork();
        if (pid == 0) {
            signal(SIGINT, SIG_DFL);
            // Child process
            execl("/bin/sh", "sh", "-c", cmd.c_str(), nullptr);
            exit(1);
        } else if (pid < 0) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to fork process");
            return -1;
        }

        return pid;
    }

    void control_logger(const std::string& log_prefix) {
        auto msg = ace_interfaces::msg::LoggerControl();
        std::copy(log_prefix.begin(),
                  log_prefix.begin() + std::min(log_prefix.size(), msg.log_prefix.size()),
                  msg.log_prefix.begin());

        loggercontrol_pub_->publish(msg);
        RCLCPP_INFO(node_->get_logger(), "Sent LoggerControl(log_prefix=\"%s\") message over ROS",
                    log_prefix.c_str());
    }

    void close_processes() {
        std::cout<<"Closing launched processes" << std::endl;
        std::cout<<"Closing Player Process" << std::endl;
        if (player_pose_pid_ > 0) {
            kill(player_pose_pid_, SIGTERM);
            waitpid(player_pose_pid_, nullptr, 0);
        }
        std::cout<<"Closing Racket Process" << std::endl;
        if (racket_pose_pid_ > 0) {
            kill(racket_pose_pid_, SIGTERM);
            waitpid(racket_pose_pid_, nullptr, 0);
        }
        std::cout<<"Launched processes closed" << std::endl;
    }

    bool run_cluster(const std::string& cluster_name) {
        // Stop any existing reader threads
        for (auto& [name, topic] : topics_) {
            if (topic->reader_thread && topic->reader_thread->joinable()) {
                topic->stop_reading = true;
                topic->reader_thread->join();
            }
        }

        std::string log_prefix = fs::path(path_).filename().string();
        std::string log_dirname = std::getenv("ACE_DATALOGGER_PATH")
            ? std::getenv("ACE_DATALOGGER_PATH")
            : "./ace_logs";
        log_dirname = log_dirname + "OfflineProcessing/" + log_prefix;
        log_dirname = log_dirname + "/" + log_prefix + cluster_name;

        if (fs::exists(log_dirname + "_player_pose.ace")) {
            std::cout << "Recording exists, skipping" << std::endl;
            return false;
        }

        control_logger(log_dirname);

        // Launch player_pose process
        std::string cmd = "ros2 launch player_pose player_pose_estimation.launch.py config:=tyo01 cluster:=\"" +
                         cluster_name + "\"";
        player_pose_pid_ = launch_process(cmd);

        // Launch racket_pose process
        cmd = "ros2 launch racket_pose_estimation racket_pose_estimation.launch.py config:=tyo01 cluster:=\"" +
              cluster_name + "\"";
        racket_pose_pid_ = launch_process(cmd);

        std::this_thread::sleep_for(std::chrono::seconds(4));

        // Reset state
        topics_.clear();
        synchronized_topics_.clear();
        all_frames_.clear();
        current_frame_ = 0;

        // Load videos for cluster
        if (clusters_.find(cluster_name) == clusters_.end()) {
            RCLCPP_ERROR(node_->get_logger(), "Unknown cluster: %s", cluster_name.c_str());
            return false;
        }

        for (const auto& camera_name : clusters_[cluster_name]) {
            std::string video_path = (fs::path(path_) / (camera_name + ".mp4")).string();
            std::string frames_path = (fs::path(path_) / (camera_name + "_frames.txt")).string();
            std::string published_topic_name = "/sensors/" + camera_name + "/image";

            if (!fs::exists(frames_path)) {
                continue;
            }

            auto topic = add_publisher_internal(camera_name, published_topic_name);
            topic->camera_name = camera_name;
            topic->video_reader = std::make_shared<cv::VideoCapture>(video_path);

            if (!topic->video_reader->isOpened()) {
                RCLCPP_ERROR(node_->get_logger(), "Failed to open video: %s", video_path.c_str());
                continue;
            }

            // Load frame numbers
            std::ifstream frames_file(frames_path);
            std::string line;
            while (std::getline(frames_file, line)) {
                if (!line.empty()) {
                    topic->frames.push_back(std::stoi(line));
                }
            }

            topic->curr_frame = 0;

            // Start background reader thread
            topic->reader_thread = std::make_unique<std::thread>(
                FrameReaderThread(topic, 10));

            // Build synchronized frame map
            for (int frame : topic->frames) {
                synchronized_topics_[frame].push_back(camera_name);
            }
        }

        // Create sorted list of all frames
        for (const auto& [frame, cameras] : synchronized_topics_) {
            all_frames_.push_back(frame);
        }
        std::sort(all_frames_.begin(), all_frames_.end());

        return true;
    }

    void destroy() {
        // Stop all reader threads
        for (auto& [name, topic] : topics_) {
            if (topic->reader_thread && topic->reader_thread->joinable()) {
                topic->stop_reading = true;
                topic->reader_thread->join();
            }
            if (topic->video_reader && topic->video_reader->isOpened()) {
                topic->video_reader->release();
            }
        }

        close_processes();
    }

    bool update() {
        if (current_frame_ >= all_frames_.size()) {
            return false;
        }

        int frame = all_frames_[current_frame_];
        bool drop = synchronized_topics_[frame].size() == 1;
        if (synchronized_topics_[frame].size()!=5) {
            RCLCPP_WARN(node_->get_logger(), "Frame %d has %zu synchronized topics", frame,
                        synchronized_topics_[frame].size());
        }

        for (const auto& camera : synchronized_topics_[frame]) {
            auto topic = topics_[camera];
            publish_image(topic, frame, drop);
        }

        current_frame_++;

        // Calculate update frame rate
        auto now = std::chrono::steady_clock::now();
        if (last_update_time_.time_since_epoch().count() != 0) {
            auto delta = std::chrono::duration_cast<std::chrono::duration<double>>(
                now - last_update_time_).count();
            update_count_++;

            if (delta > 1.0) {
                update_fps_ = update_count_ / delta;
                update_count_ = 0;
                RCLCPP_INFO(node_->get_logger(), "update_fps: %.2f", update_fps_);
                last_update_time_ = now;
            }
        } else {
            update_count_ = 1;
            last_update_time_ = now;
        }

        return true;
    }

private:
    std::shared_ptr<TopicData> add_publisher_internal(
        const std::string& camera_name,
        const std::string& topic_name) {

        rclcpp::QoS qos(rclcpp::KeepLast(5));
        auto publisher = node_->create_publisher<ace_interfaces::msg::ImageData>(
            topic_name, qos.reliable());

        auto topic = std::make_shared<TopicData>();
        topic->topic = topic_name;
        topic->publisher = publisher;

        // Initialize message
        topic->message.offset_x = 0;
        topic->message.offset_y = 0;
        std::string encoding = "bayer_rggb8";
        std::copy(encoding.begin(), encoding.end(), topic->message.encoding.begin());

        topics_[camera_name] = topic;
        return topic;
    }

    void publish_image(std::shared_ptr<TopicData> topic, int sequence_number, bool drop) {
        // Try to get frame from cache
        cv::Mat image_data;
        while (!topic->eof) {
            std::lock_guard<std::mutex> lock(topic->cache_lock);
            if (!topic->frame_cache.empty()) {
                image_data = topic->frame_cache.front();
                topic->frame_cache.pop_front();
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        if (image_data.empty()) {
            if (topic->eof) {
                std::cout << "EOF frame " << sequence_number << " for topic " << topic->topic << std::endl;
                return;
            }
            throw std::runtime_error("Frame cache is empty and direct read is not implemented");
        }

        if (drop) {
            std::cout << "Dropping frame " << sequence_number << " for topic " << topic->topic << std::endl;
            return;
        }

        auto loaned_msg = topic->publisher->borrow_loaned_message();

        auto& msg = loaned_msg.get();
        // auto& msg= topic->message;

        msg.header.sequence_number = sequence_number;
        msg.frame_rate = 100;
        msg.encoding = std::array<unsigned char, 20>{"bayer_rggb8"};

        msg.height = static_cast<uint16_t>(image_data.rows);
        msg.width = static_cast<uint16_t>(image_data.cols);
        msg.offset_x = static_cast<uint16_t>(0);
        msg.offset_y = static_cast<uint16_t>(0);
        msg.step = static_cast<uint16_t>(image_data.cols);

        std::memcpy(msg.data.data(), image_data.data, msg.height * msg.step);

        topic->publisher->publish(msg);
        // std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::string path_;
    rclcpp::Node::SharedPtr node_;
    std::map<std::string, std::shared_ptr<TopicData>> topics_;
    std::map<std::string, std::vector<std::string>> clusters_;
    std::map<int, std::vector<std::string>> synchronized_topics_;
    std::vector<int> all_frames_;
    size_t current_frame_;

    pid_t player_pose_pid_ = -1;
    pid_t racket_pose_pid_ = -1;

    rclcpp::Publisher<ace_interfaces::msg::LoggerControl>::SharedPtr loggercontrol_pub_;

    // FPS tracking
    std::chrono::steady_clock::time_point last_update_time_;
    int update_count_;
    double update_fps_ = 0.0;
};

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " --path <path> --cluster <cluster_name>" << std::endl;
        return 1;
    }

    std::string path;
    std::string cluster;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--path" && i + 1 < argc) {
            path = argv[++i];
        } else if (arg == "--cluster" && i + 1 < argc) {
            cluster = argv[++i];
        }
    }

    if (path.empty() || cluster.empty()) {
        std::cerr << "Both --path and --cluster arguments are required" << std::endl;
        return 1;
    }

    // Initialize ROS2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("play_videos");

    try {
        Context context(path, node);

        if (!context.run_cluster(cluster)) {
            return -1;
        }

        while (rclcpp::ok()) {
            if (!context.update()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
