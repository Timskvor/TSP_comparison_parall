#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <tuple>
#include <unordered_set>
#include <mutex>

namespace fs = std::filesystem;

struct Point {
    double x, y;
};

double euclidean_distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}

std::vector<Point> read_tsp_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Failed to open file");

    std::string line;
    std::getline(file, line);
    size_t n = std::stoi(line);

    std::vector<Point> points;
    points.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        double x, y;
        if (!(iss >> x >> y)) throw std::runtime_error("Invalid coordinate");
        points.push_back({x, y});
    }
    return points;
}

double total_distance(const std::vector<Point>& points, const std::vector<size_t>& path) {
    double dist = 0.0;
    for (size_t i = 0; i < path.size() - 1; ++i)
        dist += euclidean_distance(points[path[i]], points[path[i+1]]);
    dist += euclidean_distance(points[path.back()], points[path[0]]);
    return dist;
}

std::vector<size_t> nearest_neighbor(const std::vector<Point>& points) {
    size_t n = points.size();
    std::vector<size_t> path = {0};
    std::unordered_set<size_t> unvisited;
    for (size_t i = 1; i < n; ++i) unvisited.insert(i);

    while (!unvisited.empty()) {
        size_t last = path.back();
        double min_dist = INFINITY;
        size_t nearest = *unvisited.begin();

        for (size_t candidate : unvisited) {
            double dist = euclidean_distance(points[last], points[candidate]);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = candidate;
            }
        }
        path.push_back(nearest);
        unvisited.erase(nearest);
    }
    return path;
}

void two_opt(const std::vector<Point>& points, std::vector<size_t>& path) {
    bool improved = true;
    while (improved) {
        improved = false;
        size_t len = path.size();
        if (len < 4) break;

        unsigned threads_num = std::thread::hardware_concurrency();
        std::vector<std::thread> workers;
        std::vector<std::tuple<size_t, size_t, double>> bests(threads_num, {0, 0, 0.0});

        auto process_range = [&](size_t start, size_t end, size_t thread_id) {
            double best_delta = 0;
            size_t best_i = 0, best_j = 0;

            for (size_t i = start; i <= end; ++i) {
                for (size_t j = i + 2; j < len; ++j) {
                    const Point& a = points[path[i-1]];
                    const Point& b = points[path[i]];
                    const Point& c = points[path[j]];
                    const Point& d = (j == len-1) ? points[path[0]] : points[path[j+1]];

                    double original = euclidean_distance(a, b) + euclidean_distance(c, d);
                    double new_dist = euclidean_distance(a, c) + euclidean_distance(b, d);
                    double delta = original - new_dist;

                    if (delta > best_delta) {
                        best_delta = delta;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
            bests[thread_id] = {best_i, best_j, best_delta};
        };

        // Split work
        size_t range = (len - 2 - 1) / threads_num;
        for (size_t t = 0; t < threads_num; ++t) {
            size_t start = 1 + t * range;
            size_t end = (t == threads_num-1) ? len-2 : start + range - 1;
            workers.emplace_back(process_range, start, end, t);
        }

        for (auto& t : workers) t.join();

        // Find global best
        auto [best_i, best_j, best_d] = *std::max_element(bests.begin(), bests.end(),
            [](auto& a, auto& b) { return std::get<2>(a) < std::get<2>(b); });

        if (best_d > 0) {
            std::reverse(path.begin() + best_i, path.begin() + best_j + 1);
            improved = true;
        }
    }
}

void three_opt(const std::vector<Point>& points, std::vector<size_t>& path) {
    bool improved = true;
    while (improved) {
        improved = false;
        size_t len = path.size();
        if (len < 5) break;

        unsigned threads_num = std::thread::hardware_concurrency();
        std::vector<std::thread> workers;
        std::vector<std::tuple<size_t, size_t, size_t, double>> bests(threads_num, {0, 0, 0, 0.0});

        auto process_range = [&](size_t start, size_t end, size_t thread_id) {
            double best_delta = 0;
            size_t best_i = 0, best_j = 0, best_k = 0;
            double original = total_distance(points, path);

            for (size_t i = start; i <= end; ++i) {
                for (size_t j = i+1; j < len-2; ++j) {
                    for (size_t k = j+1; k < len; ++k) {
                        auto new_path = path;
                        std::reverse(new_path.begin() + i, new_path.begin() + j + 1);
                        std::reverse(new_path.begin() + j + 1, new_path.begin() + k + 1);
                        double new_dist = total_distance(points, new_path);
                        double delta = original - new_dist;

                        if (delta > best_delta) {
                            best_delta = delta;
                            best_i = i;
                            best_j = j;
                            best_k = k;
                        }
                    }
                }
            }
            bests[thread_id] = {best_i, best_j, best_k, best_delta};
        };

        // Split work
        size_t range = (len - 4) / threads_num;
        for (size_t t = 0; t < threads_num; ++t) {
            size_t start = t * range;
            size_t end = (t == threads_num-1) ? len-4 : start + range - 1;
            workers.emplace_back(process_range, start, end, t);
        }

        for (auto& t : workers) t.join();

        // Find global best
        auto [best_i, best_j, best_k, best_d] = *std::max_element(bests.begin(), bests.end(),
            [](auto& a, auto& b) { return std::get<3>(a) < std::get<3>(b); });

        if (best_d > 0) {
            std::reverse(path.begin() + best_i, path.begin() + best_j + 1);
            std::reverse(path.begin() + best_j + 1, path.begin() + best_k + 1);
            improved = true;
        }
    }
}

std::pair<unsigned, unsigned> extract_numbers(const std::string& path) {
    std::string filename = fs::path(path).filename().string();
    std::replace(filename.begin(), filename.end(), '.', '_');
    std::istringstream iss(filename);
    std::string part;
    std::vector<std::string> parts;

    while (std::getline(iss, part, '_')) parts.push_back(part);
    if (parts.size() < 3) return {0, 0};

    try {
        unsigned x = std::stoi(parts[1]);
        unsigned y = std::stoi(parts[2]);
        return {x, y};
    } catch (...) {
        return {0, 0};
    }
}

void process_file(const std::string& filename, int file_index) {
    auto points = read_tsp_file(filename);

    auto start = std::chrono::high_resolution_clock::now();
    auto path = nearest_neighbor(points);
    auto nn_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

    start = std::chrono::high_resolution_clock::now();
    two_opt(points, path);
    auto t2_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    double t2_dist = total_distance(points, path);

    double t3_dist = 0, t3_time = 0;
    if (file_index < 30 && file_index > 0) {
        start = std::chrono::high_resolution_clock::now();
        three_opt(points, path);
        t3_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        t3_dist = total_distance(points, path);
    }

    printf("%s\t%.5f\t%.5f\t%.5f\t%.5f\n",
           fs::path(filename).filename().c_str(),
           t2_dist, t2_time,
           t3_dist, t3_time);
}

void process_directory(const std::string& directory) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file())
            files.push_back(entry.path().string());
    }

    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return extract_numbers(a) < extract_numbers(b);
    });

    for (size_t i = 0; i < files.size(); ++i)
        process_file(files[i], i);
}

int main() {
    std::cout << "Enter 1 for single file, 2 for directory: ";
    int choice;
    std::cin >> choice;
    std::cin.ignore();

    if (choice == 1) {
        std::cout << "Enter file path: ";
        std::string path;
        std::getline(std::cin, path);
        process_file(path, 0);
    } else if (choice == 2) {
        std::cout << "Enter directory path: ";
        std::string dir;
        std::getline(std::cin, dir);
        process_directory(dir);
    } else {
        std::cout << "Invalid choice!\n";
    }
    return 0;
}
