use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

fn read_tsp_file(filename: &str) -> Vec<Point> {
    let file = fs::File::open(filename).expect("Failed to open file");
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    let n: usize = lines.next().unwrap().unwrap().parse().expect("Invalid number format");
    let mut points = Vec::with_capacity(n);

    for line in lines.take(n) {
        let coords: Vec<f64> = line.unwrap()
            .split_whitespace()
            .map(|s| s.parse().expect("Invalid coordinate"))
            .collect();
        points.push(Point { x: coords[0], y: coords[1] });
    }
    points
}

fn euclidean_distance(a: Point, b: Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn total_distance(points: &[Point], path: &[usize]) -> f64 {
    let mut distance = 0.0;
    for pair in path.windows(2) {
        distance += euclidean_distance(points[pair[0]], points[pair[1]]);
    }
    distance += euclidean_distance(points[path[path.len()-1]], points[path[0]]);
    distance
}

fn nearest_neighbor(points: &[Point]) -> Vec<usize> {
    let n = points.len();
    let mut unvisited: Vec<_> = (1..n).collect();
    let mut path = vec![0];

    while !unvisited.is_empty() {
        let last = path[path.len()-1];
        let (nearest, _) = unvisited.iter()
            .map(|&i| (i, euclidean_distance(points[last], points[i])))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        path.push(nearest);
        unvisited.retain(|&x| x != nearest);
    }
    path
}

fn two_opt(points: &[Point], path: &mut Vec<usize>) {
    let mut improved = true;
    let len = path.len();

    while improved {
        improved = false;
        let mut best = (0.0, 0, 0);

        // Parallel delta calculation without path cloning
        best = (1..len-1)
            .into_par_iter()
            .map(|i| {
                let mut local_best = (0.0, 0, 0);
                for j in (i+1)..len {
                    if j - i <= 1 { continue }

                    // Calculate delta using only 4 points
                    let a = points[path[i-1]];
                    let b = points[path[i]];
                    let c = points[path[j]];
                    let d = points[*path.get(j+1).unwrap_or(&path[0])];

                    let original = euclidean_distance(a, b) + euclidean_distance(c, d);
                    let modified = euclidean_distance(a, c) + euclidean_distance(b, d);
                    let delta = original - modified;

                    if delta > local_best.0 {
                        local_best = (delta, i, j);
                    }
                }
                local_best
            })
            .reduce(|| best, |a, b| if a.0 > b.0 { a } else { b });

        if best.0 > 0.0 {
            path[best.1..=best.2].reverse();
            improved = true;
        }
    }
}

fn three_opt(points: &[Point], path: &mut Vec<usize>) {
    let len = path.len();
    if len < 6 { return; }

    let mut improved = true;
    let mut counter = 0;
    let max_iterations = len * 2;

    while improved && counter < max_iterations {
        improved = false;
        counter += 1;

        let mut best = (0.0, 0, 0, 0, 0);
        let current_len = path.len();

        best = (0..current_len-5)
            .into_par_iter()
            .map(|i| {
                let mut local_best = (0.0, 0, 0, 0, 0);
                for j in (i+2)..(current_len-3) {
                    for k in (j+2)..(current_len-1) {
                        let a = points[path[i]];
                        let b = points[path[i+1]];
                        let c = points[path[j]];
                        let d = points[path[j+1]];
                        let e = points[path[k]];
                        let f = points[path[(k+1) % current_len]];

                        let original = euclidean_distance(a, b)
                            + euclidean_distance(c, d)
                            + euclidean_distance(e, f);

                        // Consider all possible 3-opt permutations
                        let permutations = [
                            // Case 1: Reverse middle segment
                            (euclidean_distance(a, c) + euclidean_distance(b, d) + euclidean_distance(e, f), 0),
                            // Case 2: Reverse last segment
                            (euclidean_distance(a, b) + euclidean_distance(c, e) + euclidean_distance(d, f), 1),
                            // Case 3: Swap segments
                            (euclidean_distance(a, e) + euclidean_distance(d, c) + euclidean_distance(b, f), 2),
                        ];

                        if let Some((case, delta)) = permutations.iter()
                            .map(|&(new, case)| (case, original - new))
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        {
                            if delta > local_best.0 {
                                local_best = (delta, i, j, k, case);
                            }
                        }
                    }
                }
                local_best
            })
            .reduce(|| best, |a, b| if a.0 > b.0 { a } else { b });

        if best.0 > 0.0 {
            let (delta, i, j, k, case) = best;
            let mut new_path = path.clone();

            match case {
                0 => new_path[i+1..j+1].reverse(),
                1 => new_path[j+1..k+1].reverse(),
                2 => {
                    // Safe segment reconstruction using slice indices
                    let part1 = &path[0..=i];
                    let part2 = &path[j+1..=k];
                    let part3 = &path[i+1..=j];
                    let part4 = &path[k+1..];

                    new_path.clear();
                    new_path.extend_from_slice(part1);
                    new_path.extend_from_slice(part2);
                    new_path.extend_from_slice(part3);
                    new_path.extend_from_slice(part4);
                },
                _ => unreachable!()
            }

            if new_path.len() == path.len() {
                *path = new_path;
                improved = true;
            }
        }
    }
}

fn process_file(filename: &str, file_index: usize) {
    let points = read_tsp_file(filename);
    let start_nn = std::time::Instant::now();
    let mut path = nearest_neighbor(&points);
    let _end_nn = start_nn.elapsed().as_secs_f64();
    let start_2opt = std::time::Instant::now();
    two_opt(&points, &mut path);
    let end_2opt = start_2opt.elapsed().as_secs_f64();
    let len_2opt = total_distance(&points, &path);
    let (len_3opt, time_3opt) = if file_index < 50 {
        let start_3opt = std::time::Instant::now();
        three_opt(&points, &mut path);
        let end_3opt = start_3opt.elapsed().as_secs_f64();
        (Some(total_distance(&points, &path)), Some(end_3opt))
    } else { (None, None) };
    println!("{}	{}	{}	{}	{}", filename, len_2opt, end_2opt, len_3opt.unwrap_or_default(), time_3opt.unwrap_or_default());
}

fn process_directory(directory: &str) {
    let mut paths: Vec<String> = fs::read_dir(directory)
        .expect("Directory not found")
        .filter_map(|entry| {
            entry.ok().map(|e| e.path().to_string_lossy().into_owned())
        })
        .collect();

    // Natural sort for tsp_X_Y filenames
    paths.sort_by(|a, b| {
        let num_a = extract_numbers(&a);
        let num_b = extract_numbers(&b);
        num_a.cmp(&num_b)
    });

    if paths.is_empty() {
        println!("No TSP files found in the directory!");
        return;
    }

    // Process files sequentially in sorted order
    for (i, file) in paths.iter().enumerate() {
        process_file(file, i);
    }
}

fn extract_numbers(path: &str) -> (u32, u32) {
    let filename = std::path::Path::new(path)
        .file_name()
        .unwrap()
        .to_string_lossy();

    let parts: Vec<&str> = filename.split('_').collect();
    if parts.len() >= 3 {
        let x = parts[1].parse().unwrap_or(0);
        let y = parts[2].split('.').next().unwrap_or("0").parse().unwrap_or(0);
        (x, y)
    } else {
        (0, 0)
    }
}

fn main() {
    let num_threads = std::thread::available_parallelism().unwrap().get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();
    println!("Enter 1 to process a single file, 2 to process a directory:");
    let mut mode = String::new();
    io::stdin().read_line(&mut mode).expect("Failed to read input");
    match mode.trim() {
        "1" => {
            println!("Enter the TSP file path:");
            let mut filename = String::new();
            io::stdin().read_line(&mut filename).expect("Failed to read input");
            let filename = filename.trim();
            if Path::new(filename).exists() {
                process_file(filename, 0);
            } else {
                println!("Error: File not found.");
            }
        },
        "2" => {
            println!("Enter the directory containing TSP files:");
            let mut directory = String::new();
            io::stdin().read_line(&mut directory).expect("Failed to read input");
            let directory = directory.trim();
            if Path::new(directory).is_dir() {
                process_directory(directory);
            } else {
                println!("Error: Directory not found.");
            }
        },
        _ => println!("Invalid option!"),
    }
}
