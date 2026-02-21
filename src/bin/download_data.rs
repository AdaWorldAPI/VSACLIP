//! Download Tiny ImageNet dataset.
//!
//! Run: cargo run --bin download-data
//!
//! This is a convenience wrapper around wget/curl.
//! For the Python embedding fallback, run scripts/embed_images.py after download.

use std::process::Command;
use std::path::Path;

const DATASET_URL: &str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip";
const DATA_DIR: &str = "data";
const ZIP_NAME: &str = "tiny-imagenet-200.zip";

fn main() {
    println!("VSACLIP — Tiny ImageNet Download");
    println!("=================================\n");

    let data_dir = Path::new(DATA_DIR);
    let train_dir = data_dir.join("tiny-imagenet-200").join("train");
    let zip_path = data_dir.join(ZIP_NAME);

    // Check if already extracted
    if train_dir.exists() {
        let class_count = std::fs::read_dir(&train_dir)
            .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)).count())
            .unwrap_or(0);
        if class_count >= 200 {
            println!("Dataset already exists: {} ({} classes)", train_dir.display(), class_count);
            println!("Nothing to do.\n");
            print_next_steps();
            return;
        }
    }

    // Create data directory
    std::fs::create_dir_all(data_dir).expect("Failed to create data directory");

    // Download
    if !zip_path.exists() {
        println!("[1/3] Downloading Tiny ImageNet (237 MB)...");
        println!("  URL: {}\n", DATASET_URL);

        let status = Command::new("wget")
            .args(["-q", "--show-progress", "-O"])
            .arg(zip_path.to_str().unwrap())
            .arg(DATASET_URL)
            .status()
            .or_else(|_| {
                // Fallback to curl
                Command::new("curl")
                    .args(["-L", "-o"])
                    .arg(zip_path.to_str().unwrap())
                    .arg(DATASET_URL)
                    .status()
            });

        match status {
            Ok(s) if s.success() => println!("  Download complete.\n"),
            _ => {
                eprintln!("ERROR: Download failed. Please download manually:");
                eprintln!("  wget -O {} {}", zip_path.display(), DATASET_URL);
                std::process::exit(1);
            }
        }
    } else {
        println!("[1/3] ZIP already downloaded: {}", zip_path.display());
    }

    // Extract
    println!("[2/3] Extracting...");
    let status = Command::new("unzip")
        .args(["-q", "-o"])
        .arg(zip_path.to_str().unwrap())
        .arg("-d")
        .arg(data_dir.to_str().unwrap())
        .status();

    match status {
        Ok(s) if s.success() => println!("  Extracted to {}\n", data_dir.display()),
        _ => {
            eprintln!("ERROR: Extraction failed. Please install unzip and retry.");
            std::process::exit(1);
        }
    }

    // Clean up
    println!("[3/3] Cleaning up ZIP...");
    if let Err(e) = std::fs::remove_file(&zip_path) {
        eprintln!("  Warning: could not remove ZIP: {}", e);
    } else {
        println!("  Removed {}\n", zip_path.display());
    }

    // Verify
    let class_count = std::fs::read_dir(&train_dir)
        .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false)).count())
        .unwrap_or(0);
    println!("Dataset ready: {} classes in {}", class_count, train_dir.display());

    print_next_steps();
}

fn print_next_steps() {
    println!("\nNext steps:");
    println!("  1. Pre-compute CLIP embeddings (Python fallback):");
    println!("     python3 scripts/embed_images.py");
    println!();
    println!("  2. Run the POC:");
    println!("     cargo run --release --bin poc");
}
