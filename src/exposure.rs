[package]
name = "vsaclip"
version = "0.1.0"
edition = "2024"
rust-version = "1.88"
license = "Apache-2.0"
description = "HDR POPCNT Sweep — Visual recognition via Hamming resonance on ladybug-rs Containers"
repository = "https://github.com/AdaWorldAPI/VSACLIP"
authors = ["Jan Hübener", "Ada Consciousness Project"]
keywords = ["vsa", "clip", "hamming", "simd", "resonance", "hdr", "popcnt"]

[lib]
name = "vsaclip"

[[bin]]
name = "vsaclip-proof"
path = "src/bin/proof.rs"

[[bin]]
name = "vsaclip-bench"
path = "src/bin/bench.rs"

# =============================================================================
# FEATURES
# =============================================================================
[features]
default = ["simd"]
simd = []
fastembed = ["dep:fastembed"]
load-image = ["dep:image"]

# =============================================================================
# DEPENDENCIES
# =============================================================================
[dependencies]
ladybug-contract = { git = "https://github.com/AdaWorldAPI/ladybug-rs.git", branch = "main" }

serde = { version = "1", features = ["derive"] }
serde_json = "1"

fastembed = { version = "4", optional = true }
image = { version = "0.25", optional = true, default-features = false, features = ["jpeg", "png"] }

rand = "0.8"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "hdr_sweep"
harness = false
