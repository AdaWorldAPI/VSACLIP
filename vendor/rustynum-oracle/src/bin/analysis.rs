use rand::SeedableRng;
use rand::rngs::StdRng;
use rustynum_oracle::sweep::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    println!("=== EXPERIMENT 1: Recovery Error vs D for each Base (K=13, axes=1, depth=1) ===");
    println!("d,base,mean_error,max_error,noise_floor,cancellation,bits_per_concept");
    for &d in &[1024usize, 2048, 4096, 8192, 16384, 32768] {
        for &base in BASES {
            let r = measure_recovery(d, base, 13, &mut rng);
            println!("{},{},{:.6},{:.6},{:.6},{:.4},{:.1}",
                d, base.name(), r.mean_error, r.max_error, r.noise_floor, r.cancellation, r.bits_per_concept);
        }
    }

    println!("\n=== EXPERIMENT 2: Recovery Error vs K for best bases (D=8192, axes=1, depth=1) ===");
    println!("k,base,mean_error,max_error,gram_cond,noise_floor,cancellation");
    for &k in BUNDLE_SIZES {
        if k > 819 { continue; }
        for &base in &[Base::Binary, Base::Signed(5), Base::Signed(7), Base::Signed(9), Base::Unsigned(7)] {
            let r = measure_recovery(8192, base, k, &mut rng);
            println!("{},{},{:.6},{:.6},{:.2},{:.6},{:.4}",
                k, base.name(), r.mean_error, r.max_error, r.gram_condition, r.noise_floor, r.cancellation);
        }
    }

    println!("\n=== EXPERIMENT 3: Multi-Axis Benefit (D=4096, K=13) ===");
    println!("axes,base,bind_depth,combined_error,bell_coeff");
    for &base in &[Base::Binary, Base::Signed(5), Base::Signed(7), Base::Unsigned(7)] {
        for &axes in AXES {
            for &bind_depth in &[1usize, 2, 3] {
                if bind_depth > axes { continue; }
                let r = measure_recovery_multiaxis(4096, base, axes, 13, bind_depth, &mut rng);
                println!("{},{},{},{:.6},{:.4}",
                    axes, base.name(), bind_depth, r.combined_error, r.bell_coefficient);
            }
        }
    }

    println!("\n=== EXPERIMENT 4: Bell Coefficient by Base (D=4096, K=8, axes=2) ===");
    println!("base,bell_coeff_avg");
    for &base in BASES {
        let mut bell_sum = 0.0f32;
        let n = 10;
        for _ in 0..n {
            let r = measure_recovery_multiaxis(4096, base, 2, 8, 1, &mut rng);
            bell_sum += r.bell_coefficient;
        }
        println!("{},{:.4}", base.name(), bell_sum / n as f32);
    }

    println!("\n=== EXPERIMENT 5: Signed vs Unsigned Head-to-Head (D=8192, K=21) ===");
    println!("base,mean_error,cancellation,noise_floor");
    for &b in &[3u8, 5, 7, 9] {
        let signed = measure_recovery(8192, Base::Signed(b), 21, &mut rng);
        let unsigned = measure_recovery(8192, Base::Unsigned(b), 21, &mut rng);
        println!("signed({}),{:.6},{:.4},{:.6}", b, signed.mean_error, signed.cancellation, signed.noise_floor);
        println!("unsigned({}),{:.6},{:.4},{:.6}", b, unsigned.mean_error, 0.0, unsigned.noise_floor);
    }

    println!("\n=== EXPERIMENT 6: Capacity Sweet Spot — Pareto Frontier (axes=1, depth=1) ===");
    println!("d,base,k,mean_error,bits_per_concept,storage_bytes");
    let target_bases = [Base::Binary, Base::Signed(3), Base::Signed(5), Base::Signed(7), Base::Signed(9)];
    for &d in &[2048usize, 4096, 8192, 16384, 32768] {
        for &base in &target_bases {
            for &k in BUNDLE_SIZES {
                if k > d / 10 { continue; }
                let r = measure_recovery(d, base, k, &mut rng);
                // Only show configurations with reasonable recovery
                if r.mean_error < 1.0 {
                    println!("{},{},{},{:.6},{:.1},{}",
                        d, base.name(), k, r.mean_error, r.bits_per_concept, base.storage_bytes(d, 1));
                }
            }
        }
    }

    println!("\n=== EXPERIMENT 7: Gram Condition Number vs K (D=8192) ===");
    println!("k,base,gram_condition");
    for &k in &[1usize, 3, 5, 8, 13, 21, 34, 55, 89] {
        if k > 819 { continue; }
        for &base in &[Base::Binary, Base::Signed(5), Base::Signed(9)] {
            let r = measure_recovery(8192, base, k, &mut rng);
            println!("{},{},{:.2}", k, base.name(), r.gram_condition);
        }
    }

    println!("\n=== EXPERIMENT 8: Bind Depth Impact (D=8192, K=8) ===");
    println!("axes,bind_depth,base,combined_error");
    for &axes in &[2usize, 3] {
        for &bind_depth in &[1usize, 2, 3] {
            if bind_depth > axes { continue; }
            for &base in &[Base::Signed(5), Base::Signed(7)] {
                let r = measure_recovery_multiaxis(8192, base, axes, 8, bind_depth, &mut rng);
                println!("{},{},{},{:.6}", axes, bind_depth, base.name(), r.combined_error);
            }
        }
    }
}
