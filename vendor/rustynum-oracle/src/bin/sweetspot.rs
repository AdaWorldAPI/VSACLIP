use rand::SeedableRng;
use rand::Rng;
use rand::rngs::StdRng;
use rustynum_oracle::sweep::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Focus: what's the BEST recovery error achievable at each K?
    // Test the top performers from experiment data
    println!("=== SWEET SPOT: Best base at each (D, K) with 5 reps ===");
    println!("d,base,k,mean_error_avg,mean_error_min,mean_error_max");

    for &d in &[2048usize, 4096, 8192, 16384] {
        for &k in &[1usize, 3, 5, 8, 13, 21, 34, 55] {
            if k > d / 10 { continue; }
            for &base in &[Base::Signed(3), Base::Signed(5), Base::Signed(7), Base::Signed(9)] {
                let mut errors = Vec::new();
                for _ in 0..5 {
                    let r = measure_recovery(d, base, k, &mut rng);
                    errors.push(r.mean_error);
                }
                let avg: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
                let min = errors.iter().cloned().fold(f32::MAX, f32::min);
                let max = errors.iter().cloned().fold(0.0f32, f32::max);
                println!("{},{},{},{:.6},{:.6},{:.6}", d, base.name(), k, avg, min, max);
            }
        }
    }

    // The key efficiency question: at error < 0.1, what's the best bits_per_concept?
    println!("\n=== EFFICIENCY FRONTIER: configs with error < 0.1 ===");
    println!("d,base,k,mean_error,bits_per_concept,bytes_total");

    for &d in &[2048usize, 4096, 8192, 16384, 32768] {
        for &base in &[Base::Signed(3), Base::Signed(5), Base::Signed(7), Base::Signed(9)] {
            for &k in &[1usize, 3, 5, 8, 13, 21, 34, 55, 89] {
                if k > d / 10 { continue; }
                let mut sum = 0.0f32;
                let reps = 5;
                for _ in 0..reps {
                    let r = measure_recovery(d, base, k, &mut rng);
                    sum += r.mean_error;
                }
                let avg = sum / reps as f32;
                if avg < 0.1 {
                    let bpc = base.storage_bits(d) as f32 / k as f32;
                    println!("{},{},{},{:.6},{:.1},{}",
                        d, base.name(), k, avg, bpc, base.storage_bytes(d, 1));
                }
            }
        }
    }

    // Oracle round-trip test at sweet spot parameters
    println!("\n=== ORACLE ROUND-TRIP at sweet spot (D=8192, Signed(5), K=3-13) ===");
    use rustynum_oracle::oracle::*;

    for &k in &[3usize, 5, 8, 13, 21] {
        let lib = TemplateLibrary::generate(
            k + 5, // library slightly larger than K
            8192, 16384, Base::Signed(5), 2, &mut rng,
        );

        let mut oracle = Oracle::new();
        let mut original_coeffs = Vec::new();
        for i in 0..k {
            let c: f32 = rng.gen_range(-0.8f32..0.8f32);
            oracle.add_concept(i as u32, c);
            original_coeffs.push(c);
        }

        // Hot round trip
        let hot = oracle.materialize_hot(&lib);
        oracle.surgical_cool(&hot, &lib);

        let mut max_err = 0.0f32;
        for i in 0..k {
            let err = (oracle.coefficients[i] - original_coeffs[i]).abs();
            max_err = max_err.max(err);
        }

        // Warm overexposure
        let warm = oracle.materialize_warm(&lib);
        let exposure = oracle.check_overexposure(&warm);

        println!("K={}: hot_roundtrip_max_err={:.6}, overexposure={:.4}, flush={:?}",
            k, max_err, exposure, oracle.flush_decision());
    }
}
