use rand::SeedableRng;
use rand::rngs::StdRng;
use rustynum_oracle::sweep::*;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let reps = 3; // 3 reps for statistical confidence without taking forever

    println!("d,base,signed,cardinality,axes,k,bind_depth,mean_error,max_error,gram_cond,noise_floor,cancellation,bell_coeff,bits_per_concept,storage_bytes");

    for &d in DIMS {
        for &base in BASES {
            for &axes in AXES {
                for &k in BUNDLE_SIZES {
                    for &bind_depth in BIND_DEPTHS {
                        if bind_depth > axes { continue; }
                        if k > d / 10 { continue; }

                        let mut sum_error = 0.0f32;
                        let mut sum_max_error = 0.0f32;
                        let mut sum_bell = 0.0f32;

                        for _ in 0..reps {
                            let result = measure_recovery_multiaxis(
                                d, base, axes, k, bind_depth, &mut rng,
                            );
                            sum_error += result.combined_error;
                            sum_bell += result.bell_coefficient;
                            // Get max error from per-axis
                            let max_ax = result.per_axis.iter()
                                .map(|a| a.mean_error)
                                .fold(0.0f32, f32::max);
                            sum_max_error += max_ax;
                        }

                        let mean_error = sum_error / reps as f32;
                        let max_error = sum_max_error / reps as f32;
                        let bell = sum_bell / reps as f32;

                        // Also get single-axis details for noise/cancellation
                        let single = measure_recovery(d, base, k, &mut rng);

                        println!("{},{},{},{},{},{},{},{:.6},{:.6},{:.2},{:.6},{:.4},{:.4},{:.1},{}",
                            d,
                            base.name(),
                            base.is_signed(),
                            base.cardinality(),
                            axes,
                            k,
                            bind_depth,
                            mean_error,
                            max_error,
                            single.gram_condition,
                            single.noise_floor,
                            single.cancellation,
                            bell,
                            single.bits_per_concept,
                            base.storage_bytes(d, axes),
                        );
                    }
                }
            }
        }
    }
}
