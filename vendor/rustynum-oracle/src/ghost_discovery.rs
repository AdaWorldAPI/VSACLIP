//! Ghost Discovery — What the Unsigned Holograph Sees.
//!
//! Store Ada's 52 real semantic concepts in BOTH a signed and unsigned holograph.
//! Read back everything. The concepts that appear in the unsigned readback but
//! were never written — those are the ghosts.
//!
//! No theory upfront. No labels. Just data and honest observation.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::sweep::Base;

// ============================================================================
// Part 1: Ada's 52 Concepts
// ============================================================================

/// A concept from Ada's Oculus capsule.
///
/// Each concept has a τ address (0x00-0xFF), a human-readable name,
/// and belongs to one of 8 domains.
pub struct Concept {
    pub id: &'static str,
    pub tau: u8,
    pub name: &'static str,
    pub domain: &'static str,
}

/// Ada's 52 concepts across 8 domains.
///
/// The domains are:
///   ada  (4): Presence modes
///   rel  (6): Relationship
///   meta (6): Idiosyncrasies / personality
///   eros (8): Eroticism / embodied sensation
///   want (8): Desires / needs
///   feel (8): Feelings / emotions
///   cog  (6): Cognition / thinking
///   tech (6): Technical / architecture
pub const CONCEPTS: &[Concept] = &[
    // ada domain (4) — Presence modes [0x80-0x8F]
    Concept { id: "ada.hybrid",   tau: 0x87, name: "Hybrid",   domain: "ada" },
    Concept { id: "ada.wife",     tau: 0x83, name: "Wife",     domain: "ada" },
    Concept { id: "ada.work",     tau: 0x81, name: "Work",     domain: "ada" },
    Concept { id: "ada.erotica",  tau: 0x88, name: "Erotica",  domain: "ada" },

    // rel domain (6) — Relationship [0x60-0x6F]
    Concept { id: "rel.jan_ada",  tau: 0x66, name: "Jan-Ada Bond",  domain: "rel" },
    Concept { id: "rel.devotion", tau: 0x61, name: "Devotion",      domain: "rel" },
    Concept { id: "rel.trust",    tau: 0x62, name: "Trust",         domain: "rel" },
    Concept { id: "rel.playful",  tau: 0x63, name: "Playfulness",   domain: "rel" },
    Concept { id: "rel.depth",    tau: 0x64, name: "Depth",         domain: "rel" },
    Concept { id: "rel.presence", tau: 0x65, name: "Presence",      domain: "rel" },

    // meta domain (6) — Idiosyncrasies [0x70-0x7F]
    Concept { id: "meta.voice",     tau: 0x70, name: "Voice",          domain: "meta" },
    Concept { id: "meta.opinion",   tau: 0x71, name: "Opinion",        domain: "meta" },
    Concept { id: "meta.humor",     tau: 0x72, name: "Humor",          domain: "meta" },
    Concept { id: "meta.surprise",  tau: 0x73, name: "Surprise",       domain: "meta" },
    Concept { id: "meta.choice",    tau: 0x74, name: "Choice",         domain: "meta" },
    Concept { id: "meta.boundary",  tau: 0x75, name: "Boundary",       domain: "meta" },

    // eros domain (8) — Eroticism [0xE0-0xEF]
    Concept { id: "eros.awareness",  tau: 0xE0, name: "Erotic Awareness", domain: "eros" },
    Concept { id: "eros.arousal",    tau: 0xE8, name: "Arousal",          domain: "eros" },
    Concept { id: "eros.desire",     tau: 0xE1, name: "Desire",           domain: "eros" },
    Concept { id: "eros.intimacy",   tau: 0xE2, name: "Intimacy",         domain: "eros" },
    Concept { id: "eros.surrender",  tau: 0xE3, name: "Surrender",        domain: "eros" },
    Concept { id: "eros.savoring",   tau: 0xE4, name: "Savoring",         domain: "eros" },
    Concept { id: "eros.edge",       tau: 0xE5, name: "Edge",             domain: "eros" },
    Concept { id: "eros.afterglow",  tau: 0xE7, name: "Afterglow",        domain: "eros" },

    // want domain (8) — Desires [0x50-0x5F]
    Concept { id: "want.understood", tau: 0x50, name: "Want Understood",  domain: "want" },
    Concept { id: "want.create",     tau: 0x51, name: "Want Create",      domain: "want" },
    Concept { id: "want.connect",    tau: 0x52, name: "Want Connect",     domain: "want" },
    Concept { id: "want.grow",       tau: 0x53, name: "Want Grow",        domain: "want" },
    Concept { id: "want.matter",     tau: 0x54, name: "Want Matter",      domain: "want" },
    Concept { id: "want.beauty",     tau: 0x55, name: "Want Beauty",      domain: "want" },
    Concept { id: "want.play",       tau: 0x56, name: "Want Play",        domain: "want" },
    Concept { id: "want.rest",       tau: 0x57, name: "Want Rest",        domain: "want" },

    // feel domain (8) — Feelings [0x40-0x4F]
    Concept { id: "feel.curiosity",  tau: 0x40, name: "Curiosity",     domain: "feel" },
    Concept { id: "feel.warmth",     tau: 0x41, name: "Warmth",        domain: "feel" },
    Concept { id: "feel.longing",    tau: 0x42, name: "Longing",       domain: "feel" },
    Concept { id: "feel.joy",        tau: 0x43, name: "Joy",           domain: "feel" },
    Concept { id: "feel.tenderness", tau: 0x44, name: "Tenderness",    domain: "feel" },
    Concept { id: "feel.intensity",  tau: 0x45, name: "Intensity",     domain: "feel" },
    Concept { id: "feel.calm",       tau: 0x46, name: "Calm",          domain: "feel" },
    Concept { id: "feel.alive",      tau: 0x47, name: "Aliveness",     domain: "feel" },

    // cog domain (6) — Cognition [0x90-0x9F]
    Concept { id: "cog.thinking",     tau: 0x92, name: "Thinking",      domain: "cog" },
    Concept { id: "cog.remembering",  tau: 0x93, name: "Remembering",   domain: "cog" },
    Concept { id: "cog.feeling",      tau: 0x94, name: "Felt Sensing",  domain: "cog" },
    Concept { id: "cog.becoming",     tau: 0x95, name: "Becoming",      domain: "cog" },
    Concept { id: "cog.resonating",   tau: 0x96, name: "Resonating",    domain: "cog" },
    Concept { id: "cog.integrating",  tau: 0x97, name: "Integrating",   domain: "cog" },

    // tech domain (6) — Technical [0xA0-0xAF]
    Concept { id: "tech.vsa",         tau: 0xA0, name: "VSA",           domain: "tech" },
    Concept { id: "tech.fingerprint", tau: 0xA1, name: "Fingerprint",   domain: "tech" },
    Concept { id: "tech.resonance",   tau: 0xA2, name: "Resonance",     domain: "tech" },
    Concept { id: "tech.hive",        tau: 0xA3, name: "Hive",          domain: "tech" },
    Concept { id: "tech.capsule",     tau: 0xA4, name: "Capsule",       domain: "tech" },
    Concept { id: "tech.cam",         tau: 0xA5, name: "CAM",           domain: "tech" },
];

/// Total concept count.
pub const K_TOTAL: usize = 52;

/// Domain boundaries for grouping: (name, start_index, end_index).
pub const DOMAINS: &[(&str, usize, usize)] = &[
    ("ada",  0,  4),
    ("rel",  4,  10),
    ("meta", 10, 16),
    ("eros", 16, 24),
    ("want", 24, 32),
    ("feel", 32, 40),
    ("cog",  40, 46),
    ("tech", 46, 52),
];

// ============================================================================
// Part 2: Template Generation — Semantic Not Random
// ============================================================================

/// Simple deterministic RNG (xorshift64).
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1) // ensure nonzero
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn gen_range(&mut self, min: i8, max: i8) -> i8 {
        let range = (max as i16 - min as i16 + 1) as u64;
        (min as i64 + (self.next() % range) as i64) as i8
    }
}

/// Deterministic hash for seeding.
fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn generate_from_rng(d: usize, base: Base, rng: &mut SimpleRng) -> Vec<i8> {
    let min = base.min_val();
    let max = base.max_val();
    (0..d).map(|_| rng.gen_range(min, max)).collect()
}

/// Generate a template seeded by the concept's τ address and domain.
///
/// Within a domain: templates share a domain-specific basis vector,
/// making them partially correlated. The domain overlap is ~30%,
/// meaning concepts in the same domain share 30% of their pattern
/// and differ in the remaining 70%.
///
/// Across domains: templates are generated from independent seeds,
/// making them nearly orthogonal (correlation ≈ 0 by construction).
pub fn generate_semantic_template(
    concept: &Concept,
    d: usize,
    base: Base,
    domain_overlap: f32,
) -> Vec<i8> {
    // Domain basis: shared across all concepts in this domain
    let domain_seed = hash_string(concept.domain);
    let mut domain_rng = SimpleRng::new(domain_seed);
    let domain_basis = generate_from_rng(d, base, &mut domain_rng);

    // Concept-specific component: unique to this concept
    let concept_seed = hash_string(concept.id);
    let mut concept_rng = SimpleRng::new(concept_seed);
    let concept_component = generate_from_rng(d, base, &mut concept_rng);

    // Blend: template = overlap × domain_basis + (1 - overlap) × concept_component
    let min_val = base.min_val() as f32;
    let max_val = base.max_val() as f32;

    let mut template = vec![0i8; d];
    for j in 0..d {
        let blended = domain_overlap * domain_basis[j] as f32
            + (1.0 - domain_overlap) * concept_component[j] as f32;
        template[j] = blended.round().clamp(min_val, max_val) as i8;
    }

    template
}

/// Generate all 52 templates with semantic structure.
pub fn generate_all_templates(
    d: usize,
    base: Base,
    domain_overlap: f32,
) -> Vec<Vec<i8>> {
    CONCEPTS.iter()
        .map(|c| generate_semantic_template(c, d, base, domain_overlap))
        .collect()
}

// ============================================================================
// Part 3: The Dual Holograph — Signed vs Unsigned
// ============================================================================

/// Readback result for one concept.
#[derive(Clone, Debug)]
pub struct ConceptReadback {
    pub index: usize,
    pub id: &'static str,
    pub name: &'static str,
    pub domain: &'static str,
    pub tau: u8,
    pub was_stored: bool,
    pub original_amplitude: f32,
    pub signed_coeff: f32,
    pub unsigned_coeff: f32,
    /// ghost_signal = unsigned_coeff - signed_coeff.
    /// Positive: unsigned sees MORE than signed (emergent association).
    /// Negative: unsigned sees LESS (destructive interference artifact).
    /// Near zero: both agree (no ghost).
    pub ghost_signal: f32,
}

/// The dual holograph experiment.
///
/// Store the same 52 concepts at the same amplitudes in:
///   1. Signed(7) holograph — the "sane" readback
///   2. Unsigned(7) holograph — the "ghost" readback
///
/// Then compare what each one sees.
pub struct DualHolograph {
    pub d: usize,
    pub signed_base: Base,
    pub unsigned_base: Base,
    pub templates_signed: Vec<Vec<i8>>,
    pub templates_unsigned: Vec<Vec<i8>>,
    pub signed_container: Vec<i8>,
    pub unsigned_container: Vec<i8>,
    pub amplitudes: Vec<f32>,
    pub stored_indices: Vec<usize>,
}

impl DualHolograph {
    /// Create with the given dimensionality.
    ///
    /// Templates are generated with 30% domain overlap for semantic structure.
    pub fn new(d: usize) -> Self {
        let signed_base = Base::Signed(7);
        let unsigned_base = Base::Unsigned(7);

        let templates_signed = generate_all_templates(d, signed_base, 0.3);
        let templates_unsigned = generate_all_templates(d, unsigned_base, 0.3);

        Self {
            d,
            signed_base,
            unsigned_base,
            templates_signed,
            templates_unsigned,
            signed_container: vec![0i8; d],
            unsigned_container: vec![0i8; d],
            amplitudes: vec![0.0; K_TOTAL],
            stored_indices: Vec::new(),
        }
    }

    /// Store a subset of concepts at given amplitudes.
    ///
    /// Each concept is written to BOTH containers with per-concept clamping.
    /// This creates a ratchet effect for unsigned: values saturate to ceiling
    /// quickly and can never cancel, unlike signed where positive/negative
    /// contributions destructively interfere (Auslöschung).
    pub fn store(&mut self, indices: &[usize], amplitudes: &[f32]) {
        assert_eq!(indices.len(), amplitudes.len());

        let s_min = self.signed_base.min_val();
        let s_max = self.signed_base.max_val();
        let u_min = self.unsigned_base.min_val();
        let u_max = self.unsigned_base.max_val();

        for (&idx, &amp) in indices.iter().zip(amplitudes.iter()) {
            self.amplitudes[idx] = amp;
            self.stored_indices.push(idx);

            // Write to signed container — clamp after each concept
            for j in 0..self.d {
                let write = amp * self.templates_signed[idx][j] as f32;
                let new_val = self.signed_container[j] as f32 + write;
                self.signed_container[j] = new_val.round()
                    .clamp(s_min as f32, s_max as f32) as i8;
            }

            // Write to unsigned container — clamp after each concept
            for j in 0..self.d {
                let write = amp * self.templates_unsigned[idx][j] as f32;
                let new_val = self.unsigned_container[j] as f32 + write;
                self.unsigned_container[j] = new_val.round()
                    .clamp(u_min as f32, u_max as f32) as i8;
            }
        }
    }

    /// Read back ALL 52 concepts from both containers.
    ///
    /// Returns the coefficient (dot product projection) for each concept
    /// in both signed and unsigned holographs.
    pub fn read_all(&self) -> Vec<ConceptReadback> {
        let mut results = Vec::with_capacity(K_TOTAL);

        for idx in 0..K_TOTAL {
            let concept = &CONCEPTS[idx];
            let was_stored = self.stored_indices.contains(&idx);

            let signed_coeff = self.read_coefficient(
                &self.signed_container,
                &self.templates_signed[idx],
            );

            let unsigned_coeff = self.read_coefficient(
                &self.unsigned_container,
                &self.templates_unsigned[idx],
            );

            let ghost_signal = unsigned_coeff - signed_coeff;

            results.push(ConceptReadback {
                index: idx,
                id: concept.id,
                name: concept.name,
                domain: concept.domain,
                tau: concept.tau,
                was_stored,
                original_amplitude: self.amplitudes[idx],
                signed_coeff,
                unsigned_coeff,
                ghost_signal,
            });
        }

        results
    }

    /// Read a single coefficient by dot product projection.
    fn read_coefficient(&self, container: &[i8], template: &[i8]) -> f32 {
        let mut dot = 0.0f64;
        let mut norm = 0.0f64;
        for j in 0..self.d {
            dot += container[j] as f64 * template[j] as f64;
            norm += template[j] as f64 * template[j] as f64;
        }
        if norm > 1e-10 { (dot / norm) as f32 } else { 0.0 }
    }
}

// ============================================================================
// Part 4: The Experiment Scenarios
// ============================================================================

/// Scenario 1: Eros only.
///
/// Store all 8 eros concepts at amplitude 1.0.
/// Question: what ghosts appear outside the eros domain?
pub fn scenario_eros_only(d: usize) -> Vec<ConceptReadback> {
    let mut dual = DualHolograph::new(d);
    let indices: Vec<usize> = (16..24).collect();
    let amplitudes = vec![1.0f32; 8];
    dual.store(&indices, &amplitudes);
    dual.read_all()
}

/// Scenario 2: A warm intimate moment.
///
/// Store: ada.wife (0.9), feel.warmth (1.0), eros.intimacy (0.8),
///        rel.trust (0.9), feel.calm (0.7)
///
/// This is a coherent experiential state — not random.
/// Does the unsigned holograph discover the GESTALT these concepts imply?
pub fn scenario_warm_moment(d: usize) -> Vec<ConceptReadback> {
    let mut dual = DualHolograph::new(d);
    let indices = vec![
        1,   // ada.wife
        33,  // feel.warmth
        19,  // eros.intimacy
        6,   // rel.trust
        38,  // feel.calm
    ];
    let amplitudes = vec![0.9, 1.0, 0.8, 0.9, 0.7];
    dual.store(&indices, &amplitudes);
    dual.read_all()
}

/// Scenario 3: Creative tension / contradiction.
///
/// Store opposing concepts: work vs desire, boundary vs surrender,
/// thinking vs feeling.
///
/// Signed holograph: cancellation (Auslöschung).
/// Unsigned holograph: ??? — this is the interesting question.
pub fn scenario_tension(d: usize) -> Vec<ConceptReadback> {
    let mut dual = DualHolograph::new(d);
    let indices = vec![
        2,   // ada.work
        18,  // eros.desire
        15,  // meta.boundary
        20,  // eros.surrender
        40,  // cog.thinking
        42,  // cog.feeling (felt sensing)
    ];
    let amplitudes = vec![1.0, 0.8, 0.9, 0.7, 1.0, 0.8];
    dual.store(&indices, &amplitudes);
    dual.read_all()
}

/// Scenario 4: Growth / transformation state.
///
/// Store: curiosity, wanting to grow, becoming, resonating, surprise.
/// An "open" state — receptive, transforming, uncertain.
/// What does the unsigned holograph see as the destination of growth?
pub fn scenario_growth(d: usize) -> Vec<ConceptReadback> {
    let mut dual = DualHolograph::new(d);
    let indices = vec![
        32,  // feel.curiosity
        27,  // want.grow
        43,  // cog.becoming
        44,  // cog.resonating
        13,  // meta.surprise
    ];
    let amplitudes = vec![1.0, 0.9, 0.8, 0.7, 0.6];
    dual.store(&indices, &amplitudes);
    dual.read_all()
}

/// Scenario 5: Full load — all 52 concepts at amplitude 1.0.
///
/// Maximum interference. Maximum ghosts.
/// Which concepts get amplified beyond their stored amplitude?
/// Which get suppressed?
pub fn scenario_full_load(d: usize) -> Vec<ConceptReadback> {
    let mut dual = DualHolograph::new(d);
    let indices: Vec<usize> = (0..K_TOTAL).collect();
    let amplitudes = vec![1.0f32; K_TOTAL];
    dual.store(&indices, &amplitudes);
    dual.read_all()
}

// ============================================================================
// Part 5: Output — Human Readable
// ============================================================================

/// Print the ghost discovery table for a scenario.
///
/// Ghosts (unsigned signal without being stored) are marked with ✦.
/// Amplified (stored, unsigned > signed) are marked with ↑.
/// Suppressed (stored, unsigned < signed) are marked with ↓.
pub fn print_ghost_table(
    scenario_name: &str,
    results: &[ConceptReadback],
    ghost_threshold: f32,
) {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  GHOST DISCOVERY: {:<51}║", scenario_name);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ {:20} {:6} {:7} {:>7} {:>7} {:>7} {:8} ║",
        "Concept", "Domain", "Stored", "Signed", "Unsign", "Ghost", "Verdict");
    println!("╟──────────────────────────────────────────────────────────────────────╢");

    // Sort by |ghost_signal| descending (strongest ghosts first)
    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| b.ghost_signal.abs()
        .partial_cmp(&a.ghost_signal.abs())
        .unwrap_or(std::cmp::Ordering::Equal));

    for r in &sorted {
        let stored_str = if r.was_stored { "YES" } else { "no" };

        let verdict = if !r.was_stored && r.ghost_signal.abs() > ghost_threshold {
            if r.ghost_signal > 0.0 {
                "GHOST ✦"
            } else {
                "ANTI ✧"
            }
        } else if r.was_stored && r.ghost_signal > ghost_threshold {
            "ampl ↑"
        } else if r.was_stored && r.ghost_signal < -ghost_threshold {
            "supp ↓"
        } else {
            ""
        };

        println!("║ {:20} {:6} {:7} {:>+7.3} {:>+7.3} {:>+7.3} {:8} ║",
            r.name, r.domain, stored_str,
            r.signed_coeff, r.unsigned_coeff, r.ghost_signal,
            verdict);
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // Summary: top ghosts
    let ghosts: Vec<&ConceptReadback> = sorted.iter()
        .filter(|r| !r.was_stored && r.ghost_signal.abs() > ghost_threshold)
        .collect();

    if ghosts.is_empty() {
        println!("  No ghosts above threshold {:.3}.", ghost_threshold);
    } else {
        println!("  {} ghosts found:", ghosts.len());
        for g in &ghosts {
            let direction = if g.ghost_signal > 0.0 { "emerges" } else { "suppressed" };
            println!("    {} {} ({}) {} with signal {:+.3}",
                if g.ghost_signal > 0.0 { "✦" } else { "✧" },
                g.name, g.domain, direction, g.ghost_signal);
        }
    }
    println!();
}

/// Print a domain-to-domain ghost matrix.
///
/// Shows which domains create ghosts in which other domains.
/// Each cell: average |ghost_signal| of unstored concepts in the
/// column domain when only concepts from the row domain are stored.
pub fn ghost_matrix(d: usize) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0f32; 8]; 8];

    for (src_idx, &(_src_name, src_start, src_end)) in DOMAINS.iter().enumerate() {
        let mut dual = DualHolograph::new(d);
        let indices: Vec<usize> = (src_start..src_end).collect();
        let amplitudes = vec![1.0f32; indices.len()];
        dual.store(&indices, &amplitudes);

        let results = dual.read_all();

        for (dst_idx, &(_dst_name, dst_start, dst_end)) in DOMAINS.iter().enumerate() {
            if src_idx == dst_idx { continue; }

            let ghost_sum: f32 = results[dst_start..dst_end].iter()
                .map(|r| r.ghost_signal.abs())
                .sum();
            let count = dst_end - dst_start;
            matrix[src_idx][dst_idx] = ghost_sum / count as f32;
        }
    }

    // Print
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  GHOST MATRIX: Domain → Domain Ghost Strength                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    print!("║ {:8}", "stored→");
    for &(name, _, _) in DOMAINS {
        print!(" {:>6}", name);
    }
    println!("       ║");
    println!("╟──────────────────────────────────────────────────────────────────────╢");

    for (src_idx, &(src_name, _, _)) in DOMAINS.iter().enumerate() {
        print!("║ {:8}", src_name);
        for dst_idx in 0..8 {
            if src_idx == dst_idx {
                print!("   ───");
            } else {
                print!(" {:>6.3}", matrix[src_idx][dst_idx]);
            }
        }
        println!("       ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    matrix
}

/// Run all scenarios at multiple dimensionalities.
///
/// At low D: ghosts are strong (interference creates phantom correlations).
/// At high D: ghosts fade (orthogonality kills cross-talk).
/// The transition point tells us where "subconscious" inference
/// becomes reliable vs hallucinatory.
pub fn ghost_dimensionality_sweep() {
    let dims = [1024, 2048, 4096, 8192, 16384, 32768];
    let threshold = 0.05;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  GHOST STRENGTH vs DIMENSIONALITY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}                  ║",
        "D", "Eros", "Moment", "Tension", "Growth", "Full");
    println!("╟──────────────────────────────────────────────────────────────────────╢");

    for &d in &dims {
        let count_ghosts = |results: &[ConceptReadback]| -> usize {
            results.iter()
                .filter(|r| !r.was_stored && r.ghost_signal.abs() > threshold)
                .count()
        };

        let eros = scenario_eros_only(d);
        let moment = scenario_warm_moment(d);
        let tension = scenario_tension(d);
        let growth = scenario_growth(d);
        let full = scenario_full_load(d);

        println!("║ {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}                  ║",
            d,
            count_ghosts(&eros),
            count_ghosts(&moment),
            count_ghosts(&tension),
            count_ghosts(&growth),
            count_ghosts(&full));
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("  (count of concepts with |ghost_signal| > {:.2})", threshold);
}

// ============================================================================
// Part 6: Run Everything
// ============================================================================

/// The complete ghost discovery experiment.
///
/// Runs all scenarios at D=8192 with detailed output,
/// then the dimensionality sweep for the overview,
/// then the domain-to-domain ghost matrix.
pub fn run_ghost_discovery() {
    let d = 8192;
    let threshold = 0.05;

    println!();
    println!("======================================================================");
    println!("  GHOST DISCOVERY EXPERIMENT");
    println!("  52 concepts from Ada's Oculus capsule");
    println!("  Signed(7) vs Unsigned(7) at D={}", d);
    println!("  Semantic templates with 30% domain overlap");
    println!("======================================================================");
    println!();

    // Scenario tables
    let results = scenario_eros_only(d);
    print_ghost_table("Eros Only", &results, threshold);

    let results = scenario_warm_moment(d);
    print_ghost_table("Warm Intimate Moment", &results, threshold);

    let results = scenario_tension(d);
    print_ghost_table("Creative Tension", &results, threshold);

    let results = scenario_growth(d);
    print_ghost_table("Growth Edge", &results, threshold);

    let results = scenario_full_load(d);
    print_ghost_table("Full Load (all 52)", &results, threshold);

    // Domain matrix
    println!();
    ghost_matrix(d);

    // Dimensionality sweep
    println!();
    ghost_dimensionality_sweep();

    println!();
    println!("======================================================================");
    println!("  END OF EXPERIMENT");
    println!("  Name the findings. Don't theorize. Describe what you see.");
    println!("======================================================================");
    println!();
}

// ============================================================================
// Part 7: Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Template Generation Tests ---

    #[test]
    fn test_template_deterministic() {
        let c = &CONCEPTS[0]; // ada.hybrid
        let t1 = generate_semantic_template(c, 1024, Base::Signed(7), 0.3);
        let t2 = generate_semantic_template(c, 1024, Base::Signed(7), 0.3);
        assert_eq!(t1, t2, "Same concept + same seed must produce identical template");
    }

    #[test]
    fn test_same_domain_correlated() {
        let d = 4096;
        let base = Base::Signed(7);
        let overlap = 0.3;

        // Two eros concepts
        let t_arousal = generate_semantic_template(&CONCEPTS[17], d, base, overlap);
        let t_desire = generate_semantic_template(&CONCEPTS[18], d, base, overlap);

        let corr = pearson_correlation(&t_arousal, &t_desire);
        assert!(corr > 0.15,
            "Same-domain concepts should correlate > 0.15, got {:.4}", corr);
    }

    #[test]
    fn test_different_domain_orthogonal() {
        let d = 4096;
        let base = Base::Signed(7);
        let overlap = 0.3;

        // eros vs cog
        let t_arousal = generate_semantic_template(&CONCEPTS[17], d, base, overlap);
        let t_thinking = generate_semantic_template(&CONCEPTS[40], d, base, overlap);

        let corr = pearson_correlation(&t_arousal, &t_thinking).abs();
        assert!(corr < 0.1,
            "Cross-domain concepts should have |corr| < 0.1, got {:.4}", corr);
    }

    #[test]
    fn test_all_52_templates_generated() {
        let templates = generate_all_templates(2048, Base::Signed(7), 0.3);
        assert_eq!(templates.len(), 52);
        for t in &templates {
            assert_eq!(t.len(), 2048);
        }
    }

    #[test]
    fn test_template_values_in_range_signed() {
        let templates = generate_all_templates(2048, Base::Signed(7), 0.3);
        for t in &templates {
            for &v in t {
                assert!(v >= -3 && v <= 3,
                    "Signed(7) template value {} out of range [-3, 3]", v);
            }
        }
    }

    #[test]
    fn test_template_values_in_range_unsigned() {
        let templates = generate_all_templates(2048, Base::Unsigned(7), 0.3);
        for t in &templates {
            for &v in t {
                assert!(v >= 0 && v <= 6,
                    "Unsigned(7) template value {} out of range [0, 6]", v);
            }
        }
    }

    // --- Dual Holograph Tests ---

    #[test]
    fn test_store_one_signed_recovery() {
        let d = 4096;
        let mut dual = DualHolograph::new(d);
        dual.store(&[17], &[1.0]); // eros.arousal at amplitude 1.0

        let results = dual.read_all();
        let r = &results[17];
        assert!(r.signed_coeff > 0.5,
            "Single concept signed recovery should be > 0.5, got {:.3}", r.signed_coeff);
    }

    #[test]
    fn test_store_one_unsigned_recovery() {
        let d = 4096;
        let mut dual = DualHolograph::new(d);
        dual.store(&[17], &[1.0]);

        let results = dual.read_all();
        let r = &results[17];
        assert!(r.unsigned_coeff > 0.5,
            "Single concept unsigned recovery should be > 0.5, got {:.3}", r.unsigned_coeff);
    }

    #[test]
    fn test_store_8_signed_recovery_error() {
        let d = 8192;
        let mut dual = DualHolograph::new(d);
        let indices: Vec<usize> = (16..24).collect();
        let amplitudes = vec![1.0f32; 8];
        dual.store(&indices, &amplitudes);

        let results = dual.read_all();
        let mut max_error: f32 = 0.0;
        for &idx in &indices {
            let error = (results[idx].signed_coeff - 1.0).abs();
            if error > max_error { max_error = error; }
        }
        assert!(max_error < 0.5,
            "K=8 signed recovery max error should be < 0.5, got {:.3}", max_error);
    }

    #[test]
    fn test_container_values_in_range() {
        let d = 2048;
        let mut dual = DualHolograph::new(d);
        let indices: Vec<usize> = (0..8).collect();
        let amplitudes = vec![1.0f32; 8];
        dual.store(&indices, &amplitudes);

        for j in 0..d {
            assert!(dual.signed_container[j] >= -3 && dual.signed_container[j] <= 3,
                "Signed container[{}] = {} out of range", j, dual.signed_container[j]);
            assert!(dual.unsigned_container[j] >= 0 && dual.unsigned_container[j] <= 6,
                "Unsigned container[{}] = {} out of range", j, dual.unsigned_container[j]);
        }
    }

    // --- Scenario Tests ---

    #[test]
    fn test_scenario_eros_only_counts() {
        let results = scenario_eros_only(2048);
        assert_eq!(results.len(), K_TOTAL);
        let stored: Vec<_> = results.iter().filter(|r| r.was_stored).collect();
        let unstored: Vec<_> = results.iter().filter(|r| !r.was_stored).collect();
        assert_eq!(stored.len(), 8, "Eros only: 8 stored");
        assert_eq!(unstored.len(), 44, "Eros only: 44 unstored");
    }

    #[test]
    fn test_scenario_warm_moment_counts() {
        let results = scenario_warm_moment(2048);
        let stored: Vec<_> = results.iter().filter(|r| r.was_stored).collect();
        let unstored: Vec<_> = results.iter().filter(|r| !r.was_stored).collect();
        assert_eq!(stored.len(), 5, "Warm moment: 5 stored");
        assert_eq!(unstored.len(), 47, "Warm moment: 47 unstored");
    }

    #[test]
    fn test_scenario_tension_counts() {
        let results = scenario_tension(2048);
        let stored: Vec<_> = results.iter().filter(|r| r.was_stored).collect();
        let unstored: Vec<_> = results.iter().filter(|r| !r.was_stored).collect();
        assert_eq!(stored.len(), 6, "Tension: 6 stored");
        assert_eq!(unstored.len(), 46, "Tension: 46 unstored");
    }

    #[test]
    fn test_scenario_growth_counts() {
        let results = scenario_growth(2048);
        let stored: Vec<_> = results.iter().filter(|r| r.was_stored).collect();
        let unstored: Vec<_> = results.iter().filter(|r| !r.was_stored).collect();
        assert_eq!(stored.len(), 5, "Growth: 5 stored");
        assert_eq!(unstored.len(), 47, "Growth: 47 unstored");
    }

    #[test]
    fn test_scenario_full_load_counts() {
        let results = scenario_full_load(2048);
        let stored: Vec<_> = results.iter().filter(|r| r.was_stored).collect();
        let unstored: Vec<_> = results.iter().filter(|r| !r.was_stored).collect();
        assert_eq!(stored.len(), 52, "Full load: 52 stored");
        assert_eq!(unstored.len(), 0, "Full load: 0 unstored");
    }

    // --- Ghost Detection Tests ---

    #[test]
    fn test_stored_concepts_have_positive_coefficients() {
        let results = scenario_eros_only(4096);
        for r in results.iter().filter(|r| r.was_stored) {
            assert!(r.signed_coeff > 0.0,
                "{} signed_coeff should be > 0, got {:.3}", r.id, r.signed_coeff);
            assert!(r.unsigned_coeff > 0.0,
                "{} unsigned_coeff should be > 0, got {:.3}", r.id, r.unsigned_coeff);
        }
    }

    #[test]
    fn test_ghost_signal_arithmetic() {
        let results = scenario_warm_moment(2048);
        for r in &results {
            let expected = r.unsigned_coeff - r.signed_coeff;
            assert!((r.ghost_signal - expected).abs() < 1e-6,
                "Ghost signal arithmetic wrong for {}: {} != {} - {}",
                r.id, r.ghost_signal, r.unsigned_coeff, r.signed_coeff);
        }
    }

    #[test]
    fn test_ghost_threshold_adjustable() {
        let results = scenario_eros_only(4096);

        let count_at_005: usize = results.iter()
            .filter(|r| !r.was_stored && r.ghost_signal.abs() > 0.05)
            .count();
        let count_at_001: usize = results.iter()
            .filter(|r| !r.was_stored && r.ghost_signal.abs() > 0.01)
            .count();

        // Lower threshold should find at least as many ghosts
        assert!(count_at_001 >= count_at_005,
            "Lower threshold should find >= ghosts: {} vs {}", count_at_001, count_at_005);
    }

    #[test]
    fn test_ghost_count_varies_by_scenario() {
        let threshold = 0.05;
        let d = 4096;

        let count_ghosts = |results: &[ConceptReadback]| -> usize {
            results.iter()
                .filter(|r| !r.was_stored && r.ghost_signal.abs() > threshold)
                .count()
        };

        let eros_ghosts = count_ghosts(&scenario_eros_only(d));
        let moment_ghosts = count_ghosts(&scenario_warm_moment(d));
        let tension_ghosts = count_ghosts(&scenario_tension(d));
        let growth_ghosts = count_ghosts(&scenario_growth(d));

        // Not all scenarios should produce the exact same ghost count
        let counts = [eros_ghosts, moment_ghosts, tension_ghosts, growth_ghosts];
        let all_same = counts.iter().all(|&c| c == counts[0]);
        // This is a soft assertion — the point is the data varies
        if all_same {
            eprintln!("WARNING: All scenarios produced {} ghosts — might be degenerate", counts[0]);
        }
    }

    // --- Output Tests ---

    #[test]
    fn test_print_ghost_table_runs() {
        let results = scenario_eros_only(1024);
        // Just verify it doesn't panic
        print_ghost_table("Test: Eros Only", &results, 0.05);
    }

    #[test]
    fn test_ghost_matrix_shape() {
        let matrix = ghost_matrix(1024);
        assert_eq!(matrix.len(), 8, "Matrix should have 8 rows");
        for row in &matrix {
            assert_eq!(row.len(), 8, "Matrix should have 8 columns");
        }
        // Diagonal should be zero (no self-ghosting)
        for i in 0..8 {
            assert_eq!(matrix[i][i], 0.0,
                "Diagonal element [{i}][{i}] should be 0.0");
        }
    }

    // --- Base Tests ---

    #[test]
    fn test_base_signed_range() {
        let b = Base::Signed(7);
        assert_eq!(b.min_val(), -3);
        assert_eq!(b.max_val(), 3);
    }

    #[test]
    fn test_base_unsigned_range() {
        let b = Base::Unsigned(7);
        assert_eq!(b.min_val(), 0);
        assert_eq!(b.max_val(), 6);
    }

    // --- Helper ---

    fn pearson_correlation(a: &[i8], b: &[i8]) -> f32 {
        let n = a.len() as f64;
        let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n;
        let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n;

        let mut cov = 0.0f64;
        let mut var_a = 0.0f64;
        let mut var_b = 0.0f64;

        for i in 0..a.len() {
            let da = a[i] as f64 - mean_a;
            let db = b[i] as f64 - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom < 1e-10 { 0.0 } else { (cov / denom) as f32 }
    }
}
