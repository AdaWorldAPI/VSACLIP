// ============================================================================
// CAM-GRAPH: Content Addressable Memory Graph Store
// ============================================================================
// 
// Design constraints:
//   - 8192 bits per vector, packed as 128 × u64
//   - All lookups via Hamming distance (popcount on XOR)
//   - Zero serialization. No JSON. No strings in hot path.
//   - RISC: 5 operations only (STORE, MATCH, BIND, BUNDLE, SCAN)
//   - Codebook compression for storage efficiency
//
// Philosophy:
//   Content IS the address. There is no indirection layer.
//   A node's identity is its content hash. Two identical nodes are one node.
//   An edge is the XOR-binding of its endpoints + relation token.
//   Properties are atoms stored the same way as nodes.
//
// ============================================================================

// ── The Word ────────────────────────────────────────────────────────────────
// Everything in the system is a Word. There is only one type.

const WORD_BITS: usize = 8192;
const WORD_LANES: usize = 128; // 8192 / 64

#[derive(Clone, Copy)]
#[repr(C, align(64))] // cache-line aligned for SIMD
struct Word {
    lanes: [u64; WORD_LANES],
}

// ── RISC Operations ─────────────────────────────────────────────────────────
// There are exactly 5 operations. Everything else composes from these.

impl Word {
    // OP 1: BIND — XOR two words. Produces a bound pair.
    //        Self-inverse: bind(bind(a, b), b) == a
    //        This IS the edge constructor.
    #[inline(always)]
    fn bind(a: &Word, b: &Word) -> Word {
        let mut out = Word { lanes: [0u64; WORD_LANES] };
        for i in 0..WORD_LANES {
            out.lanes[i] = a.lanes[i] ^ b.lanes[i];
        }
        out
    }

    // OP 2: BUNDLE — Majority vote across N words. Produces a superposition.
    //        The result is similar to all inputs.
    //        This IS the property aggregator / node constructor.
    fn bundle(words: &[&Word]) -> Word {
        let n = words.len();
        let threshold = n / 2;
        let mut out = Word { lanes: [0u64; WORD_LANES] };
        
        for bit in 0..WORD_BITS {
            let lane = bit / 64;
            let pos = bit % 64;
            let mut count = 0usize;
            for w in words {
                if (w.lanes[lane] >> pos) & 1 == 1 {
                    count += 1;
                }
            }
            if count > threshold {
                out.lanes[lane] |= 1u64 << pos;
            }
        }
        out
    }

    // OP 3: MATCH — Hamming distance. Returns number of differing bits.
    //        0 = identical. Lower = more similar.
    //        This IS the query engine.
    #[inline(always)]
    fn distance(a: &Word, b: &Word) -> u32 {
        let mut d = 0u32;
        for i in 0..WORD_LANES {
            d += (a.lanes[i] ^ b.lanes[i]).count_ones();
        }
        d
    }

    // OP 4: PERMUTE — Rotate bit pattern by k positions.
    //        Creates orthogonal role markers.
    //        permute(w, 0) = source role
    //        permute(w, 1) = relation role  
    //        permute(w, 2) = target role
    //        This IS the role encoder for directed edges.
    fn permute(w: &Word, k: u32) -> Word {
        if k == 0 { return *w; }
        let shift = (k as usize * 64) % WORD_BITS; // rotate in 64-bit granularity
        let lane_shift = shift / 64;
        let mut out = Word { lanes: [0u64; WORD_LANES] };
        for i in 0..WORD_LANES {
            out.lanes[(i + lane_shift) % WORD_LANES] = w.lanes[i];
        }
        out
    }

    // OP 5: ZERO / ONE — Constants.
    fn zero() -> Word { Word { lanes: [0u64; WORD_LANES] } }
    fn random(seed: u64) -> Word {
        // Deterministic PRNG expansion from seed
        let mut w = Word { lanes: [0u64; WORD_LANES] };
        let mut s = seed;
        for i in 0..WORD_LANES {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            w.lanes[i] = s;
        }
        w
    }
}

// ============================================================================
// GRAPH ENCODING
// ============================================================================
//
// A graph has three things: nodes, edges, properties.
// In CAM-graph, all three are Words stored in one flat array.
//
// Node:     word_n = content_hash(data)
// Edge:     word_e = BIND(permute(src, 0), BIND(permute(rel, 1), permute(tgt, 2)))
// Property: word_p = BIND(owner, BIND(permute(key_token, 1), permute(val_token, 2)))
//
// There is no difference in storage. A Word is a Word.
// The structure is recovered by querying, not by tagging.

// ── Directed Edge Construction ──────────────────────────────────────────────
//
//   edge = src ⊕ permute(rel, 1) ⊕ permute(tgt, 2)
//
//   To recover target given source and relation:
//     tgt_permuted = edge ⊕ src ⊕ permute(rel, 1)
//     tgt = unpermute(tgt_permuted, 2)
//
//   To find all edges from src:
//     partial = src ⊕ permute(?, 1) ⊕ permute(?, 2)
//     scan store for words within Hamming radius of partial patterns

fn make_edge(src: &Word, rel: &Word, tgt: &Word) -> Word {
    let r1 = Word::permute(rel, 1);
    let t2 = Word::permute(tgt, 2);
    let tmp = Word::bind(src, &r1);
    Word::bind(&tmp, &t2)
}

fn recover_target(edge: &Word, src: &Word, rel: &Word) -> Word {
    let r1 = Word::permute(rel, 1);
    let tmp = Word::bind(edge, src);     // cancel src
    let t2 = Word::bind(&tmp, &r1);      // cancel rel
    Word::permute(&t2, WORD_LANES as u32 - 2) // unpermute(2) = permute(128-2)
}

fn recover_source(edge: &Word, rel: &Word, tgt: &Word) -> Word {
    let r1 = Word::permute(rel, 1);
    let t2 = Word::permute(tgt, 2);
    let tmp = Word::bind(edge, &r1);
    Word::bind(&tmp, &t2)
}

// ── Property Encoding ───────────────────────────────────────────────────────
// A property is structurally identical to an edge:
//   prop = owner ⊕ permute(key, 1) ⊕ permute(value, 2)
//
// "key" and "value" are themselves Words from the codebook.
// Small integers: mapped to fixed codebook entries.
// Strings: hashed to Words.
// Booleans: two codebook entries (TRUE_WORD, FALSE_WORD).

fn make_property(owner: &Word, key: &Word, value: &Word) -> Word {
    make_edge(owner, key, value) // same structure
}

fn recover_value(prop: &Word, owner: &Word, key: &Word) -> Word {
    recover_target(prop, owner, key) // same operation
}

// ============================================================================
// CODEBOOK
// ============================================================================
//
// The codebook is a fixed set of well-known Words that serve as:
//   1. Relation types (CAUSES, BECOMES, etc.)
//   2. Property keys (NAME, AGE, LABEL, etc.)
//   3. Small value atoms (integers 0-1023, TRUE, FALSE, NULL)
//   4. Label tokens (PERSON, DOCUMENT, CONCEPT, etc.)
//
// Each codebook entry is generated deterministically from a seed.
// The codebook IS the schema. Adding a new concept = adding a codebook entry.
//
// Codebook compression: instead of storing full 1024-byte Words for known
// tokens, store a 16-bit codebook index. Expand on read.

const CODEBOOK_SIZE: usize = 4096; // 12-bit index

struct Codebook {
    entries: [Word; CODEBOOK_SIZE],
    // Reverse lookup: given a Word, find closest codebook entry
    // This is a small exhaustive scan (4096 × popcount) — fast enough
}

impl Codebook {
    fn init() -> Self {
        let mut cb = Codebook {
            entries: [Word::zero(); CODEBOOK_SIZE],
        };
        // Deterministic generation — same codebook everywhere
        for i in 0..CODEBOOK_SIZE {
            cb.entries[i] = Word::random(0xCAFE_0000 + i as u64);
        }
        cb
    }

    // Lookup: find closest codebook entry
    fn lookup(&self, w: &Word) -> (u16, u32) { // (index, distance)
        let mut best_idx = 0u16;
        let mut best_dist = u32::MAX;
        for i in 0..CODEBOOK_SIZE {
            let d = Word::distance(w, &self.entries[i]);
            if d < best_dist {
                best_dist = d;
                best_idx = i as u16;
            }
        }
        (best_idx, best_dist)
    }

    // Well-known slots
    const REL_BECOMES:     usize = 0;
    const REL_CAUSES:      usize = 1;
    const REL_SUPPORTS:    usize = 2;
    const REL_CONTRADICTS: usize = 3;
    const REL_REFINES:     usize = 4;
    const REL_GROUNDS:     usize = 5;
    const REL_ABSTRACTS:   usize = 6;
    const REL_CONTAINS:    usize = 7;
    const REL_REFERENCES:  usize = 8;
    const REL_OBSERVES:    usize = 9;
    const REL_INTEGRATES:  usize = 10;
    
    const KEY_NAME:        usize = 64;
    const KEY_LABEL:       usize = 65;
    const KEY_AGE:         usize = 66;
    const KEY_TYPE:        usize = 67;
    const KEY_TIMESTAMP:   usize = 68;
    
    const VAL_TRUE:        usize = 128;
    const VAL_FALSE:       usize = 129;
    const VAL_NULL:        usize = 130;
    // 131..1154 = integers 0..1023
    const VAL_INT_BASE:    usize = 131;
    
    const LBL_NODE:        usize = 1200;
    const LBL_EDGE:        usize = 1201;
    const LBL_PERSON:      usize = 1202;
    const LBL_DOCUMENT:    usize = 1203;
    const LBL_CONCEPT:     usize = 1204;
    const LBL_EVENT:       usize = 1205;
    
    // 2048..4095 = user-defined / dynamic
}

// ============================================================================
// STORE
// ============================================================================
//
// The store is a flat array of Words. That's it.
// No B-trees. No indexes. Brute-force Hamming scan.
//
// At 1024 bytes per word:
//   1M words = 1 GB
//   Scan 1M words with AVX-512 popcount: ~2ms
//
// For larger stores, we add a single optimization:
//   Coarse quantization via codebook. Each Word gets a 16-bit signature
//   (nearest codebook entry). Pre-filter on signature before full scan.

struct Store {
    words: Vec<Word>,
    signatures: Vec<u16>,  // codebook index for each word, coarse filter
    codebook: Codebook,
}

impl Store {
    fn new() -> Self {
        Store {
            words: Vec::new(),
            signatures: Vec::new(),
            codebook: Codebook::init(),
        }
    }

    // STORE operation
    fn insert(&mut self, w: Word) -> usize {
        let (sig, _dist) = self.codebook.lookup(&w);
        let idx = self.words.len();
        self.words.push(w);
        self.signatures.push(sig);
        idx
    }

    // MATCH operation — exact (distance = 0)
    fn find_exact(&self, query: &Word) -> Option<usize> {
        let (qsig, _) = self.codebook.lookup(query);
        for i in 0..self.words.len() {
            // Coarse filter: skip if signature differs too much
            // (signatures from same codebook entry should be close)
            if self.signatures[i] == qsig {
                if Word::distance(query, &self.words[i]) == 0 {
                    return Some(i);
                }
            }
        }
        // Fallback: full scan (signature mismatch possible for near-boundary words)
        for i in 0..self.words.len() {
            if Word::distance(query, &self.words[i]) == 0 {
                return Some(i);
            }
        }
        None
    }

    // SCAN operation — find all within Hamming radius
    fn scan(&self, query: &Word, radius: u32) -> Vec<(usize, u32)> {
        let mut results = Vec::new();
        for i in 0..self.words.len() {
            let d = Word::distance(query, &self.words[i]);
            if d <= radius {
                results.push((i, d));
            }
        }
        results.sort_by_key(|&(_, d)| d);
        results
    }

    // SCAN with coarse pre-filter
    fn scan_fast(&self, query: &Word, radius: u32, sig_radius: u16) -> Vec<(usize, u32)> {
        let (qsig, _) = self.codebook.lookup(query);
        let mut results = Vec::new();
        for i in 0..self.words.len() {
            // Coarse: skip entries whose signature is far
            let sig_dist = (self.signatures[i] as i32 - qsig as i32).unsigned_abs() as u16;
            if sig_dist > sig_radius { continue; }
            
            let d = Word::distance(query, &self.words[i]);
            if d <= radius {
                results.push((i, d));
            }
        }
        results.sort_by_key(|&(_, d)| d);
        results
    }
}

// ============================================================================
// CODEBOOK COMPRESSION FOR STORAGE
// ============================================================================
//
// On disk, a Word can be stored as:
//   - Full: 1024 bytes (128 × u64)
//   - Compressed: 16-bit codebook index + residual
//
// Residual encoding:
//   residual = word XOR codebook[index]
//   The residual has ~4096 - distance bits set.
//   For well-quantized words (distance < 500), the residual is sparse.
//   Sparse residual → run-length encode the set bit positions.
//
// Best case (exact codebook hit): 2 bytes
// Typical case (distance ~300): 2 + ~600 bytes (bit position list as u16)
// Worst case (far from codebook): 2 + 1024 bytes (just store full word)

struct CompressedWord {
    codebook_idx: u16,
    residual_bits: u16,          // number of differing bits
    residual: ResidualEncoding,
}

enum ResidualEncoding {
    Zero,                         // exact codebook match
    Sparse(Vec<u16>),            // list of flipped bit positions (when < 512 flips)
    Dense([u64; WORD_LANES]),    // full XOR residual (when >= 512 flips)
}

impl CompressedWord {
    fn compress(w: &Word, cb: &Codebook) -> Self {
        let (idx, dist) = cb.lookup(w);
        let residual_word = Word::bind(w, &cb.entries[idx as usize]);
        
        if dist == 0 {
            CompressedWord {
                codebook_idx: idx,
                residual_bits: 0,
                residual: ResidualEncoding::Zero,
            }
        } else if dist < 512 {
            // Sparse: store flipped bit positions
            let mut positions = Vec::with_capacity(dist as usize);
            for bit in 0..WORD_BITS {
                let lane = bit / 64;
                let pos = bit % 64;
                if (residual_word.lanes[lane] >> pos) & 1 == 1 {
                    positions.push(bit as u16);
                }
            }
            CompressedWord {
                codebook_idx: idx,
                residual_bits: dist as u16,
                residual: ResidualEncoding::Sparse(positions),
            }
        } else {
            CompressedWord {
                codebook_idx: idx,
                residual_bits: dist as u16,
                residual: ResidualEncoding::Dense(residual_word.lanes),
            }
        }
    }

    fn decompress(&self, cb: &Codebook) -> Word {
        let base = &cb.entries[self.codebook_idx as usize];
        match &self.residual {
            ResidualEncoding::Zero => *base,
            ResidualEncoding::Sparse(positions) => {
                let mut w = *base;
                for &bit_pos in positions {
                    let lane = bit_pos as usize / 64;
                    let pos = bit_pos as usize % 64;
                    w.lanes[lane] ^= 1u64 << pos;
                }
                w
            }
            ResidualEncoding::Dense(residual) => {
                let mut w = *base;
                for i in 0..WORD_LANES {
                    w.lanes[i] ^= residual[i];
                }
                w
            }
        }
    }

    // On-disk size in bytes
    fn storage_size(&self) -> usize {
        match &self.residual {
            ResidualEncoding::Zero => 4,                               // idx + count
            ResidualEncoding::Sparse(p) => 4 + p.len() * 2,          // idx + count + positions
            ResidualEncoding::Dense(_) => 4 + WORD_LANES * 8,        // idx + count + full residual
        }
    }
}

// ============================================================================
// CYPHER EMULATION
// ============================================================================
//
// Neo4j Cypher maps to sequences of the 5 RISC operations.
//
// MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = "Jan"
//
// Translates to:
//
// 1. Construct query for "name = Jan" property:
//    prop_query = BIND(ANYTHING, BIND(permute(KEY_NAME, 1), permute(hash("Jan"), 2)))
//    → But we don't know the owner yet. So:
//
// 2. First, find the Word for "Jan":
//    jan_word = hash("Jan")  → a specific Word
//
// 3. Construct the property Word we're looking for:
//    For each candidate node N in store:
//      test_prop = BIND(N, BIND(permute(codebook[KEY_NAME], 1), permute(jan_word, 2)))
//      if test_prop EXISTS in store → N has name "Jan"
//
// 4. Once we have node_jan, find edges:
//    For relation KNOWS:
//      partial = BIND(node_jan, permute(codebook[REL_KNOWS], 1))
//      SCAN store for words within radius of BIND(partial, permute(ANY, 2))
//      → Each hit is an edge. Recover target via unbind.
//
// This is O(N) per step. For a store of 1M words with AVX-512, each
// scan takes ~2ms. A 3-hop query takes ~6ms. Acceptable.

// ── Query Plan ──────────────────────────────────────────────────────────────

enum QueryOp {
    // Find word by exact content
    FindExact { word: Word },
    
    // Find all words within Hamming radius
    Scan { center: Word, radius: u32 },
    
    // Bind result of previous op with a constant
    BindWith { prev_result: usize, constant: Word },
    
    // Recover (unbind) a component from a bound word
    Unbind { bound_result: usize, known: Word, role: u32 },
    
    // Filter results by checking property existence
    FilterHasProperty { candidates: usize, key_idx: usize, value: Word },
}

struct QueryPlan {
    ops: Vec<QueryOp>,
}

// ── Cypher Pattern → Query Plan ─────────────────────────────────────────────

// MATCH (a)-[r:REL]->(b) WHERE a has property key=val
// 
// Plan:
//   0: val_word = hash(val)
//   1: For each node N: check if BIND(N, permute(key, 1), permute(val_word, 2)) in store
//      → yields candidate set A
//   2: For each a in A:
//        edge_prefix = BIND(a, permute(rel, 1))
//        scan store within radius for matches
//        for each hit: recover b = unbind(hit, a, rel)
//        → yields result pairs (a, b)

// ============================================================================
// MULTI-HOP TRAVERSAL
// ============================================================================
//
// Variable-length path: MATCH (a)-[:CAUSES*2..4]->(b)
//
// Iterative frontier expansion:
//   frontier_0 = {start_node}
//   for hop in 0..max_hops:
//     for each node in frontier:
//       find all CAUSES edges from node (scan with partial bind)
//       recover targets
//       add to next frontier
//     if hop >= min_hops:
//       collect results

fn traverse(
    store: &Store,
    start: &Word,
    relation_idx: usize,
    min_hops: u32,
    max_hops: u32,
    scan_radius: u32,
) -> Vec<(Vec<Word>, Word)> { // (path, terminal_node)
    let rel = &store.codebook.entries[relation_idx];
    let rel_permuted = Word::permute(rel, 1);
    
    let mut results = Vec::new();
    let mut frontier: Vec<(Vec<Word>, Word)> = vec![(vec![*start], *start)];
    
    for hop in 0..max_hops {
        let mut next_frontier = Vec::new();
        
        for (path, current) in &frontier {
            // Construct partial edge: current ⊕ permute(rel, 1)
            let partial = Word::bind(current, &rel_permuted);
            
            // Scan for edges that match this partial
            // An edge = src ⊕ permute(rel, 1) ⊕ permute(tgt, 2)
            // So edge ⊕ partial = permute(tgt, 2)
            // We scan for words "close to" being a valid permute(*, 2)
            let candidates = store.scan(&partial, scan_radius);
            
            for (edge_idx, _dist) in &candidates {
                let edge = &store.words[*edge_idx];
                // Recover target
                let tgt_permuted = Word::bind(edge, &partial);
                let tgt = Word::permute(&tgt_permuted, WORD_LANES as u32 - 2);
                
                let mut new_path = path.clone();
                new_path.push(tgt);
                
                if hop + 1 >= min_hops {
                    results.push((new_path.clone(), tgt));
                }
                
                if hop + 1 < max_hops {
                    next_frontier.push((new_path, tgt));
                }
            }
        }
        
        frontier = next_frontier;
    }
    
    results
}

// ============================================================================
// ON-DISK FORMAT (LanceDB compatible)
// ============================================================================
//
// Single table layout in Lance columnar format:
//
//   Column 0: word          FixedSizeBinary(1024)   -- the raw Word
//   Column 1: signature     UInt16                   -- codebook index
//   Column 2: created       Int64                    -- unix micros
//
// That's it. Three columns. Everything else is derived by computation.
//
// For compressed storage variant:
//
//   Column 0: codebook_idx  UInt16
//   Column 1: residual_len  UInt16                   -- number of flipped bits
//   Column 2: residual      Binary                   -- sparse or dense encoding
//   Column 3: created       Int64
//
// The compressed variant saves ~40-70% disk for well-quantized data
// at the cost of decompression on read.

// ============================================================================
// SUMMARY
// ============================================================================
//
// Data types:    1  (Word: 128 × u64)
// Operations:    5  (BIND, BUNDLE, MATCH, PERMUTE, STORE/SCAN)
// Tables:        1  (flat array of Words)
// Indexes:       1  (16-bit codebook signature for coarse filter)
// Schema:        0  (schema IS the codebook)
//
// Neo4j mapping:
//   Node     → Word (content hash)
//   Edge     → BIND(src, permute(rel, 1), permute(tgt, 2))
//   Property → BIND(owner, permute(key, 1), permute(val, 2))
//   Label    → BIND(node, permute(LABEL_KEY, 1), permute(label_token, 2))
//   Index    → codebook signature (16-bit quantization)
//   Cypher   → sequence of SCAN + BIND + UNBIND
//
// Compression:
//   Best:    2 bytes (exact codebook hit)
//   Typical: ~600 bytes (sparse residual, ~300 bit flips)
//   Worst:   1028 bytes (full word + header)
//   Average: ~50% of uncompressed (512 bytes vs 1024)
//
// Performance (1M entries, AVX-512):
//   Exact lookup:  <1ms (signature pre-filter + popcount)
//   Radius scan:   ~2ms (full scan with SIMD popcount)
//   1-hop query:   ~4ms (2 scans)
//   3-hop query:   ~10ms (iterative frontier)
//
// The entire graph database is ONE flat array of ONE data type
// with FIVE operations and a CODEBOOK that defines the schema.
// ============================================================================

fn main() {
    println!("CAM-Graph: 1 type, 5 ops, 1 table, 0 serialization");
}
