#!/usr/bin/env python3
"""
HDR POPCNT Sweep — Proof of Concept v1.1
Fix: d=8192 (matching actual container spec), majority-vote superposition
"""

import numpy as np
from time import perf_counter

# ============================================================
# CORE PRIMITIVES
# ============================================================

D_DEFAULT = 8192  # matches our actual container Q3 size: 128 × u64

def random_binary(d=D_DEFAULT):
    n = d // 64
    hi = np.random.randint(0, 2**32, size=n, dtype=np.uint64) << np.uint64(32)
    lo = np.random.randint(0, 2**32, size=n, dtype=np.uint64)
    return hi | lo

_POP8 = np.zeros(256, dtype=np.uint32)
for _i in range(256):
    _POP8[_i] = bin(_i).count('1')

def hamming(a, b):
    return int(np.sum(_POP8[np.bitwise_xor(a, b).view(np.uint8)]))

def hamming_partial(a, b, start_u64, count_u64):
    seg = np.bitwise_xor(a[start_u64:start_u64+count_u64],
                          b[start_u64:start_u64+count_u64])
    return int(np.sum(_POP8[seg.view(np.uint8)]))

def xor_superpose(vectors):
    """XOR fold — works when K < sqrt(d)"""
    result = vectors[0].copy()
    for v in vectors[1:]:
        np.bitwise_xor(result, v, out=result)
    return result

def majority_superpose(vectors, d=D_DEFAULT):
    """Majority-vote superposition — robust for any K.
    Each bit is set if majority of input vectors have it set.
    Cost: O(K × d/8) but only used at ingest, not query time."""
    K = len(vectors)
    if K == 1:
        return vectors[0].copy()
    
    # Count bits via byte-level accumulation
    n_u64 = d // 64
    n_bytes = n_u64 * 8
    
    # For each bit position, count how many vectors have it set
    # Work byte-by-byte for memory efficiency
    result = np.zeros(n_u64, dtype=np.uint64)
    
    # Stack all vectors as byte arrays
    byte_arrays = [v.view(np.uint8) for v in vectors]
    
    for byte_idx in range(n_bytes):
        byte_val = np.uint8(0)
        for bit in range(8):
            count = 0
            mask = np.uint8(1 << bit)
            for arr in byte_arrays:
                if arr[byte_idx] & mask:
                    count += 1
            if count > K // 2:  # majority vote
                byte_val |= mask
        # Write to result
        result_bytes = result.view(np.uint8)
        result_bytes[byte_idx] = byte_val
    
    return result

def make_near(ref, target_dist, d=D_DEFAULT):
    c = ref.copy()
    bits = np.random.choice(d, min(target_dist, d), replace=False)
    for bit in bits:
        c[bit // 64] ^= np.uint64(1 << (bit % 64))
    return c


# ============================================================
# TEST 1: SUPERPOSITION — XOR vs MAJORITY
# ============================================================

def test_superposition():
    print("=" * 60)
    print("TEST 1: Superposition — XOR vs Majority Vote")
    print(f"  d = {D_DEFAULT}")
    print("=" * 60)

    d = D_DEFAULT
    n_features = 1000
    features = [random_binary(d) for _ in range(n_features)]
    target_idx = 42
    target = features[target_idx]

    for K in [1, 5, 10, 25, 49]:
        close_patch = make_near(target, int(d * 0.10), d)
        random_patches = [random_binary(d) for _ in range(K - 1)]
        all_patches = random_patches + [close_patch]

        # XOR superposition
        xor_sup = xor_superpose(all_patches)
        xor_dist = hamming(xor_sup, target)
        xor_dists = sorted([(hamming(xor_sup, f), i) for i, f in enumerate(features)])
        xor_rank = next(r for r, (_, i) in enumerate(xor_dists) if i == target_idx)

        # Majority superposition  
        maj_sup = majority_superpose(all_patches, d)
        maj_dist = hamming(maj_sup, target)
        maj_dists = sorted([(hamming(maj_sup, f), i) for i, f in enumerate(features)])
        maj_rank = next(r for r, (_, i) in enumerate(maj_dists) if i == target_idx)

        print(f"  K={K:3d} | XOR: dist={xor_dist:5d} rank={xor_rank:4d} {'✓' if xor_rank<20 else '⚠'} | "
              f"MAJ: dist={maj_dist:5d} rank={maj_rank:4d} {'✓' if maj_rank<20 else '⚠'}")

    print()
    return True


# ============================================================
# TEST 2: EARLY EXIT
# ============================================================

def test_early_exit():
    print("=" * 60)
    print("TEST 2: Early Exit — Zero False Negatives")
    print(f"  d = {D_DEFAULT}")
    print("=" * 60)

    d = D_DEFAULT
    u64s = d // 64
    n = 10_000
    threshold = int(d * 0.30)  # 30% = 2458 for d=8192
    safety = 1.5

    containers = [random_binary(d) for _ in range(n)]
    query = random_binary(d)

    # Plant 50 matches
    for i in range(50):
        target_d = np.random.randint(int(d*0.05), int(d*0.28))
        containers[i * 200] = make_near(query, target_d, d)

    # Full sweep
    t0 = perf_counter()
    full_matches = {}
    for i, c in enumerate(containers):
        dist = hamming(query, c)
        if dist < threshold:
            full_matches[i] = dist
    full_ops = n * u64s
    t_full = perf_counter() - t0

    # Three-stage early exit
    t0 = perf_counter()
    early_ops = 0

    # Stage 1: 64-bit
    s1_t = int(threshold * (64 / d) * safety)
    stage1 = [i for i, c in enumerate(containers) if hamming_partial(query, c, 0, 1) <= s1_t]
    early_ops += n

    # Stage 2: 512-bit (8 u64) — wider than before for d=8192
    s2_t = int(threshold * (512 / d) * safety)
    stage2 = [i for i in stage1 if hamming_partial(query, containers[i], 0, 8) <= s2_t]
    early_ops += len(stage1) * 8

    # Stage 3: full
    early_matches = {}
    for i in stage2:
        dist = hamming(query, containers[i])
        if dist < threshold:
            early_matches[i] = dist
    early_ops += len(stage2) * u64s
    t_early = perf_counter() - t0

    missed = set(full_matches) - set(early_matches)

    print(f"  Threshold: {threshold} ({threshold/d*100:.0f}% of {d})")
    print(f"  Full:  {len(full_matches):4d} matches | {full_ops:>10,} ops | {t_full*1000:.1f}ms")
    print(f"  Early: {len(early_matches):4d} matches | {early_ops:>10,} ops | {t_early*1000:.1f}ms")
    print(f"  Stage 1: {n:,} → {len(stage1):,} ({len(stage1)/n*100:.1f}%)")
    print(f"  Stage 2: {len(stage1):,} → {len(stage2):,}")
    print(f"  Instruction speedup: {full_ops/max(early_ops,1):.1f}×")
    print(f"  False negatives: {len(missed)}")
    print(f"  {'✓ ZERO FALSE NEGATIVES' if not missed else '✗ MISSED ' + str(len(missed))}")
    print()
    return len(missed) == 0


# ============================================================
# TEST 3: HDR SCORING
# ============================================================

def test_hdr_scoring():
    print("=" * 60)
    print("TEST 3: HDR Exposure Scoring")
    print(f"  d = {D_DEFAULT}")
    print("=" * 60)

    d = D_DEFAULT
    n = 5000
    containers = [random_binary(d) for _ in range(n)]
    query = random_binary(d)

    plants = [
        (0, int(d*0.05), "blazing"),   # 410
        (1, int(d*0.15), "strong"),     # 1229
        (2, int(d*0.25), "medium"),     # 2048
        (3, int(d*0.44), "weak"),       # 3604
    ]
    for idx, td, _ in plants:
        containers[idx] = make_near(query, td, d)

    def hdr(pc, d=D_DEFAULT):
        hot  = 3 if pc < d * 0.10 else 0   # < 819
        mid  = 2 if pc < d * 0.30 else 0   # < 2458
        cold = 1 if pc < d * 0.49 else 0   # < 4014
        return hot + mid + cold

    for idx, td, label in plants:
        dist = hamming(query, containers[idx])
        print(f"  '{label}' (target={td:5d}): actual={dist:5d}, HDR={hdr(dist)}")

    dist_map = {}
    for c in containers:
        s = hdr(hamming(query, c))
        dist_map[s] = dist_map.get(s, 0) + 1

    print(f"\n  HDR distribution ({n} containers):")
    for s in sorted(dist_map, reverse=True):
        cnt = dist_map[s]
        print(f"    HDR={s}: {cnt:5d} ({cnt/n*100:5.1f}%) {'█' * min(cnt//5, 50)}")

    noise = dist_map.get(0, 0) / n * 100
    print(f"\n  Noise floor: {noise:.1f}%  {'✓' if noise > 85 else '⚠'}")
    print()
    return noise > 85


# ============================================================
# TEST 4: RESONANCE CASCADE
# ============================================================

def test_resonance_cascade():
    print("=" * 60)
    print("TEST 4: Resonance Cascade — Organic Growth")
    print(f"  d = {D_DEFAULT}, using majority-vote superposition")
    print("=" * 60)

    d = D_DEFAULT
    threshold = int(d * 0.35)  # 35% = 2867
    n_seeds = 20
    n_classes = 5

    features = [random_binary(d) for _ in range(n_seeds)]
    activations = [0] * n_seeds
    protos = [random_binary(d) for _ in range(n_classes)]
    new_count = 0
    bind_count = 0

    for img in range(200):
        cls = img % n_classes
        # 9 patches, each 20% away from prototype
        patches = [make_near(protos[cls], int(d * 0.20), d) for _ in range(9)]
        # Majority-vote: robust superposition
        sup = majority_superpose(patches, d)

        matched = False
        for i, f in enumerate(features):
            if hamming(sup, f) < threshold:
                matched = True
                activations[i] += 1
                bind_count += 1

        if not matched:
            features.append(sup.copy())
            activations.append(1)
            new_count += 1

    print(f"  Seeds: {n_seeds} → Total: {len(features)} (+{new_count} organic)")
    print(f"  Binds: {bind_count}")
    print(f"  Active (>5): {sum(1 for a in activations if a > 5)}")
    print(f"  Dead (0):    {sum(1 for a in activations if a == 0)}")

    matched_classes = 0
    for c in range(n_classes):
        dists = [(hamming(protos[c], f), i) for i, f in enumerate(features)]
        closest_d, closest_i = min(dists)
        ok = closest_d < threshold
        if ok: matched_classes += 1
        print(f"    Class {c}: closest={closest_d:5d} (feat #{closest_i}) {'✓' if ok else '✗'}")

    print(f"\n  {matched_classes}/{n_classes} classes matched  {'✓' if matched_classes >= 4 else '✗'}")
    print()
    return matched_classes >= 4


# ============================================================
# TEST 5: THREE-LAYER PIPELINE
# ============================================================

def test_full_pipeline():
    print("=" * 60)
    print("TEST 5: Three-Layer Pipeline")
    print(f"  d = {D_DEFAULT}")
    print("=" * 60)

    d = D_DEFAULT
    n_feat = 100
    features = [random_binary(d) for _ in range(n_feat)]

    # Parts = majority superposition of 2-3 features
    parts, part_r = [], []
    for _ in range(20):
        ing = list(np.random.choice(n_feat, np.random.randint(2, 4), replace=False))
        parts.append(majority_superpose([features[j] for j in ing], d))
        part_r.append(ing)

    # Objects = majority superposition of 3-5 parts
    objects, obj_r = [], []
    for _ in range(5):
        ing = list(np.random.choice(20, np.random.randint(3, 6), replace=False))
        objects.append(majority_superpose([parts[j] for j in ing], d))
        obj_r.append(ing)

    target = 2
    t_parts = obj_r[target]
    t_feats = list(set(f for p in t_parts for f in part_r[p]))

    # Patches: features with 10% noise + 5 random
    patches = [make_near(features[f], int(d * 0.10), d) for f in t_feats]
    patches += [random_binary(d) for _ in range(5)]

    print(f"  Target: object {target} ← parts {t_parts} ← {len(t_feats)} features")

    # L1: majority superpose patches → sweep features
    sup1 = majority_superpose(patches, d)
    l1_thresh = int(d * 0.38)
    l1 = sorted([(hamming(sup1, f), i) for i, f in enumerate(features)])
    l1_act = [(i, dd) for dd, i in l1 if dd < l1_thresh]
    t_found = sum(1 for i, _ in l1_act if i in t_feats)
    print(f"\n  L1: {len(l1_act)} features ({t_found}/{len(t_feats)} target) [thresh={l1_thresh}]")

    if not l1_act:
        # Fallback: take top-20 closest
        l1_act = [(i, dd) for dd, i in l1[:20]]
        print(f"  (fallback: top 20, closest dist={l1_act[0][1]})")

    # L2: majority superpose activated features → sweep parts
    sup2 = majority_superpose([features[i] for i, _ in l1_act[:20]], d)
    l2_thresh = int(d * 0.40)
    l2 = sorted([(hamming(sup2, p), i) for i, p in enumerate(parts)])
    l2_act = [(i, dd) for dd, i in l2 if dd < l2_thresh]
    if l2_act:
        tp_found = sum(1 for i, _ in l2_act if i in t_parts)
        print(f"  L2: {len(l2_act)} parts ({tp_found}/{len(t_parts)} target) [thresh={l2_thresh}]")
    else:
        l2_act = [(i, dd) for dd, i in l2[:5]]
        print(f"  L2: (fallback top 5, closest dist={l2_act[0][1]})")

    # L3: majority superpose parts → sweep objects
    sup3 = majority_superpose([parts[i] for i, _ in l2_act[:8]], d)
    l3 = sorted([(hamming(sup3, o), i) for i, o in enumerate(objects)])
    best_d, best_i = l3[0]
    recognized = best_i == target

    print(f"  L3: best = object {best_i} dist={best_d}")
    for dd, ii in l3[:3]:
        tag = " ← TARGET" if ii == target else ""
        print(f"      object {ii}: {dd}{tag}")

    print(f"\n  {'✓ RECOGNIZED' if recognized else '✗ not recognized'}")
    print()
    return recognized


# ============================================================
# TEST 6: INSTRUCTION COUNT
# ============================================================

def test_instruction_count():
    print("=" * 60)
    print("TEST 6: Instruction Count at Scale")
    print(f"  d = {D_DEFAULT}")
    print("=" * 60)

    d = D_DEFAULT
    u64s = d // 64  # 128

    for n in [10_000, 100_000, 1_000_000, 10_000_000]:
        full = n * u64s
        s1 = int(n * 0.05)
        s2 = int(s1 * 0.10)
        early = n + s1 * 8 + s2 * u64s  # stage 1=1, stage 2=8, stage 3=128
        sp = full / early

        us_scalar = early / 3_000
        us_avx512 = early / 24_000

        print(f"\n  N={n:>12,}")
        print(f"    Full: {full/3_000:>10.0f} μs | Early: {us_scalar:>8.0f} μs | "
              f"AVX-512: {us_avx512:>6.0f} μs | {sp:.0f}× speedup")

    print()
    return True


# ============================================================
# BONUS: ANTI-RESONANCE
# ============================================================

def test_anti_resonance():
    print("=" * 60)
    print("BONUS: Anti-Resonance")
    print("=" * 60)

    d = D_DEFAULT
    orig = random_binary(d)
    anti = np.bitwise_not(orig)
    close = make_near(orig, int(d * 0.10), d)
    neutral = random_binary(d)

    print(f"  Close:   {hamming(orig, close):5d} / {d}  (resonance)")
    print(f"  Neutral: {hamming(orig, neutral):5d} / {d}  (noise)")
    print(f"  Anti:    {hamming(orig, anti):5d} / {d}  (inhibition)")
    print()
    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("\n" + "═" * 60)
    print("  HDR POPCNT SWEEP — PROOF v1.1")
    print(f"  d = {D_DEFAULT} (8KB containers)")
    print("═" * 60 + "\n")

    tests = [
        ("Superposition",     test_superposition),
        ("Early Exit",        test_early_exit),
        ("HDR Scoring",       test_hdr_scoring),
        ("Resonance Cascade", test_resonance_cascade),
        ("Three-Layer Pipe",  test_full_pipeline),
        ("Instruction Count", test_instruction_count),
        ("Anti-Resonance",    test_anti_resonance),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  ✗ EXCEPTION in {name}: {e}")
            import traceback; traceback.print_exc()
            results.append((name, False))

    print("═" * 60)
    print("  RESULTS")
    print("═" * 60)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print()
    ok = all(p for _, p in results)
    print("  " + ("ALL PASSED. Port to Rust." if ok else "Fix failures."))
    print()
