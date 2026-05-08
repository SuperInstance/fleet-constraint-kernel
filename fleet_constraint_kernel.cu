#include "fleet_constraint.h"
#include <cmath>
#include <cstdio>

// ══════════════════════════════════════════════════════════════════════
//  Eisenstein beam map — hex-lattice beam indexing
// ══════════════════════════════════════════════════════════════════════
// Basis: e1 = (1, 0),  e2 = (−½, √3/2)
// Point on lattice: P = a·e1 + b·e2
// Map angle θ → nearest Eisenstein integer pair (a, b).
// Beam steering angle becomes an exact hex-grid index.

__host__ __device__
EisensteinCoord eisenstein_beam_index(float angle)
{
    // Project onto dual basis to recover (a, b) as reals, then round.
    // P = a·(1,0) + b·(−½, √3/2)
    //   = (a − b/2,  b·√3/2)
    // Inverse:
    //   x = a − b/2        →  a = x + b/2
    //   y = b·√3/2         →  b = 2y/√3
    float x = cosf(angle);
    float y = sinf(angle);
    float b_real = 2.0f * y / sqrtf(3.0f);
    float a_real = x + 0.5f * b_real;

    EisensteinCoord c;
    c.a = (int)roundf(a_real);
    c.b = (int)roundf(b_real);
    return c;
}

// ══════════════════════════════════════════════════════════════════════
//  Single-constraint evaluator (per thread)
// ══════════════════════════════════════════════════════════════════════
// Returns: margin > 0 means satisfied, ≤ 0 means violated.

__device__
float evaluate_constraint(
    const DeviceState    &dev,
    const FleetConstraint &con,
    bool                 &satisfied)
{
    float margin = 1e30f;   // start huge, narrow down

    switch (con.type) {

    case CONSTRAINT_RANGE: {
        // params: {joint_index, min, max, pad}
        // Evaluate joint range limit
        int ji = (int)con.params[0];
        float lo = con.params[1];
        float hi = con.params[2];
        float val = dev.joint_values[ji];
        margin = fminf(val - lo, hi - val);
        break;
    }

    case CONSTRAINT_TORQUE: {
        // params: {joint_index, max_torque, 0, 0}
        int ji = (int)con.params[0];
        float limit = con.params[1];
        float val = fabsf(dev.torques[ji]);
        margin = limit - val;
        break;
    }

    case CONSTRAINT_VELOCITY: {
        // params: {joint_index, max_velocity, 0, 0}
        int ji = (int)con.params[0];
        float limit = con.params[1];
        float val = fabsf(dev.velocities[ji]);
        margin = limit - val;
        break;
    }

    case CONSTRAINT_EISENSTEIN: {
        // params: {radius, scale, 0, 0}
        // End-effector must lie within Eisenstein lattice radius
        float r   = con.params[0];
        float scl = con.params[1];
        float ex  = dev.end_effector[0] * scl;
        float ey  = dev.end_effector[1] * scl;
        float dist = sqrtf(ex * ex + ey * ey);
        margin = r - dist;
        break;
    }

    case CONSTRAINT_BLOOM: {
        // params: {max_reach, 0, 0, 0}
        // Bloom filter–style reachability: end-effector within max sphere
        float reach = con.params[0];
        float dx = dev.end_effector[0];
        float dy = dev.end_effector[1];
        float dz = dev.end_effector[2];
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        margin = reach - dist;
        break;
    }

    default:
        margin = 0.0f;
        break;
    }

    satisfied = (margin > 0.0f);
    return margin;
}

// ══════════════════════════════════════════════════════════════════════
//  Fleet constraint kernel — beamformer pattern
// ══════════════════════════════════════════════════════════════════════
//
//  Sonar beamformer:  N signals × M beams → delay-and-sum per beam
//  Fleet constraint:  N devices × M constraints → evaluate-and-reduce per device
//
//  Thread mapping:
//    - blockIdx.x  = device index  (one block per device, like one block per hydrophone)
//    - threadIdx.x = constraint index within block
//    - Shared memory holds constraint definitions (like beamformer's delay table)
//
//  Reduction pattern:
//    - Each thread evaluates one constraint, writes margin to shared array
//    - Block-level reduction computes violations, worst severity, min margin
//    - Atomic add to global safety accumulator (like beamformer's atomic sum)

template <int MAX_CONSTRAINTS>   // compile-time bound for shared memory sizing
__global__
void fleet_constraint_kernel(
    const DeviceState     *__restrict__ devices,       // [N]
    const FleetConstraint *__restrict__ constraints,   // [M]
    DeviceResult          *__restrict__ results,        // [N] output
    int                                N,
    int                                M)
{
    // One block per device
    int dev_idx = blockIdx.x;
    if (dev_idx >= N) return;

    int tid = threadIdx.x;

    // ── Shared memory: constraint cache (beamformer delay-table analog) ──
    __shared__ FleetConstraint s_constraints[MAX_CONSTRAINTS];
    __shared__ float           s_margins[MAX_CONSTRAINTS];
    __shared__ int             s_satisfied[MAX_CONSTRAINTS];
    __shared__ int             s_severities[MAX_CONSTRAINTS];

    // Cooperative load of constraints into shared memory
    for (int i = tid; i < M; i += blockDim.x) {
        s_constraints[i] = constraints[i];
    }
    __syncthreads();

    // ── Per-thread constraint evaluation (like per-thread beam delay) ──
    // Initialize shared accumulators
    if (tid < MAX_CONSTRAINTS) {
        s_margins[tid]    = 1e30f;
        s_satisfied[tid]  = 1;     // assume satisfied
        s_severities[tid] = 0;
    }
    __syncthreads();

    // Each thread evaluates one constraint for this device
    // If more constraints than threads, loop (stride = blockDim.x)
    for (int ci = tid; ci < M; ci += blockDim.x) {
        bool sat = false;
        float margin = evaluate_constraint(devices[dev_idx], s_constraints[ci], sat);
        s_margins[ci]    = margin;
        s_satisfied[ci]  = sat ? 1 : 0;
        s_severities[ci] = sat ? 0 : s_constraints[ci].severity;
    }
    __syncthreads();

    // ── Block-level reduction (like beamformer's sum reduction) ──
    // We need: violation count, worst severity, min margin, weighted score
    int   local_violations = 0;
    int   local_worst_sev  = 0;
    float local_min_margin = 1e30f;
    float local_weight_sum = 0.0f;
    float local_satisfied_weight = 0.0f;

    for (int ci = tid; ci < M; ci += blockDim.x) {
        if (!s_satisfied[ci]) {
            local_violations++;
            local_worst_sev = max(local_worst_sev, s_severities[ci]);
        } else {
            local_satisfied_weight += s_constraints[ci].weight;
        }
        local_min_margin   = fminf(local_min_margin, s_margins[ci]);
        local_weight_sum  += s_constraints[ci].weight;
    }

    // Final reduction: thread 0 writes the result (simple serial reduction per block)
    __syncthreads();
    // Use shared vars for the reduction
    __shared__ int   s_total_violations;
    __shared__ int   s_total_worst_sev;
    __shared__ float s_total_min_margin;
    __shared__ float s_total_weight_sum;
    __shared__ float s_total_sat_weight;

    if (tid == 0) {
        s_total_violations = 0;
        s_total_worst_sev  = 0;
        s_total_min_margin = 1e30f;
        s_total_weight_sum = 0.0f;
        s_total_sat_weight = 0.0f;
    }
    __syncthreads();

    // Atomic accumulate from each thread's local results
    atomicAdd(&s_total_violations, local_violations);
    atomicAdd((unsigned int*)&s_total_worst_sev, (unsigned int)local_worst_sev);
    // For min-margin and weight sums we need atomics on floats
    // CUDA has atomicAdd for float; for min we use a CAS loop

    // Float atomic min via CAS (cast to int for atomicCAS)
    {
        unsigned int *addr = (unsigned int*)&s_total_min_margin;
        unsigned int old_int = *addr;
        float old_f = *(float*)&old_int;
        float desired = fminf(old_f, local_min_margin);
        while (desired < old_f) {
            unsigned int assumed_int = old_int;
            old_int = atomicCAS(addr, assumed_int, __float_as_uint(desired));
            if (old_int == assumed_int) break;
            old_f = *(float*)&old_int;
            desired = fminf(old_f, local_min_margin);
        }
    }
    atomicAdd(&s_total_weight_sum, local_weight_sum);
    atomicAdd(&s_total_sat_weight, local_satisfied_weight);

    __syncthreads();

    // Thread 0 writes the final DeviceResult (like beamformer's output sample)
    if (tid == 0) {
        float safety = (s_total_weight_sum > 0.0f)
            ? s_total_sat_weight / s_total_weight_sum
            : 1.0f;

        results[dev_idx].safety_score  = safety;
        results[dev_idx].violations    = s_total_violations;
        results[dev_idx].worst_severity = s_total_worst_sev;
        results[dev_idx].min_margin    = s_total_min_margin;
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Host-side launcher wrapper
// ══════════════════════════════════════════════════════════════════════

static const int MAX_CON = 32;   // max constraints (shared-mem tile)

void launch_fleet_constraint_kernel(
    const DeviceState     *d_devices,
    const FleetConstraint *d_constraints,
    DeviceResult          *d_results,
    int N, int M,
    cudaStream_t stream = 0)
{
    int threads = min(M, MAX_CON);
    if (threads < 1) threads = 1;
    // Pad threads to warp size for efficiency
    threads = ((threads + 31) / 32) * 32;

    fleet_constraint_kernel<MAX_CON><<<N, threads, 0, stream>>>(
        d_devices, d_constraints, d_results, N, M);
}
