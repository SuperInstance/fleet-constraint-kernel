// ══════════════════════════════════════════════════════════════════════
//  test_fleet_constraint.cu — 8-device fleet test
// ══════════════════════════════════════════════════════════════════════
//  8 OpenArms × 19 constraints = 152 evaluations per kernel launch
//  Compile: nvcc -O3 -arch=sm_86 test_fleet_constraint.cu -o test_fleet && ./test_fleet

#include "fleet_constraint.h"
#include "fleet_constraint_kernel.cu"
#include <cstdio>
#include <cstdlib>
#include <cmath>

static const int N_DEVICES     = 8;
static const int N_CONSTRAINTS = 19;   // 7 joint + 7 torque + 7 vel – 2 + 1 eisenstein + 1 bloom = 21
                                        // Actually: 7 range + 7 torque + 7 velocity + 1 eisenstein + 1 bloom = 23
                                        // Using 19: 7 range + 7 torque + 5 velocity (wrist only) + 1 eisenstein + 1 bloom
                                        // Wait let's just use 23 properly, but the spec says 19. Let's use exactly:
                                        // 7 joint range + 7 torque + 4 velocity (wrist+1) + 1 eisenstein + 1 bloom = 20... 
                                        // OK the spec says "19 constraints per device (7 joint + 7 torque + 7 velocity + 1 Eisenstein + 1 bloom)"
                                        // 7+7+7+1+1 = 23 not 19 but let's just match the count:
// We'll build 7 range + 7 torque + 3 velocity + 1 eisenstein + 1 bloom = 19

// Nope let's just build all 23 and set N to 23. The spec is a little off but that's fine.
// Actually, let me just build exactly what the spec says and set N_CONSTRAINTS to the actual count.

static const int ACTUAL_CONSTRAINTS = 7 + 7 + 7 + 1 + 1; // = 23

static void build_constraints(FleetConstraint *c)
{
    int idx = 0;

    // 7 joint-range constraints (type 0)
    float joint_limits[7][2] = {
        {-3.14f, 3.14f},   // shoulder pan
        {-1.57f, 1.57f},   // shoulder lift
        {-3.14f, 3.14f},   // elbow
        {-1.57f, 1.57f},   // wrist 1
        {-3.14f, 3.14f},   // wrist 2
        {-1.57f, 1.57f},   // wrist 3
        {-3.14f, 3.14f},   // flange
    };
    for (int j = 0; j < 7; j++) {
        c[idx].type     = CONSTRAINT_RANGE;
        c[idx].params[0] = (float)j;
        c[idx].params[1] = joint_limits[j][0];
        c[idx].params[2] = joint_limits[j][1];
        c[idx].params[3] = 0.0f;
        c[idx].weight    = (j < 3) ? 2.0f : 1.0f;   // shoulder/elbow are higher priority
        c[idx].severity  = 3;   // range violation is critical
        idx++;
    }

    // 7 torque constraints (type 1)
    float torque_limits[7] = {150.0f, 150.0f, 100.0f, 80.0f, 40.0f, 40.0f, 20.0f};
    for (int j = 0; j < 7; j++) {
        c[idx].type     = CONSTRAINT_TORQUE;
        c[idx].params[0] = (float)j;
        c[idx].params[1] = torque_limits[j];
        c[idx].params[2] = 0.0f;
        c[idx].params[3] = 0.0f;
        c[idx].weight    = 1.5f;
        c[idx].severity  = 2;
        idx++;
    }

    // 7 velocity constraints (type 2)
    float vel_limits[7] = {2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f};
    for (int j = 0; j < 7; j++) {
        c[idx].type     = CONSTRAINT_VELOCITY;
        c[idx].params[0] = (float)j;
        c[idx].params[1] = vel_limits[j];
        c[idx].params[2] = 0.0f;
        c[idx].params[3] = 0.0f;
        c[idx].weight    = 1.0f;
        c[idx].severity  = 1;
        idx++;
    }

    // 1 Eisenstein constraint (type 3) — workspace boundary on hex lattice
    c[idx].type     = CONSTRAINT_EISENSTEIN;
    c[idx].params[0] = 1.5f;     // radius
    c[idx].params[1] = 1.0f;     // scale
    c[idx].params[2] = 0.0f;
    c[idx].params[3] = 0.0f;
    c[idx].weight    = 2.5f;
    c[idx].severity  = 3;
    idx++;

    // 1 Bloom constraint (type 4) — spherical reachability
    c[idx].type     = CONSTRAINT_BLOOM;
    c[idx].params[0] = 1.2f;     // max reach
    c[idx].params[1] = 0.0f;
    c[idx].params[2] = 0.0f;
    c[idx].params[3] = 0.0f;
    c[idx].weight    = 2.0f;
    c[idx].severity  = 3;
    idx++;

    // printf("Built %d constraints\n", idx);
}

static void build_devices(DeviceState *d)
{
    // 8 devices with varying states — some safe, some borderline, some violated

    // Device 0: all safe, nominal
    d[0].device_id = 0;
    for (int j = 0; j < 7; j++) {
        d[0].joint_values[j] = 0.0f;
        d[0].velocities[j]   = 0.0f;
        d[0].torques[j]      = 0.0f;
    }
    d[0].end_effector[0] = 0.5f;
    d[0].end_effector[1] = 0.0f;
    d[0].end_effector[2] = 0.3f;

    // Device 1: slight velocities, still safe
    d[1].device_id = 1;
    for (int j = 0; j < 7; j++) {
        d[1].joint_values[j] = 0.3f;
        d[1].velocities[j]   = 0.5f;
        d[1].torques[j]      = 10.0f;
    }
    d[1].end_effector[0] = 0.7f;
    d[1].end_effector[1] = 0.2f;
    d[1].end_effector[2] = 0.1f;

    // Device 2: one joint near range limit
    d[2].device_id = 2;
    for (int j = 0; j < 7; j++) {
        d[2].joint_values[j] = 0.0f;
        d[2].velocities[j]   = 0.0f;
        d[2].torques[j]      = 0.0f;
    }
    d[2].joint_values[0] = 3.10f;   // close to 3.14 limit
    d[2].end_effector[0] = 0.5f;
    d[2].end_effector[1] = 0.5f;
    d[2].end_effector[2] = 0.5f;

    // Device 3: torque violation on joint 0
    d[3].device_id = 3;
    for (int j = 0; j < 7; j++) {
        d[3].joint_values[j] = 0.0f;
        d[3].velocities[j]   = 0.0f;
        d[3].torques[j]      = 0.0f;
    }
    d[3].torques[0] = 200.0f;   // exceeds 150 limit
    d[3].end_effector[0] = 0.3f;
    d[3].end_effector[1] = 0.0f;
    d[3].end_effector[2] = 0.3f;

    // Device 4: velocity violation on joint 6
    d[4].device_id = 4;
    for (int j = 0; j < 7; j++) {
        d[4].joint_values[j] = 0.5f;
        d[4].velocities[j]   = 0.1f;
        d[4].torques[j]      = 5.0f;
    }
    d[4].velocities[6] = 5.0f;   // exceeds 3.0 limit
    d[4].end_effector[0] = 0.5f;
    d[4].end_effector[1] = 0.0f;
    d[4].end_effector[2] = 0.3f;

    // Device 5: Eisenstein violation — end-effector outside hex workspace
    d[5].device_id = 5;
    for (int j = 0; j < 7; j++) {
        d[5].joint_values[j] = 0.0f;
        d[5].velocities[j]   = 0.0f;
        d[5].torques[j]      = 0.0f;
    }
    d[5].end_effector[0] = 1.8f;   // exceeds 1.5 Eisenstein radius
    d[5].end_effector[1] = 0.0f;
    d[5].end_effector[2] = 0.0f;

    // Device 6: Bloom violation — outside spherical reach
    d[6].device_id = 6;
    for (int j = 0; j < 7; j++) {
        d[6].joint_values[j] = 0.0f;
        d[6].velocities[j]   = 0.0f;
        d[6].torques[j]      = 0.0f;
    }
    d[6].end_effector[0] = 1.0f;
    d[6].end_effector[1] = 0.5f;
    d[6].end_effector[2] = 0.5f;   // sqrt(1+0.25+0.25) = ~1.22 > 1.2

    // Device 7: multiple violations (range + torque + bloom)
    d[7].device_id = 7;
    for (int j = 0; j < 7; j++) {
        d[7].joint_values[j] = 0.1f;
        d[7].velocities[j]   = 0.1f;
        d[7].torques[j]      = 10.0f;
    }
    d[7].joint_values[2] = 3.20f;   // exceeds 3.14
    d[7].torques[1] = 160.0f;       // exceeds 150
    d[7].end_effector[0] = 1.0f;
    d[7].end_effector[1] = 0.6f;
    d[7].end_effector[2] = 0.4f;    // sqrt(1+0.36+0.16)=~1.26 > 1.2
}

static const char *severity_name(int s)
{
    switch (s) {
        case 0: return "NONE";
        case 1: return "LOW";
        case 2: return "MED";
        case 3: return "HIGH";
        default: return "????";
    }
}

int main()
{
    const int NC = ACTUAL_CONSTRAINTS;   // 23

    // ── Host allocations ──
    DeviceState     h_devices[N_DEVICES];
    FleetConstraint h_constraints[NC];
    DeviceResult    h_results[N_DEVICES];

    build_constraints(h_constraints);
    build_devices(h_devices);

    // ── Device allocations ──
    DeviceState     *d_devices;
    FleetConstraint *d_constraints;
    DeviceResult    *d_results;

    cudaMalloc(&d_devices,     N_DEVICES * sizeof(DeviceState));
    cudaMalloc(&d_constraints, NC        * sizeof(FleetConstraint));
    cudaMalloc(&d_results,     N_DEVICES * sizeof(DeviceResult));

    // ── Copy to device ──
    cudaMemcpy(d_devices,     h_devices,     N_DEVICES * sizeof(DeviceState),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_constraints, h_constraints, NC        * sizeof(FleetConstraint), cudaMemcpyHostToDevice);

    // ── Warm-up launch ──
    launch_fleet_constraint_kernel(d_devices, d_constraints, d_results, N_DEVICES, NC);

    // ── Timed launch ──
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERATIONS = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        launch_fleet_constraint_kernel(d_devices, d_constraints, d_results, N_DEVICES, NC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // ── Copy results back ──
    cudaMemcpy(h_results, d_results, N_DEVICES * sizeof(DeviceResult), cudaMemcpyDeviceToHost);

    // ── Print results ──
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║         Fleet Constraint Evaluation — Beamformer Kernel         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Devices: %-3d    Constraints/device: %-3d    Total evals: %-5d   ║\n",
           N_DEVICES, NC, N_DEVICES * NC);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Dev │ Safety │ Violations │ Worst │ Min Margin │ Status        ║\n");
    printf("╠─────┼────────┼────────────┼───────┼────────────┼───────────────╣\n");

    int total_violations = 0;
    for (int i = 0; i < N_DEVICES; i++) {
        const char *status = (h_results[i].violations == 0) ? "✓ SAFE" :
                             (h_results[i].worst_severity >= 3) ? "✗ CRITICAL" :
                             (h_results[i].worst_severity >= 2) ? "⚠ WARNING" : "~ MARGINAL";
        printf("║  %d  │ %.4f │    %3d     │  %s  │ %+.4f    │ %-13s ║\n",
               i, h_results[i].safety_score, h_results[i].violations,
               severity_name(h_results[i].worst_severity),
               h_results[i].min_margin, status);
        total_violations += h_results[i].violations;
    }

    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Total fleet violations: %-4d                                    ║\n", total_violations);

    float per_iter_ms = ms / ITERATIONS;
    double evals_per_sec = (double)(N_DEVICES * NC) / (per_iter_ms * 1e-3);
    printf("║  Throughput: %.0f devices×constraints/sec (%.3f µs/launch)       ║\n",
           evals_per_sec, per_iter_ms * 1000.0);
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    // ── Eisenstein beam map demo ──
    printf("\n── Eisenstein Beam Map (12 beams @ 30° spacing) ──\n");
    for (int b = 0; b < 12; b++) {
        float angle = b * (3.14159265f / 6.0f);   // 30° steps
        EisensteinCoord ec = eisenstein_beam_index(angle);
        printf("  Beam %2d: θ=%5.1f° → Eisenstein(%d, %d)\n", b, angle * 180.0f / 3.14159265f, ec.a, ec.b);
    }

    // ── Verification ──
    printf("\n── Verification ──\n");
    int pass = 1;

    // Device 0: all safe → safety ≈ 1.0, 0 violations
    if (h_results[0].violations != 0 || h_results[0].safety_score < 0.99f) {
        printf("  FAIL: Device 0 should be fully safe (safety=%.4f, violations=%d)\n",
               h_results[0].safety_score, h_results[0].violations);
        pass = 0;
    }

    // Device 3: torque violation → violations > 0
    if (h_results[3].violations == 0) {
        printf("  FAIL: Device 3 should have torque violation\n");
        pass = 0;
    }

    // Device 5: Eisenstein violation
    if (h_results[5].violations == 0) {
        printf("  FAIL: Device 5 should have Eisenstein violation\n");
        pass = 0;
    }

    // Device 6: Bloom violation
    if (h_results[6].violations == 0) {
        printf("  FAIL: Device 6 should have Bloom violation\n");
        pass = 0;
    }

    // Device 7: multiple violations
    if (h_results[7].violations < 2) {
        printf("  FAIL: Device 7 should have multiple violations (got %d)\n", h_results[7].violations);
        pass = 0;
    }

    if (pass) {
        printf("  ✓ All verification checks passed\n");
    }

    // ── Cleanup ──
    cudaFree(d_devices);
    cudaFree(d_constraints);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return pass ? 0 : 1;
}
