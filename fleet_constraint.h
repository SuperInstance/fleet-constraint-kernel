#pragma once

#include <cstdint>

// ── Device state (input per fleet member) ──────────────────────────
typedef struct {
    int    device_id;
    float  joint_values[7];      // 7-DOF joint positions (rad)
    float  velocities[7];        // joint velocities (rad/s)
    float  torques[7];           // joint torques (Nm)
    float  end_effector[3];      // end-effector x, y, z (m)
} DeviceState;

// ── Constraint definitions ─────────────────────────────────────────
// type: 0=range, 1=torque, 2=velocity, 3=eisenstein, 4=bloom
typedef struct {
    int    type;                 // constraint type selector
    float  params[4];            // {min,max,...} or {radius,scale,...}
    float  weight;               // priority weight for fleet scoring
    int    severity;             // 0-3 severity level
} FleetConstraint;

// ── Per-device evaluation result ───────────────────────────────────
typedef struct {
    float  safety_score;         // 0.0 (all violated) – 1.0 (all satisfied)
    int    violations;           // count of violated constraints
    int    worst_severity;       // max severity among violations
    float  min_margin;           // smallest margin to any violation (negative = violated)
} DeviceResult;

// ── Eisenstein beam mapping ────────────────────────────────────────
// Maps a beam angle (radians) to Eisenstein integer lattice coordinates.
// The Eisenstein lattice uses basis vectors (1, 0) and (−½, √3/2).
// This maps 360° beam steering onto an exact hex grid — zero float drift
// in the lattice index once computed.
typedef struct {
    int a;   // first lattice coordinate
    int b;   // second lattice coordinate
} EisensteinCoord;

// ── Constraint type constants ──────────────────────────────────────
enum ConstraintType {
    CONSTRAINT_RANGE      = 0,
    CONSTRAINT_TORQUE     = 1,
    CONSTRAINT_VELOCITY   = 2,
    CONSTRAINT_EISENSTEIN = 3,
    CONSTRAINT_BLOOM      = 4,
};
