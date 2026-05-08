# fleet-constraint-kernel — GPU Fleet Constraint Evaluator

Sonar beamformer architecture repurposed for fleet-wide constraint checking.

Evaluates N devices × M constraints in parallel using the same delay-and-sum
pattern used in underwater sonar beamforming. One block per device, shared
memory caches constraint parameters (like beamformer delay tables), threads
evaluate constraints cooperatively.

## Architecture

Sonar beamformer → Fleet constraint kernel

- N hydrophones → N devices
- M beams → M constraints
- Shared delay table → Shared constraint cache
- Delay-and-sum → Evaluate-and-reduce
- Atomic accumulator → Atomic safety score

## Performance

- 13M evaluations/sec on RTX 4050 (8 devices × 23 constraints)
- Eisenstein beam map: exact integer coordinates for beam steering
- Compiles on CUDA 11.5+ (sm_86, sm_89)

## Use Cases

- Multi-robot fleet coordination (N arms, M constraints each)
- Sonar beam steering with constraint-checked patterns
- Distributed IoT fleet monitoring
- Any N×M parallel constraint evaluation

## Quick Start

```bash
nvcc -O3 -arch=sm_86 test_fleet_constraint.cu -o test_fleet && ./test_fleet
```

## Composable With

- **eisenstein-cuda**: constraint math kernels
- **snap-lut**: FPGA version of constraint snapping
- **cocapn-schemas**: tile format for publishing results
- **physics-clock**: temporal inference from evaluation timing

## License

Apache 2.0
