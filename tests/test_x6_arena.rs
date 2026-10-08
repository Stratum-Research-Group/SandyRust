//! Tests for Extension X6: Typed Buffer Reuse & In-Place Scratch Arena Allocator (§4.2 Item 6, §9.2 GX6).

use sandy_rust::arena::{
    assert_disjoint, gx6, with_thread_scratch_arena, ArenaAllocationError, ScratchArena,
    ARENA_ALIGNMENT, DEFAULT_MAX_ARENA_BYTES,
};
use sandy_rust::config::Config;

#[test]
fn test_x6_config_validation_and_fingerprint() {
    let base = Config::reference_v2();
    assert!(!base.ext_x6_arena);

    let arena_cfg = Config::arena_v2(1024 * 1024);
    assert!(arena_cfg.ext_x6_arena);
    assert_eq!(arena_cfg.max_arena_bytes, 1024 * 1024);
    assert!(arena_cfg.validate().is_ok());

    // Baseline parameter count and fingerprint unchanged
    assert_eq!(base.param_count(), Some(886_272));
    assert_ne!(base.fingerprint(), arena_cfg.fingerprint());

    // Validation rejection of zero capacity
    let mut invalid_cfg = arena_cfg.clone();
    invalid_cfg.max_arena_bytes = 0;
    assert!(invalid_cfg.validate().is_err());
}

#[test]
fn test_x6_strict_64_byte_alignment() {
    let arena = ScratchArena::new(1024 * 1024).expect("arena creation");

    // Test across diverse element counts including primes and odd numbers
    let test_counts = [1, 2, 3, 5, 7, 11, 13, 17, 19, 31, 64, 127, 256];
    for count in test_counts {
        let buf = arena.alloc::<f32>(count).expect("allocation");
        let ptr_val = buf.as_ptr() as usize;
        assert_eq!(
            ptr_val % ARENA_ALIGNMENT,
            0,
            "Allocation of {} f32s at address 0x{:x} is not 64-byte aligned",
            count,
            ptr_val
        );
    }
}

#[test]
fn test_x6_retained_disjoint_views() {
    let arena = ScratchArena::new(1024 * 1024).expect("arena creation");

    // Allocate 3 buffers simultaneously and retain views
    let mut b1 = arena.alloc::<f32>(32).expect("alloc b1");
    let mut b2 = arena.alloc::<f32>(64).expect("alloc b2");
    let mut b3 = arena.alloc::<f32>(128).expect("alloc b3");

    // Verify disjoint non-overlapping address ranges
    assert!(assert_disjoint(&b1, &b2));
    assert!(assert_disjoint(&b2, &b3));
    assert!(assert_disjoint(&b1, &b3));

    // Fill with distinct patterns
    for (i, x) in b1.iter_mut().enumerate() {
        *x = (i as f32) + 1.0;
    }
    for (i, x) in b2.iter_mut().enumerate() {
        *x = ((i * 2) as f32) + 100.0;
    }
    for (i, x) in b3.iter_mut().enumerate() {
        *x = ((i * 3) as f32) + 1000.0;
    }

    // Verify contents were preserved without memory overwrites
    for (i, &x) in b1.iter().enumerate() {
        assert_eq!(x, (i as f32) + 1.0);
    }
    for (i, &x) in b2.iter().enumerate() {
        assert_eq!(x, ((i * 2) as f32) + 100.0);
    }
    for (i, &x) in b3.iter().enumerate() {
        assert_eq!(x, ((i * 3) as f32) + 1000.0);
    }
}

#[test]
fn test_x6_arena_tensor_and_resets() {
    let mut arena = ScratchArena::new(1024 * 1024).expect("arena creation");

    for iter in 0..5 {
        {
            let mut t_q = arena.alloc_tensor(&[2, 4, 32]).expect("alloc t_q");
            let mut t_k = arena.alloc_tensor(&[2, 4, 32]).expect("alloc t_k");

            assert_eq!(t_q.shape(), &[2, 4, 32]);
            assert_eq!(t_k.shape(), &[2, 4, 32]);

            t_q.data_mut()[0] = 42.0 + iter as f32;
            t_k.data_mut()[0] = 84.0 + iter as f32;

            assert_eq!(t_q.data()[0], 42.0 + iter as f32);
            assert_eq!(t_k.data()[0], 84.0 + iter as f32);
            assert!(arena.offset() > 0);
        }
        arena.reset();
        assert_eq!(arena.offset(), 0);
    }
}

#[test]
fn test_x6_artificial_oom_resilience() {
    // Artificial tight capacity: 512 bytes
    let mut arena = ScratchArena::new(512).expect("arena creation");

    // Request 1024 bytes (256 * 4) -> must trigger OutOfMemory
    let res = arena.alloc::<f32>(256);
    match res {
        Err(ArenaAllocationError::OutOfMemory {
            requested_bytes,
            available_bytes,
            capacity_bytes,
        }) => {
            assert_eq!(requested_bytes, 1024);
            assert_eq!(available_bytes, 512);
            assert_eq!(capacity_bytes, 512);
        }
        other => panic!("Expected OutOfMemory, got: {:?}", other),
    }

    // Verify arena state was NOT corrupted (offset is still 0)
    assert_eq!(arena.offset(), 0);

    // Subsequent valid allocation (64 * 4 = 256 bytes <= 512) must succeed
    let mut valid_buf = arena.alloc::<f32>(64).expect("valid allocation after OOM");
    valid_buf[0] = 123.456;
    assert_eq!(valid_buf[0], 123.456);

    // Now request another 100 f32s (400 bytes, but remaining is < 400 after alignment) -> OOM again
    let res2 = arena.alloc::<f32>(100);
    assert!(matches!(res2, Err(ArenaAllocationError::OutOfMemory { .. })));

    // Resetting recovers full capacity
    arena.reset();
    assert_eq!(arena.offset(), 0);
    let mut fresh_buf = arena.alloc::<f32>(128).expect("fresh allocation");
    fresh_buf[127] = 789.0;
    assert_eq!(fresh_buf[127], 789.0);
}

#[test]
fn test_x6_multi_step_gradient_accumulation() {
    let mut arena = ScratchArena::new(DEFAULT_MAX_ARENA_BYTES).expect("arena creation");

    let num_microsteps = 8;
    let mut observed_hwms = Vec::with_capacity(num_microsteps);

    for _step in 0..num_microsteps {
        {
            // Simulate forward + backward scratch allocations
            let _q = arena.alloc::<f32>(128).unwrap();
            let _k = arena.alloc::<f32>(128).unwrap();
            let _v = arena.alloc::<f32>(128).unwrap();
            let _attn_weights = arena.alloc::<f32>(128 * 128).unwrap();
            let _grad_q = arena.alloc::<f32>(128).unwrap();
            let _grad_k = arena.alloc::<f32>(128).unwrap();
            let _grad_v = arena.alloc::<f32>(128).unwrap();
        }
        observed_hwms.push(arena.high_water_mark());
        arena.reset();
        assert_eq!(arena.offset(), 0);
    }

    // High water mark must be identical across all microsteps (zero memory creep)
    let first_hwm = observed_hwms[0];
    assert!(first_hwm > 0);
    for &hwm in &observed_hwms {
        assert_eq!(hwm, first_hwm, "Memory creep detected across microsteps");
    }
}

#[test]
fn test_x6_thread_isolation() {
    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            std::thread::spawn(move || {
                with_thread_scratch_arena(64 * 1024, |arena| {
                    let mut b = arena.alloc::<f32>(128).expect("thread alloc");
                    b[0] = thread_id as f32 * 10.0;
                    assert_eq!(b[0], thread_id as f32 * 10.0);
                    assert_eq!((b.as_ptr() as usize) % ARENA_ALIGNMENT, 0);
                })
                .expect("with_thread_scratch_arena");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread join");
    }
}

#[test]
fn test_x6_gate_gx6_execution() {
    let rep = gx6();
    assert_eq!(rep.gate_id, "GX6");
    assert_eq!(rep.scope, "acceptance");
    assert_eq!(rep.status, "PASS", "Gate GX6 failed: {:?}", rep.reason);
}
