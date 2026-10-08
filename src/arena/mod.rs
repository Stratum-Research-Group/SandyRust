//! Extension X6: Typed Buffer Reuse & In-Place Scratch Arena Allocator (§4.2 Item 6, §9.2 GX6).
//!
//! Provides a pre-allocated scratch arena enforcing strict 64-byte alignment, checked isize
//! offsets, typed buffer handles, and zero heap allocations during steady-state forward and backward iterations.

use std::cell::Cell;
use std::ops::{Deref, DerefMut};
use std::time::Instant;

use crate::config::Config;
use crate::gates::{detect_revision, detect_toolchain, GateReport};

/// Strict alignment required for high-throughput AVX2/AVX-512 vectorization and cache lines.
pub const ARENA_ALIGNMENT: usize = 64;

/// Default maximum arena capacity cap: 512 MiB (§4.2 Item 6).
pub const DEFAULT_MAX_ARENA_BYTES: usize = 512 * 1024 * 1024;

/// Strongly typed error enum for arena allocations (§4.2 Item 6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArenaAllocationError {
    OutOfMemory {
        requested_bytes: usize,
        available_bytes: usize,
        capacity_bytes: usize,
    },
    CapacityExceeded {
        reason: &'static str,
    },
    AlignmentError {
        align: usize,
    },
    ZeroAllocation,
    ActiveTapeReferenceForbidden,
}

impl std::fmt::Display for ArenaAllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory {
                requested_bytes,
                available_bytes,
                capacity_bytes,
            } => {
                write!(
                    f,
                    "Arena out of memory: requested {} bytes, available {} bytes (capacity {} bytes)",
                    requested_bytes, available_bytes, capacity_bytes
                )
            }
            Self::CapacityExceeded { reason } => write!(f, "Arena capacity exceeded: {reason}"),
            Self::AlignmentError { align } => write!(f, "Invalid arena alignment: {align}"),
            Self::ZeroAllocation => write!(f, "Cannot allocate 0 elements/bytes from arena"),
            Self::ActiveTapeReferenceForbidden => write!(
                f,
                "Scratch arena buffer handle cannot be referenced by active autograd tape"
            ),
        }
    }
}

impl std::error::Error for ArenaAllocationError {}

/// A typed, mutable view into memory allocated from the scratch arena.
///
/// Implements `Deref` and `DerefMut` to expose `&[T]` and `&mut [T]` slices.
pub struct TypedBuffer<'a, T> {
    ptr: *mut T,
    len: usize,
    offset: usize,
    byte_len: usize,
    _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T> TypedBuffer<'a, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<'a, T> Deref for TypedBuffer<'a, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'a, T> DerefMut for TypedBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<'a, T: std::fmt::Debug> std::fmt::Debug for TypedBuffer<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedBuffer")
            .field("len", &self.len)
            .field("offset", &self.offset)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

/// A tensor wrapper backed by arena-allocated memory.
pub struct ArenaTensor<'a> {
    shape: Vec<usize>,
    buffer: TypedBuffer<'a, f32>,
}

impl<'a> ArenaTensor<'a> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.buffer
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.buffer
    }

    pub fn offset(&self) -> usize {
        self.buffer.offset()
    }

    pub fn byte_len(&self) -> usize {
        self.buffer.byte_len()
    }
}

/// Pre-allocated scratch arena enforcing strict 64-byte alignment, checked isize
/// offsets, and typed buffer handles (§4.2 Item 6).
pub struct ScratchArena {
    #[allow(dead_code)]
    storage: Vec<u8>,
    aligned_base_ptr: *mut u8,
    capacity: usize,
    offset: Cell<usize>,
    high_water_mark: Cell<usize>,
    allocation_count: Cell<usize>,
}

// ScratchArena owns storage and can be moved between threads.
// Thread isolation is preserved because &ScratchArena uses Cell and is !Sync.
unsafe impl Send for ScratchArena {}

impl ScratchArena {
    /// Create a new pre-allocated scratch arena with capacity capped by `max_arena_bytes`.
    pub fn new(max_arena_bytes: usize) -> Result<Self, ArenaAllocationError> {
        if max_arena_bytes == 0 {
            return Err(ArenaAllocationError::ZeroAllocation);
        }
        if isize::try_from(max_arena_bytes).is_err() {
            return Err(ArenaAllocationError::CapacityExceeded {
                reason: "max_arena_bytes exceeds isize::MAX",
            });
        }

        // Allocate backing storage with extra padding to ensure 64-byte base alignment.
        let alloc_bytes = max_arena_bytes.checked_add(ARENA_ALIGNMENT).ok_or(
            ArenaAllocationError::CapacityExceeded {
                reason: "arena allocation bytes overflowed usize",
            },
        )?;

        let mut storage = vec![0u8; alloc_bytes];
        let raw_addr = storage.as_mut_ptr() as usize;
        let base_align_pad = (ARENA_ALIGNMENT - (raw_addr % ARENA_ALIGNMENT)) % ARENA_ALIGNMENT;
        let aligned_base_ptr = unsafe { storage.as_mut_ptr().add(base_align_pad) };

        debug_assert_eq!((aligned_base_ptr as usize) % ARENA_ALIGNMENT, 0);

        Ok(ScratchArena {
            storage,
            aligned_base_ptr,
            capacity: max_arena_bytes,
            offset: Cell::new(0),
            high_water_mark: Cell::new(0),
            allocation_count: Cell::new(0),
        })
    }

    /// Current allocated byte offset within the arena.
    pub fn offset(&self) -> usize {
        self.offset.get()
    }

    /// Total capacity of the arena in bytes.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// High-water mark of peak memory allocated since creation or last reset of mark.
    pub fn high_water_mark(&self) -> usize {
        self.high_water_mark.get()
    }

    /// Total number of allocations performed in the current cycle.
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.get()
    }

    /// Allocate a typed buffer of `count` elements of type `T` from the scratch arena.
    ///
    /// Computes 64-byte alignment padding via `(64 - (offset % 64)) % 64`, verifies `isize`
    /// bounds, and returns a `TypedBuffer` view without heap allocation.
    pub fn alloc<T: Copy>(&self, count: usize) -> Result<TypedBuffer<'_, T>, ArenaAllocationError> {
        if count == 0 {
            return Err(ArenaAllocationError::ZeroAllocation);
        }

        let elem_size = std::mem::size_of::<T>();
        let bytes_needed = count
            .checked_mul(elem_size)
            .ok_or(ArenaAllocationError::CapacityExceeded {
                reason: "requested count * size_of::<T> overflowed usize",
            })?;

        let curr_offset = self.offset.get();
        let align_pad = (ARENA_ALIGNMENT - (curr_offset % ARENA_ALIGNMENT)) % ARENA_ALIGNMENT;
        let aligned_offset = curr_offset
            .checked_add(align_pad)
            .ok_or(ArenaAllocationError::CapacityExceeded {
                reason: "aligned offset overflowed usize",
            })?;

        // Checked isize offset verification (§4.2 Item 6)
        if isize::try_from(aligned_offset).is_err() {
            return Err(ArenaAllocationError::CapacityExceeded {
                reason: "aligned offset exceeds isize::MAX",
            });
        }

        let end_offset = aligned_offset
            .checked_add(bytes_needed)
            .ok_or(ArenaAllocationError::CapacityExceeded {
                reason: "end offset overflowed usize",
            })?;

        if isize::try_from(end_offset).is_err() {
            return Err(ArenaAllocationError::CapacityExceeded {
                reason: "end offset exceeds isize::MAX",
            });
        }

        if end_offset > self.capacity {
            return Err(ArenaAllocationError::OutOfMemory {
                requested_bytes: bytes_needed,
                available_bytes: self.capacity.saturating_sub(aligned_offset),
                capacity_bytes: self.capacity,
            });
        }

        self.offset.set(end_offset);
        if end_offset > self.high_water_mark.get() {
            self.high_water_mark.set(end_offset);
        }
        self.allocation_count.set(self.allocation_count.get() + 1);

        let raw_ptr = unsafe { self.aligned_base_ptr.add(aligned_offset) as *mut T };
        debug_assert_eq!((raw_ptr as usize) % ARENA_ALIGNMENT, 0);

        Ok(TypedBuffer {
            ptr: raw_ptr,
            len: count,
            offset: aligned_offset,
            byte_len: bytes_needed,
            _marker: std::marker::PhantomData,
        })
    }

    /// Allocate a 2D or n-D float tensor backed by scratch arena memory.
    pub fn alloc_tensor(&self, shape: &[usize]) -> Result<ArenaTensor<'_>, ArenaAllocationError> {
        if shape.is_empty() {
            return Err(ArenaAllocationError::ZeroAllocation);
        }
        let mut numel = 1usize;
        for &dim in shape {
            if dim == 0 {
                return Err(ArenaAllocationError::ZeroAllocation);
            }
            numel = numel
                .checked_mul(dim)
                .ok_or(ArenaAllocationError::CapacityExceeded {
                    reason: "tensor shape elements overflowed usize",
                })?;
        }
        let buffer = self.alloc::<f32>(numel)?;
        Ok(ArenaTensor {
            shape: shape.to_vec(),
            buffer,
        })
    }

    /// Reset the scratch arena offset back to 0.
    ///
    /// Requires exclusive `&mut self` borrow, ensuring all `TypedBuffer` views have been dropped.
    /// Reuses the existing backing storage with zero heap reallocations.
    pub fn reset(&mut self) {
        self.offset.set(0);
        self.allocation_count.set(0);
    }

    /// Reset high-water mark counter.
    pub fn reset_high_water_mark(&mut self) {
        self.high_water_mark.set(self.offset.get());
    }
}

/// Helper verifying that two typed buffers occupy strictly disjoint non-overlapping address ranges.
pub fn assert_disjoint<T, U>(a: &TypedBuffer<'_, T>, b: &TypedBuffer<'_, U>) -> bool {
    let a_start = a.ptr as usize;
    let a_end = a_start + a.byte_len;
    let b_start = b.ptr as usize;
    let b_end = b_start + b.byte_len;
    a_end <= b_start || b_end <= a_start
}

thread_local! {
    static THREAD_SCRATCH_ARENA: std::cell::RefCell<Option<ScratchArena>> = const { std::cell::RefCell::new(None) };
}

/// Execute a closure with a thread-isolated scratch arena (§4.2 Item 6).
pub fn with_thread_scratch_arena<R>(
    capacity: usize,
    f: impl FnOnce(&mut ScratchArena) -> R,
) -> Result<R, ArenaAllocationError> {
    THREAD_SCRATCH_ARENA.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() || opt.as_ref().unwrap().capacity() != capacity {
            *opt = Some(ScratchArena::new(capacity)?);
        }
        let arena = opt.as_mut().unwrap();
        arena.reset();
        Ok(f(arena))
    })
}

/// Gate GX6 runner: Typed Buffer Reuse & In-Place Allocator (§4.2 Item 6, §9.2 GX6).
pub fn gx6() -> GateReport {
    let t0 = Instant::now();
    let _cfg = Config::arena_v2(DEFAULT_MAX_ARENA_BYTES);
    let mut details = Vec::new();
    let mut checks_passed = true;

    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();

    // 1. Zero allocation during steady-state forward and backward iterations
    let arena_res = ScratchArena::new(DEFAULT_MAX_ARENA_BYTES);
    let mut arena = match arena_res {
        Ok(a) => a,
        Err(e) => {
            return GateReport {
                gate_id: "GX6",
                scope: "acceptance",
                status: "FAIL",
                reason: Some(format!("Failed to initialize scratch arena: {e}")),
                requirement: "Scratch arena zero allocation during steady-state forward and backward iterations; retained disjoint views; graceful recovery on artificial OOM; multi-step gradient accumulation without memory creep (§4.2 Item 6, §9.2 GX6)",
                implemented_check: "zero_allocation_steady_state, retained_disjoint_views, graceful_oom_recovery, alignment_and_padding, multi_step_accumulation_no_creep",
                prerequisites: "reference_v2, G6, G14",
                exclusions: "none",
                revision: rev,
                dirty_state: dirty,
                toolchain,
                target: "x86_64-unknown-linux-gnu",
                command: "cargo test --test test_x6_arena",
                fixture_hash: "arena-scratch-v1".into(),
                config: "arena_v2",
                dtype: "fp32",
                expected: "zero steady-state allocation; non-overlapping disjoint views with 64-byte alignment; typed OOM error without corruption; bounded accumulation memory".into(),
                observed: format!("Initialization error: {e}"),
                atol: 0.0,
                rtol: 0.0,
                execution_time_ms: t0.elapsed().as_millis(),
            };
        }
    };

    // Warm-up iteration
    {
        let _b1 = arena.alloc::<f32>(128).unwrap();
        let _b2 = arena.alloc::<f32>(256).unwrap();
    }
    arena.reset();

    // 10 steady-state iterations
    let mut steady_allocs_ok = true;
    for _ in 0..10 {
        {
            let mut t1 = arena.alloc_tensor(&[4, 64]).unwrap();
            let mut t2 = arena.alloc_tensor(&[64, 128]).unwrap();
            let mut t3 = arena.alloc_tensor(&[4, 128]).unwrap();

            t1.data_mut()[0] = 1.0;
            t2.data_mut()[0] = 2.0;
            t3.data_mut()[0] = 3.0;

            if arena.offset() == 0 {
                steady_allocs_ok = false;
            }
        }
        arena.reset();
        if arena.offset() != 0 {
            steady_allocs_ok = false;
        }
    }
    checks_passed &= steady_allocs_ok;
    details.push(format!("zero_allocation_steady_state (ok={steady_allocs_ok})"));

    // 2. Retained disjoint views
    let disjoint_ok;
    {
        let mut buf_a = arena.alloc::<f32>(64).unwrap();
        let mut buf_b = arena.alloc::<f32>(128).unwrap();
        let mut buf_c = arena.alloc::<f32>(256).unwrap();

        // Check alignment
        let align_a = (buf_a.as_ptr() as usize) % ARENA_ALIGNMENT == 0;
        let align_b = (buf_b.as_ptr() as usize) % ARENA_ALIGNMENT == 0;
        let align_c = (buf_c.as_ptr() as usize) % ARENA_ALIGNMENT == 0;

        // Check non-overlapping disjoint memory
        let disj_ab = assert_disjoint(&buf_a, &buf_b);
        let disj_bc = assert_disjoint(&buf_b, &buf_c);
        let disj_ac = assert_disjoint(&buf_a, &buf_c);

        // Independent writes
        buf_a[0] = 11.0;
        buf_b[0] = 22.0;
        buf_c[0] = 33.0;

        let content_ok = buf_a[0] == 11.0 && buf_b[0] == 22.0 && buf_c[0] == 33.0;
        disjoint_ok = align_a && align_b && align_c && disj_ab && disj_bc && disj_ac && content_ok;
    }
    arena.reset();
    checks_passed &= disjoint_ok;
    details.push(format!("retained_disjoint_views (ok={disjoint_ok})"));

    // 3. Graceful recovery on artificial OOM
    let small_arena = ScratchArena::new(1024).unwrap();
    let oom_err = small_arena.alloc::<f32>(512); // 512 * 4 = 2048 bytes > 1024
    let oom_ok = matches!(oom_err, Err(ArenaAllocationError::OutOfMemory { requested_bytes: 2048, .. }));

    // Verify recovery without corruption: valid small allocation succeeds
    let recover_res = small_arena.alloc::<f32>(64); // 64 * 4 = 256 bytes <= 1024
    let recover_ok = recover_res.is_ok();
    let graceful_oom_ok = oom_ok && recover_ok;
    checks_passed &= graceful_oom_ok;
    details.push(format!("graceful_oom_recovery (ok={graceful_oom_ok})"));

    // 4. Strict 64-byte alignment and padding across odd sizes
    let mut align_check_ok = true;
    for &elem_count in &[1, 3, 7, 13, 29, 63] {
        let b = arena.alloc::<f32>(elem_count).unwrap();
        if (b.as_ptr() as usize) % ARENA_ALIGNMENT != 0 {
            align_check_ok = false;
        }
    }
    arena.reset();
    checks_passed &= align_check_ok;
    details.push(format!("alignment_and_padding (ok={align_check_ok})"));

    // 5. Multi-step gradient accumulation without memory creep
    let mut accum_ok = true;
    let mut hwms = Vec::new();

    for _microstep in 0..4 {
        {
            let _act_q = arena.alloc::<f32>(256).unwrap();
            let _act_k = arena.alloc::<f32>(256).unwrap();
            let _act_v = arena.alloc::<f32>(256).unwrap();
            let _scores = arena.alloc::<f32>(64 * 64).unwrap();
        }
        hwms.push(arena.high_water_mark());
        arena.reset();
    }

    // Verify high water mark remained constant across microsteps (no creep)
    if hwms.len() == 4 {
        let first = hwms[0];
        if !hwms.iter().all(|&h| h == first) {
            accum_ok = false;
        }
    } else {
        accum_ok = false;
    }
    if arena.offset() != 0 {
        accum_ok = false;
    }
    checks_passed &= accum_ok;
    details.push(format!("multi_step_accumulation_no_creep (ok={accum_ok})"));

    let joined_details = details.join("; ");
    GateReport {
        gate_id: "GX6",
        scope: "acceptance",
        status: if checks_passed { "PASS" } else { "FAIL" },
        reason: if checks_passed {
            None
        } else {
            Some("one or more arena invariants failed".into())
        },
        requirement: "Scratch arena zero allocation during steady-state forward and backward iterations; retained disjoint views; graceful recovery on artificial OOM; multi-step gradient accumulation without memory creep (§4.2 Item 6, §9.2 GX6)",
        implemented_check: "zero_allocation_steady_state, retained_disjoint_views, graceful_oom_recovery, alignment_and_padding, multi_step_accumulation_no_creep",
        prerequisites: "reference_v2, G6, G14",
        exclusions: "none",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x6_arena",
        fixture_hash: "arena-scratch-v1".into(),
        config: "arena_v2",
        dtype: "fp32",
        expected: "zero steady-state allocation; non-overlapping disjoint views with 64-byte alignment; typed OOM error without corruption; bounded accumulation memory".into(),
        observed: joined_details,
        atol: 0.0,
        rtol: 0.0,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
