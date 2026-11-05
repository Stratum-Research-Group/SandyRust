//! Extension X11: Paged KV-Cache Management (§4.2 Item 11, §9.2 GX11).
//!
//! Provides:
//! 1. Fixed-size block paging ($B_{\text{page}} = 16$) with logical-to-physical block table mapping.
//! 2. Reserve-before-append transaction safety with atomic rollback on pool exhaustion.
//! 3. Generational pool allocation preventing stale-handle access and double-free corruption.
//! 4. Multi-session isolation and unreferenced page reclamation without rebasing or evicting live positions.
//! 5. Gate GX11 verification.

use std::collections::HashMap;
use std::time::Instant;

use crate::gates::{detect_revision, detect_toolchain, GateReport};
use crate::sha256::{hex, sha256};

pub const DEFAULT_PAGE_BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionHandle {
    pub session_id: usize,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedKvError {
    OutOfMemory { required: usize, available: usize },
    SessionNotFound(usize),
    StaleHandle { session_id: usize, handle_gen: u64, active_gen: u64 },
    DoubleFree(usize),
    InvalidBlockSize,
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for PagedKvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfMemory { required, available } => {
                write!(f, "KV block pool exhausted: required {required} blocks, available {available}")
            }
            Self::SessionNotFound(id) => write!(f, "KV session {id} not found"),
            Self::StaleHandle { session_id, handle_gen, active_gen } => {
                write!(f, "Stale handle for session {session_id}: handle gen {handle_gen}, active gen {active_gen}")
            }
            Self::DoubleFree(id) => write!(f, "Double free rejected for session {id}"),
            Self::InvalidBlockSize => write!(f, "Block size must be > 0"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "KV dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for PagedKvError {}

/// Physical cache block holding fixed number of tokens for all KV heads.
#[derive(Debug, Clone)]
pub struct KvBlock {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

impl KvBlock {
    pub fn new(capacity_elements: usize) -> Self {
        Self {
            keys: vec![0.0f32; capacity_elements],
            values: vec![0.0f32; capacity_elements],
        }
    }
}

/// Physical block pool with generational counters and free list.
#[derive(Debug, Clone)]
pub struct BlockPool {
    pub block_size: usize,
    pub token_dim: usize, // n_kv_heads * head_dim
    pub blocks: Vec<KvBlock>,
    pub free_list: Vec<BlockId>,
    pub block_generations: Vec<u64>,
}

impl BlockPool {
    pub fn new(total_blocks: usize, block_size: usize, token_dim: usize) -> Result<Self, PagedKvError> {
        if block_size == 0 {
            return Err(PagedKvError::InvalidBlockSize);
        }
        let cap = block_size * token_dim;
        let mut blocks = Vec::with_capacity(total_blocks);
        let mut free_list = Vec::with_capacity(total_blocks);
        let mut block_generations = Vec::with_capacity(total_blocks);

        for i in 0..total_blocks {
            blocks.push(KvBlock::new(cap));
            free_list.push(BlockId(i));
            block_generations.push(1);
        }

        Ok(Self {
            block_size,
            token_dim,
            blocks,
            free_list,
            block_generations,
        })
    }

    pub fn available_blocks(&self) -> usize {
        self.free_list.len()
    }
}

/// Logical-to-physical block table for a single session.
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub block_ids: Vec<BlockId>,
    pub seq_len: usize,
}

impl BlockTable {
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            seq_len: 0,
        }
    }

    pub fn capacity_tokens(&self, block_size: usize) -> usize {
        self.block_ids.len() * block_size
    }
}

impl Default for BlockTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Paged KV Cache Manager coordinating sessions, logical-to-physical block tables,
/// transactional allocation, and block recycling.
#[derive(Debug, Clone)]
pub struct PagedKvManager {
    pub pool: BlockPool,
    sessions: HashMap<usize, (BlockTable, u64)>,
    freed_sessions: HashMap<usize, u64>,
    next_session_id: usize,
}

impl PagedKvManager {
    pub fn new(total_blocks: usize, block_size: usize, token_dim: usize) -> Result<Self, PagedKvError> {
        let pool = BlockPool::new(total_blocks, block_size, token_dim)?;
        Ok(Self {
            pool,
            sessions: HashMap::new(),
            freed_sessions: HashMap::new(),
            next_session_id: 1,
        })
    }

    /// Allocate a new session handle with generation 1.
    pub fn create_session(&mut self) -> SessionHandle {
        let sid = self.next_session_id;
        self.next_session_id += 1;
        self.sessions.insert(sid, (BlockTable::new(), 1));
        SessionHandle {
            session_id: sid,
            generation: 1,
        }
    }


    fn get_session(&self, handle: &SessionHandle) -> Result<&(BlockTable, u64), PagedKvError> {
        if let Some(entry) = self.sessions.get(&handle.session_id) {
            if entry.1 != handle.generation {
                return Err(PagedKvError::StaleHandle {
                    session_id: handle.session_id,
                    handle_gen: handle.generation,
                    active_gen: entry.1,
                });
            }
            Ok(entry)
        } else if self.freed_sessions.contains_key(&handle.session_id) {
            Err(PagedKvError::StaleHandle {
                session_id: handle.session_id,
                handle_gen: handle.generation,
                active_gen: 0,
            })
        } else {
            Err(PagedKvError::SessionNotFound(handle.session_id))
        }
    }

    /// Reserve-before-append transactional append of new tokens.
    ///
    /// If pool has insufficient free blocks, returns `PagedKvError::OutOfMemory`
    /// and leaves the session table and pool completely unchanged (atomic rollback).
    pub fn append_kv(
        &mut self,
        handle: &SessionHandle,
        new_keys: &[f32],
        new_values: &[f32],
        num_tokens: usize,
    ) -> Result<(), PagedKvError> {
        let expected_len = num_tokens * self.pool.token_dim;
        if new_keys.len() != expected_len {
            return Err(PagedKvError::DimensionMismatch {
                expected: expected_len,
                got: new_keys.len(),
            });
        }
        if new_values.len() != expected_len {
            return Err(PagedKvError::DimensionMismatch {
                expected: expected_len,
                got: new_values.len(),
            });
        }

        // 1. Transaction pre-check: verify session and calculate block demand
        let block_size = self.pool.block_size;
        let token_dim = self.pool.token_dim;
        let (current_seq_len, current_cap) = {
            let (table, _) = self.get_session(handle)?;
            (table.seq_len, table.capacity_tokens(block_size))
        };

        let new_total_tokens = current_seq_len + num_tokens;
        let needed_blocks = if new_total_tokens > current_cap {
            let additional_tokens = new_total_tokens - current_cap;
            additional_tokens.div_ceil(block_size)
        } else {
            0
        };

        // 2. Reserve-before-append: verify pool capacity before allocating
        if needed_blocks > self.pool.free_list.len() {
            return Err(PagedKvError::OutOfMemory {
                required: needed_blocks,
                available: self.pool.free_list.len(),
            });
        }

        // 3. Allocate physical blocks from free_list
        let mut allocated_blocks = Vec::with_capacity(needed_blocks);
        for _ in 0..needed_blocks {
            let bid = self.pool.free_list.pop().unwrap();
            allocated_blocks.push(bid);
        }

        // 4. Append to session table and copy token values
        let table_blocks = {
            let (table, _) = self.sessions.get_mut(&handle.session_id).unwrap();
            table.block_ids.extend_from_slice(&allocated_blocks);
            table.block_ids.clone()
        };

        for t in 0..num_tokens {
            let logical_idx = current_seq_len + t;
            let block_idx_in_session = logical_idx / block_size;
            let offset_in_block = logical_idx % block_size;

            let physical_bid = table_blocks[block_idx_in_session];
            let start = offset_in_block * token_dim;
            let end = start + token_dim;

            let src_start = t * token_dim;
            let src_end = src_start + token_dim;

            self.pool.blocks[physical_bid.0].keys[start..end]
                .copy_from_slice(&new_keys[src_start..src_end]);
            self.pool.blocks[physical_bid.0].values[start..end]
                .copy_from_slice(&new_values[src_start..src_end]);
        }

        let (table, _) = self.sessions.get_mut(&handle.session_id).unwrap();
        table.seq_len = new_total_tokens;
        Ok(())
    }

    /// Read all accumulated KV data for session in contiguous logical order.
    pub fn read_kv(&self, handle: &SessionHandle) -> Result<(Vec<f32>, Vec<f32>), PagedKvError> {
        let (table, _) = self.get_session(handle)?;
        let token_dim = self.pool.token_dim;
        let block_size = self.pool.block_size;
        let total_elements = table.seq_len * token_dim;

        let mut out_keys = Vec::with_capacity(total_elements);
        let mut out_values = Vec::with_capacity(total_elements);

        for t in 0..table.seq_len {
            let block_idx_in_session = t / block_size;
            let offset_in_block = t % block_size;

            let physical_bid = table.block_ids[block_idx_in_session];
            let start = offset_in_block * token_dim;
            let end = start + token_dim;

            out_keys.extend_from_slice(&self.pool.blocks[physical_bid.0].keys[start..end]);
            out_values.extend_from_slice(&self.pool.blocks[physical_bid.0].values[start..end]);
        }

        Ok((out_keys, out_values))
    }

    /// Release session: reclaims physical blocks back to pool without rebasing live sessions.
    /// Re-releasing or accessing the released session triggers stale-handle/double-free rejection.
    pub fn release_session(&mut self, handle: SessionHandle) -> Result<(), PagedKvError> {
        let (table, gen) = if let Some(entry) = self.sessions.remove(&handle.session_id) {
            if entry.1 != handle.generation {
                // Re-insert to keep state consistent
                self.sessions.insert(handle.session_id, entry);
                return Err(PagedKvError::StaleHandle {
                    session_id: handle.session_id,
                    handle_gen: handle.generation,
                    active_gen: 0,
                });
            }
            entry
        } else if self.freed_sessions.contains_key(&handle.session_id) {
            return Err(PagedKvError::DoubleFree(handle.session_id));
        } else {
            return Err(PagedKvError::SessionNotFound(handle.session_id));
        };

        // Return physical blocks to free list and increment generation
        for bid in table.block_ids {
            self.pool.block_generations[bid.0] += 1;
            self.pool.free_list.push(bid);
        }

        self.freed_sessions.insert(handle.session_id, gen);
        Ok(())
    }
}

// ===========================================================================
// Verification Gate GX11
// ===========================================================================

/// Gate GX11 verification for Extension X11 (§4.2 Item 11, §9.2 GX11).
pub fn gx11() -> GateReport {
    let t0 = Instant::now();

    let block_size = 16;
    let token_dim = 8;
    let total_blocks = 10;

    let mut manager = PagedKvManager::new(total_blocks, block_size, token_dim).expect("create manager");

    // 1. Session 1 allocation spanning multiple block pages (35 tokens across 3 blocks)
    let s1 = manager.create_session();
    let num_tokens1 = 35;
    let s1_keys: Vec<f32> = (0..num_tokens1 * token_dim).map(|i| (i as f32) + 1.0).collect();
    let s1_values: Vec<f32> = (0..num_tokens1 * token_dim).map(|i| (i as f32) * 2.0 + 1.0).collect();

    manager
        .append_kv(&s1, &s1_keys, &s1_values, num_tokens1)
        .expect("append S1");

    let s1_table_len = manager.get_session(&s1).unwrap().0.block_ids.len();
    let s1_blocks_ok = s1_table_len == 3; // ceil(35 / 16) = 3 blocks
    let pool_left_1 = manager.pool.available_blocks(); // 10 - 3 = 7

    // Read parity verification
    let (read_k1, read_v1) = manager.read_kv(&s1).expect("read S1");
    let read_parity_ok = read_k1 == s1_keys && read_v1 == s1_values;

    // 2. Transaction Safety & Atomic Rollback on Pool Exhaustion
    // Currently 7 blocks left. Attempt to allocate 8 blocks (128 tokens) in Session 2
    let s2 = manager.create_session();
    let big_tokens = 120; // ceil(120 / 16) = 8 blocks > 7 available
    let big_keys = vec![1.0f32; big_tokens * token_dim];
    let big_values = vec![2.0f32; big_tokens * token_dim];

    let oom_res = manager.append_kv(&s2, &big_keys, &big_values, big_tokens);
    let oom_rejected = matches!(oom_res, Err(PagedKvError::OutOfMemory { required: 8, available: 7 }));

    // Verify atomic rollback: S2 has 0 blocks allocated, pool still has exactly 7 blocks!
    let s2_empty_ok = manager.get_session(&s2).unwrap().0.block_ids.is_empty();
    let pool_unleaked_ok = manager.pool.available_blocks() == 7;

    // 3. Multi-Session Isolation (Interleaved S1 and S2 appends)
    let s2_small_tokens = 20; // 2 blocks
    let s2_keys: Vec<f32> = (0..s2_small_tokens * token_dim).map(|i| (i as f32) + 500.0).collect();
    let s2_values: Vec<f32> = (0..s2_small_tokens * token_dim).map(|i| (i as f32) + 1000.0).collect();
    manager
        .append_kv(&s2, &s2_keys, &s2_values, s2_small_tokens)
        .expect("append S2");

    let pool_left_2 = manager.pool.available_blocks(); // 7 - 2 = 5 blocks

    // Verify S1 and S2 do not crosstalk
    let (read_k1_again, read_v1_again) = manager.read_kv(&s1).expect("read S1 again");
    let (read_k2, read_v2) = manager.read_kv(&s2).expect("read S2");
    let isolation_ok = read_k1_again == s1_keys
        && read_v1_again == s1_values
        && read_k2 == s2_keys
        && read_v2 == s2_values;

    // 4. Lifecycle Reclaim without rebasing live sessions
    // Release S1: 3 blocks returned to pool (5 + 3 = 8 blocks available).
    manager.release_session(s1).expect("release S1");
    let pool_reclaimed_ok = manager.pool.available_blocks() == 8;

    // S2 must be completely unaffected and read exact data
    let (read_k2_post, read_v2_post) = manager.read_kv(&s2).expect("read S2 post release");
    let s2_unaffected_ok = read_k2_post == s2_keys && read_v2_post == s2_values;

    // S2 can continue appending further tokens
    let s2_more = 10;
    let s2_more_keys = vec![99.0f32; s2_more * token_dim];
    let s2_more_vals = vec![100.0f32; s2_more * token_dim];
    let append_more_ok = manager
        .append_kv(&s2, &s2_more_keys, &s2_more_vals, s2_more)
        .is_ok();

    // 5. Double-Free and Stale-Handle Assertions
    let double_free_rejected = matches!(manager.release_session(s1), Err(PagedKvError::DoubleFree(_)));
    let stale_access_rejected = matches!(manager.read_kv(&s1), Err(PagedKvError::StaleHandle { .. }));

    let all_ok = s1_blocks_ok
        && pool_left_1 == 7
        && read_parity_ok
        && oom_rejected
        && s2_empty_ok
        && pool_unleaked_ok
        && pool_left_2 == 5
        && isolation_ok
        && pool_reclaimed_ok
        && s2_unaffected_ok
        && append_more_ok
        && double_free_rejected
        && stale_access_rejected;

    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();

    GateReport {
        gate_id: "GX11",
        scope: "acceptance",
        status: if all_ok { "PASS" } else { "FAIL" },
        reason: None,
        requirement: "X11 Paged KV Cache Management: fixed block paging (B=16), logical-to-physical mapping, atomic reserve-before-append rollback, multi-session isolation, unreferenced page reclaim without live rebasing, double-free and stale-handle rejection",
        implemented_check: "Page block allocation + exact KV gather parity + pool exhaustion rollback + inter-session non-interference + block recycle + double-free guard",
        prerequisites: "Config::reference_v2",
        exclusions: "vLLM C++/CUDA custom kernels (pure safe Rust std-only paged memory management)",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x11_paged_kv",
        fixture_hash: hex(&sha256(format!("gx11:blocks={}:avail={}:parity={}", total_blocks, manager.pool.available_blocks(), read_parity_ok).as_bytes())),
        config: "reference_v2_x11",
        dtype: "f32",
        expected: "35 tokens in 3 blocks, read parity exact, rollback on 8 blocks demand with 7 avail, S1 release reclaims 3 blocks while S2 continues, double-free rejected".to_string(),
        observed: format!(
            "s1_blocks={}, read_parity={}, oom_rollback={}, isolation={}, reclaimed={}, s2_continued={}, double_free_rejected={}",
            s1_blocks_ok, read_parity_ok, oom_rejected && pool_unleaked_ok, isolation_ok, pool_reclaimed_ok, append_more_ok, double_free_rejected
        ),
        atol: 0.0,
        rtol: 0.0,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
