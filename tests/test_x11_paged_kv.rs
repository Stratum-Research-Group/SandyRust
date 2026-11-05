use sandy_rust::gates;
use sandy_rust::paged_kv::{PagedKvError, PagedKvManager};

#[test]
fn test_gx11_gate_pass() {
    let rep = gates::run(&["gx11"]).expect("run gx11");
    assert_eq!(rep.len(), 1);
    assert_eq!(rep[0].status, "PASS", "Gate GX11 failed: {:?}", rep[0]);
}

#[test]
fn test_paged_kv_basic_and_multi_session() {
    let mut manager = PagedKvManager::new(8, 16, 4).expect("create manager");
    assert_eq!(manager.pool.available_blocks(), 8);

    // Create session 1
    let s1 = manager.create_session();
    let s1_tokens = 30; // 2 blocks
    let s1_k = vec![1.0f32; s1_tokens * 4];
    let s1_v = vec![2.0f32; s1_tokens * 4];

    manager.append_kv(&s1, &s1_k, &s1_v, s1_tokens).unwrap();
    assert_eq!(manager.pool.available_blocks(), 6);

    // Create session 2
    let s2 = manager.create_session();
    let s2_tokens = 15; // 1 block
    let s2_k = vec![3.0f32; s2_tokens * 4];
    let s2_v = vec![4.0f32; s2_tokens * 4];

    manager.append_kv(&s2, &s2_k, &s2_v, s2_tokens).unwrap();
    assert_eq!(manager.pool.available_blocks(), 5);

    // Read back both
    let (k1, v1) = manager.read_kv(&s1).unwrap();
    assert_eq!(k1, s1_k);
    assert_eq!(v1, s1_v);

    let (k2, v2) = manager.read_kv(&s2).unwrap();
    assert_eq!(k2, s2_k);
    assert_eq!(v2, s2_v);

    // Release session 1
    manager.release_session(s1).unwrap();
    assert_eq!(manager.pool.available_blocks(), 7); // 5 + 2

    // S2 is still intact
    let (k2_post, v2_post) = manager.read_kv(&s2).unwrap();
    assert_eq!(k2_post, s2_k);
    assert_eq!(v2_post, s2_v);

    // Double free on s1 rejected
    assert!(matches!(manager.release_session(s1), Err(PagedKvError::DoubleFree(_))));
}

#[test]
fn test_atomic_rollback_on_pool_exhaustion() {
    let mut manager = PagedKvManager::new(2, 16, 4).expect("create manager");
    let s = manager.create_session();

    // 1 block of 16 tokens
    let k = vec![1.0f32; 16 * 4];
    let v = vec![2.0f32; 16 * 4];
    manager.append_kv(&s, &k, &v, 16).unwrap();
    assert_eq!(manager.pool.available_blocks(), 1);

    // Try to append 32 tokens (requires 2 blocks, only 1 available)
    let big_k = vec![9.0f32; 32 * 4];
    let big_v = vec![9.0f32; 32 * 4];
    let err = manager.append_kv(&s, &big_k, &big_v, 32);
    assert!(matches!(err, Err(PagedKvError::OutOfMemory { required: 2, available: 1 })));

    // Verify session still has exactly 16 tokens and 1 block available
    assert_eq!(manager.pool.available_blocks(), 1);
    let (read_k, read_v) = manager.read_kv(&s).unwrap();
    assert_eq!(read_k, k);
    assert_eq!(read_v, v);
}

