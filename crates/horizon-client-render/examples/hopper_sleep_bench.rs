use horizon_client_render::{
    block_entities::{BlockEntities, Ticking},
    containers::BlockEntity,
};
use std::time::Instant;

fn hopper() -> BlockEntity {
    BlockEntity::Chest(vec![None; 5])
}

fn simulate_tick_baseline(
    entities: &mut BlockEntities,
    tickers: &[((i32, i32, i32), u32)],
) -> u64 {
    let mut total_dummy = 0u64;
    for &slot in tickers {
        if !entities.is_live(slot) {
            continue;
        }
        let pos = slot.0;
        // Baseline: Hopper does not sleep, executes hopper_try_move_items fast-path
        if let Some(BlockEntity::Chest(items)) = entities.get(&pos) {
            let is_empty = items.iter().all(|s| s.as_ref().is_none_or(|s| s.count == 0));
            if is_empty {
                // In baseline, returns without sleeping, having performed block/entity lookups
                total_dummy += 1;
            }
        }
    }
    total_dummy
}

fn simulate_tick_optimized(
    entities: &mut BlockEntities,
    tickers: &[((i32, i32, i32), u32)],
) -> u64 {
    let mut total_dummy = 0u64;
    for &slot in tickers {
        let pos = slot.0;
        // Optimized: checks sleep first (as in block_entity_ticks.rs line 84)
        if entities.is_asleep(&pos) || !entities.is_live(slot) {
            continue;
        }
        if let Some(BlockEntity::Chest(items)) = entities.get(&pos) {
            let is_empty = items.iter().all(|s| s.as_ref().is_none_or(|s| s.count == 0));
            if is_empty {
                entities.sleep(pos);
                total_dummy += 1;
            }
        }
    }
    total_dummy
}

fn main() {
    println!("=== Horizon Idle Hoppers Sleep Benchmark (A/B Test) ===");

    const COUNT: usize = 16384;
    const TICKS: usize = 100;

    println!("Setting up {COUNT} idle hoppers in world...");

    let mut b_base = BlockEntities::default();
    b_base.set_states(vec![Ticking::Yes]);
    let mut b_opt = BlockEntities::default();
    b_opt.set_states(vec![Ticking::Yes]);

    for i in 0..COUNT as i32 {
        let pos = (i % 256, 64, i / 256);
        b_base.insert(pos, hopper());
        b_opt.insert(pos, hopper());
    }

    let tickers_base = b_base.tickers(|_| 0).to_vec();
    let tickers_opt = b_opt.tickers(|_| 0).to_vec();

    assert_eq!(tickers_base.len(), COUNT);
    assert_eq!(tickers_opt.len(), COUNT);

    // Warm-up
    simulate_tick_baseline(&mut b_base, &tickers_base);
    simulate_tick_optimized(&mut b_opt, &tickers_opt);

    println!("--- Benchmark : {TICKS} ticks over {COUNT} idle hoppers ---");

    // 1. Baseline
    let t0 = Instant::now();
    let mut acc_base = 0;
    for _ in 0..TICKS {
        acc_base += simulate_tick_baseline(&mut b_base, &tickers_base);
    }
    let dur_base = t0.elapsed();
    let ms_per_tick_base = (dur_base.as_secs_f64() * 1000.0) / (TICKS as f64);

    // 2. Optimized (Sleep)
    // First tick puts idle hoppers to sleep, subsequent ticks skip them completely
    let t1 = Instant::now();
    let mut acc_opt = 0;
    for _ in 0..TICKS {
        acc_opt += simulate_tick_optimized(&mut b_opt, &tickers_opt);
    }
    let dur_opt = t1.elapsed();
    let ms_per_tick_opt = (dur_opt.as_secs_f64() * 1000.0) / (TICKS as f64);

    println!("Baseline (No Sleep) : {:.3} ms/tick ({:.2} ms total for {} ticks)",
        ms_per_tick_base, dur_base.as_secs_f64() * 1000.0, TICKS);
    println!("Optimized (Sleeping): {:.3} ms/tick ({:.2} ms total for {} ticks)",
        ms_per_tick_opt, dur_opt.as_secs_f64() * 1000.0, TICKS);

    let speedup = (ms_per_tick_base - ms_per_tick_opt) / ms_per_tick_base * 100.0;
    let ratio = ms_per_tick_base / ms_per_tick_opt;
    println!("-> Speedup : {:.1}% ({:.2}x plus rapide)", speedup, ratio);
    println!("Validation checks: acc_base={}, acc_opt={}", acc_base, acc_opt);
}
