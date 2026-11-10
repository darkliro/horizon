//! Persistent background worker pool for parallel chunk compression and NBT encoding.
use std::{
    collections::VecDeque,
    io,
    sync::{
        Arc, Condvar, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use horizon_world::{compression::Compression, nbt::Limits, region::ChunkRecord};

use super::Record;

struct Batch {
    work: Box<dyn Fn(usize) + Send + Sync>,
    count: usize,
    next: AtomicUsize,
    remaining: AtomicUsize,
    done_lock: Mutex<bool>,
    done: Condvar,
}

struct Inner {
    queue: Mutex<VecDeque<Arc<Batch>>>,
    wake: Condvar,
    shutdown: AtomicBool,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.wake.notify_all();
    }
}

pub struct CompressionPool {
    inner: Arc<Inner>,
    workers: usize,
}

static POOL: OnceLock<CompressionPool> = OnceLock::new();

pub fn pool() -> &'static CompressionPool {
    POOL.get_or_init(|| {
        let workers = std::env::var("HORIZON_SAVE_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| horizon_threads::worker_count().div_ceil(2).max(1));
        CompressionPool::new(workers)
    })
}

impl CompressionPool {
    pub fn new(workers: usize) -> Self {
        let inner = Arc::new(Inner {
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        for i in 0..workers {
            let worker_inner = Arc::clone(&inner);
            let _ = std::thread::Builder::new()
                .name(format!("horizon-save-compression-{i}"))
                .spawn(move || worker_loop(&worker_inner));
        }
        Self { inner, workers }
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    pub fn run(&self, count: usize, work: impl Fn(usize) + Send + Sync + 'static) {
        if count == 0 {
            return;
        }
        if count == 1 || self.workers == 0 {
            work(0);
            return;
        }
        let batch = Arc::new(Batch {
            work: Box::new(work),
            count,
            next: AtomicUsize::new(0),
            remaining: AtomicUsize::new(count),
            done_lock: Mutex::new(false),
            done: Condvar::new(),
        });
        {
            let mut queue = self.inner.queue.lock().unwrap_or_else(|e| e.into_inner());
            queue.push_back(Arc::clone(&batch));
            self.inner.wake.notify_all();
        }
        while let Some(i) = next_job(&batch) {
            run_job(&batch, i);
        }
        let mut finished = batch.done_lock.lock().unwrap_or_else(|e| e.into_inner());
        while !*finished {
            finished = batch.done.wait(finished).unwrap_or_else(|e| e.into_inner());
        }
        let mut queue = self.inner.queue.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = queue.iter().position(|b| Arc::ptr_eq(b, &batch)) {
            queue.remove(pos);
        }
    }
}

fn next_job(batch: &Batch) -> Option<usize> {
    let i = batch.next.fetch_add(1, Ordering::Relaxed);
    if i < batch.count { Some(i) } else { None }
}

fn run_job(batch: &Arc<Batch>, i: usize) {
    struct Guard<'a>(&'a Arc<Batch>);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            if self.0.remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
                let mut finished = self.0.done_lock.lock().unwrap_or_else(|e| e.into_inner());
                *finished = true;
                self.0.done.notify_all();
            }
        }
    }
    let _guard = Guard(batch);
    (batch.work)(i);
}

fn worker_loop(inner: &Arc<Inner>) {
    loop {
        let batch = {
            let mut queue = inner.queue.lock().unwrap_or_else(|e| e.into_inner());
            while queue.is_empty() && !inner.shutdown.load(Ordering::Relaxed) {
                queue = inner.wake.wait(queue).unwrap_or_else(|e| e.into_inner());
            }
            if inner.shutdown.load(Ordering::Relaxed) && queue.is_empty() {
                return;
            }
            queue.front().cloned()
        };
        if let Some(batch) = batch {
            let mut did_work = false;
            while let Some(i) = next_job(&batch) {
                did_work = true;
                run_job(&batch, i);
            }
            if !did_work {
                let mut queue = inner.queue.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(front) = queue.front()
                    && Arc::ptr_eq(front, &batch)
                    && front.next.load(Ordering::Relaxed) >= front.count
                {
                    queue.pop_front();
                }
            }
        }
    }
}

pub fn compress_records_with_pool(
    records: &[((i32, i32), &Record)],
    timestamp: u32,
    pool: &CompressionPool,
) -> Vec<io::Result<Option<ChunkRecord>>> {
    let count = records.len();
    if count == 0 {
        return Vec::new();
    }
    let compression = Compression::selected();
    let compress = move |record: &Record| {
        record
            .tag()
            .map(|tag| ChunkRecord::from_nbt(&tag, compression, timestamp, Limits::saved()))
            .transpose()
    };
    if count == 1 || pool.workers() == 0 {
        return vec![compress(records[0].1)];
    }
    let records_arc: Arc<[Record]> = records.iter().map(|(_, r)| (*r).clone()).collect();
    let results: Arc<Mutex<Vec<Option<io::Result<Option<ChunkRecord>>>>>> =
        Arc::new(Mutex::new((0..count).map(|_| None).collect()));

    let recs = Arc::clone(&records_arc);
    let res_lock = Arc::clone(&results);
    pool.run(count, move |i| {
        let res = compress(&recs[i]);
        res_lock.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(res);
    });

    let mut guard = results.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .drain(..)
        .map(|opt| opt.expect("all records compressed"))
        .collect()
}

pub fn compress_records(
    records: &[((i32, i32), &Record)],
    timestamp: u32,
) -> Vec<io::Result<Option<ChunkRecord>>> {
    compress_records_with_pool(records, timestamp, pool())
}
