"""Bounded real-data NFS pressure test for 64 logical training ranks.

Launch on the CPU developer host with torchrun --nproc_per_node=64. Uses the
production data stream, sampler, retries, worker timeout, global skip and cursor.
This measures storage supply, not 64-GPU computation or cross-node networking.
"""
import argparse
import ast
import contextlib
import datetime
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
import yaml
from data.lmdb_dataset import GigaLMDBDataset, collate_image_batch
from data.samplers.bucket_sampler import BucketBatchSampler


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered)-1)*q
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(k-lo)


def stream_helpers():
    source = ROOT/'scripts/train/train_giga_exp.py'
    tree = ast.parse(source.read_text())
    names = {'TrainingDataStream', '_data_error_category', 'next_training_batch', 'sampler_signature',
             'make_training_data_stream'}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = dict(__name__=__name__, torch=torch, dist=dist, time=time,
                 logging=logging, hashlib=hashlib, json=json, collate=collate_image_batch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope['make_training_data_stream'], scope['next_training_batch']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ranks-per-node', type=int, default=8,
                        help='Logical topology for local I/O slots; does not simulate network links')
    parser.add_argument('--io-slots-per-node', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--micro-batches', type=int, default=128)
    parser.add_argument('--max-seconds', type=float, default=900)
    parser.add_argument('--target-micro-seconds', type=float, default=0)
    parser.add_argument('--check-watermark', action='store_true', help='Exercise the training startup readiness check before reading')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if args.ranks_per_node <= 0 or world % args.ranks_per_node:
        raise ValueError('world size must be divisible by positive ranks-per-node')
    os.environ['YISU_TAR_IO_NAMESPACE'] = f'nfs-benchmark-node-{rank // args.ranks_per_node}'
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.io_slots_per_node is not None:
        cfg['dataset']['tar_io_concurrency_per_node'] = args.io_slots_per_node
    if not cfg['dataset'].get('freeze_index_signature', False):
        raise ValueError('64-rank test requires the verified frozen index signature')
    if args.workers is not None:
        cfg['training']['num_workers'] = args.workers
    logger = logging.getLogger(f'nfs-rank-{rank}')
    logging.basicConfig(level=logging.WARNING, format=f'rank={rank} %(levelname)s %(message)s')
    dist.init_process_group('gloo', timeout=datetime.timedelta(seconds=300))
    TrainingDataStream, next_training_batch = stream_helpers()
    stream = None
    stage = 'index_loading'
    started = time.perf_counter()
    try:
        # Avoid 64 copies of the large distribution table in one log file.
        with contextlib.redirect_stdout(io.StringIO()):
            data = GigaLMDBDataset(cfg['dataset'], seed=int(cfg.get('seed', 42)),
                index_cache_dir=str(Path(cfg['output_dir'])/'dataset_index_cache'))
        sampler = BucketBatchSampler(data.bucket_indices, cfg['training']['batch_size_per_device'],
            weights=cfg['dataset'].get('bucket_sampling_weights'),
            bucket_batch_sizes=cfg['dataset'].get('bucket_batch_sizes'),
            seed=int(cfg.get('seed', 42)), rank=rank, world_size=world,
            randomize_order_on_start=True, drop_last=True)
        stream = TrainingDataStream(data, sampler, cfg['training'], logger)
        watermark = None
        if args.check_watermark:
            try:
                local_status = stream.wait_ready(cfg['training']['ready_min_batches'],
                                                 cfg['training']['ready_warmup_timeout_seconds'])
            except Exception as exc:
                local_status = {'ready': False, 'error': str(exc)}
            watermark = [None] * world
            dist.all_gather_object(watermark, local_status)
            if not all(entry['ready'] for entry in watermark):
                if rank == 0:
                    result = dict(ok=False, stage='startup_watermark', world_size=world,
                                  ready_ranks=sum(entry['ready'] for entry in watermark),
                                  rank_status=watermark, global_samples=0, measured_attempts=0)
                    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
                    print('NFS_WATERMARK_FAILED', result['ready_ranks'], '/', world, flush=True)
                return
        original_next = stream.next_batch
        local_fetch = 0.
        def measured_next():
            nonlocal local_fetch
            begin = time.perf_counter()
            try:
                return original_next()
            finally:
                local_fetch = time.perf_counter()-begin
        stream.next_batch = measured_next
        startup = time.perf_counter()-started
        dist.barrier()
        measured = []
        shapes, samples, skipped = {}, 0, 0
        timed_start = None
        stage = 'real_tar_read'
        for attempt in range(args.warmup+args.micro_batches):
            stop = torch.tensor([time.perf_counter()-started > args.max_seconds], dtype=torch.int32)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                break
            if attempt == args.warmup:
                timed_start = time.perf_counter()
            tick = time.perf_counter()
            local_fetch = 0.
            batch = next_training_batch(stream, 'cpu', logger, 0, attempt)
            fetch_and_global_status = time.perf_counter()-tick
            local_samples = 0 if batch is None else len(batch['captions'])
            maxima = torch.tensor([local_fetch, fetch_and_global_status], dtype=torch.float64)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
            total = torch.tensor([local_samples], dtype=torch.int64)
            dist.all_reduce(total)
            if rank == 0 and attempt < args.warmup and (attempt+1) % 8 == 0:
                print('NFS_WARMUP', json.dumps(dict(attempt=attempt+1,
                    slowest_fetch_seconds=float(maxima[0]), elapsed_seconds=time.perf_counter()-started)), flush=True)
            if attempt >= args.warmup:
                skipped += int(batch is None)
                samples += local_samples
                if batch is not None:
                    shape = str(list(batch['pixel_values'].shape))
                    shapes[shape] = shapes.get(shape, 0)+1
                measured.append(dict(fetch_seconds=local_fetch,
                    fetch_and_status_seconds=fetch_and_global_status,
                    slowest_fetch_seconds=float(maxima[0]),
                    slowest_fetch_and_status_seconds=float(maxima[1]),
                    global_samples=int(total[0]), skipped=batch is None))
                if rank == 0:
                    print('NFS_REAL_READ', json.dumps(dict(attempt=attempt-args.warmup+1,
                        world_size=world, samples=int(total[0]),
                        slowest_fetch_seconds=float(maxima[0]),
                        slowest_fetch_and_status_seconds=float(maxima[1]),
                        skipped=batch is None)), flush=True)
            if args.target_micro_seconds:
                time.sleep(max(0., args.target_micro_seconds-(time.perf_counter()-tick)))
        elapsed = time.perf_counter()-timed_start if timed_start else None
        local = dict(rank=rank, startup_seconds=startup, timed_seconds=elapsed,
            measured_attempts=len(measured), samples=samples, globally_skipped_attempts=skipped,
            shapes=shapes, cursor=stream.state_dict(), measurements=measured)
        ranks = [None]*world
        dist.all_gather_object(ranks, local)
        if rank == 0:
            slowest = [row['slowest_fetch_and_status_seconds'] for row in measured]
            fetches = [row['fetch_seconds'] for result in ranks for row in result['measurements']]
            global_samples = sum(result['samples'] for result in ranks)
            wall = max(result['timed_seconds'] or 0. for result in ranks)
            output = dict(ok=len(measured)==args.micro_batches and skipped==0,
                test='real_NFS_logical_rank_pressure', world_size=world,
                seed=int(cfg.get('seed', 42)),
                worker_count_per_rank=cfg['training'].get('num_workers', 4),
                logical_ranks_per_node=args.ranks_per_node,
                simulated_nodes=world // args.ranks_per_node,
                io_slots_per_node=cfg['dataset'].get('tar_io_concurrency_per_node', 0),
                ready_first=bool(cfg['training'].get('ready_data_pipeline', False)),
                startup_watermark=watermark,
                total_workers=world*cfg['training'].get('num_workers', 4),
                measured_attempts=len(measured), requested_attempts=args.micro_batches,
                global_samples=global_samples, timed_seconds=wall,
                samples_per_second=global_samples/wall if wall else None,
                global_skip_attempts=skipped, target_micro_seconds=args.target_micro_seconds,
                slowest_rank_wait_p50=percentile(slowest, .5),
                slowest_rank_wait_p95=percentile(slowest, .95),
                slowest_rank_wait_max=max(slowest, default=None),
                slowest_read_wait_p95=percentile([r['slowest_fetch_seconds'] for r in measured], .95),
                local_fetch_p50=percentile(fetches, .5), local_fetch_p95=percentile(fetches, .95),
                local_fetch_p99=percentile(fetches, .99),
                limitations=['One CPU host; excludes cross-node NCCL and GPU compute.',
                    'Host CPU limits and existing OS/NFS caches affect results; no caches were dropped.',
                    'Paced tests include compute-like delay; unpaced tests measure maximum input supply.'],
                ranks=ranks)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(output, indent=2)+'\n')
            print('NFS_RANK_TEST_COMPLETE', json.dumps({key:value for key,value in output.items() if key!='ranks'}), flush=True)
    except Exception as exc:
        print('NFS_RANK_TEST_FAILED', json.dumps(dict(rank=rank, stage=stage, error=str(exc))), flush=True)
        raise
    finally:
        if stream is not None:
            stream.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
