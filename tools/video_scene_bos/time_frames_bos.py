#!/usr/bin/env python3
"""Sample videos by time, deduplicate with CPU pHash, upload JPEGs to BOS."""
import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
import sqlite3
import tempfile
import sys

DEFAULT_SOURCE = '/mnt/nfs/users/boyuan.wang/yisu_data_lake/manifests/train/boyuan.wang/20261013_all_2_10s_captioned_usable_delta'
ROOT = 'users/zhanqian.wu/unified_images_ori/'

def records(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'manifest.parquet'
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def make_bos(args):
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.services.bos.bos_client import BosClient
    cred = json.loads(Path(args.credentials).read_text()) if args.credentials else {}
    ak = cred.get('ak', cred.get('accesskey', os.environ.get('BOS_ACCESS_KEY_ID')))
    sk = cred.get('sk', cred.get('secretkey', os.environ.get('BOS_SECRET_ACCESS_KEY')))
    if not ak or not sk:
        raise ValueError('Provide --credentials or BOS_ACCESS_KEY_ID/BOS_SECRET_ACCESS_KEY')
    config = BceClientConfiguration(credentials=BceCredentials(ak, sk), endpoint=args.endpoint)
    token = cred.get('security_token') or cred.get('session_token') or cred.get('sessionToken')
    if token:
        config.security_token = token
    return BosClient(config)


def phash(frame):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small.astype(np.float32))[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return sum(int(bit) << (63-i) for i, bit in enumerate(bits))


QUALITY_FIELDS = ['min_short_side', 'min_blur_score', 'min_brightness',
                  'max_brightness', 'min_contrast', 'max_gray_chroma']


def quality(frame, args):
    """Metrics use an aspect-preserving max-512 thumbnail for stable CPU cost."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    scale = min(1., 512. / max(h, w))
    thumb = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    chroma = np.max(thumb, axis=2).astype(np.int16) - np.min(thumb, axis=2).astype(np.int16)
    metrics = {'short_side': min(h, w),
               'blur_score': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
               'brightness': float(gray.mean()), 'contrast': float(gray.std()),
               'chroma_p95': float(np.percentile(chroma, 95))}
    reasons = []
    if metrics['short_side'] <= args.min_short_side:
        reasons.append('resolution')
    if metrics['blur_score'] < args.min_blur_score:
        reasons.append('blur')
    if metrics['brightness'] < args.min_brightness:
        reasons.append('dark')
    if metrics['brightness'] > args.max_brightness:
        reasons.append('bright')
    if metrics['contrast'] < args.min_contrast:
        reasons.append('low_contrast')
    if metrics['chroma_p95'] <= args.max_gray_chroma:
        reasons.append('grayscale')
    return metrics, reasons


def video_directory(prefix, row, identity):
    source = str(row.get('source') or 'unknown')
    safe = re.sub(r'[^a-zA-Z0-9_.-]+', '_', source).strip('._')[:80] or 'unknown'
    # Hash suffix avoids collisions between source names after sanitization.
    safe += '_' + hashlib.sha256(source.encode()).hexdigest()[:8]
    return f'{prefix}/{safe}/{identity[:2]}/{identity[2:4]}/{identity}'


def run(args):
    import cv2
    cv2.setNumThreads(args.cpu_threads)
    logging.getLogger('baidubce').setLevel(logging.CRITICAL)
    bos = None if args.dry_run else make_bos(args)
    settings = {k: getattr(args, k) for k in ['source', 'bucket', 'prefix', 'interval', 'phash_distance', 'jpeg_quality', 'num_shards', 'shard_index'] + QUALITY_FIELDS}
    settings['pipeline_version'] = 'quality-hierarchy-v3-strict'
    settings['algorithm'] = 'opencv-dct-area32-phash64-v1'
    namespace = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    work = Path(args.work_dir) / namespace
    work.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(work / 'progress.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS done (id TEXT PRIMARY KEY)')
    processed = failed = skipped = kept = duplicates = 0
    rejected_total = Counter()
    candidate_total = rejected_frames = 0
    for index, row in enumerate(records(args.source)):
        if index % args.num_shards != args.shard_index:
            continue
        if args.limit and processed + failed >= args.limit:
            break
        uri = row.get('video_uri')
        identity = hashlib.sha256((str(row.get('clip_id', '')) + '\0' + str(uri)).encode()).hexdigest()
        if not args.dry_run and db.execute('SELECT 1 FROM done WHERE id=?', (identity,)).fetchone():
            skipped += 1
            continue
        cap = None
        directory = video_directory(args.prefix, row, identity)
        try:
            if not uri or not Path(uri).is_file():
                raise FileNotFoundError('video_uri is not an accessible file')
            cap = cv2.VideoCapture(uri)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError('Cannot open video or determine FPS')
            hashes, items = [], []
            rejection_counts = Counter()
            candidates = rejected = 0
            frame_index, next_time, last_time, video_duplicates = 0, 0.0, -1.0, 0
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    # Prefer presentation timestamps; fall back to frame/FPS for broken backends.
                    if not math.isfinite(timestamp) or timestamp < 0 or (frame_index and timestamp <= last_time):
                        timestamp = frame_index / fps
                    last_time = timestamp
                    if timestamp + 1e-6 >= next_time:
                        next_time = (math.floor((timestamp + 1e-6) / args.interval) + 1) * args.interval
                        candidates += 1
                        metrics, reasons = quality(frame, args)
                        if reasons:
                            rejected += 1
                            rejection_counts.update(reasons)
                            frame_index += 1
                            continue
                        value = phash(frame)
                        if any((value ^ old).bit_count() <= args.phash_distance for old in hashes):
                            video_duplicates += 1
                        else:
                            hashes.append(value)
                            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                            if not ok:
                                raise ValueError('JPEG encoding failed')
                            payload = encoded.tobytes()
                            key = f'{directory}/frame_{frame_index:09d}.jpg'
                            item = {'id': f'{identity}_{frame_index:09d}', 'video_uri': uri,
                                    'frame_index': frame_index, 'timestamp_seconds': timestamp,
                                    'w': frame.shape[1], 'h': frame.shape[0], 'quality': metrics,
                                    'phash64': f'{value:016x}', 'phash_algorithm': settings['algorithm'],
                                    'sha256': hashlib.sha256(payload).hexdigest(),
                                    'image_uri': f'bos://{args.bucket}/{key}', 'source_metadata': row}
                            image = Path(tmp) / 'frame.jpg'
                            image.write_bytes(payload)
                            if bos:
                                bos.put_object_from_file(args.bucket, key, str(image), content_type='image/jpeg')
                            if args.save_images:
                                local = Path(args.save_images) / key.removeprefix(args.prefix + '/')
                                local.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copyfile(image, local)
                            items.append(item)
                    frame_index += 1
                if not frame_index:
                    raise ValueError('No decodable frames')
                # Detect clearly truncated reads when the container advertises a frame count.
                expected = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if expected > 0 and frame_index + 1 < expected:
                    raise ValueError('Video decode ended before advertised frame count')
                metadata = Path(tmp) / 'metadata.jsonl'
                metadata.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
                summary = {'id': identity, 'video_uri': uri, 'candidates': candidates,
                           'kept': len(items), 'duplicates': video_duplicates,
                           'rejected_frames': rejected, 'rejection_reasons': dict(rejection_counts),
                           'settings': settings}
                summary_file = Path(tmp) / 'summary.json'
                summary_file.write_text(json.dumps(summary, ensure_ascii=False))
                if args.save_images:
                    local_dir = Path(args.save_images) / directory.removeprefix(args.prefix + '/')
                    local_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(metadata, local_dir/'metadata.jsonl')
                    shutil.copyfile(summary_file, local_dir/'summary.json')
                if bos:
                    bos.put_object_from_file(args.bucket, f'{directory}/summary.json', str(summary_file))
                    bos.put_object_from_file(args.bucket, f'{directory}/metadata.jsonl', str(metadata))
                    db.execute('INSERT OR IGNORE INTO done VALUES (?)', (identity,))
                    db.commit()
                with (work / ('preview.jsonl' if args.dry_run else 'uploaded.jsonl')).open('a') as log:
                    log.write(metadata.read_text())
                with (work / 'video_summaries.jsonl').open('a') as summaries:
                    summaries.write(json.dumps(summary, ensure_ascii=False) + '\n')
                candidate_total += candidates
                rejected_frames += rejected
                rejected_total.update(rejection_counts)
                processed += 1
                kept += len(items)
                duplicates += video_duplicates
                print(json.dumps({'row': index, 'kept': len(items), 'duplicates': video_duplicates, 'rejected': rejected, 'reasons': dict(rejection_counts)}), flush=True)
        except Exception as exc:
            failed += 1
            error = {'row': index, 'video_uri': uri, 'error_type': type(exc).__name__}
            with (work / 'errors.jsonl').open('a') as log:
                log.write(json.dumps(error) + '\n')
            print(json.dumps(error), flush=True)
        finally:
            if cap is not None:
                cap.release()
    db.close()
    print(json.dumps({'processed': processed, 'failed': failed, 'skipped': skipped, 'kept': kept,
                      'duplicates': duplicates, 'candidates': candidate_total, 'rejected_frames': rejected_frames,
                      'rejection_reasons': dict(rejected_total), 'work_dir': str(work)}), flush=True)
    return int(failed > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=DEFAULT_SOURCE)
    p.add_argument('--bucket', default='giga-video-pretrain-dataset')
    p.add_argument('--prefix', default=ROOT + '20261013_all_2_10s_captioned_usable_delta_timeframes_quality_v3_strict')
    p.add_argument('--credentials', help='Private JSON containing ak/sk; alternatively use BOS environment variables')
    p.add_argument('--endpoint', default='https://bj.bcebos.com')
    p.add_argument('--interval', type=float, default=2.0)
    p.add_argument('--phash-distance', type=int, default=14, help='Within-video Hamming distance <= threshold is duplicate')
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--min-short-side', type=int, default=512, help='Strictly greater than this value')
    p.add_argument('--min-blur-score', type=float, default=200., help='Laplacian variance on max-512 thumbnail')
    p.add_argument('--min-brightness', type=float, default=25.)
    p.add_argument('--max-brightness', type=float, default=230.)
    p.add_argument('--min-contrast', type=float, default=20., help='Grayscale standard deviation')
    p.add_argument('--max-gray-chroma', type=float, default=8., help='Reject chroma p95 at or below threshold')
    p.add_argument('--save-images', help='Optional local image root; preserves source/hash/video hierarchy')
    p.add_argument('--cpu-threads' , type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--work-dir', default=str(Path(__file__).resolve().parent / 'time_work'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    args.prefix = args.prefix.strip('/')
    if not args.prefix.startswith(ROOT):
        p.error('prefix must be a subfolder under ' + ROOT)
    if not math.isfinite(args.interval) or args.interval <= 0 or not 0 <= args.phash_distance <= 64:
        p.error('invalid interval or hash distance')
    if not 1 <= args.jpeg_quality <= 100 or args.cpu_threads < 1:
        p.error('invalid JPEG quality or CPU threads')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.limit < 0:
        p.error('invalid shard/limit')
    if any(not math.isfinite(getattr(args, k)) for k in QUALITY_FIELDS):
        p.error('quality thresholds must be finite')
    if args.min_short_side < 0 or args.min_blur_score < 0 or args.min_contrast < 0 or not 0 <= args.max_gray_chroma <= 255:
        p.error('invalid quality threshold')
    if not 0 <= args.min_brightness < args.max_brightness <= 255:
        p.error('invalid brightness range')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
