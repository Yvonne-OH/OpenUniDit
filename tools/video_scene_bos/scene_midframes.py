#!/usr/bin/env python3
"""TransNetV2 scene midpoint extraction from Parquet/JSONL manifests to BOS."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

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


def run(args):
    import cv2
    import numpy as np
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    sys.path.insert(0, str(Path(args.transnet_dir).resolve()))
    from transnetv2 import TransNetV2
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    model = TransNetV2(args.weights)
    bos = None if args.dry_run else make_bos(args)
    config = {'source': str(Path(args.source).resolve()), 'prefix': args.prefix,
              'threshold': args.threshold, 'quality': args.jpeg_quality,
              'weights': args.weights or str(Path(args.transnet_dir).resolve()), 'version': 1}
    namespace = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    work = Path(args.work_dir) / namespace
    work.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(work / 'progress.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS done (id TEXT PRIMARY KEY)')
    processed = failed = skipped = 0
    logging.getLogger('baidubce').setLevel(logging.CRITICAL)
    with (work / ('preview.jsonl' if args.dry_run else 'uploaded.jsonl')).open('a') as log:
        for row_index, row in enumerate(records(args.source)):
            if row_index % args.num_shards != args.shard_index:
                continue
            if args.limit and processed + failed >= args.limit:
                break
            uri = row.get('video_uri')
            identity = hashlib.sha256((str(row.get('clip_id', '')) + '\0' + str(uri)).encode()).hexdigest()
            if not args.dry_run and db.execute('SELECT 1 FROM done WHERE id=?', (identity,)).fetchone():
                skipped += 1
                continue
            cap = None
            try:
                if not uri or not Path(uri).is_file():
                    raise FileNotFoundError('video_uri is not an accessible local file')
                with tempfile.TemporaryDirectory(dir=work) as tmp:
                    # Decode reduced RGB frames to disk: bounded RAM for long inputs.
                    raw = Path(tmp) / 'frames.rgb'
                    with raw.open('wb') as output:
                        subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-i', uri,
                                        '-map', '0:v:0', '-an', '-vf', 'scale=48:27',
                                        '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                                       stdout=output, stderr=subprocess.PIPE, check=True)
                    n, remainder = divmod(raw.stat().st_size, 27 * 48 * 3)
                    if not n or remainder:
                        raise ValueError('Empty or truncated video decode')
                    frames = np.memmap(raw, dtype=np.uint8, mode='r', shape=(n, 27, 48, 3))
                    # Official 100-frame window, 25-frame context, 50 retained predictions.
                    predictions = np.empty(n, np.float32)
                    for start in range(0, n, 50):
                        indices = np.clip(np.arange(start - 25, start + 75), 0, n - 1)
                        single, _ = model.predict_raw(frames[indices][None])
                        length = min(50, n - start)
                        predictions[start:start+length] = single.numpy()[0, 25:25+length, 0]
                    scenes = model.predictions_to_scenes(predictions, threshold=args.threshold)
                    del frames
                    targets = {(int(a) + int(b)) // 2: (i, int(a), int(b))
                               for i, (a, b) in enumerate(scenes)}
                    # Sequential decoding preserves exact frame indices, including VFR inputs.
                    cap = cv2.VideoCapture(uri)
                    items = []
                    for frame_index in range(max(targets) + 1):
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError('Full resolution decode ended before midpoint')
                        if frame_index not in targets:
                            continue
                        scene_index, begin, end = targets[frame_index]
                        key = f'{args.prefix}/{identity[:2]}/{identity}/scene_{scene_index:05d}_frame_{frame_index:09d}.jpg'
                        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                        if not ok:
                            raise ValueError('JPEG encode failed')
                        image = Path(tmp) / 'frame.jpg'
                        image.write_bytes(encoded.tobytes())
                        item = {'id': f'{identity}_{scene_index:05d}', 'source_metadata': row,
                                'video_uri': uri, 'scene_index': scene_index,
                                'scene_start_frame': begin, 'scene_end_frame_inclusive': end,
                                'frame_index': frame_index, 'timestamp_seconds': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                'w': frame.shape[1], 'h': frame.shape[0],
                                'sha256': hashlib.sha256(encoded.tobytes()).hexdigest(),
                                'image_uri': f'bos://{args.bucket}/{key}'}
                        if bos:
                            bos.put_object_from_file(args.bucket, key, str(image), content_type='image/jpeg')
                        items.append(item)
                    cap.release()
                    cap = None
                    manifest = Path(tmp) / 'metadata.jsonl'
                    manifest.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
                    if bos:
                        # Per-video metadata is committed last; retries reuse deterministic keys.
                        bos.put_object_from_file(args.bucket, f'{args.prefix}/{identity[:2]}/{identity}/metadata.jsonl', str(manifest))
                    for item in items:
                        log.write(json.dumps(item, ensure_ascii=False) + '\n')
                    log.flush()
                    if bos:
                        db.execute('INSERT OR IGNORE INTO done VALUES (?)', (identity,))
                        db.commit()
                    processed += 1
                    print(json.dumps({'row': row_index, 'scenes': len(items), 'processed': processed}), flush=True)
            except Exception as exc:
                failed += 1
                # Do not log raw SDK exceptions, which can include signed request headers.
                with (work / 'errors.jsonl').open('a') as errors:
                    errors.write(json.dumps({'row': row_index, 'uri': uri, 'error_type': type(exc).__name__}) + '\n')
                print(json.dumps({'row': row_index, 'error_type': type(exc).__name__}), flush=True)
            finally:
                if cap is not None:
                    cap.release()
    db.close()
    print(json.dumps({'processed': processed, 'failed': failed, 'skipped': skipped, 'work_dir': str(work)}))
    return 1 if failed else 0


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=DEFAULT_SOURCE)
    p.add_argument('--bucket', default='giga-video-pretrain-dataset')
    p.add_argument('--prefix', default=ROOT + '20261013_all_2_10s_captioned_usable_delta_transnet_midframes')
    p.add_argument('--credentials')
    p.add_argument('--endpoint', default='https://bj.bcebos.com')
    p.add_argument('--transnet-dir', default=str(here / 'vendor/TransNetV2/inference'))
    p.add_argument('--weights')
    p.add_argument('--work-dir', default=str(here / 'work'))
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--limit', type=int, default=0, help='0 processes all assigned videos')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--dry-run', action='store_true', help='Infer and encode, without BOS writes')
    args = p.parse_args()
    args.prefix = args.prefix.strip('/')
    if not args.prefix.startswith(ROOT) or args.prefix == ROOT.rstrip('/'):
        p.error('prefix must be a subfolder under ' + ROOT)
    if not 0 <= args.threshold <= 1 or not 1 <= args.jpeg_quality <= 100:
        p.error('invalid threshold or JPEG quality')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.limit < 0:
        p.error('invalid shard or limit')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
