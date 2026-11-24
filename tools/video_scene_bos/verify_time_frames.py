"""Bounded integration validation; never traverses the full dataset."""
import argparse
import contextlib
import hashlib
import io
import json
import sqlite3
import time
from pathlib import Path
import cv2
import numpy as np
import time_frames_bos as m

root = Path(__file__).resolve().parent
work = root / 'validation_20261124'
work.mkdir(exist_ok=True)
source = work / 'sample20.jsonl'
import itertools
rows = list(itertools.islice(m.records(m.DEFAULT_SOURCE), 20))
source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
args = argparse.Namespace(source=str(source), bucket='giga-video-pretrain-dataset',
    prefix=m.ROOT+'20261013_all_2_10s_captioned_usable_delta_timeframes', interval=2.,
    phash_distance=6, jpeg_quality=95, num_shards=1, shard_index=0, cpu_threads=2,
    limit=0, work_dir=str(work/'checkpoints'), dry_run=False,
    credentials=str(root/'.bos_credentials.json'), endpoint='https://bj.bcebos.com')
start=time.monotonic()
assert m.run(args)==0, 'real sample processing failed'
elapsed=time.monotonic()-start
log=next((work/'checkpoints').glob('*/uploaded.jsonl'))
output=[json.loads(line) for line in log.read_text().splitlines()]
by_video={}
for item in output:
    by_video.setdefault(item['video_uri'], []).append(item)
assert len(by_video)==20
bos=m.make_bos(args)
verified=removed=payload_bytes=0
for row in rows:
    items=by_video[row['video_uri']]
    cap=cv2.VideoCapture(row['video_uri'])
    fps=cap.get(cv2.CAP_PROP_FPS)
    # Independent selection at 0,2,4,... seconds from presentation timestamps.
    next_sample=0.
    idx=0
    prior=-1.
    selected=[]
    retained=[]
    expected=[]
    while True:
        ok, frame=cap.read()
        if not ok:
            break
        ts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if not np.isfinite(ts) or ts<0 or (idx and ts<=prior):
            ts=idx/fps
        prior=ts
        if ts+1e-6>=next_sample:
            while next_sample<=ts+1e-6:
                next_sample+=2.
            selected.append(idx)
            h=m.phash(frame)
            if all((h^old).bit_count()>6 for old in retained):
                retained.append(h)
                expected.append(idx)
                item=items[len(expected)-1]
                assert item['frame_index']==idx
                assert abs(item['timestamp_seconds']-ts)<1e-6
                assert item['source_metadata']==row
                assert item['phash64']==f'{h:016x}'
                ok, encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
                assert ok
                key=item['image_uri'].split('/',3)[3]
                data=bos.get_object_as_string(args.bucket,key)
                assert data==encoded.tobytes(), 'source frame differs from uploaded JPEG'
                assert hashlib.sha256(data).hexdigest()==item['sha256']
                decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                assert decoded.shape[:2]==(item['h'],item['w'])
                payload_bytes+=len(data)
                verified+=1
        idx+=1
    cap.release()
    assert expected==[item['frame_index'] for item in items]
    removed+=len(selected)-len(expected)
    meta_key=items[0]['image_uri'].split('/',3)[3].rsplit('/',1)[0]+'/metadata.jsonl'
    remote_meta=bos.get_object_as_string(args.bucket,meta_key)
    assert [json.loads(x) for x in remote_meta.splitlines()]==items
# Exact same bounded input: no new successful writes/rows should be produced.
before=log.read_bytes()
resume=io.StringIO()
with contextlib.redirect_stdout(resume):
    assert m.run(args)==0
summary=json.loads(resume.getvalue().splitlines()[-1])
assert summary['processed']==0 and summary['skipped']==20
assert log.read_bytes()==before
# Synthetic freeze -> different scene: five candidates, exactly two retained.
synthetic=work/'repeat.avi'
writer=cv2.VideoWriter(str(synthetic),cv2.VideoWriter_fourcc(*'MJPG'),10,(96,64))
assert writer.isOpened()
a=np.random.default_rng(42).integers(0,256,(64,96,3),dtype=np.uint8)
b=np.zeros_like(a); b[:,48:]=255
for frame in [a]*30+[b]*20:
    writer.write(frame)
writer.release()
manifest=work/'synthetic.jsonl'
manifest.write_text(json.dumps({'clip_id':'validation','video_uri':str(synthetic)})+'\n')
args.source=str(manifest); args.interval=1.; args.dry_run=True
args.work_dir=str(work/'synthetic_checkpoints')
capture=io.StringIO()
with contextlib.redirect_stdout(capture):
    assert m.run(args)==0
synthetic_result=json.loads(capture.getvalue().splitlines()[-1])
assert synthetic_result['kept']==2 and synthetic_result['duplicates']==3
report={'videos':20,'verified_images':verified,'removed_similar_frames':removed,
        'uploaded_jpeg_bytes':payload_bytes,'processing_seconds':round(elapsed,2),
        'source_jpeg_exact_match':True,'bos_sha256_match':True,'dimensions_match':True,
        'remote_metadata_match':True,'sampling_timestamps_match':True,
        'resume_skipped':summary['skipped'],'synthetic_dedup':{'candidates':5,'kept':2,'removed':3},
        'dedup_scope':'within each video','full_dataset_started':False}
(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
