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
work = root / 'validation_quality_v3_strict'
work.mkdir(exist_ok=True)
source = work / 'sample20.jsonl'
import itertools
rows = list(itertools.islice(m.records(m.DEFAULT_SOURCE), 20))
source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
args = argparse.Namespace(source=str(source), bucket='giga-video-pretrain-dataset',
    prefix=m.ROOT+'20261013_all_2_10s_captioned_usable_delta_timeframes_quality_v3_strict', interval=2.,
    phash_distance=14, jpeg_quality=95, num_shards=1, shard_index=0, cpu_threads=2,
    limit=0, work_dir=str(work/'checkpoints'), dry_run=False,
    min_short_side=512, min_blur_score=200., min_brightness=25., max_brightness=230.,
    min_contrast=20., max_gray_chroma=8., save_images=str(work/'images'),
    credentials=str(root/'.bos_credentials.json'), endpoint='https://bj.bcebos.com')
start=time.monotonic()
assert m.run(args)==0, 'real sample processing failed'
elapsed=time.monotonic()-start
log=next((work/'checkpoints').glob('*/uploaded.jsonl'))
output=[json.loads(line) for line in log.read_text().splitlines()]
by_video={}
for item in output:
    by_video.setdefault(item['video_uri'], []).append(item)
summaries=[json.loads(x) for x in next((work/'checkpoints').glob('*/video_summaries.jsonl')).read_text().splitlines()]
assert len(summaries)==20
bos=m.make_bos(args)
verified=removed=payload_bytes=0
for row in rows:
    items=by_video.get(row['video_uri'], [])
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
            metrics, reasons=m.quality(frame,args)
            if reasons:
                idx+=1
                continue
            h=m.phash(frame)
            if all((h^old).bit_count()>args.phash_distance for old in retained):
                retained.append(h)
                expected.append(idx)
                item=items[len(expected)-1]
                assert item['frame_index']==idx
                assert abs(item['timestamp_seconds']-ts)<1e-6
                assert item['source_metadata']==row
                assert item['phash64']==f'{h:016x}'
                assert item['quality']==metrics
                assert min(item['w'],item['h'])>512
                assert len(item['image_uri'].split('/',3)[3].removeprefix(args.prefix+'/').split('/'))==5
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
    identity=hashlib.sha256((str(row.get('clip_id',''))+'\0'+row['video_uri']).encode()).hexdigest()
    meta_key=m.video_directory(args.prefix,row,identity)+'/metadata.jsonl'
    remote_meta=bos.get_object_as_string(args.bucket,meta_key)
    assert [json.loads(x) for x in remote_meta.splitlines()]==items
assert all(not (r['id'].startswith('cbc73c3410f3') and r['frame_index']==0) for r in output), 'Previously blurry frame survived'
assert sum(x['duplicates'] for x in summaries)>2
assert sum(x['rejection_reasons'].get('blur',0) for x in summaries)>3
# Exact same bounded input: no new successful writes/rows should be produced.
before=log.read_bytes()
resume=io.StringIO()
with contextlib.redirect_stdout(resume):
    assert m.run(args)==0
summary=json.loads(resume.getvalue().splitlines()[-1])
assert summary['processed']==0 and summary['skipped']==20
assert log.read_bytes()==before
# Synthetic quality gates, including the strict 512 boundary.
base=np.random.default_rng(42).integers(0,256,(640,960,3),dtype=np.uint8)
assert not m.quality(base,args)[1]
checks={}
for side in [512,513]:
    image=cv2.resize(base,(768,side))
    reasons=m.quality(image,args)[1]
    assert ('resolution' in reasons)==(side==512)
checks['strict_short_side_512_513']=True
examples={
    'blur': cv2.GaussianBlur(base,(0,0),12),
    'dark': np.zeros_like(base),
    'bright': np.full_like(base,255),
    'grayscale': cv2.cvtColor(cv2.cvtColor(base,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR),
    'low_contrast': np.full_like(base,128),
}
for reason,image in examples.items():
    assert reason in m.quality(image,args)[1], reason
    checks[reason]=True
# Identical high-quality frames are still treated as duplicates.
assert (m.phash(base)^m.phash(base.copy())).bit_count()==0
checks['identical_phash']=True
# Entirely rejected videos must be completed, rather than retried forever.
synthetic=work/'reject.avi'
writer=cv2.VideoWriter(str(synthetic),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480))
assert writer.isOpened()
for _ in range(30):
    writer.write(np.zeros((480,640,3),np.uint8))
writer.release()
manifest=work/'reject.jsonl'
manifest.write_text(json.dumps({'clip_id':'reject','video_uri':str(synthetic),'source':'synthetic'})+'\n')
args.source=str(manifest); args.dry_run=True
args.work_dir=str(work/'reject_checkpoints')
capture=io.StringIO()
with contextlib.redirect_stdout(capture):
    assert m.run(args)==0
rejection=json.loads(capture.getvalue().splitlines()[-1])
assert rejection['processed']==1 and rejection['kept']==0 and rejection['rejected_frames']==2
checks['all_rejected_video_completed']=True
report={'videos':20,'verified_images':verified,
        'min_blur_score':200,'phash_distance':14,'previous_blurry_frame_removed':True,
        'candidates':sum(x['candidates'] for x in summaries),
        'quality_rejected':sum(x['rejected_frames'] for x in summaries),
        'dedup_removed':sum(x['duplicates'] for x in summaries),
        'rejection_reasons':dict(__import__('functools').reduce(lambda a,b:a+__import__('collections').Counter(b['rejection_reasons']),summaries,__import__('collections').Counter())),
        'uploaded_jpeg_bytes':payload_bytes,'processing_seconds':round(elapsed,2),
        'source_jpeg_exact_match':True,'bos_sha256_match':True,'dimensions_match':True,
        'remote_metadata_match':True,'sampling_and_quality_match':True,
        'hierarchy':'source/hash2/hash2/video_id/frame.jpg',
        'resume_skipped':summary['skipped'],'quality_tests':checks,
        'full_dataset_started':False}
(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
