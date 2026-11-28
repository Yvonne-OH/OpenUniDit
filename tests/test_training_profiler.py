import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizations.training_profiler import summarize_trace, duration, overlap

class TraceTests(unittest.TestCase):
    def test_overlapping_kernels_are_not_double_counted(self):
        events=[]
        for device,name,start,length in [(0,'nccl_allreduce',0,10000),(0,'nccl_reduce',5000,10000),(0,'gemm',7000,5000),(1,'gemm',0,10000)]:
            events.append(dict(ph='X',cat='kernel',name=name,ts=start,dur=length,args={'device':device}))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.json';p.write_text(json.dumps({'traceEvents':events}))
            s=summarize_trace(p)
        self.assertEqual(s['devices']['0']['nccl_union_ms'],15)
        self.assertEqual(s['devices']['0']['overlap_ms'],5)
        self.assertEqual(s['devices']['0']['nccl_without_compute_ms'],10)
        self.assertEqual(s['devices']['1']['nccl_union_ms'],0)
        self.assertIn('NOT measured critical-path wait',s['interpretation'])
    def test_empty_and_adjacent(self):
        self.assertEqual(duration([]),0)
        self.assertEqual(duration([(0,5),(5,10)]),10)
        self.assertEqual(overlap([(0,5)],[(5,10)]),0)
if __name__=='__main__':unittest.main()
