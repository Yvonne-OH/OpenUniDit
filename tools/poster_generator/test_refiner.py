# -*- coding: utf-8 -*-
"""单独测试 Qwen3-VL 润色模块: CUDA_VISIBLE_DEVICES=7 python test_refiner.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_opencole import load_opencole
from prompt_refiner import build_user_message, generate_layouts, load_vlm

recs = load_opencole("test", 1)
print(build_user_message(recs[0])[:400])
processor, model = load_vlm()
layouts, reply = generate_layouts(processor, model, recs[0], 2)
import json
print(json.dumps(layouts, ensure_ascii=False, indent=2))
