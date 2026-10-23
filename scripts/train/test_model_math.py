"""CPU 数学回归：python -m unittest scripts.train.test_model_math -v。"""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from models.giga_transformer import GigaExpTransformer, SpacetimeRoPE, rotate_half
import scripts.train.train_giga_exp as train_exp


class ModelMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_rope_precision_conversion(self):
        r = SpacetimeRoPE(128)
        original = r.inv_freq.clone()
        reference = r(2, 6, 8, 512, 'cpu', torch.float32)
        for dtype in (torch.bfloat16, torch.float16, torch.float64, torch.float32):
            r.to(dtype=dtype)
            self.assertEqual(r.inv_freq.dtype, torch.float32)
            torch.testing.assert_close(r.inv_freq, original, rtol=0, atol=0)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = r(2, 6, 8, 512, 'cpu', dtype)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertNotIn('inv_freq', r.state_dict())

    def test_rope_independent_double_reference(self):
        r = SpacetimeRoPE(128).bfloat16()
        cos, sin = r(2, 3, 4, 512, 'cpu', torch.bfloat16)
        positions = [(t, h, w, 0) for t in range(2) for h in range(3) for w in range(4)]
        positions += [(0, 0, 0, j + 1) for j in range(512)]
        phase = torch.tensor([[p[a] * math.pow(10000., -2 * j / d)
                               for a, d in enumerate(r.axes_dims) for j in range(d // 2)]
                              for p in positions], dtype=torch.float64)
        phase = torch.cat([phase, phase], -1)
        torch.testing.assert_close(cos.double(), phase.cos(), atol=2e-5, rtol=0)
        torch.testing.assert_close(sin.double(), phase.sin(), atol=2e-5, rtol=0)
        x, y = torch.randn(128), torch.randn(128)
        p, q = 25, 500
        rx = x * cos[p] + rotate_half(x) * sin[p]
        ry = y * cos[q] + rotate_half(y) * sin[q]
        torch.testing.assert_close(rx.square().sum(), x.square().sum(), atol=2e-5, rtol=1e-6)
        delta = phase[q] - phase[p]
        relative = y.double() * delta.cos() + rotate_half(y.double()) * delta.sin()
        torch.testing.assert_close((rx * ry).sum().double(), (x.double() * relative).sum(), atol=1e-4, rtol=0)

    def test_patch_inverse_with_index_oracle(self):
        # 关闭非排列操作，直接检验模型中的 patchify/unpatchify 互为逆映射。
        m = GigaExpTransformer(in_channels=2, out_channels=2, text_dim=8, dim=8,
                              heads=1, layers=0, rope_axes_dims=(2,2,2,2), patch_size=(1,2,2))
        m.patch_in = nn.Identity()
        m.norm = nn.Identity()
        m.out = nn.Identity()
        x = torch.arange(2*2*3*6*8).reshape(2,2,3,6,8).float()
        captured = []
        handle = m.patch_in.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        y = m(x, torch.zeros(2), torch.zeros(2,1,8))
        handle.remove()
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        oracle = torch.stack([torch.stack([torch.stack([x[b,c,t,2*h+dh,2*w+dw]
                    for dh in range(2) for dw in range(2) for c in range(2)])
                    for t in range(3) for h in range(3) for w in range(4)]) for b in range(2)])
        torch.testing.assert_close(captured[0], oracle, rtol=0, atol=0)

    def model(self):
        return GigaExpTransformer(in_channels=4, out_channels=4, text_dim=12, dim=32,
                                 heads=2, ffn_dim=64, layers=2,
                                 rope_axes_dims=(4,4,4,4), patch_size=(1,2,2))

    def test_qk_norm_is_per_head_and_active(self):
        m = self.model()
        self.assertEqual(tuple(m.blocks[0].q_norm.weight.shape), (16,))
        self.assertEqual(tuple(m.blocks[0].k_norm.weight.shape), (16,))
        with torch.no_grad():
            m.blocks[0].q_norm.weight.fill_(2.0)
            m.blocks[0].k_norm.weight.fill_(3.0)
        x = torch.randn(1, 4, 1, 6, 8)
        text = torch.randn(1, 3, 12)
        captured = {}
        m.blocks[0].q_norm.register_forward_hook(lambda _, __, output: captured.setdefault("q", output.detach()))
        m.blocks[0].k_norm.register_forward_hook(lambda _, __, output: captured.setdefault("k", output.detach()))
        m(x, torch.tensor([100.]), text, torch.ones(1, 3, dtype=torch.bool))
        self.assertTrue(torch.isfinite(captured["q"]).all())
        self.assertTrue(torch.isfinite(captured["k"]).all())
        q_rms = captured["q"].square().mean(-1).sqrt()
        k_rms = captured["k"].square().mean(-1).sqrt()
        torch.testing.assert_close(q_rms, torch.full_like(q_rms, 2.0), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(k_rms, torch.full_like(k_rms, 3.0), atol=2e-5, rtol=2e-5)

    def test_mask_gradient_and_batch_invariance(self):
        m = self.model()
        x = torch.randn(2,4,1,6,8, requires_grad=True)
        text = torch.randn(2,5,12, requires_grad=True)
        mask = torch.tensor([[1,1,0,0,0],[1,1,1,1,1]], dtype=torch.bool)
        ts = torch.tensor([100.,900.])
        y = m(x,ts,text,mask)
        single = m(x[:1],ts[:1],text[:1,:2],torch.ones(1,2,dtype=torch.bool))
        torch.testing.assert_close(y[:1],single,atol=2e-6,rtol=2e-5)
        altered = text.detach().clone()
        altered[~mask] = 1000
        torch.testing.assert_close(y,m(x,ts,altered,mask),rtol=0,atol=0)
        y.square().mean().backward()
        torch.testing.assert_close(text.grad[~mask],torch.zeros_like(text.grad[~mask]),rtol=0,atol=0)
        self.assertGreater(text.grad[mask].abs().sum().item(),0)
        self.assertGreater(x.grad.abs().sum().item(),0)
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        m.bfloat16()
        out = m(x.detach().bfloat16(),ts,text.detach().bfloat16(),mask)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(m.rope.inv_freq.dtype,torch.float32)
        clone = self.model().bfloat16()
        clone.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(out,clone(x.detach().bfloat16(),ts,text.detach().bfloat16(),mask),rtol=0,atol=0)

    def test_sdpa_against_softmax_definition(self):
        q,k,v = [torch.randn(2,2,7,8,dtype=torch.float64) for _ in range(3)]
        mask = torch.tensor([1,1,1,1,0,0,0],dtype=torch.bool)[None,None,None,:]
        logits = (q @ k.transpose(-1,-2)) / math.sqrt(8)
        expected = logits.masked_fill(~mask,-torch.inf).softmax(-1) @ v
        actual = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_semantic_metadata_warning_and_roundtrip(self):
        model = self.model()
        metadata = model.semantic_metadata()
        self.assertEqual(metadata["heads"], 2)
        with self.assertNoLogs("giga_exp", level="WARNING"), mock.patch("builtins.print") as console:
            train_exp._warn_semantic_mismatch(model, metadata, "same")
        console.assert_not_called()
        altered = dict(metadata, heads=4, rope_theta=123.0)
        with self.assertLogs("giga_exp", level="WARNING") as logs:
            train_exp._warn_semantic_mismatch(model, altered, "different")
        self.assertIn("heads", logs.output[0])
        self.assertIn("rope_theta", logs.output[0])
        with self.assertLogs("giga_exp", level="WARNING") as logs:
            train_exp._warn_semantic_mismatch(model, None, "legacy")
        self.assertIn("missing/unknown", logs.output[0])

        class Engine:
            module = model
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            train_exp.save_pure_weights(Engine(), output, 3)
            weight_dir = output / "checkpoints" / "weights"
            for path in (weight_dir / "step-00000003.safetensors", weight_dir / "step-00000003.pt"):
                if path.exists():
                    loaded = train_exp._read_weight_metadata(path)
                    self.assertEqual(loaded, metadata)
            self.assertEqual(train_exp._read_weight_metadata(weight_dir / "latest.pt"), metadata)

    def test_deepspeed_client_state_metadata_check(self):
        model = self.model()
        engine = mock.Mock(module=model)
        for metadata in (model.semantic_metadata(), {"heads": 99}, None):
            client_state = {"step": 1, "giga_semantic_metadata": metadata}
            engine.load_checkpoint.return_value = ("checkpoint/step-1", client_state)
            with mock.patch.object(train_exp, "_warn_semantic_mismatch") as check:
                result = train_exp.load_training_checkpoint(engine, Path("checkpoint"), "step-1")
            self.assertEqual(result[1], client_state)
            check.assert_called_once_with(model, metadata, "DeepSpeed step-1")
        engine.load_checkpoint.assert_called_with("checkpoint", tag="step-1",
                                                   load_optimizer_states=True, load_lr_scheduler_states=True)

    def test_same_shape_semantics_and_load_paths(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            engine = mock.Mock(module=model)
            train_exp.save_pure_weights(engine, output, 3)
            path = output / "checkpoints/weights/latest.pt"
            for changes in ({"rope_theta": 123.0}, {"heads": 4, "rope_axes_dims": (2,2,2,2)}):
                config = dict(in_channels=4, out_channels=4, text_dim=12, dim=32,
                              heads=2, ffn_dim=64, layers=2, rope_axes_dims=(4,4,4,4), patch_size=(1,2,2))
                config.update(changes)
                target = GigaExpTransformer(**config)
                if changes == {"rope_theta": 123.0}:
                    self.assertEqual({k: v.shape for k,v in model.state_dict().items()},
                                     {k: v.shape for k,v in target.state_dict().items()})
                with self.assertLogs("giga_exp", level="WARNING") as logs, mock.patch("builtins.print") as console:
                    train_exp.load_init_weights(target, path)
                self.assertIn("old=", logs.output[0])
                self.assertIn("current=", logs.output[0])
                self.assertNotIn("\033", logs.output[0])
                self.assertIn("\033[1;31m", console.call_args.args[0])
            with self.assertNoLogs("giga_exp", level="WARNING"):
                train_exp.load_pure_weights(engine, output, str(path))
            legacy = output / "legacy.pt"
            torch.save(model.state_dict(), legacy)
            with self.assertLogs("giga_exp", level="WARNING") as logs:
                train_exp.load_init_weights(model, legacy)
            self.assertIn("unknown", logs.output[0])

    def test_pure_weights_refresh_zero_master(self):
        from types import SimpleNamespace
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

        # 使用真实 ZeRO 刷新方法，CPU 模拟两个 rank 的本地分片，无需启动进程组。
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = self.model().bfloat16()
            train_exp.save_pure_weights(SimpleNamespace(module=source), output, 7)
            weight_dir = output / "checkpoints/weights"
            paths = [weight_dir / "latest.pt"] + list(weight_dir.glob("step-*.safetensors"))
            for rank in (0, 1):
                for path in paths:
                    target = self.model().bfloat16()
                    parameter = target.patch_in.weight
                    parts = list(parameter.detach().view(-1).chunk(2))
                    master = nn.Parameter(torch.full_like(parts[rank], -10, dtype=torch.float32))
                    zero = object.__new__(DeepSpeedZeroOptimizer)
                    zero.parallel_partitioned_bit16_groups = [parts]
                    zero.single_partition_of_fp32_groups = [master]
                    zero.real_dp_process_group = [None]
                    engine = SimpleNamespace(module=target, optimizer=zero)
                    with mock.patch.object(train_exp, "_dist_rank", return_value=rank), \
                         mock.patch("deepspeed.runtime.zero.stage_1_and_2.dist.get_rank", return_value=rank):
                        with mock.patch.object(zero, "refresh_fp32_params", wraps=zero.refresh_fp32_params) as refresh:
                            train_exp.load_pure_weights(engine, output, str(path))
                            refresh.assert_called_once_with()
                        expected = source.patch_in.weight.detach().view(-1).chunk(2)[rank].float()
                        torch.testing.assert_close(master, expected, rtol=0, atol=0)
                        # g=1 时首次 SGD 更新必须为 W_checkpoint-lr，而不是 W_old-lr。
                        optimizer = torch.optim.SGD([master], lr=0.125)
                        master.grad = torch.ones_like(master)
                        optimizer.step()
                        torch.testing.assert_close(master, expected - 0.125, rtol=0, atol=0)
                        with torch.no_grad():
                            parts[rank].copy_(master)
                        torch.testing.assert_close(parts[rank], (expected - 0.125).bfloat16(), rtol=0, atol=0)
                        # auto 恢复也经过相同同步路径。
                        with mock.patch.object(zero, "refresh_fp32_params", wraps=zero.refresh_fp32_params) as refresh:
                            self.assertEqual(train_exp.load_pure_weights(engine, output), 7)
                            refresh.assert_called_once_with()
                            torch.testing.assert_close(master, expected, rtol=0, atol=0)
            empty_engine = mock.Mock()
            self.assertEqual(train_exp.load_pure_weights(empty_engine, output / "empty"), 0)
            empty_engine.optimizer.refresh_fp32_params.assert_not_called()

    def test_flow_target_shift_and_euler(self):
        u = torch.linspace(0,1,101,dtype=torch.float64)
        s = 3.
        sigma = s*u/(1+(s-1)*u)
        recovered = sigma/(s-(s-1)*sigma)
        torch.testing.assert_close(recovered,u,atol=1e-14,rtol=0)
        # p_sigma(f(u))*f'(u)=1，是变量替换密度的独立恒等式。
        density = s/(s-(s-1)*sigma).square()
        derivative = s/(1+(s-1)*u).square()
        torch.testing.assert_close(density*derivative,torch.ones_like(u),atol=1e-14,rtol=0)
        z,eps = torch.randn(4,dtype=torch.float64),torch.randn(4,dtype=torch.float64)
        velocity = eps-z
        x = eps.clone()
        for a,b in zip(sigma.flip(0)[:-1],sigma.flip(0)[1:]):
            x = x+(b-a)*velocity
        torch.testing.assert_close(x,z,atol=1e-14,rtol=0)


if __name__ == '__main__':
    unittest.main()
