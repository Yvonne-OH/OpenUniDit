"""patchify 变更续训兼容性回归: python -m unittest scripts.test.test_patchify_resume -v

覆盖 load_compatible_state_dict 的 patchify 兼容语义:
- (1,2,2) ckpt → (1,1,1) 模型: patch_in.weight / out.weight / out.bias shape 不匹配,
  保持重新随机初始化 (红字警告); 其余参数全部正常继承;
- 无 semantic metadata 的旧格式 ckpt: 由 patch_in 输入维反推 patch volume, 仍触发警告;
- patch_size 一致: 全量继承, 不触发 patchify 警告 (回归);
- 模型 patchify (1,2,2) + 旧未 patchify ckpt: 复制扩展迁移, 不触发警告 (回归)。
"""
import io
import unittest
from contextlib import redirect_stdout

import torch

from models.giga_transformer import GigaExpTransformer
from scripts.train.train_giga_exp import load_compatible_state_dict

PATCHIFY_KEYS = ("patch_in.weight", "out.weight", "out.bias")


class PatchifyResumeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def model(self, patch_size):
        return GigaExpTransformer(in_channels=4, out_channels=4, text_dim=12, dim=32,
                                  heads=2, ffn_dim=64, layers=2,
                                  rope_axes_dims=(4, 4, 4, 4), patch_size=patch_size)

    def load(self, model, state, metadata=None):
        captured = io.StringIO()
        with redirect_stdout(captured):
            matched, missing, unexpected = load_compatible_state_dict(model, state, metadata)
        return matched, missing, unexpected, captured.getvalue()

    def test_ckpt_patchify_to_unpatchified_reinit_and_warn(self):
        # (1,2,2) ckpt → (1,1,1) 模型: patchify 层重新初始化, 其余全量继承
        src = self.model((1, 2, 2))
        with torch.no_grad():
            for p in src.parameters():
                p.fill_(1.0)
        state = {k: v.clone() for k, v in src.state_dict().items()}
        metadata = src.semantic_metadata()
        self.assertEqual(metadata["patch_size"], [1, 2, 2])

        dst = self.model((1, 1, 1))
        init_state = {k: v.clone() for k, v in dst.state_dict().items()}
        matched, missing, unexpected, output = self.load(dst, state, metadata)

        self.assertEqual(unexpected, 3)                       # 3 个 patchify 参数无法继承
        self.assertEqual(matched, len(state) - 3)
        # strict=False 下 3 个未继承的 patchify 参数计入 missing_keys = 保持重新随机初始化
        self.assertEqual(missing, 3)
        for key, value in state.items():
            if key in PATCHIFY_KEYS:
                # patchify 入口/出口保持重新随机初始化, 未被 ckpt 覆盖
                self.assertTrue(torch.equal(dst.state_dict()[key], init_state[key]), key)
            else:
                self.assertTrue(torch.equal(dst.state_dict()[key], value), key)
        # 红字警告: patch_size 对照 + 重新初始化参数清单 + 其余继承说明
        self.assertIn("\033[1;31m", output)
        self.assertIn("patchify", output)
        self.assertIn("[1, 2, 2]", output)
        self.assertIn("[1, 1, 1]", output)
        self.assertIn("patch_in.weight", output)
        self.assertIn("out.weight", output)
        self.assertIn("重新随机初始化", output)
        self.assertIn("全部正常继承", output)

    def test_ckpt_without_metadata_infers_patch_volume_and_warns(self):
        src = self.model((1, 2, 2))
        state = {k: v.clone() for k, v in src.state_dict().items()}
        dst = self.model((1, 1, 1))
        init_state = {k: v.clone() for k, v in dst.state_dict().items()}
        matched, missing, unexpected, output = self.load(dst, state, metadata=None)
        self.assertEqual((matched, missing, unexpected), (len(state) - 3, 3, 3))
        for key in PATCHIFY_KEYS:
            self.assertTrue(torch.equal(dst.state_dict()[key], init_state[key]), key)
        self.assertIn("\033[1;31m", output)
        self.assertIn("patch volume=4", output)
        self.assertIn("重新随机初始化", output)

    def test_same_patch_size_full_inherit_no_warning(self):
        src = self.model((1, 2, 2))
        state = {k: v.clone() for k, v in src.state_dict().items()}
        dst = self.model((1, 2, 2))
        matched, missing, unexpected, output = self.load(dst, state, src.semantic_metadata())
        self.assertEqual((unexpected, missing), (0, 0))
        self.assertEqual(matched, len(state))
        self.assertNotIn("patchify 不兼容", output)
        for key, value in state.items():
            self.assertTrue(torch.equal(dst.state_dict()[key], value), key)

    def test_unpatchified_ckpt_to_patchified_model_migrates_no_warning(self):
        # 旧路径回归: (1,1,1) ckpt → (1,2,2) 模型, patch_in/out 复制扩展迁移
        src = self.model((1, 1, 1))
        state = {k: v.clone() for k, v in src.state_dict().items()}
        dst = self.model((1, 2, 2))
        matched, missing, unexpected, output = self.load(dst, state, src.semantic_metadata())
        self.assertEqual((unexpected, missing), (0, 0))
        self.assertEqual(matched, len(state))
        self.assertNotIn("patchify 不兼容", output)
        torch.testing.assert_close(
            dst.state_dict()["patch_in.weight"],
            state["patch_in.weight"].repeat(1, 4) / 4, rtol=0, atol=0)
        torch.testing.assert_close(
            dst.state_dict()["out.weight"],
            state["out.weight"].repeat(4, 1), rtol=0, atol=0)
        torch.testing.assert_close(
            dst.state_dict()["out.bias"], state["out.bias"].repeat(4), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
