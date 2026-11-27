"""CPU-only regression tests for the actual stage-1 LR scheduling helper."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/train/train_giga_exp.py"
tree = ast.parse(SOURCE.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "set_lr")
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
set_lr = namespace["set_lr"]


class LearningRateScheduleTests(unittest.TestCase):
    def schedule(self, step, minimum=None):
        cfg = dict(learning_rate=1e-4, lr_warmup_steps=500, total_steps=1000)
        if minimum is not None:
            cfg["min_learning_rate"] = minimum
        wrapper = SimpleNamespace(param_groups=[{}, {"_lr_scale": 3.0}])
        engine = SimpleNamespace(param_groups=[{}, {"_lr_scale": 3.0}])
        lr = set_lr(wrapper, engine, step, cfg)
        for opt in (wrapper, engine):
            self.assertAlmostEqual(opt.param_groups[0]["lr"], lr)
            self.assertAlmostEqual(opt.param_groups[1]["lr"], 3 * lr)
        return lr

    def test_floor_and_warmup_boundaries(self):
        for step, expected in [(0, 0), (250, 5e-5), (500, 1e-4),
                               (750, 7.5e-5), (1000, 5e-5), (1200, 5e-5)]:
            with self.subTest(step=step):
                self.assertAlmostEqual(self.schedule(step, 5e-5), expected)

    def test_legacy_default_still_decays_to_zero(self):
        self.assertAlmostEqual(self.schedule(750), 5e-5)
        self.assertEqual(self.schedule(1000), 0)

    def test_decay_is_monotone_and_bounded(self):
        values = [self.schedule(s, 5e-5) for s in range(500, 1101)]
        self.assertTrue(all(5e-5 <= v <= 1e-4 for v in values))
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_invalid_floor(self):
        for minimum in [-1e-5, 2e-4, float("nan")]:
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                self.schedule(750, minimum)


if __name__ == "__main__":
    unittest.main()
