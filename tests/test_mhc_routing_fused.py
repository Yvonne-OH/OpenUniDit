"""Inductor routing checks, with CUDA coverage when a GPU is available."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch._dynamo.backends.registry import lookup_backend
from models import mhc_routing_fused as routing


class FusedRoutingTests(unittest.TestCase):
    def test_inductor_forward_and_backward(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        torch.set_num_threads(1)
        compile_fn = torch.compile
        inductor = lookup_backend("inductor")
        compiled_names = set()

        def tracked_compile(fn, **kwargs):
            def backend(graph, inputs):
                compiled_names.add(fn.__name__)
                return inductor(graph, inputs)
            return compile_fn(fn, backend=backend, **kwargs)

        routing.compiled_routing.cache_clear()
        self.addCleanup(routing.compiled_routing.cache_clear)
        with patch.object(torch, "compile", tracked_compile):
            compiled = routing.compiled_routing()
        for dtype in (torch.float32, torch.bfloat16):
            for batch, tokens in ((2, 7), (2, 11), (1, 5)):
                def randn(*shape):
                    return torch.randn(*shape, device=device, dtype=dtype)
                h = randn(batch, tokens, 4, 16)
                cases = (
                    (routing._mapping_logits, compiled[0],
                     [h, randn(24, 64) * .1, randn(24) * .1, randn(3)], (4, 1e-6)),
                    (routing._pre_mix, compiled[1],
                     [h, randn(batch, tokens, 4).float()], ()),
                    (routing._post_mix, compiled[2],
                     [h, randn(batch, tokens, 16), randn(batch, tokens, 4).float(),
                      randn(batch, tokens, 4, 4).float()], ()),
                )
                for eager, fused, values, extras in cases:
                    with self.subTest(function=eager.__name__, dtype=dtype,
                                      batch=batch, tokens=tokens, device=device):
                        expected = self.evaluate(eager, values, extras)
                        actual = self.evaluate(fused, values, extras)
                        rtol, atol = ((.03, .002) if dtype == torch.bfloat16
                                      else (2e-4, 2e-6))
                        for reference, result in zip(expected, actual):
                            torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
        self.assertEqual(compiled_names, {"_mapping_logits", "_pre_mix", "_post_mix"})
        print(f"Inductor forward/backward passed on {device}: {sorted(compiled_names)}", flush=True)

    @staticmethod
    def evaluate(fn, values, extras):
        inputs = [value.clone().requires_grad_() for value in values]
        outputs = fn(*inputs, *extras)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        loss = sum(output.float().square().mean() for output in outputs)
        gradients = torch.autograd.grad(loss, inputs)
        return (*outputs, *gradients)

    def test_runtime_errors_propagate(self):
        # A failed fused call must not permanently route subsequent calls to eager.
        for error in (torch.OutOfMemoryError("test OOM"), RuntimeError("test runtime error")):
            with self.subTest(error=type(error).__name__):
                failure = unittest.mock.Mock(side_effect=error)
                with patch.object(routing, "compiled_routing", return_value=(failure,) * 3):
                    for fn in (routing.mapping_logits, routing.pre_mix, routing.post_mix):
                        with self.assertRaises(type(error)) as caught:
                            fn()
                        self.assertIs(caught.exception, error)
                self.assertEqual(failure.call_count, 3)


if __name__ == "__main__":
    unittest.main()
