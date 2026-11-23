from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from h3v2 import pipeline


def test_only_array_elements_missing_a_registered_arm_are_retried():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        with patch.object(pipeline, "ROOT", root):
            registered = pipeline.inner_tasks("r1")
            assert [len(registered[backend]) for backend in ("gbdt", "torch")] == [90, 180]
            assert sum(len(receipts) for tasks in registered.values() for receipts in tasks.values()) == 990
            assert {file.parts[-5] for file in registered["torch"][0]} == {"drug-0"}
            assert any("/drug-1/inner-0/film/" in str(file) for file in registered["torch"][12])

            present = root / "complete"
            missing_a = root / "cpu-incomplete"
            missing_b = root / "gpu-incomplete"
            present.touch()
            tasks = {"gbdt": {0: [present], 2: [present, missing_a]}, "torch": {3: [missing_b]}}
            with patch.object(pipeline, "inner_tasks", return_value=tasks), \
                    patch.object(pipeline, "submit", side_effect=["37101", "37102", "37103"]) as submit, \
                    patch.object(pipeline, "advance") as advance:
                pipeline.settle_inner(["r1"], root, retried=False)
                advance.assert_not_called()
                cpu, gpu, controller = submit.call_args_list
                assert cpu.kwargs["array"] == "2%5" and not cpu.kwargs["gpu"]
                assert gpu.kwargs["array"] == "3%3" and gpu.kwargs["gpu"]
                assert controller.kwargs["deps"] == ("37101", "37102")
                assert controller.kwargs["dependency"] == "afterany" and "--retried" in controller.args[2]

                try:
                    pipeline.settle_inner(["r1"], root, retried=True)
                except RuntimeError as exc:
                    assert "still has incomplete trials" in str(exc)
                else:
                    raise AssertionError("An unfinished retry was accepted")
                assert submit.call_count == 3

                missing_a.touch()
                missing_b.touch()
                pipeline.settle_inner(["r1"], root, retried=True)
                advance.assert_called_once_with(["r1"], root, dispatch_only=False)
                assert submit.call_count == 3


if __name__ == "__main__":
    test_only_array_elements_missing_a_registered_arm_are_retried()
    print("Registered 990 arms and verified bounded recovery, dependencies and the three-GPU cap", flush=True)
