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
                assert gpu.kwargs["array"] == "3%4" and gpu.kwargs["gpu"]
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


def test_cpu_training_arrays_share_the_c5_budget():
    indices = range(13)
    with patch.object(pipeline, "submit", side_effect=["37101", "37102"]) as submit:
        jobs = pipeline.submit_training_arrays("boundary-gbdt", Path("inner.sbatch"), [], "gbdt", indices)
    assert jobs == ["37101", "37102"]
    planned = set()
    training_cores = 0
    for call, threads in zip(submit.call_args_list, (8, 6), strict=True):
        ids, concurrency = call.kwargs["array"].split("%")
        batch = set(map(int, ids.split(",")))
        assert not (planned & batch)
        assert call.kwargs["threads"] == threads and not call.kwargs["gpu"]
        planned.update(batch)
        training_cores += threads * int(concurrency)
    assert planned == set(indices) and training_cores + 2 == 48


def test_completed_early_gpu_arrays_are_not_resubmitted():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        present = root / "complete"
        present.touch()
        tasks = {"gbdt": {0: [root / "cpu"]}, "torch": {3: [present], 4: [present, root / "gpu"]}}
        with patch.object(pipeline, "inner_tasks", return_value=tasks), \
                patch.object(pipeline, "submit", side_effect=["37101", "37102", "37103"]) as submit:
            assert pipeline.schedule_inner(["r1", "r2", "boundary-1", "boundary-2"], root) == "37103"
            cpu, gpu, selector = submit.call_args_list
            assert cpu.kwargs["array"] == "0%5" and gpu.kwargs["array"] == "4%4"
            assert selector.kwargs["deps"] == ("37101", "37102")


def test_followup_gpu_arrays_use_only_g3_a40():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        (root / "logs").mkdir()
        with patch.object(pipeline, "ROOT", root), \
                patch.object(pipeline, "check_output", return_value="37104\n") as sbatch:
            jobs = pipeline.submit_training_arrays("final-torch", Path("job.sbatch"), [], "torch", range(5))
        assert jobs == ["37104"]
        command = sbatch.call_args.args[0]
        assert "--nodelist=g3" in command and "--gres=gpu:a40:1" in command and "--array=0,1,2,3,4%4" in command
        assert (root / "logs/trajectory.tsv").read_text().split("\t")[:3] == ["final-torch", "37104", "a40"]


if __name__ == "__main__":
    test_only_array_elements_missing_a_registered_arm_are_retried()
    test_cpu_training_arrays_share_the_c5_budget()
    test_completed_early_gpu_arrays_are_not_resubmitted()
    test_followup_gpu_arrays_use_only_g3_a40()
    print("Registered 990 arms and verified bounded recovery, dependencies and the training caps", flush=True)
