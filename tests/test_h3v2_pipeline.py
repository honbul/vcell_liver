import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from h3v2 import pipeline
from h3v2.final_fit import final_output


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
                assert cpu.kwargs["array"] == "2%16" and cpu.kwargs["node"] == "g4"
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


def test_cpu_training_arrays_share_the_expanded_budget():
    indices = range(51)
    with patch.object(pipeline, "submit", side_effect=map(str, range(37101, 37106))) as submit:
        jobs = pipeline.submit_training_arrays("boundary-gbdt", Path("inner.sbatch"), [], "gbdt", indices)
    assert jobs == list(map(str, range(37101, 37106)))
    planned = set()
    training_cores = {}
    for call in submit.call_args_list:
        ids, concurrency = call.kwargs["array"].split("%")
        batch = set(map(int, ids.split(",")))
        assert not (planned & batch)
        assert not call.kwargs.get("gpu", False)
        planned.update(batch)
        node = call.kwargs["node"]
        training_cores[node] = training_cores.get(node, 0) + call.kwargs["threads"] * int(concurrency)
    assert planned == set(indices)
    assert training_cores == {"g4": 128, "g3": 48, "c4": 48, "c5": 30}


def test_completed_early_gpu_arrays_are_not_resubmitted():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        present = root / "complete"
        present.touch()
        tasks = {"gbdt": {0: [root / "cpu"]}, "torch": {3: [present], 4: [present, root / "gpu"]}}
        with patch.object(pipeline, "ROOT", root), \
                patch.object(pipeline, "inner_tasks", return_value=tasks), \
                patch.object(pipeline, "submit", side_effect=["37101", "37102", "37103"]) as submit:
            assert pipeline.schedule_inner(["r1", "r2", "boundary-1", "boundary-2"], root) == "37103"
            cpu, gpu, selector = submit.call_args_list
            assert cpu.kwargs["array"] == "0%16" and gpu.kwargs["array"] == "4%4"
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


def test_g4_cpu_fit_requests_no_gpu():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        (root / "logs").mkdir()
        with patch.object(pipeline, "ROOT", root), \
                patch.object(pipeline, "check_output", return_value="37104\n") as sbatch:
            pipeline.submit_training_arrays("final-gbdt", Path("job.sbatch"), [], "gbdt", range(1))
        command = sbatch.call_args.args[0]
        assert command[command.index("-p") + 1] == "gpu" and "--nodelist=g4" in command
        assert all(not argument.startswith("--gres=") for argument in command)
        assert (root / "logs/trajectory.tsv").read_text().split("\t")[:3] == ["final-gbdt-g4-8core", "37104", "cpu"]


def test_next_cpu_round_advances_while_registered_gpu_fits_continue():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        tasks = {"gbdt": {0: [root / "cpu"]}, "torch": {4: [root / "early"], 7: [root / "hybrid"]}}
        stages = ["r1", "r2", "boundary-1", "boundary-2"]
        with patch.object(pipeline, "inner_tasks", return_value=tasks), \
                patch.object(pipeline, "controller_dependency", return_value=("37100",)), \
                patch.object(pipeline, "active_early_array", side_effect=[("37104", {4}), None]), \
                patch.object(pipeline, "submit", side_effect=["37101", "37102", "37103"]) as submit:
            assert pipeline.schedule_inner(stages, root) == "37103"
            cpu, gpu, watcher = submit.call_args_list
            assert cpu.kwargs["deps"] == ("37100",) and not cpu.kwargs["terminal_deps"]
            assert gpu.kwargs["array"] == "7%4" and gpu.kwargs["terminal_deps"] == ("37104",)
            assert watcher.kwargs["deps"] == ("37104", "37101", "37102") and watcher.kwargs["dependency"] == "afterany"
        (root / "logs").mkdir()
        with patch.object(pipeline, "ROOT", root), \
                patch.object(pipeline, "check_output", return_value="37102\n") as sbatch:
            pipeline.submit("torch", Path("job.sbatch"), [], gpu=True, deps=("37100",), terminal_deps=("37104",))
        assert "--dependency=afterok:37100,afterany:37104" in sbatch.call_args.args[0]


def test_early_primary_tree_job_owns_its_indices_and_only_delays_g3():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        tasks = {"gbdt": {i: [root / f"cpu-{i}"] for i in range(4)}, "torch": {}}
        stages = ["r1", "r2", "boundary-1", "boundary-2"]
        with patch.object(pipeline, "ROOT", root), \
                patch.object(pipeline, "inner_tasks", return_value=tasks), \
                patch.object(pipeline, "controller_dependency", return_value=("37100",)), \
                patch.object(pipeline, "active_early_tree_array", side_effect=[("37104", {0}), None]), \
                patch.object(pipeline, "submit", side_effect=["37101", "37102", "37103", "37105"]) as submit:
            assert pipeline.schedule_inner(stages, root) == "37105"
            g4, g3, c4, controller = (call.kwargs for call in submit.call_args_list)
            assert (g4["node"], g4["array"], g4["terminal_deps"]) == ("g4", "1%16", ())
            assert (g3["node"], g3["array"], g3["terminal_deps"]) == ("g3", "2%6", ("37104",))
            assert (c4["node"], c4["array"], c4["terminal_deps"]) == ("c4", "3%6", ())
            assert controller["deps"] == ("37104", "37101", "37102", "37103")

        with patch.object(pipeline, "ROOT", root), \
                patch.object(pipeline, "controller_dependency", return_value=("37100",)), \
                patch.object(pipeline, "active_early_tree_array", side_effect=[None, ("37104", {0})]), \
                patch.object(pipeline, "submit", side_effect=["37101", "37102", "37103", "37105"]) as submit:
            retry = {"gbdt": {i: [root / f"cpu-{i}"] for i in range(3)}, "torch": {}}
            assert pipeline.schedule_inner(stages[:-1], root, retry=True, tasks=retry) == "37105"
            g4, g3, c4, controller = (call.kwargs for call in submit.call_args_list)
            assert g4["array"] == "0%16" and not g4["terminal_deps"]
            assert g3["array"] == "1%6" and g3["terminal_deps"] == ("37104",)
            assert c4["array"] == "2%6" and not c4["terminal_deps"]
            assert controller["deps"] == ("37101", "37102", "37103")


def test_failed_final_jobs_are_retried_before_the_only_score_submission():
    def touch(root, relative):
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}")

    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        root = Path(scratch)
        paths = ["final/drug-0/gbdt/11711/receipt.json", "final/drug-0/gbdt/11729/receipt.json",
                 "final/external/film/11711/receipt.json", "final/external/references/receipt.json"]
        for relative in (paths[0], "features/external/receipt.json", "final/drug-0/references/receipt.json", "locks/frozen.json"):
            touch(root, relative)
        job_lock = {"gbdt": [{"split": "drug-0", "family": "gbdt", "seed": seed} for seed in (11711, 11729)],
                    "torch": [{"split": "external", "family": "film", "seed": 11711}]}
        (root / "locks/final-jobs.json").write_text(json.dumps(job_lock))
        with patch.multiple(pipeline, ROOT=root, FINAL_PARTITIONS=("drug-0", "external")), \
                patch.object(pipeline, "controller_dependency", return_value=("37100",)), \
                patch.object(pipeline, "submit", side_effect=map(str, range(37101, 37120))) as submit:
            assert pipeline.settle_final(root, retried=False) == "37104"
            cpu, gpu, refs, watcher = submit.call_args_list
            assert cpu.kwargs["array"] == "1%16" and gpu.kwargs["array"] == "0%4"
            assert refs.kwargs["array"] == "1%3" and refs.kwargs["deps"] == ("37101",)
            assert watcher.kwargs["dependency"] == "afterany" and "--retried" in watcher.args[2]
            assert not (root / "locks/score-dispatch.json").exists()
            try:
                pipeline.settle_final(root, retried=True)
            except RuntimeError as exc:
                assert "after the registered retry" in str(exc)
            else:
                raise AssertionError("Missing final artifacts reached scoring")
            for relative in paths[1:]:
                touch(root, relative)
            assert pipeline.settle_final(root, retried=True) == "37107"
            before, score, report = submit.call_args_list[4:]
            assert before.args[0] == "preflight" and score.kwargs["deps"] == ("37105",)
            assert score.args[0] == "score-once" and report.kwargs["deps"] == ("37106",)
            try:
                pipeline.settle_final(root, retried=True)
            except FileExistsError:
                pass
            else:
                raise AssertionError("A duplicate score submission was accepted")
            assert submit.call_count == 7


def test_final_output_preserves_interrupted_and_completed_artifacts():
    with TemporaryDirectory(dir=pipeline.ROOT / "tmp") as scratch:
        target = Path(scratch) / "final" / "fit"
        target.mkdir(parents=True)
        (target / "legacy").touch()
        try:
            with final_output(target) as stage:
                stage.mkdir()
                (stage / "checkpoint").touch()
        except RuntimeError:
            pass
        else:
            raise AssertionError("An artifact without a receipt was published")
        assert (target / "legacy").is_file()
        with final_output(target) as stage:
            stage.mkdir()
            (stage / "receipt.json").write_text("complete")
        assert (target / "receipt.json").read_text() == "complete"
        assert len(list(target.parent.glob(".fit-interrupted-*/fit/legacy"))) == 1
        try:
            with final_output(target):
                raise AssertionError("A completed artifact was writable")
        except FileExistsError:
            pass


if __name__ == "__main__":
    test_only_array_elements_missing_a_registered_arm_are_retried()
    test_cpu_training_arrays_share_the_expanded_budget()
    test_completed_early_gpu_arrays_are_not_resubmitted()
    test_followup_gpu_arrays_use_only_g3_a40()
    test_g4_cpu_fit_requests_no_gpu()
    test_next_cpu_round_advances_while_registered_gpu_fits_continue()
    test_early_primary_tree_job_owns_its_indices_and_only_delays_g3()
    test_failed_final_jobs_are_retried_before_the_only_score_submission()
    test_final_output_preserves_interrupted_and_completed_artifacts()
    print("Registered 990 arms and verified bounded recovery, dependencies and the training caps", flush=True)
