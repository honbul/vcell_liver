import numpy as np
import torch

from h3_pcgrad.train import Arm, TwoAxisNet, component_losses, predict, project, schedule, task_update, train, vector_grads


def test_masked_losses_have_exact_axis_invariances():
    gen = torch.Generator().manual_seed(124)
    alpha = torch.randn(12, 1, generator=gen)
    delta = torch.randn(12, 7, generator=gen)
    target = torch.randn(12, 7, generator=gen)
    anchor = torch.randn(7, generator=gen)
    i, j = torch.where(torch.rand(12, 7, generator=gen) > .12)
    target[1, 2] = torch.inf
    keep = (i != 1) | (j != 2)
    i, j = i[keep], j[keep]
    first, _ = component_losses(alpha, delta, target[i, j], anchor, i, j)
    shifted_columns, _ = component_losses(alpha, delta + anchor[None, :], target[i, j], anchor, i, j)
    shifted_rows, _ = component_losses(alpha * 4 + 1, delta, target[i, j], anchor, i, j)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first[1], shifted_columns[1])
    torch.testing.assert_close(first[2], shifted_rows[2], rtol=0, atol=0)


def test_batch_schedules_have_identical_observed_budget():
    mask = np.random.default_rng(22).uniform(size=(100, 40)) > .3
    expected = set(zip(*np.where(mask)))
    for sampler in ("pairs", "block4x3", "block6x2"):
        batches = list(schedule(mask, np.random.default_rng(271), sampler, 12))
        observed = [(c[ii], d[jj]) for c, d, i, j in batches for ii, jj in zip(i, j)]
        assert len(batches) == 12
        assert set(observed) == expected and len(observed) == len(expected)


def test_projection_removes_pairwise_conflict_and_preserves_private_gradient():
    g = torch.tensor([[1., 0., 3.], [-1., 1., 4.], [0., 0., 2.]])
    w = torch.tensor([1., 1., .2])
    orders = [[1, 2], [0, 2], [0, 1]]
    result = project(g[:, :2], orders)
    torch.testing.assert_close(result, torch.tensor([[.5, .5], [0., 1.], [0., 0.]]))
    v = task_update(g, w, "pcgrad", orders, 0, torch.tensor([0, 1]), None, None)
    torch.testing.assert_close(v, torch.tensor([.5, 1.5, 7.4]))
    no_conflict = torch.abs(g)
    result = task_update(no_conflict, w, "pcgrad", orders, 0, torch.tensor([0, 1]), None, None)
    torch.testing.assert_close(result, w @ no_conflict)


def test_weighted_sum_matches_direct_autograd():
    model = TwoAxisNet("native")
    x, d = torch.rand(10, 512), torch.rand(4, 512)
    i, j = torch.arange(10).repeat_interleave(4), torch.arange(4).repeat(10)

    def losses():
        a, delta = model(x, d, torch.arange(10), torch.arange(4))
        return component_losses(a, delta, torch.sin(i + j), torch.arange(4) / 4, i, j)[0]

    params = list(model.parameters())
    w = torch.tensor([1., .08, .035])
    grads = vector_grads(losses(), params)
    (losses() @ w).backward()
    expected = torch.cat([p.grad.reshape(-1) for p in params])
    torch.testing.assert_close(w @ grads, expected, atol=2e-7, rtol=1e-5)


def test_g0_g1_replay_and_structural_single_query():
    rng = np.random.default_rng(43)
    y = rng.normal(size=(26, 17))
    y[2, 4], y[7, 9] = np.nan, np.nan
    view = {"x": rng.normal(size=(26, 512)), "drug": rng.normal(size=(17, 512)),
            "q": rng.normal(size=(4, 512)), "y": y, "anchor": rng.normal(size=17),
            "scale": 1., "mapping": (np.arange(4), np.arange(4))}
    config = {"epochs": 1, "updates_per_epoch": 12, "weights": [[1, .08, .035]]}
    for arm in (Arm("g0", "smooth"), Arm("g1", "sum")):
        one, a, _ = train(view, arm, config, 19, device="cpu", diagnostics=True)
        two, b, _ = train(view, arm, config, 19, device="cpu", diagnostics=True)
        assert a["shared_params"] and a["state_sha256"] == b["state_sha256"]
        assert a["batch_order_sha256"] == b["batch_order_sha256"]
        np.testing.assert_array_equal(predict(one, view, "cpu"), predict(two, view, "cpu"))
    for head in ("shared", "separate"):
        model, _, _ = train(view, Arm("parts", "sum", head=head), config, 19, device="cpu")
        prediction = predict(model, view, "cpu", matrix=True)
        single = predict(model, view | {"q": view["q"][[2]]}, "cpu", matrix=True)
        np.testing.assert_allclose(prediction[[2]], single, atol=4e-7)
        x = torch.from_numpy(view["x"]).float()
        drug = torch.from_numpy(view["drug"]).float()
        _, delta = model(x, drug, torch.arange(len(x)), torch.arange(len(drug)))
        torch.testing.assert_close(delta.mean(0), torch.zeros(len(drug)), atol=1e-7, rtol=0)
        torch.testing.assert_close(delta.mean(1), torch.zeros(len(x)), atol=1e-7, rtol=0)


if __name__ == "__main__":
    test_masked_losses_have_exact_axis_invariances()
    test_batch_schedules_have_identical_observed_budget()
    test_projection_removes_pairwise_conflict_and_preserves_private_gradient()
    test_weighted_sum_matches_direct_autograd()
    test_g0_g1_replay_and_structural_single_query()
    print("5 H3 Step 0 mathematical and deterministic replay tests passed")
