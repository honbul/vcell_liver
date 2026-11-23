"""Fixed two-axis model, masked objectives, samplers and gradient update rules."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vc_build.response_components import orthogonal_parts

PAIRS = tuple(combinations(range(3), 2))
LOSS_NAMES = ("smooth_l1", "drug_pearson", "cell_listnet")
BLOCKS = {"block4x3": (4, 3), "block6x2": (6, 2)}


@dataclass(frozen=True)
class Arm:
    name: str
    method: str
    weight: int = 0
    batch: str = "pairs"
    head: str = "native"

    def record(self):
        return asdict(self)


class TwoAxisNet(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.cell = self.encoder(512)
        self.drug = self.encoder(512)
        self.additive = nn.Linear(16, 1)
        if head == "separate":
            self.alpha_cell = self.encoder(512)

    @staticmethod
    def encoder(width):
        return nn.Sequential(nn.Linear(width, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 16))

    def forward(self, x, d, cells, drugs, reference=None):
        molecule = self.drug(d)
        if self.head == "native":
            cell = self.cell(x[cells])
        else:
            ref = self.cell(x if reference is None else reference)
            cell = (ref[cells] if reference is None else self.cell(x[cells])) - ref.mean(dim=0)
            molecule = molecule - molecule.mean(dim=0)
        if self.head == "separate":
            marginal_features = self.alpha_cell(x[cells])
        elif self.head == "shared":
            marginal_features = cell  # Fit-centred cell representation.
        else:
            marginal_features = cell
        alpha = self.additive(marginal_features)
        delta = (cell @ molecule[drugs].T) / 4
        return alpha, delta


def schedule(mask, rng, kind, chunks):
    """Yield local valid positions. All observed pairs occur exactly once per epoch."""
    if kind == "pairs":
        row, col = np.nonzero(mask)
        for part in np.array_split(rng.permutation(len(row)), chunks):
            cells, i = np.unique(row[part], return_inverse=True)
            drugs, j = np.unique(col[part], return_inverse=True)
            yield cells, drugs, i, j
        return

    n_cells, n_drugs = BLOCKS[kind]
    by_cell = np.array_split(rng.permutation(mask.shape[0]), n_cells)
    by_drug = np.array_split(rng.permutation(mask.shape[1]), n_drugs)
    tiles = [(cells, drugs) for cells in by_cell for drugs in by_drug]
    for index in rng.permutation(len(tiles)):
        cells, drugs = tiles[index]
        i, j = np.nonzero(mask[np.ix_(cells, drugs)])
        if not len(i):
            raise ValueError("Empty registered block would change the sample budget")
        yield cells, drugs, i, j


def component_losses(alpha, delta, y, anchors, i, j, parts=None):
    """Operate on gathered observed positions only. Logit cells contain no missing-label targets."""
    p = (alpha + delta)[i, j]
    if parts is None:
        regression = F.smooth_l1_loss(p, y, beta=.45)
    else:
        fitted_alpha, residual = parts
        regression = (F.smooth_l1_loss(alpha[i, 0], fitted_alpha, beta=.45)
                      + F.smooth_l1_loss(delta[i, j], residual, beta=.45))
    n = torch.bincount(j, minlength=delta.shape[1]).to(p.dtype).clamp_min(1)

    def colsum(values):
        return torch.zeros_like(n).scatter_add(0, j, values)

    yp, pp = y - (colsum(y) / n)[j], p - (colsum(p) / n)[j]
    denom = (colsum(yp.square()) * colsum(pp.square())).clamp_min(1e-8).sqrt()
    valid_drugs = (n >= 8) & (colsum(yp.square()) > 1e-8)
    cor = (colsum(yp * pp) / denom * valid_drugs).sum() / valid_drugs.sum().clamp_min(1)
    pearson = 1 - cor

    # Alpha is a constant across each row and cancels algebraically in log_softmax.
    expected = (y + anchors[j]) / .17
    logits = (delta[i, j] + anchors[j]) / .17
    sentinel = -1e4
    score_matrix = torch.full_like(delta, sentinel).index_put((i, j), logits)
    truth_matrix = torch.full_like(delta, sentinel).index_put((i, j), expected)
    list_support = torch.bincount(i, minlength=delta.shape[0]) >= 2
    row_loss = -(truth_matrix.softmax(dim=1) * score_matrix.log_softmax(dim=1)).sum(dim=1)
    listnet = (row_loss * list_support).sum() / list_support.sum().clamp_min(1)
    stats = torch.stack((valid_drugs.sum(), list_support.sum(), torch.bincount(
        i, minlength=delta.shape[0]).gt(0).sum()))
    return torch.stack((regression, pearson, listnet)), stats


def vector_grads(losses, params):
    def gradient(loss, retain):
        result = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        return torch.cat([(torch.zeros_like(param) if grad is None else grad).reshape(-1)
                          for grad, param in zip(result, params)])

    return torch.stack([gradient(loss, index < 2) for index, loss in enumerate(losses)])


def project(gradients, orders):
    """Project each weighted task against other ORIGINAL task gradients in a seeded order."""
    result = []
    for index, order in enumerate(orders):
        current = gradients[index]
        for other in order:
            reference = gradients[other]
            coefficient = (current @ reference).clamp_max(0) / reference.square().sum().clamp_min(1e-16)
            current = current - coefficient * reference
        result.append(current)
    return torch.stack(result)


def cagrad(gradients, strength=.4):
    """Solve the CAGrad three-task convex simplex objective with fixed Frank-Wolfe steps."""
    gram = gradients @ gradients.T
    uniform = torch.full_like(gram[0], 1 / len(gradients))
    mean_projection = gram @ uniform
    radius = strength * (uniform @ mean_projection).clamp_min(1e-16).sqrt()
    mixture = uniform.clone()
    vertices = torch.eye(len(gradients), device=gradients.device)
    for step in range(12):
        prod = gram @ mixture
        slope = mean_projection + radius * prod / (mixture @ prod).clamp_min(1e-16).sqrt()
        eta = 2 / (3 + step)
        mixture = (1 - eta) * mixture + eta * (vertices * (slope == slope.min())[:, None]).sum(
            dim=0) / (slope == slope.min()).sum()
    worst = mixture @ gradients
    combined = gradients.mean(0) + radius / worst.norm().clamp_min(1e-8) * worst
    return len(gradients) * combined / (1 + strength**2)


class GradNorm:
    def __init__(self, device):
        self.logits = nn.Parameter(torch.zeros(3, device=device))
        self.optimizer = torch.optim.Adam([self.logits], lr=.01)
        self.reference = None

    def weights(self, loss, weighted):
        if self.reference is None:
            self.reference = loss.detach().clamp_min(1e-4)
        rate = loss.detach() / self.reference
        rate = rate / rate.mean().clamp_min(1e-8)
        multipliers = self.logits.softmax(0) * len(loss)
        norm = weighted.norm(dim=1) * multipliers
        target = norm.detach().mean() * rate.clamp_min(0).pow(.5)
        update = (norm - target).abs().sum()
        self.optimizer.zero_grad(set_to_none=True)
        update.backward()
        self.optimizer.step()
        return multipliers.detach()


def task_update(gradients, weights, method, orders, update, shared, loss, gradnorm):
    weighted = gradients * weights[:, None]
    if method == "smooth":
        return weighted[0]
    if method == "alternating":
        return 3 * weighted[update % 3]
    direct = weighted.sum(0)
    if method == "sum":
        return direct
    active = weighted[:, shared]
    if method == "gradnorm":
        return gradnorm.weights(loss, active) @ weighted
    if method == "cagrad":
        combined = cagrad(active)
    elif method == "normalized":
        norms = active.norm(dim=1)
        normalized = active / norms[:, None].clamp_min(1e-8)
        combined = project(normalized, orders).sum(0) * norms.mean()
    elif method == "pcgrad":
        combined = project(active, orders).sum(0)
    else:
        raise ValueError(f"Unrecognized method: {method}")
    direct[shared] = combined
    return direct


def gradient_diagnostics(gradients, weights):
    norms = gradients.norm(dim=1)
    dots = gradients @ gradients.T
    details = []
    for i, j in PAIRS:
        valid = (norms[i] > 1e-8) & (norms[j] > 1e-8)
        cosine = dots[i, j] / (norms[i] * norms[j]).clamp_min(1e-16)
        ratio = norms[i] / norms[j].clamp_min(1e-8)
        weighted = ratio * weights[i] / weights[j]
        details.extend((cosine, ratio, weighted, valid, norms[i], norms[j]))
    return torch.stack(details)


def weight_stamp(model):
    return sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()


def train(view, arm, config, seed, *, device, diagnostics=False, task_vectors=False):
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("No assigned CUDA device")
        torch.cuda.synchronize()
    started = perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    permutations = np.random.default_rng(seed + 34173)
    model = TwoAxisNet(arm.head).to(device)
    initial = weight_stamp(model)
    params = list(model.parameters())
    shared = torch.cat([torch.full((p.numel(),), not name.startswith(("additive", "alpha_cell")),
                                  device=device, dtype=torch.bool) for name, p in model.named_parameters()])
    # Integer selector avoids a dynamic shape and CPU synchronisation at every update.
    shared = torch.where(shared)[0]
    if not len(shared):
        raise RuntimeError("Expected shared learnable encoder parameters")
    optimizer = torch.optim.AdamW(params, lr=.0022, weight_decay=.08)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=.00025)
    gradnorm = GradNorm(device) if arm.method == "gradnorm" else None

    def float_tensor(values):
        return torch.as_tensor(values, device=device, dtype=torch.float32)

    x, d, y = [float_tensor(view[key]) for key in ("x", "drug", "y")]
    mask = np.isfinite(view["y"])
    a = float_tensor((view["anchor"] - np.mean(view["anchor"])) / view["scale"])
    weights = float_tensor(config["weights"][arm.weight])
    if arm.head != "native":
        alpha, _, interaction = orthogonal_parts(view["y"], mask.astype(float))
        alpha, interaction = float_tensor(alpha), float_tensor(interaction)

    use_task_vectors = diagnostics or task_vectors or arm.method not in ("sum", "smooth", "alternating")
    audits, gradient_events, timings = [], [], []
    steps, covered = 0, 0
    batch_hash = sha256()
    for epoch in range(config["epochs"]):
        for cells, drugs, i, j in schedule(mask, rng, arm.batch, config["updates_per_epoch"]):
            batch_hash.update(np.stack((cells[i], drugs[j])).tobytes())
            covered += len(i)
            batch = [torch.as_tensor(v, device=device) for v in (cells, drugs, i, j)]
            ci, di, ri, dj = batch
            truth = y[ci[ri], di[dj]]
            pred_alpha, delta = model(x, d, ci, di)
            parts = None if arm.head == "native" else (alpha[ci[ri]], interaction[ci[ri], di[dj]])
            loss, counts = component_losses(pred_alpha, delta, truth, a[di], ri, dj, parts)
            orders = [permutations.permutation([j for j in range(3) if j != task]).tolist()
                      for task in range(3)]
            optimizer.zero_grad(set_to_none=True)
            if diagnostics and device == "cuda":
                before = torch.cuda.Event(enable_timing=True)
                after = torch.cuda.Event(enable_timing=True)
                before.record()
            if use_task_vectors:
                gradients = vector_grads(loss, params)
                update = task_update(gradients, weights, arm.method, orders, steps, shared, loss, gradnorm)
                for param, part in zip(params, update.split([p.numel() for p in params])):
                    param.grad = part.view_as(param)
            else:
                total = (loss * weights).sum() if arm.method == "sum" else weights[0] * loss[0]
                if arm.method == "alternating":
                    total = 3 * (loss * weights)[steps % 3]
                total.backward()
                for param in params:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
            if diagnostics and device == "cuda":
                after.record()
                gradient_events.append((before, after))
            nn.utils.clip_grad_norm_(params, 3.)
            optimizer.step()
            if diagnostics:
                audits.append(gradient_diagnostics(gradients[:, shared], weights))
                timings.append((epoch + 1, steps + 1, len(i), len(cells) * len(drugs), counts))
            steps += 1
        decay.step()
    if covered != int(mask.sum()) * config["epochs"] or steps != config["epochs"] * config["updates_per_epoch"]:
        raise ValueError("Training changed the locked observation or update budget")
    if device == "cuda":
        torch.cuda.synchronize()
    runtime = perf_counter() - started
    records = []
    if audits:
        audited = torch.stack(audits).cpu().numpy()
        support = torch.stack([value[4] for value in timings]).cpu().numpy()
        for index, ((epoch, step, pairs, possible, _), diag, (n_drugs, n_lists, n_cells)) in enumerate(
            zip(timings, audited, support)
        ):
            time_ms = gradient_events[index][0].elapsed_time(gradient_events[index][1]) if gradient_events else None
            for (a_idx, b_idx), (cos, ratio, weighted_ratio, valid, a_norm, b_norm) in zip(
                PAIRS, diag.reshape(3, 6)
            ):
                records.append({
                    "epoch": epoch, "update": step, "loss_a": LOSS_NAMES[a_idx],
                    "loss_b": LOSS_NAMES[b_idx], "gradient_cosine": float(cos) if valid else None,
                    "norm_ratio": float(ratio) if valid else None,
                    "weighted_norm_ratio": float(weighted_ratio) if valid else None,
                    "norm_a": float(a_norm), "norm_b": float(b_norm),
                    "conflict": bool(cos < 0) if valid else None, "batch_pairs": pairs,
                    "rectangular_slots": possible, "eligible_drug_groups": int(n_drugs),
                    "eligible_cell_groups": int(n_lists), "present_cell_groups": int(n_cells),
                    "gradient_update_ms": time_ms,
                })
    model.eval()
    return model, {
        "arm": arm.record(), "seed": seed, "updates": steps, "pair_exposures": covered,
        "observed_cell_drug_pairs": int(mask.sum()), "missing_slots": int((~mask).sum()),
        "shared_params": len(shared), "trainable_params": sum(p.numel() for p in params),
        "initial_sha256": initial, "state_sha256": weight_stamp(model), "batch_order_sha256": batch_hash.hexdigest(),
        "wall_seconds": runtime, "gpu_hours": runtime / 3600 if device == "cuda" else 0.,
        "diagnostics": diagnostics, "explicit_task_vectors": use_task_vectors,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
    }, records


@torch.inference_mode()
def predict(model, view, device, *, matrix=False):
    x = torch.as_tensor(view["q"], dtype=torch.float32, device=device)
    d = torch.as_tensor(view["drug"], dtype=torch.float32, device=device)
    ref = torch.as_tensor(view["x"], dtype=torch.float32, device=device)
    alpha, delta = model(x, d, torch.arange(len(x), device=device), torch.arange(len(d), device=device), ref)
    scaled = (alpha.double() + delta.double()).cpu().numpy() * view["scale"] + view["anchor"]
    if not np.isfinite(scaled).all():
        raise ValueError("Nonfinite supported forecast")
    return scaled if matrix else scaled[view["mapping"]]
