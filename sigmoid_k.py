
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from kerr_teukolsky_study import (
    AngularPINN, reference, residual, LO, HI
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "kerr_sigmoid_k.csv"
SUMMARY = BASE / "kerr_sigmoid_k_summary.csv"
CONFIG = BASE / "kerr_sigmoid_k_metadata.json"

K_VALUES = (0.5, 1.0, 2.0, 4.0)
SPINS = (0.2, 0.3, 0.5, 0.7, 0.9)
SEEDS = range(5)

OMEGA = 0.5
EPOCHS = 800
LBFGS_ITER = 50
TEST_ONLY = False

torch.set_num_threads(1)


def fit(a_true, k, seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    x = np.linspace(-0.9, 0.9, 81)
    y = reference(a_true, x, OMEGA, 22)

    xd = torch.tensor(
        x, dtype=torch.float64
    ).reshape(-1, 1)

    yd = torch.tensor(
        y, dtype=torch.float64
    ).reshape(-1, 1)

    xf = torch.linspace(
        -0.92, 0.92, 80,
        dtype=torch.float64
    ).reshape(-1, 1)

    model = AngularPINN(
        width=24,
        depth=2,
        activation="tanh",
        hard=True
    )

    spin_initial = 0.495

    p = (spin_initial - LO) / (HI - LO)
    q = p ** (1.0 / k)

    raw = torch.nn.Parameter(
        torch.tensor(
            math.log(q / (1.0 - q)),
            dtype=torch.float64
        )
    )

    lam = torch.nn.Parameter(
        torch.tensor(
            4.0,
            dtype=torch.float64
        )
    )

    params = list(model.parameters()) + [raw, lam]

    def losses():

        a = LO + (HI - LO) * (
            torch.sigmoid(raw).pow(k)
        )

        lp = residual(
            model, xf, a, lam, OMEGA
        ).square().mean()

        ld = (
            model(xd) - yd
        ).square().mean()

        return lp + 20.0 * ld, lp, ld, a

    opt = torch.optim.Adam(
        params, lr=1e-3
    )

    for epoch in range(EPOCHS):

        opt.zero_grad()
        loss, _, _, _ = losses()

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite Adam loss"
            )

        loss.backward()
        opt.step()

    if LBFGS_ITER:

        opt2 = torch.optim.LBFGS(
            params,
            lr=0.5,
            max_iter=LBFGS_ITER,
            line_search_fn="strong_wolfe"
        )

        def closure():

            opt2.zero_grad()
            loss, _, _, _ = losses()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite L-BFGS loss"
                )

            loss.backward()
            return loss

        opt2.step(closure)

    with torch.enable_grad():
        loss, lp, ld, a = losses()

    predicted = float(a.detach())

    return {
        "k": k,
        "a_true": a_true,
        "seed": seed,
        "a_hat": predicted,
        "absolute_error": abs(predicted - a_true),
        "lambda_hat": float(lam.detach()),
        "total_loss": float(loss.detach()),
        "pde_loss": float(lp.detach()),
        "data_loss": float(ld.detach()),
        "epochs": EPOCHS,
        "lbfgs": LBFGS_ITER,
        "sigma": 0.0
    }


def write_rows(path, rows):

    if not rows:
        return

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)


def run():

    combinations = [
        (k, a, seed)
        for k in K_VALUES
        for a in SPINS
        for seed in SEEDS
    ]

    if TEST_ONLY:
        combinations = [(1.0, 0.3, 0)]

    rows = []

    if OUT.exists() and not TEST_ONLY:

        with OUT.open(
            newline="", encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))

    done = {
        (
            float(r["k"]),
            float(r["a_true"]),
            int(r["seed"]),
            int(r["epochs"]),
            int(r["lbfgs"])
        )
        for r in rows
    }

    print("SIGMOID EXPONENT SENSITIVITY")
    print("Test only:", TEST_ONLY)
    print("Configurations:", len(combinations))

    for index, (k, a, seed) in enumerate(
        combinations, 1
    ):

        key = (
            k, a, seed, EPOCHS, LBFGS_ITER
        )

        if key in done:
            print("SKIP", key, flush=True)
            continue

        result = fit(a, k, seed)

        print(
            f"{index}/{len(combinations)} "
            f"k={k:g} "
            f"true={a:.1f} "
            f"seed={seed} "
            f"prediction={result['a_hat']:.6f} "
            f"error={result['absolute_error']:.6g}",
            flush=True
        )

        if not TEST_ONLY:
            rows.append(result)
            write_rows(OUT, rows)

    if TEST_ONLY:
        print("TEST COMPLETED SUCCESSFULLY")
        return

    summary = []

    for k in K_VALUES:

        selected = [
            r for r in rows
            if float(r["k"]) == k
            and int(r["epochs"]) == EPOCHS
            and int(r["lbfgs"]) == LBFGS_ITER
        ]

        errors = np.array([
            float(r["absolute_error"])
            for r in selected
        ])

        if len(errors):

            summary.append({
                "k": k,
                "n": len(errors),
                "MAE": float(errors.mean()),
                "RMSE": float(
                    np.sqrt(np.mean(errors ** 2))
                ),
                "worst_error": float(errors.max())
            })

    write_rows(SUMMARY, summary)

    metadata = {
        "k_values": K_VALUES,
        "spins": SPINS,
        "seeds": list(SEEDS),
        "omega": OMEGA,
        "sigma": 0.0,
        "epochs": EPOCHS,
        "lbfgs_iter": LBFGS_ITER,
        "torch_version": torch.__version__,
        "spin_initial": 0.495,
        "model": "s=-2 angular Teukolsky",
        "architecture": "tanh 24x2",
        "sigmoid": "LO+(HI-LO)*sigmoid(raw)**k"
    }

    CONFIG.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("Completed records:", len(rows))
    print("Expected records:", 100)

    for item in summary:
        print(item)


if __name__ == "__main__":
    run()
