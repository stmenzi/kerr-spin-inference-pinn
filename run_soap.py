
import csv
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch

from kerr_teukolsky_study import (
    AngularPINN,
    reference,
    residual,
    LO,
    HI
)

folder = Path(__file__).resolve().parent
soap_file = folder / "soap.py"

if not soap_file.exists():
    print("Downloading SOAP optimizer...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py",
        soap_file
    )

source = soap_file.read_text(encoding="utf-8")

old1 = (
    "torch.zeros(grad.shape[0], "
    "grad.shape[0], device=grad.device)"
)

new1 = (
    "torch.zeros(grad.shape[0], "
    "grad.shape[0], device=grad.device, "
    "dtype=grad.dtype)"
)

old2 = "torch.zeros(sh, sh, device=grad.device)"

new2 = (
    "torch.zeros(sh, sh, device=grad.device, "
    "dtype=grad.dtype)"
)

changes = 0

if old1 in source:
    source = source.replace(old1, new1)
    changes += 1

if old2 in source:
    source = source.replace(old2, new2)
    changes += 1

if changes:
    soap_file.write_text(source, encoding="utf-8")
    print("SOAP float64 corrections applied.")
else:
    if new1 not in source or new2 not in source:
        raise RuntimeError(
            "SOAP version has changed. "
            "Check its dtype initialization."
        )

    print("SOAP float64 corrections already present.")

from soap import SOAP

SPINS = [0.2, 0.3, 0.5, 0.7, 0.9]
SIGMAS = [0.0, 0.01, 0.03, 0.05, 0.10]
SEEDS = range(5)

OMEGA = 0.5
EPOCHS = 800
LEARNING_RATE = 0.001
TEST_ONLY = False

torch.set_num_threads(1)

output = folder / "kerr_soap_core.csv"


def train_soap(a_true, sigma, seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    x = np.linspace(-0.9, 0.9, 81)

    true_profile = reference(
        a_true,
        x,
        OMEGA,
        22
    )

    rng = np.random.default_rng(
        seed
        + int(a_true * 1000)
        + int(sigma * 10000)
    )

    observations = (
        true_profile
        + rng.normal(0, sigma, len(x))
    )

    xd = torch.tensor(
        x,
        dtype=torch.float64
    ).reshape(-1, 1)

    yd = torch.tensor(
        observations,
        dtype=torch.float64
    ).reshape(-1, 1)

    xf = torch.linspace(
        -0.92,
        0.92,
        80,
        dtype=torch.float64
    ).reshape(-1, 1)

    model = AngularPINN(
        width=24,
        depth=2,
        activation="tanh",
        hard=True
    ).double()

    initial_spin = 0.495

    z = (
        (initial_spin - LO)
        / (HI - LO)
    )

    raw_spin = torch.nn.Parameter(
        torch.tensor(
            math.log(z / (1 - z)),
            dtype=torch.float64
        )
    )

    eigenvalue = torch.nn.Parameter(
        torch.tensor(
            4.0,
            dtype=torch.float64
        )
    )

    parameters = (
        list(model.parameters())
        + [raw_spin, eigenvalue]
    )

    optimizer = SOAP(
        parameters,
        lr=LEARNING_RATE,
        betas=(0.95, 0.95),
        weight_decay=0.0,
        precondition_frequency=10
    )

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        estimated_spin = (
            LO
            + (HI - LO)
            * torch.sigmoid(raw_spin)
        )

        pde_loss = residual(
            model,
            xf,
            estimated_spin,
            eigenvalue,
            OMEGA
        ).square().mean()

        data_loss = (
            model(xd) - yd
        ).square().mean()

        loss = (
            pde_loss
            + 20.0 * data_loss
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss at epoch "
                + str(epoch)
            )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Loss={loss.item():.8e}",
                flush=True
            )

    with torch.no_grad():

        predicted_spin = float(
            LO
            + (HI - LO)
            * torch.sigmoid(raw_spin)
        )

        predicted_eigenvalue = float(
            eigenvalue
        )

    absolute_error = abs(
        predicted_spin - a_true
    )

    return {
        "method": "SOAP",
        "a_true": a_true,
        "sigma": sigma,
        "seed": seed,
        "a_hat": predicted_spin,
        "absolute_error": absolute_error,
        "lambda_hat": predicted_eigenvalue,
        "epochs": EPOCHS
    }


rows = []

if output.exists():
    with output.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))

done = {
    (
        float(r["a_true"]),
        float(r["sigma"]),
        int(r["seed"])
    )
    for r in rows
    if int(r["epochs"]) == EPOCHS
}

combinations = [
    (a, sigma, seed)
    for a in SPINS
    for sigma in SIGMAS
    for seed in SEEDS
]

if TEST_ONLY:
    combinations = [(0.3, 0.0, 0)]

print("=" * 55)
print("SOAP OPTIMIZER EXPERIMENT")
print("PyTorch:", torch.__version__)
print("Precision: float64")
print("Epochs:", EPOCHS)
print("Test only:", TEST_ONLY)
print("=" * 55)

for a, sigma, seed in combinations:

    if (a, sigma, seed) in done:
        print("Already completed:", a, sigma, seed)
        continue

    print()
    print(
        f"Starting: spin={a}, "
        f"noise={sigma}, seed={seed}"
    )

    result = train_soap(
        a,
        sigma,
        seed
    )

    print(
        f"True spin:      {a:.6f}"
    )

    print(
        f"Predicted spin: {result['a_hat']:.6f}"
    )

    print(
        f"Absolute error: "
        f"{result['absolute_error']:.6f}"
    )

    if not TEST_ONLY:

        rows.append(result)

        with output.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(result.keys())
            )

            writer.writeheader()
            writer.writerows(rows)

if TEST_ONLY:

    print()
    print("=" * 55)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 55)

else:

    errors = np.array([
        float(r["absolute_error"])
        for r in rows
        if int(r["epochs"]) == EPOCHS
    ])

    mae = float(np.mean(errors))
    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    print()
    print("=" * 55)
    print("FINAL SOAP RESULTS")
    print("=" * 55)

    print("Completed runs:", len(errors))
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("CSV saved:", output)

    environment = {
        "optimizer": "SOAP",
        "torch_version": torch.__version__,
        "precision": "float64",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "betas": [0.95, 0.95],
        "precondition_frequency": 10,
        "soap_dtype_patch": True
    }

    metadata_file = (
        folder / "kerr_soap_environment.json"
    )

    metadata_file.write_text(
        json.dumps(
            environment,
            indent=2
        ),
        encoding="utf-8"
    )

    print("Metadata saved:", metadata_file)
