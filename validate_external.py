
import csv
import numpy as np
import spheroidal

from kerr_teukolsky_study import eigenvalue, reference

# Parameters from the manuscript
spins = [0.2, 0.3, 0.5, 0.7, 0.9]
omega = 0.5
x = np.linspace(-0.85, 0.85, 69)
tolerance = 1e-6

results = []

print("EXTERNAL TEUKOLSKY VALIDATION")
print("-" * 55)

for a in spins:
    gamma = a * omega

    # Original Jacobi-Galerkin solver
    A_original = eigenvalue(a, omega, 22)
    S_original = reference(a, x, omega, 22)

    # External spheroidal package
    lam_spectral = spheroidal.eigenvalue(
        -2, 2, 2, gamma, method="spectral"
    )

    lam_leaver = spheroidal.eigenvalue(
        -2, 2, 2, gamma, method="leaver"
    )

    # Convert the eigenvalue convention
    A_spectral = lam_spectral - gamma**2 + 4*gamma
    A_leaver = lam_leaver - gamma**2 + 4*gamma

    # External angular eigenfunction
    S_function = spheroidal.harmonic(
        -2, 2, 2, gamma, method="spectral"
    )

    S_external = np.array([
        S_function(float(np.arccos(xi)), 0)
        for xi in x
    ], dtype=complex)

    # Normalize both solutions to S(0) = 1
    S_external /= S_function(np.pi / 2, 0)

    # Compare the two implementations
    eigen_error = abs(A_original - A_spectral)
    leaver_error = abs(A_original - A_leaver)
    profile_error = np.max(
        np.abs(S_original - S_external)
    )

    results.append({
        "spin": a,
        "A_original": A_original,
        "A_spectral": A_spectral,
        "A_leaver": A_leaver,
        "eigen_error": eigen_error,
        "leaver_error": leaver_error,
        "profile_error": profile_error
    })

    print(f"Spin = {a}")
    print(f"  Spectral error: {eigen_error:.3e}")
    print(f"  Leaver error:   {leaver_error:.3e}")
    print(f"  Profile error:  {profile_error:.3e}")
    print()

# Save real experimental results
with open(
    "kerr_external_validation.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

passed = all(
    r["eigen_error"] < tolerance
    and r["leaver_error"] < tolerance
    and r["profile_error"] < tolerance
    for r in results
)

print("-" * 55)
print("RESULT:", "PASS" if passed else "FAIL")
print("Results saved to kerr_external_validation.csv")