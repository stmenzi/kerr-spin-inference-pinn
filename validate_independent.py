"""Independent SciPy BVP collocation cross-check for spin-weighted Kerr angular problem.

No training, no Jacobi basis, no reuse of the Galerkin matrices.
Uses boundary regularity conditions for F defined by S=(1+x)^2 F.
For s=-2, l=m=2, at c=a*omega the regularized equation is
(1-x^2)F'' + (4-6x)F' + (A-4+4cx+c^2x^2)F = 0.
Endpoints are approximated at +- (1-epsilon); use epsilon convergence.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import eval_jacobi, roots_legendre
from scipy.linalg import eigh

SPINS=(0.2,0.3,0.5,0.7,0.9)
OMEGA=0.5


def bvp_reference(a, omega=OMEGA, eps=1e-5, tol=1e-7):
    c=a*omega
    mesh=np.linspace(-1+eps,1-eps,151)
    def ode(x,y,p):
        A=p[0]
        f,fp=y
        fpp= -((4-6*x)*fp+(A-4+4*c*x+c*c*x*x)*f)/(1-x*x)
        return np.vstack((fp,fpp))
    def bc(ya,yb,p):
        A=p[0]
        # Leading-order regularity relations at the true endpoints x=+-1.
        return np.array([ya[0]-1,
                         10*ya[1]+(A-4-4*c+c*c)*ya[0],
                         -2*yb[1]+(A-4+4*c+c*c)*yb[0]])
    y0=np.vstack((np.ones(len(mesh)),np.zeros(len(mesh))))
    fit=solve_bvp(ode,bc,mesh,y0,p=np.array([4.]),tol=tol,max_nodes=20000)
    if fit.status!=0:
        raise RuntimeError(f'SciPy BVP failed spin={a}: {fit.message}')
    def evaluate(x):
        x=np.asarray(x,dtype=float)
        f=fit.sol(x)[0]
        return (1+x)**2*f/fit.sol(0.)[0]
    return float(fit.p[0]),evaluate,int(fit.x.size),float(max(fit.rms_residuals))


def galerkin_reference(a,x,omega=OMEGA,order=22):
    # Comparator from the manuscript implemented in a different numerical family.
    xx,w=roots_legendre(180)
    P=np.column_stack([eval_jacobi(j,0,4,xx) for j in range(order)])
    DP=np.column_stack([np.zeros_like(xx) if j==0 else .5*(j+5)*eval_jacobi(j-1,1,5,xx) for j in range(order)])
    B=(1+xx[:,None])**2*P
    DB=2*(1+xx[:,None])*P+(1+xx[:,None])**2*DP
    c=a*omega
    W=4*(1-xx)/(1+xx)+2*(-2)*c*xx-c*c*xx**2+2
    K=DB.T@((w*(1-xx**2))[:,None]*DB)+B.T@((w*W)[:,None]*B)
    M=B.T@(w[:,None]*B)
    eig,vec=eigh(K,M,subset_by_index=[0,0]);v=vec[:,0]
    q=lambda t: (1+np.asarray(t))**2*sum(v[j]*eval_jacobi(j,0,4,np.asarray(t)) for j in range(order))
    y=q(x)/q(0.)
    return float(eig[0]),np.asarray(y)


def main():
    out=Path(__file__).resolve().parent
    x=np.linspace(-.85,.85,69)
    rows=[]
    for a in SPINS:
        A_bvp,p_bvp,mesh_n,max_r=bvp_reference(a)
        A_gal,sg=galerkin_reference(a,x)
        sb=p_bvp(x)
        row=dict(a_true=a,omega=OMEGA,s=-2,ell=2,m=2,
                 A_galerkin=A_gal,A_independent_bvp=A_bvp,
                 abs_eigenvalue_difference=abs(A_gal-A_bvp),
                 max_profile_difference=float(np.max(np.abs(sb-sg))),
                 rms_profile_difference=float(np.sqrt(np.mean((sb-sg)**2))),
                 bvp_nodes=mesh_n,max_bvp_rms_residual=max_r,
                 bvp_epsilon=1e-5,bvp_tol=1e-7)
        rows.append(row)
        print(f"a={a:.1f} A_Gal={A_gal:.12f} A_BVP={A_bvp:.12f} "
              f"dA={row['abs_eigenvalue_difference']:.3e} "
              f"dS={row['max_profile_difference']:.3e}",flush=True)
    with (out/'kerr_independent_bvp_validation.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved kerr_independent_bvp_validation.csv')

if __name__=='__main__': main()
