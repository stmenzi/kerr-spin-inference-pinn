import argparse, csv, json, math, os, time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, least_squares
from scipy.special import eval_jacobi, roots_legendre
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DTYPE=torch.float64
SPINS=[0.2,0.3,0.5,0.7,0.9]
SIGMAS=[0.0,0.01,0.03,0.05,0.10]
LO,HI=0.001,0.99
S,M,ELL=-2,2,2

# Angular Teukolsky convention:
# ((1-x²) S')' + [c²x² - 2scx + s + A - (m+sx)²/(1-x²)] S=0.
# At c=0, A=ell(ell+1)-s(s+1)=4.
# For s=-2,m=2,l=2, regular solutions have (1+x)^2 at x=-1 and finite x=1.

@lru_cache(maxsize=12)
def galerkin_matrices(order=16, quad=180):
    x,w=roots_legendre(quad)
    # Use regular Jacobi basis (1+x)^2 P_k^(0,4)(x)
    k=np.arange(order)
    polys=np.column_stack([eval_jacobi(int(j),0,4,x) for j in k])
    dp=np.column_stack([np.zeros_like(x) if j==0 else 0.5*(j+5)*eval_jacobi(int(j-1),1,5,x) for j in k])
    basis=(1+x[:,None])**2*polys
    deriv=2*(1+x[:,None])*polys+(1+x[:,None])**2*dp
    # Selfadjoint stiffness K = int((1-x²) S' T' + V(x) S T) dx
    # For m=2, s=-2: ((m+s*x)^2)/(1-x²) = 4(1-x)/(1+x).
    potential0=4*(1-x)/(1+x)-S
    mass=basis.T@(w[:,None]*basis)
    base=deriv.T@((w*(1-x*x))[:,None]*deriv)+basis.T@((w*potential0)[:,None]*basis)
    # operator V includes +2*s*c*x - c²*x²
    linear=basis.T@((w*(2*S*x))[:,None]*basis)
    quadratic=basis.T@((w*(-x*x))[:,None]*basis)
    return mass,base,linear,quadratic

@lru_cache(maxsize=4096)
def eigenpair(c_rounded,order=16):
    c=float(c_rounded)
    mass,base,linear,quadratic=galerkin_matrices(order)
    K=base+c*linear+c*c*quadratic
    val,vec=eigh(K,mass,subset_by_index=[0,0],check_finite=True)
    co=vec[:,0]
    # Normalize profile at x=0 to 1; sign fixed by normalization.
    normalization=sum(co[j]*eval_jacobi(j,0,4,0.) for j in range(order))
    return float(val[0]),co/normalization

def reference(a,x,omega=0.5,order=16):
    lam,co=eigenpair(round(float(a*omega),12),order)
    xx=np.asarray(x,dtype=float)
    return (1+xx)**2*sum(co[j]*eval_jacobi(j,0,4,xx) for j in range(order))

def eigenvalue(a,omega=0.5,order=16):
    return eigenpair(round(float(a*omega),12),order)[0]

def strong_form_residual(a, x, omega=0.5, order=22):
    """Independent pointwise ODE residual from analytic Jacobi derivatives."""
    x=np.asarray(x,dtype=float)
    A,co=eigenpair(round(float(a*omega),12),order)
    def P(j):return eval_jacobi(j,0,4,x)
    def DP(j):return np.zeros_like(x) if j<1 else (j+5)/2*eval_jacobi(j-1,1,5,x)
    def DDP(j):return np.zeros_like(x) if j<2 else (j+5)*(j+6)/4*eval_jacobi(j-2,2,6,x)
    poly=sum(co[j]*P(j) for j in range(order))
    dpoly=sum(co[j]*DP(j) for j in range(order))
    ddpoly=sum(co[j]*DDP(j) for j in range(order))
    y=(1+x)**2*poly
    dy=2*(1+x)*poly+(1+x)**2*dpoly
    ddy=2*poly+4*(1+x)*dpoly+(1+x)**2*ddpoly
    c=a*omega
    return (1-x*x)*ddy-2*x*dy+(c*c*x*x-2*S*c*x+S+A-(M+S*x)**2/(1-x*x))*y

def consistency_check(omega):
    xs=np.linspace(-0.85,0.85,69)
    errors=[]
    for a in SPINS:
        p16=reference(a,xs,omega,16);p22=reference(a,xs,omega,22)
        l16=eigenvalue(a,omega,16);l22=eigenvalue(a,omega,22)
        errors.append(dict(a=a,eigenvalue=l22,eigen_error=abs(l16-l22),profile_max_difference=float(np.max(np.abs(p16-p22))),pointwise_residual_max=float(np.max(np.abs(strong_form_residual(a,xs,omega,22))))))
    return errors

class AngularPINN(torch.nn.Module):
    def __init__(self,width=24,depth=2,activation='tanh',hard=True):
        super().__init__()
        act={'tanh':torch.nn.Tanh,'silu':torch.nn.SiLU,'gelu':torch.nn.GELU}[activation]
        blocks=[];n=1
        for _ in range(depth):
            blocks.extend((torch.nn.Linear(n,width),act()));n=width
        blocks.append(torch.nn.Linear(n,1))
        self.net=torch.nn.Sequential(*blocks).double()
        self.hard=hard
        for layer in self.net:
            if isinstance(layer,torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight);torch.nn.init.zeros_(layer.bias)

    def forward(self,x):
        # Explicitly regular at x=-1 and equator normalized to unity.
        # No unjustified even-parity symmetry (s=-2,m=2 is not even).
        return (1+x)**2*(1+x*self.net(x)) if self.hard else self.net(x)

def residual(model,x,a,lam,omega):
    x=x.detach().clone().requires_grad_(True)
    y=model(x)
    d=torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]
    dd=torch.autograd.grad(d,x,torch.ones_like(d),create_graph=True)[0]
    c=a*omega
    return (1-x*x)*dd-2*x*d+(c*c*x*x-2*S*c*x+S+lam-(M+S*x)**2/(1-x*x))*y

def fit_spectral(x, y, omega, method='bounded'):
    """Fit reference-family profiles; this is NOT an observational comparator."""
    if method == 'bounded':
        r = minimize_scalar(lambda a:float(np.mean((reference(a,x,omega,22)-y)**2)),
                            bounds=(LO,HI),method='bounded',options={'xatol':1e-9})
        return float(r.x)
    if method == 'trust_region':
        r = least_squares(lambda z:reference(float(z[0]),x,omega,22)-y,
                          x0=[0.495],bounds=([LO],[HI]),method='trf',
                          max_nfev=300,xtol=1e-12,ftol=1e-12,gtol=1e-12)
        return float(r.x[0])
    raise ValueError(method)

def train(x,y,seed,omega,spin_init=0.495,epochs=800,lbfgs=50,width=24,depth=2,activation='tanh',fixed_spin=None,eigenmode='joint',pde_weight=1.,data_weight=20.,hard=True):
    np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    model=AngularPINN(width,depth,activation,hard=hard)
    z=(spin_init-LO)/(HI-LO)
    raw=torch.nn.Parameter(torch.tensor(math.log(z/(1-z)),dtype=DTYPE))
    lam=torch.nn.Parameter(torch.tensor(float(ELL*(ELL+1)-S*(S+1)),dtype=DTYPE))
    params=list(model.parameters())
    if fixed_spin is None: params.append(raw)
    if eigenmode=='joint':params.append(lam)
    xd=torch.tensor(x,dtype=DTYPE).reshape(-1,1)
    yd=torch.tensor(y,dtype=DTYPE).reshape(-1,1)
    xf=torch.linspace(-0.92,0.92,80,dtype=DTYPE).reshape(-1,1)
    afix=torch.tensor(fixed_spin,dtype=DTYPE) if fixed_spin is not None else None
    lamfix=torch.tensor(eigenvalue(fixed_spin,omega,22),dtype=DTYPE) if eigenmode=='spectral' and fixed_spin is not None else None
    if eigenmode=='spectral' and fixed_spin is None:
        grid=np.linspace(LO,HI,1001)
        ev=torch.tensor([eigenvalue(float(z),omega,22) for z in grid],dtype=DTYPE)
        gridstep=float(grid[1]-grid[0])
    if eigenmode not in ('joint','spectral','fixed'):
        raise ValueError('Unknown eigenvalue policy')
    def losses():
        a=afix if fixed_spin is not None else LO+(HI-LO)*torch.sigmoid(raw)
        if eigenmode=='joint':L=lam
        elif eigenmode=='fixed':L=torch.tensor(4.,dtype=DTYPE)
        elif fixed_spin is not None:L=lamfix
        else:
            position=(a-LO)/gridstep
            idx=torch.floor(position.detach()).long().clamp(0,len(ev)-2)
            frac=position-idx.to(DTYPE)
            L=ev[idx]*(1-frac)+ev[idx+1]*frac
        lp=residual(model,xf,a,L,omega).square().mean() if pde_weight else torch.zeros((),dtype=DTYPE)
        ld=(model(xd)-yd).square().mean()
        bc=torch.zeros((),dtype=DTYPE)
        if not hard:
            endpoints=torch.tensor([[-1.0],[0.0],[1.0]],dtype=DTYPE)
            target=torch.tensor([[0.0],[1.0],[4.0]],dtype=DTYPE)
            # At x=1 the hard ansatz is 4 (when N finite); this is not a physical
            # endpoint value requirement. For soft constraints enforce only
            # vanishing at x=-1 and unit normalization at x=0.
            bc=(model(endpoints[:2])-target[:2]).square().sum()
        total=pde_weight*lp+data_weight*ld+10.0*bc
        return total,lp,ld,a,L
    opt=torch.optim.Adam(params,lr=1e-3)
    for ep in range(epochs):
        opt.zero_grad();total,*_=losses()
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite loss')
        total.backward();opt.step()
    if lbfgs:
        opt2=torch.optim.LBFGS(params,lr=.5,max_iter=lbfgs,line_search_fn='strong_wolfe')
        def closure():
            opt2.zero_grad();total,*_=losses();total.backward();return total
        opt2.step(closure)
    with torch.enable_grad():total,lp,ld,a,L=losses()
    return dict(a_hat=float(a.detach()),lambda_hat=float(L.detach()),total_loss=float(total.detach()),pde_loss=float(lp.detach()),data_loss=float(ld.detach()))

def load_csv(path):
    if not path.exists():return []
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize_core(rows, epochs):
    output=[]
    for method in sorted({r['method'] for r in rows}):
        for sigma in sorted({float(r['sigma']) for r in rows}):
            selected=[r for r in rows if r['method']==method and float(r['sigma'])==sigma and int(r['epochs'])==epochs]
            if not selected:continue
            error=np.asarray([float(r['absolute_error']) for r in selected])
            output.append(dict(method=method,sigma=sigma,n=len(selected),MAE=float(error.mean()),
                               RMSE=float(np.sqrt(np.mean(error**2))),median_AE=float(np.median(error)),
                               worst_AE=float(error.max())))
    write_csv(Path('kerr_summary.csv'),output)

def summarize_architecture(rows, epochs):
    output=[]
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant and int(r['epochs'])==epochs]
        if not selected:continue
        error=np.asarray([float(r['absolute_error']) for r in selected])
        output.append(dict(variant=variant,n=len(selected),MAE=float(error.mean()),
                           RMSE=float(np.sqrt(np.mean(error**2)))))
    write_csv(Path('kerr_architecture_summary.csv'),output)

def save_plot(fig,name):
    fig.tight_layout();fig.savefig(f'kerr_fig_{name}.pdf');plt.close(fig)
    print('FIGURE',f'kerr_fig_{name}.pdf',flush=True)

def execute(args):
    quick=args.quick
    spins=[.3,.7] if quick else args.spins
    seeds=[0] if quick else args.seeds
    noises=[0.,.01] if quick else args.noise
    epochs=min(30,args.epochs) if quick else args.epochs
    lbfgs=min(5,args.lbfgs) if quick else args.lbfgs
    omega=args.omega
    x=np.linspace(-.9,.9,81)
    metadata={'model':'spin-weighted gravitational angular Teukolsky equation','s':S,'m':M,'ell':ELL,
              'omega':'real prescribed '+str(omega),'not_QNM':True,'quick':quick,'epochs':epochs,'lbfgs':lbfgs,'seeds':seeds,
              'data_origin':'independent higher-order Jacobi Galerkin reference (order 22); solver PINN or lower-order order16',
              'physical_scope':'Angular only, no radial/QNM eigenfrequency or observations',
              'warning':'Reference and fitting use same Galerkin-family model (orders 22 vs 16); not external astrophysical validation'}
    Path('kerr_metadata.json').write_text(json.dumps(metadata,indent=2))
    write_csv(Path('kerr_consistency.csv'),consistency_check(omega))
    if args.mode in ('all','core'):
        path=Path('kerr_core.csv');rows=[] if args.force else load_csv(path)
        done={(r['method'],float(r['a_true']),float(r['sigma']),int(r['seed']),int(r['epochs'])) for r in rows}
        for a in spins:
            ref=reference(a,x,omega,22)
            for sigma in noises:
                for seed in seeds:
                    rng=np.random.default_rng(seed+int(a*1000)+int(sigma*10000))
                    y=ref+rng.normal(0,sigma,len(x))
                    for method in ('spectral','spectral_trust_region','adam','adam_lbfgs','fixed_lambda','spectral_lambda','soft_constraints','physics_only','data_only'):
                        key=(method,a,sigma,seed,epochs)
                        if key in done:continue
                        if method.startswith('spectral') and method!='spectral_lambda':
                            r={'a_hat':fit_spectral(x,y,omega,
                               'trust_region' if method=='spectral_trust_region' else 'bounded')}
                        else:
                            r=train(x,y,seed,omega,epochs=epochs,lbfgs=0 if method=='adam' else lbfgs,
                                     eigenmode=('fixed' if method=='fixed_lambda' else
                                                'spectral' if method=='spectral_lambda' else 'joint'),
                                     data_weight=0 if method=='physics_only' else 20,
                                     pde_weight=0 if method=='data_only' else 1,
                                     hard=(method!='soft_constraints'))
                        row=dict(method=method,a_true=a,sigma=sigma,seed=seed,epochs=epochs,absolute_error=abs(r['a_hat']-a),**r)
                        rows.append(row);write_csv(path,rows)
                        print('CORE',method,'true',a,'noise',sigma,'seed',seed,'estimate',round(r['a_hat'],5),flush=True)
    if args.mode in ('all','architecture'):
        path=Path('kerr_architecture.csv');rows=[] if args.force else load_csv(path)
        done={(r['variant'],float(r['a_true']),int(r['seed']),int(r['epochs'])) for r in rows}
        variants=[('tanh_24x2','tanh',24,2),('silu_24x2','silu',24,2),('gelu_24x2','gelu',24,2),('tanh_12x1','tanh',12,1),('tanh_48x4','tanh',48,4)]
        for a in spins:
            y=reference(a,x,omega,22)
            for seed in seeds:
                for name,activation,width,depth in variants:
                    key=(name,a,seed,epochs)
                    if key in done:continue
                    r=train(x,y,seed,omega,epochs=epochs,lbfgs=lbfgs,width=width,depth=depth,activation=activation)
                    rows.append(dict(variant=name,a_true=a,seed=seed,epochs=epochs,absolute_error=abs(r['a_hat']-a),**r))
                    write_csv(path,rows);print('ARCH',name,a,seed,round(r['a_hat'],5),flush=True)
    if args.mode in ('all','fixed_profile'):
        path=Path('kerr_fixed_profiles.csv');rows=[] if args.force else load_csv(path)
        done={(float(r['a_true']),float(r['candidate']),int(r['seed']),int(r['epochs'])) for r in rows}
        candidates=np.linspace(LO,HI,5 if quick else args.profile_points)
        for a in spins:
            y=reference(a,x,omega,22)
            for seed in seeds:
                for ca in candidates:
                    key=(a,float(ca),seed,epochs)
                    if key in done:continue
                    r=train(x,y,seed,omega,epochs=epochs,lbfgs=lbfgs,fixed_spin=float(ca),eigenmode='joint')
                    rows.append(dict(a_true=a,candidate=float(ca),seed=seed,epochs=epochs,**r))
                    write_csv(path,rows);print('PROFILE',a,round(ca,3),seed,round(r['total_loss'],8),flush=True)
    if args.mode in ('all','starts'):
        path=Path('kerr_starts.csv');rows=[] if args.force else load_csv(path)
        done={(float(r['a_true']),float(r['a_init']),int(r['seed']),int(r['epochs'])) for r in rows}
        starts=[.1,.5,.9] if quick else args.starts
        for a in spins:
            y=reference(a,x,omega,22)
            for initial in starts:
                for seed in seeds:
                    key=(a,initial,seed,epochs)
                    if key in done:continue
                    r=train(x,y,seed,omega,spin_init=initial,epochs=epochs,lbfgs=lbfgs)
                    rows.append(dict(a_true=a,a_init=initial,seed=seed,epochs=epochs,absolute_error=abs(r['a_hat']-a),**r))
                    write_csv(path,rows);print('START',a,initial,seed,round(r['a_hat'],5),flush=True)
    if args.mode in ('all','figures','core','architecture','fixed_profile','starts'):
        summarize_core(load_csv(Path('kerr_core.csv')),epochs)
        summarize_architecture(load_csv(Path('kerr_architecture.csv')),epochs)
        figures(epochs,omega)

def figures(epochs,omega):
    rows=load_csv(Path('kerr_core.csv'))
    if rows:
        fig,ax=plt.subplots(figsize=(7,5));ax.plot([0,1],[0,1],'k--',label='Ideal')
        for method in ('spectral','adam','adam_lbfgs'):
            ss=[r for r in rows if r['method']==method and float(r['sigma'])==0 and int(r['epochs'])==epochs]
            xx=sorted({float(r['a_true']) for r in ss})
            if xx:
                yy=[np.array([float(r['a_hat']) for r in ss if float(r['a_true'])==a]) for a in xx]
                ax.errorbar(xx,[q.mean() for q in yy],yerr=[q.std(ddof=1) if len(q)>1 else 0 for q in yy],marker='o',capsize=3,label=method)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Reference Kerr angular spin',ylabel='Estimated spin',title='Spin-weighted Teukolsky angular recovery');ax.legend();save_plot(fig,'recovery')
        fig,ax=plt.subplots(figsize=(7,5))
        for method in ('spectral','adam','adam_lbfgs'):
            ss=[r for r in rows if r['method']==method and int(r['epochs'])==epochs]
            xx=sorted({float(r['sigma']) for r in ss})
            if xx:ax.plot(xx,[np.mean([float(r['absolute_error']) for r in ss if float(r['sigma'])==q]) for q in xx],'o-',label=method)
        ax.set(xlabel='Gaussian noise sigma',ylabel='Mean absolute spin error',title='Angular inversion: noise sensitivity');ax.legend();save_plot(fig,'noise')
    rows=load_csv(Path('kerr_architecture.csv'))
    if rows:
        ss=[r for r in rows if int(r['epochs'])==epochs];vs=sorted({r['variant'] for r in ss})
        if vs:
            fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(vs,[np.mean([float(r['absolute_error']) for r in ss if r['variant']==v]) for v in vs]);ax.set(ylabel='Mean absolute spin error',title='Teukolsky PINN architecture comparison');ax.tick_params(axis='x',labelrotation=20);save_plot(fig,'architecture')
    rows=load_csv(Path('kerr_fixed_profiles.csv'))
    if rows:
        fig,ax=plt.subplots(figsize=(7,5))
        ss=[r for r in rows if int(r['epochs'])==epochs]
        for a in sorted({float(r['a_true']) for r in ss}):
            rr=[r for r in ss if float(r['a_true'])==a];cc=sorted({float(r['candidate']) for r in rr}); vv=[np.array([float(r['total_loss']) for r in rr if abs(float(r['candidate'])-c)<1e-9]) for c in cc]
            ax.errorbar(cc,[v.mean() for v in vv],yerr=[v.std(ddof=1) if len(v)>1 else 0 for v in vv],marker='o',capsize=3,label=f'true a={a:g}')
        ax.set(xlabel='Fixed candidate spin',ylabel='Retrained PINN objective',title='Spin-weighted angular fixed-spin loss profiles');ax.legend();save_plot(fig,'fixed_profile')
    rows=load_csv(Path('kerr_starts.csv'))
    if rows:
        fig,ax=plt.subplots(figsize=(7,5));ax.plot([0,1],[0,1],'k--',label='Ideal')
        ss=[r for r in rows if int(r['epochs'])==epochs];truth=sorted({float(r['a_true']) for r in ss})
        for init in sorted({float(r['a_init']) for r in ss}):
            yy=[np.array([float(r['a_hat']) for r in ss if float(r['a_true'])==a and float(r['a_init'])==init]) for a in truth]
            ax.errorbar(truth,[v.mean() if len(v) else np.nan for v in yy],yerr=[v.std(ddof=1) if len(v)>1 else 0 for v in yy],marker='o',capsize=3,label=f'initial a={init:g}')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Reference spin',ylabel='Estimated spin',title='Teukolsky PINN initialization');ax.legend();save_plot(fig,'initialization')
    # Spectral profile of the exact same spin-weighted reference family.
    x=np.linspace(-.9,.9,81);true=.3;sigma=.01;seed=0
    rg=np.random.default_rng(seed+int(true*1000)+int(sigma*10000));y=reference(true,x,omega,22)+rg.normal(0,sigma,len(x))
    grid=np.linspace(LO,HI,100);sse=np.array([np.sum((reference(c,x,omega,22)-y)**2) for c in grid]); dev=(sse-sse.min())/sigma**2
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(grid,dev,label='Spectral profile deviance');ax.axhline(3.841459,ls='--',label='Approx. 95% threshold');ax.axvline(true,ls=':',label='True spin');ax.set(xlabel='Candidate spin',ylabel='Delta chi-squared',title='Teukolsky angular spectral identifiability');ax.legend();save_plot(fig,'profile')
    Path('kerr_profile.json').write_text(json.dumps(dict(a_true=true,sigma=sigma,best=float(grid[np.argmin(sse)]),approximate_region=[float(grid[dev<=3.841459].min()),float(grid[dev<=3.841459].max())],caution='uncalibrated bounded nonlinear diagnostic'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['all','core','architecture','fixed_profile','starts','figures'],default='all')
    p.add_argument('--quick',action='store_true',help='SMOKE TEST ONLY, not manuscript data')
    p.add_argument('--force',action='store_true')
    p.add_argument('--epochs',type=int,default=800)
    p.add_argument('--lbfgs',type=int,default=50)
    p.add_argument('--profile-points',type=int,default=9)
    p.add_argument('--spins',type=float,nargs='+',default=SPINS)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--noise',type=float,nargs='+',default=SIGMAS)
    p.add_argument('--starts',type=float,nargs='+',default=[.1,.3,.5,.7,.9])
    p.add_argument('--omega',type=float,default=.5)
    args=p.parse_args();os.chdir(Path(__file__).resolve().parent)
    if args.epochs<1 or args.lbfgs<0 or args.profile_points<3 or args.omega<=0:p.error('invalid parameters')
    begin=time.time();execute(args);print('FINISHED in',round(time.time()-begin,1),'seconds. Results in',Path.cwd())
