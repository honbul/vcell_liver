"""Separable spectral preconditioning for the same weighted ridge system."""
import torch


def separate(k):
    spectrum,vectors=torch.linalg.eigh(k.double())
    if float(spectrum[0]) <= 0.:
        raise ValueError("The spectral preconditioner needs strictly positive kernel axes")
    return spectrum.to(k.dtype),vectors.to(k.dtype).contiguous()


class SpectralBasis:
    """Transform full sample grids without constructing a pair-by-pair Gram."""
    def __init__(self, system, cell=None, drug=None):
        self.system=system
        c,self.uc=cell or separate(system.cell)
        d,self.ud=drug or separate(system.drug)
        self.task=None
        self.spectrum=c[:,None]*d[None,:]
        if system.stacked:
            n=len(system.mass)
            mat=system.tau*torch.eye(n,dtype=c.dtype)+torch.ones((n,n),dtype=c.dtype)
            eigen,self.task=separate(mat)
            self.spectrum=eigen[:,None,None]*self.spectrum
        self.average=float(system.mass.mean())

    def forward(self, x):
        x=(self.uc @ x) @ self.ud.T
        return x if self.task is None else (self.task @ x.flatten(1)).reshape_as(x)

    def backward(self,x):
        if self.task is not None:
            x=(self.task.T @ x.flatten(1)).reshape_as(x)
        return (self.uc.T @ x) @ self.ud

    def solve(self,penalty,tol=5e-5,max_steps=320):
        """Solve the preconditioned primal system and verify the original dual residual."""
        w,y=self.system.mass,self.system.values
        rscale=max(1e-12,float(self.system.y.norm()))
        d=(self.average+penalty/self.spectrum).rsqrt()
        rhs=d*self.backward(w*y)
        diagonal=penalty*d.square()/self.spectrum
        point=torch.zeros_like(rhs)
        r=rhs.clone()
        p=rhs.clone()
        norm=(r*r).sum()
        delta=w-self.average

        def physical():
            fit=self.forward(d*point)
            original=self.system.root*(y-fit)/penalty
            err=(self.system.y-self.system.product(original)-penalty*original).norm().item()/rscale
            return original,err

        for step in range(1,max_steps+1):
            ap=p + d*self.backward(delta*self.forward(d*p))
            length=norm/(ap*p).sum().clamp_min(1e-30)
            point.add_(p*length)
            r.sub_(ap*length)
            update=(r*r).sum()
            if step%12==0 or float(update)<1e-17:
                val,e=physical()
                if e<=tol:
                    return val,step,e
                exact=rhs-diagonal*point-d*self.backward(w*self.forward(d*point))
                if step%48==0:
                    r=exact
                    p=exact
                    norm=(r*r).sum()
                    continue
            p=r+(update/norm.clamp_min(1e-30))*p
            norm=update
        val,err=physical()
        return val,max_steps,err


def solve_hybrid(system,penalties,*,tol=5e-5,cell=None,drug=None):
    """Finish every spectral warm start on the original weighted problem."""
    from .kernel import pcg,solve_shifts

    if system.mass.numel() < 95000:
        return solve_shifts(system,penalties,tol=tol)
    try:
        basis=SpectralBasis(system,cell,drug)
    except ValueError:
        return solve_shifts(system,penalties,tol=tol)
    out={}
    high=None
    for p in penalties:
        initial,n,error=basis.solve(p,tol,max_steps=156)
        if error <= tol:
            out[p]=system.root*initial,n,error,"spectral"
            continue
        if high is None:
            high=system.higher_precision()
        coef,m,error=pcg(high,p,tol*.8,initial=initial.double())
        out[p]=coef.to(system.y.dtype),n+m,error,"spectral_pcg"
    return out


class BlockSpectral:
    """Restricted spectral block-Jacobi preconditioner on each observed screen grid."""
    def __init__(self,system):
        self.stacked=system.stacked
        self.parts=[]
        arrays=system.mass if self.stacked else system.mass[None]
        for w in arrays:
            cells=torch.nonzero(w.any(1))[:,0]
            drugs=torch.nonzero(w.any(0))[:,0]
            coords=(cells[:,None],drugs[None,:])
            weights=w[coords]
            c,uc=separate(system.cell[cells[:,None],cells[None,:]])
            # Per-screen row weighting is constant within each drug on observed rows.
            # Preserve that exact scale in each local spectral preconditioner.
            drug_scale=(weights.sum(0)/(weights>0).sum(0)).sqrt()
            block=system.drug[drugs[:,None],drugs[None,:]]
            d,ud=separate(block*drug_scale[:,None]*drug_scale[None,:])
            diagonal=(1.+system.tau)*(c[:,None]*d[None,:])
            self.parts.append((coords,weights>0,uc,ud,diagonal))

    def build(self,ridge):
        pieces=[(*entry[:-1],(entry[-1]+ridge).reciprocal()) for entry in self.parts]

        def inverse(r):
            view=r if self.stacked else r[None]
            z=torch.zeros_like(view)
            for i,(coords,mask,uc,ud,diagonal) in enumerate(pieces):
                v=(uc.T @ view[i][coords]) @ ud
                ans=(uc @ (v*diagonal)) @ ud.T
                z[i][coords]=ans*mask
            return z if self.stacked else z[0]
        return inverse

    def solve(self,system,ridge,tol=5e-5,max_steps=550,initial=None):
        R=self.build(ridge)
        y=system.y
        scale=max(float(y.norm()),1e-12)
        if initial is None:
            x,r=torch.zeros_like(y),y.clone()
        else:
            x=initial.clone()
            r=y-system.product(x)-ridge*x
        e=float(r.norm())/scale
        if e<=tol:
            return x,0,e
        p=R(r)
        energy=(p*r).sum()
        for step in range(1,max_steps+1):
            a=system.product(p)+ridge*p
            n=energy/(p*a).sum().clamp_min(1e-30)
            x.add_(p*n)
            r.sub_(a*n)
            if float(r.norm()) <= tol*scale:
                e=(y-system.product(x)-ridge*x).norm().item()/scale
                if e<=tol:
                    return x,step,e
                if float(r.norm()) <= .14*tol*scale:
                    return x,step,e
            z=R(r)
            new=(z*r).sum()
            p=z+(new/energy.clamp_min(1e-30))*p
            energy=new
        return x,max_steps,(y-system.product(x)-ridge*x).norm().item()/scale


def solve_block(system,penalties,*,tol=5e-5):
    """Precondition observational PCG without changing its target system."""
    from .kernel import pcg,solve_shifts

    if system.mass.numel() < 95000:
        return solve_shifts(system,penalties,tol=tol)
    try:
        basis=BlockSpectral(system)
    except ValueError:
        return solve_shifts(system,penalties,tol=tol)
    out={}
    high,block_high=None,None
    for p in penalties:
        initial,n,error=basis.solve(system,p,tol)
        if error<=tol:
            out[p]=system.root*initial,n,error,"block_pcg"
            continue
        if high is None:
            high=system.higher_precision()
            block_high=BlockSpectral(high)
        alpha,m,error=block_high.solve(high,p,tol*.8,initial=initial.double())
        if error>tol:
            coef,more,error=pcg(high,p,tol*.8,initial=alpha)
            m+=more
        else:
            coef=high.root*alpha
        out[p]=coef.to(system.y.dtype),n+m,error,"block_double_pcg"
    return out
