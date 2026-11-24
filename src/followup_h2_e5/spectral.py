"""Separable spectral preconditioning for the same weighted ridge system."""
import torch


def separate(k):
    spectrum,vectors=torch.linalg.eigh(k.double())
    if float(spectrum[0]) <= 0.:
        raise ValueError("The spectral preconditioner needs strictly positive kernel axes")
    return spectrum.to(k.dtype),vectors.to(k.dtype).contiguous()


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


def solve_block(system,penalties,*,tol=5e-5,warm=None):
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
    warm=warm or {}
    for p in penalties:
        previous=warm.get(p)
        initial_point=None if previous is None else torch.where(
            system.root>0,previous.to(system.y.dtype)/system.root.clamp_min(1e-12),0.)
        initial,n,error=basis.solve(system,p,tol,initial=initial_point)
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
        out[p]=coef,n+m,error,"block_double_pcg"
    return out
