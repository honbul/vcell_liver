"""Predeclared family grids and inner-only continuation from boundary points."""
import hashlib
import json

SEEDS=(11711,11729,11743)
FAMILIES=("gbdt","film","attention","gnn","hybrid")
MSE=(1.,0.,0.)
PEARSON=(1.,.20,0.)
JOINT=(1.,.20,.15)


def tree(family,objective,depth,leaves,rate,frac,child,l2,trees=3000):
    return dict(family=family,backend="gbdt",objective=objective,depth=depth,leaves=leaves,
                learning_rate=rate,colsample=frac,min_child=child,lambda_l2=l2,
                trees=trees,hybrid=family=="hybrid")


def neural(family,width,depth,drop,wd,epochs,*,arch=None,loss=MSE,heads=4,graph="gine",msg=3,fm=None):
    shape=arch or family
    portable=(64,64) if shape=="attention" else (96,128)
    return dict(family=family,backend="torch",arch=shape,width=width,depth=depth,dropout=drop,
                weight_decay=wd,epochs=epochs,learning_rate=.00065,heads=heads,batch=portable,
                eval_batch=6144 if shape=="attention" else 16384,graph=graph,message_depth=msg,
                fm=fm,loss=list(loss),min_epochs=32,patience=24,check_every=4,hybrid=family=="hybrid")


def initial(family):
    if family=="gbdt":
        return [tree("gbdt",objective,d,l,r,f,c,p) for objective,d,l,r,f,c,p in (
            ("l2",4,31,.1,.35,70,2.),("l2",8,127,.05,.55,45,5.),
            ("huber",6,63,.05,.5,50,5.),("huber",10,255,.02,.6,35,12.),
            ("lambdarank",6,63,.1,.45,60,5.),("lambdarank",10,127,.05,.55,50,10.))]
    if family=="film":
        return [neural("film",w,d,p,reg,epoch) for w,d,p,reg,epoch in (
            (128,2,.10,.0001,60),(256,4,.25,.001,84),(512,4,.25,.01,108),(1024,6,.40,.05,128))]
    if family=="attention":
        return [neural("attention",w,d,p,reg,epoch,heads=h) for w,d,p,reg,epoch,h in (
            (128,2,.10,.001,60,4),(128,3,.25,.01,84,8),(256,4,.25,.01,108,8))]
    if family=="gnn":
        return [neural("gnn",w,d,p,.01,epoch,graph=g,msg=m,fm=f) for w,d,p,epoch,g,m,f in (
            (128,2,.10,60,"gine",3,None),(256,4,.25,84,"gatv2",4,None),
            (256,4,.25,84,"attentivefp",5,None),(256,4,.25,84,"gine",4,"clamp"),
            (256,4,.25,84,"gatv2",4,"unimol"))]
    if family=="hybrid":
        return [tree("hybrid","l2",6,63,.05,.5,60,10.),
                tree("hybrid","l2_lambdarank",8,127,.05,.55,60,15.),
                neural("hybrid",256,4,.25,.01,84,arch="film",loss=JOINT),
                neural("hybrid",128,2,.25,.01,84,arch="attention",loss=JOINT)]
    raise ValueError(f"Unknown family {family}")


def nearby(config, *, intensive):
    cfg=config.copy()
    if cfg["backend"]=="gbdt":
        return [cfg|{"depth":max(2,cfg["depth"]+direction*2),
                     "leaves":max(15,int((cfg["leaves"]+1)*2**direction-1)),
                     "learning_rate":cfg["learning_rate"]*rate,
                     "colsample":min(.95,max(.15,cfg["colsample"]+direction*.12)),
                     "min_child":max(20,int(cfg["min_child"]*child)),
                     "lambda_l2":cfg["lambda_l2"]*child}
                for direction,rate,child in ((-1,1.4,1.5),(1,.7,.7))]
    regularized=cfg|{"dropout":min(.65,cfg["dropout"]+.12),"weight_decay":cfg["weight_decay"]*3,
                     "loss":list(PEARSON),"balance":True}
    higher_width=cfg["width"]*2
    bigger=cfg|{"width":higher_width,"depth":cfg["depth"]+1,
                "epochs":min(192,cfg["epochs"]+24),"loss":list(JOINT),"balance":True}
    if cfg["arch"]=="attention" or cfg["family"]=="gnn":
        bigger["batch"]=(48,64)
        bigger["eval_batch"]=4096
    if intensive:
        return [regularized,bigger,cfg|{"parts":True,"loss":list(JOINT),"balance":True},
                cfg|{"loss":[1.,.35,.35],"balance":True,"batch":[96,96]}]
    return [regularized,bigger]


def identify(config):
    key=json.dumps(config,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def boundary_neighbors(config, fitted, max_iteration):
    """Expand any still-winning exterior point without reading outer labels."""
    cfg=config
    keys=("depth","leaves","learning_rate","colsample","min_child","lambda_l2") if cfg["backend"]=="gbdt" else (
        "width","depth","dropout","weight_decay","learning_rate")
    seen=[c for c in fitted if c["backend"]==cfg["backend"]]
    targets=[]
    for key in keys:
        x=cfg[key]
        axis={c[key] for c in seen}
        for direction in (-1,1):
            if not ((direction<0 and x==min(axis)) or (direction>0 and x==max(axis))):
                continue
            if key in ("width","leaves"):
                proposed=max(32,int(x*2**direction)) if key=="width" else max(7,int((x+1)*2**direction-1))
            elif key=="depth":
                proposed=max(1,x+direction)
            elif key=="dropout":
                proposed=min(.7,max(.025,x+.08*direction))
            elif key=="colsample":
                proposed=min(.98,max(.1,x+.1*direction))
            elif key=="min_child":
                proposed=max(10,int(x*1.4**direction))
            else:
                proposed=x*2.**direction
            if proposed!=x:
                altered=cfg|{key:proposed}
                if key=="width" and proposed>=512 and altered.get("arch")=="attention":
                    altered|={"batch":[32,48],"eval_batch":2048}
                targets.append(altered)
    iterations="trees" if cfg["backend"]=="gbdt" else "epochs"
    if max_iteration>=cfg[iterations]*.94:
        targets.append(cfg|{iterations:int(cfg[iterations]*1.5)})
    seen_ids={identify(c) for c in fitted}
    return [c for c in targets if identify(c) not in seen_ids]
