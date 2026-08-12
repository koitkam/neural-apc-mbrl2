import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
# DISCRIMINATIVE mini RSSM: obs has a per-step random channel w that ONLY the
# posterior latent z can carry (h cannot predict it) -> if z collapses, the w
# channel recon FAILS (mse ~ var(w) ~ 1.0). Tests whether an explicit latent
# VARIANCE FLOOR cures the SimNorm collapse that tau+unimix did not.
D, K, C, H, FB = 3, 16, 16, 64, 0.5
def mlp(i, o, h=128, nl=2):
    L=[]; d=i
    for _ in range(nl-1): L+=[nn.Linear(d,h),nn.LayerNorm(h),nn.Mish()]; d=h
    L+=[nn.Linear(d,o)]; return nn.Sequential(*L)
class Mini(nn.Module):
    def __init__(s,kind,tau=1.0,unimix=0.0):
        super().__init__(); s.kind=kind; s.tau=tau; s.um=unimix
        s.enc=mlp(D,H); s.gru=nn.GRUCell(K*C+1,H); s.prior=mlp(H,K*C); s.post=mlp(2*H,K*C); s.dec=mlp(H+K*C,D)
    def forward(s,obs,u):
        B,T,_=obs.shape; h=torch.zeros(B,H); rec=[];zs=[];kls=[]
        for t in range(T):
            e=s.enc(obs[:,t]); plg=s.post(torch.cat([h,e],-1)).view(B,K,C); qlg=s.prior(h).view(B,K,C)
            sc=s.tau if s.kind=='simnorm' else 1.0
            pp=F.softmax(plg/sc,-1); qp=F.softmax(qlg/sc,-1)
            if s.um>0: pp=(1-s.um)*pp+s.um/C; qp=(1-s.um)*qp+s.um/C
            if s.kind=='cat': oh=F.one_hot(pp.argmax(-1),C).float(); z=(oh+pp-pp.detach()).reshape(B,K*C)
            else: z=pp.reshape(B,K*C)
            kl=(pp*(pp.clamp(min=1e-8).log()-qp.clamp(min=1e-8).log())).sum(-1).sum(-1)
            kls.append(kl); rec.append(s.dec(torch.cat([h,z],-1))); zs.append(z)
            h=s.gru(torch.cat([z,u[:,t:t+1]],-1),h)
        return torch.stack(rec,1),torch.stack(zs,1),torch.stack(kls,1)
def data(B=96,T=24):
    u=(torch.rand(B,T)>0.5).float()*2-1; s=torch.zeros(B); S=[]
    for t in range(T): s=0.9*s+0.1*u[:,t]; S.append(s)
    s=torch.stack(S,1); w=torch.randn(B,T)
    return torch.stack([s,u,w],-1), u
def run(kind,tau=1.0,um=0.0,vfloor=0.0,fc=0.0):
    m=Mini(kind,tau,um); opt=torch.optim.Adam(m.parameters(),2e-3)
    for _ in range(700):
        obs,u=data(); rec,zs,kls=m(obs,u)
        obs_v=obs.reshape(-1,D).var(0).mean().detach()
        util=zs.reshape(-1,K*C).var(0).mean()/obs_v
        loss=F.mse_loss(rec,obs)+0.5*torch.clamp(kls.mean(),min=FB)+fc*F.relu(vfloor-util)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        obs,u=data(256,24); rec,zs,kls=m(obs,u)
        recon_w=float(F.mse_loss(rec[...,2],obs[...,2])); evr=float(zs.reshape(-1,K*C).var(0).mean()/obs.reshape(-1,D).var(0).mean())
        return recon_w,float(F.mse_loss(rec,obs)),evr
print(f"{'config':34s} recon_w  recon_all  enc_var_ratio  (recon_w~1.0=z collapsed)", flush=True)
for tag,k,t,u,vf,fc in [('cat (baseline)','cat',1,0.01,0.0,0.0),('simnorm t0.5+unimix NO floor','simnorm',0.5,0.01,0.0,0.0),('simnorm t0.5+unimix +varfloor','simnorm',0.5,0.01,0.02,3.0)]:
    rw,ra,e=run(k,t,u,vf,fc); print(f"{tag:34s} {rw:.4f}   {ra:.4f}     {e:.4f}", flush=True)
