"""Own NumPy formulation of Kovesi monogenic phase congruency equations.

References: Peter Kovesi phasecongmono, perfft2, filtergrid, lowpassfilter.
Author methods/copyright acknowledged; no original implementation embedded.
Fixed Phase-Congruency-A-v1; no file, model, label, or mask API.
"""
import numpy as np
EPS=1e-4

def _calculate(image,forward,inverse):
    a=np.asarray(image)
    if a.ndim!=2 or min(a.shape)<2 or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a):raise ValueError('2D real grey required')
    a=np.asarray(a,np.float64)
    if not np.isfinite(a).all() or a.min()<0 or a.max()>255:raise ValueError('finite grey in [0,255] required')
    h,w=a.shape
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    edge=np.zeros_like(a);edge[0]=a[0]-a[-1];edge[-1]=-edge[0]
    edge[:,0]+=a[:,0]-a[:,-1];edge[:,-1]-=a[:,0]-a[:,-1]
    denom=2*(2-np.cos(2*np.pi*xx/w)-np.cos(2*np.pi*yy/h));denom[0,0]=1
    smooth=forward(edge)/denom;smooth[0,0]=0;periodic=forward(a)-smooth
    fy=np.fft.fftfreq(h);fx=np.fft.fftfreq(w);u,v=np.meshgrid(fx,fy);rad=np.hypot(u,v)
    low=1/(1+(rad/.45)**30);rad[0,0]=1
    riesz=(1j*u-v)/rad
    amp=np.zeros_like(a);sf=np.zeros_like(a);sh1=np.zeros_like(a);sh2=np.zeros_like(a);peak=np.zeros_like(a)
    responses=[]
    for scale in range(4):
        center=1/(3*2.1**scale);bank=np.exp(-np.log(rad/center)**2/(2*np.log(.55)**2))*low;bank[0,0]=0
        f=inverse(periodic*bank).real;hh=inverse(periodic*bank*riesz);h1=hh.real;h2=hh.imag
        an=np.sqrt(f*f+h1*h1+h2*h2)
        if scale==0:tau=np.median(an)/np.sqrt(np.log(4))
        amp+=an;sf+=f;sh1+=h1;sh2+=h2;peak=np.maximum(peak,an)
        responses.append(np.stack((f,h1,h2)))
    energy=np.sqrt(sf*sf+sh1*sh1+sh2*sh2)
    tau_total=tau*(1-(1/2.1)**4)/(1-1/2.1)
    threshold=float(tau_total*(np.sqrt(np.pi/2)+3*np.sqrt((4-np.pi)/2)))
    width=(amp/(peak+EPS)-1)/3;weight=1/(1+np.exp((.5-width)*10))
    ratio=energy/(amp+EPS);outside=float(max(0.,float(ratio.max()-1),float(-1-ratio.min())))
    if outside>1e-12:raise ArithmeticError('acos input beyond roundoff tolerance')
    pc=weight*np.maximum(1-1.5*np.arccos(np.clip(ratio,-1,1)),0)*np.maximum(energy-threshold,0)/(energy+EPS)
    if not np.isfinite(pc).all() or pc.min() < -1e-12 or pc.max()>1+1e-12:raise ArithmeticError('invalid PC')
    valid=np.hypot(sh1,sh2)>0
    if a.max()==a.min():pc=np.zeros_like(a)
    result=dict(pc=pc,amplitude=amp,energy=energy,threshold=np.array(threshold),direction_valid=valid,acos_overshoot=np.array(outside))
    for value in result.values():value.setflags(write=False)
    return result,dict(periodic=periodic,smooth=smooth,responses=np.stack(responses))

def measure(image):
    return _calculate(image,np.fft.fft2,np.fft.ifft2)[0]

def direct_reference(image):
    h,w=image.shape
    ey=np.exp(-2j*np.pi*np.outer(np.arange(h),np.arange(h))/h)
    ex=np.exp(-2j*np.pi*np.outer(np.arange(w),np.arange(w))/w)
    def forward(a):return ey@a@ex.T
    def inverse(a):return ey.conj().T@a@ex.conj()/(h*w)
    return _calculate(image,forward,inverse)
