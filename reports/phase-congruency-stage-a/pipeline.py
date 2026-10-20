"""Pure synthetic qualification; all thresholds and families frozen before run."""
import os,sys,json,time
from pathlib import Path
import numpy as np
import phase

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def gate(root):
    def audit(event,args):
        if event.startswith('socket.'):raise PermissionError('network denied')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if str(p).startswith('/remote-home') and not p.is_relative_to(root):raise PermissionError('other data denied')
            flags=args[2]
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) and not p.is_relative_to(root):raise PermissionError('outside write denied')
    sys.addaudithook(audit)
    for name in ('rgb.png','gt.png','weights.pth','scores.json'):
        try:Path('/remote-home/forbidden/'+name).read_bytes()
        except PermissionError:pass
        else:raise AssertionError('access gate')
    return 4

def run(c):
    start=time.perf_counter();root=Path(c['root']).resolve();mode=os.environ['AUDIT_MODE'];assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    access=gate(root);store=root/'private/replay';store.mkdir(exist_ok=True)
    checks=[];failures=[];peaks=[];contrast_errors=[];polarity_errors=[];dft_errors=[];inputs=0
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok:failures.append(name)
    def evaluate(name,x):
        nonlocal inputs
        before=x.copy();r=phase.measure(x);check(name+'_input_immutable',np.array_equal(x,before));check(name+'_readonly',all(not a.flags.writeable for a in r.values()))
        payload=dict(image=x,**r);path=store/(name+'.npz')
        if mode=='synthetic1':np.savez_compressed(path,**payload)
        else:
            with np.load(path) as prior:
                check(name+'_independent_exact',set(prior.files)==set(payload) and all(np.array_equal(prior[k],v) for k,v in payload.items()))
        inputs+=1;return r
    for i,value in enumerate((0.,127.5,255.)):
        r=evaluate('constant'+str(i),np.full((512,512),value));check('constant_zero'+str(i),np.max(np.abs(r['pc']))<=1e-10)
    for axis in (0,1):
        ramp=np.arange(512)[None,:] if axis==1 else np.arange(512)[:,None]
        x=np.broadcast_to(np.where(ramp<256,63.75,191.25),(512,512)).copy()
        base=evaluate('step'+str(axis),x);reverse=evaluate('reverse'+str(axis),255-x)
        trace=(lambda a:a[256,32:-32]) if axis==1 else (lambda a:a[32:-32,256])
        peak=int(np.argmax(trace(base['pc'])))+32;peaks.append(peak)
        check('step_localized'+str(axis),abs(peak-256)<=3 and np.max(trace(base['pc']))>0)
        error=float(np.max(np.abs(base['pc']-reverse['pc'])));polarity_errors.append(error);check('polarity'+str(axis),error<=1e-8)
        for n,a in enumerate((.25,.5,1.)):
            r=evaluate('contrast'+str(axis)+'_'+str(n),127.5+a*(x-127.5));pk=int(np.argmax(trace(r['pc'])))+32
            error=float(np.max(np.abs(r['pc'][32:-32,32:-32]-base['pc'][32:-32,32:-32])));contrast_errors.append(error)
            check('contrast'+str(axis)+'_'+str(n),abs(pk-peak)<=1 and error<=.01)
    xx=np.arange(512)[None,:];x=np.broadcast_to(np.where((xx>=128)&(xx<384),191.25,63.75),(512,512)).copy();r=evaluate('double',x)
    for j in (128,384):
        z=r['pc'][256,j-3:j+4];peak=j-3+int(np.argmax(z));check('double_edge'+str(j),z.max()>0 and r['pc'][256,peak]>=max(r['pc'][256,peak-1],r['pc'][256,peak+1]))
    yy,xx=np.meshgrid(np.arange(31),np.arange(35),indexing='ij');a=np.sin(xx/3.7)+np.cos(yy/4.3)+.02*xx+.015*yy;a=(a-a.min())/(a.max()-a.min())*255
    evaluate('odd',a);actual,aux=phase._calculate(a,np.fft.fft2,np.fft.ifft2);reference,refaux=phase.direct_reference(a)
    for name in ('periodic','smooth','responses'):
        error=float(np.max(np.abs(aux[name]-refaux[name])));dft_errors.append(error);check('direct_dft_'+name,error<=1e-8)
    for name in actual:
        if actual[name].dtype==bool:continue
        error=float(np.max(np.abs(actual[name]-reference[name])));dft_errors.append(error);check('direct_dft_'+name,error<=1e-8)
    for n,bad in enumerate((np.zeros((2,2,2)),np.array([[0.,np.nan],[0.,0.]]),np.array([[0.,256.],[0.,0.]]))):
        try:phase.measure(bad)
        except ValueError:check('invalid'+str(n),True)
        else:check('invalid'+str(n),False)
    check('no_model_import','torch' not in sys.modules)
    result=dict(status='PASS' if not failures else 'FAILED',source_sha=c['source_sha'],pass_number=1 if mode=='synthetic1' else 2,checks=checks,check_count=len(checks),failures=failures,synthetic_inputs=inputs,synthetic_families=5,phase_peak_error_pixels_max=max(abs(p-256) for p in peaks),contrast_pc_error_max=max(contrast_errors),polarity_pc_error_max=max(polarity_errors),direct_dft_error_max=max(dft_errors),access_negative_checks=access,real_rgb_reads=0,label_reads=0,model_weight_reads=0,model_calls=0,gpu_calls=0,independent_replay=mode=='synthetic2',matlab_parity=False,body_seconds=time.perf_counter()-start)
    write(root/'public'/('QUALIFICATION1.json' if mode=='synthetic1' else 'QUALIFICATION2.json'),result)
    if failures:raise RuntimeError('frozen qualification failed: '+','.join(failures))
