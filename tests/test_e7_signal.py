import unittest
import numpy as np
from scipy.special import expit
from dpa_ctta.srg_ctta.e7_math import masks,fields,entropy_from_logit,score_frame,stratify,NAMES

class E7Tests(unittest.TestCase):
    def test_masks_rng_counts_and_coordinates(self):
        a=list(masks(0,0));b=list(masks(0,0));c=list(masks(0,1))
        self.assertEqual(len(a),8)
        for (m,v),(mm,vv) in zip(a,b):
            self.assertTrue(np.array_equal(m,mm));self.assertEqual(v,vv)
            self.assertEqual(int(m.sum()),16*int(np.floor(16384*v['rho'])))
            self.assertEqual(len(v['patch_ids']),len(set(v['patch_ids'])))
            self.assertTrue(np.array_equal(m.reshape(128,4,128,4),np.broadcast_to(m[::4,::4][:,None,:,None],(128,4,128,4))))
        self.assertNotEqual(a[0][1],c[0][1])
    def test_entropy_limits_and_formula(self):
        z=np.array([-1000.,-10.,-1.,0.,1.,10.,1000.]);h=entropy_from_logit(z)
        self.assertTrue(np.isfinite(h).all());self.assertEqual(h[0],0);self.assertEqual(h[-1],0)
        self.assertAlmostEqual(h[3],np.log(2));self.assertTrue(np.allclose(h,h[::-1],rtol=0,atol=1e-15))
        for x in z[1:-1]:
            p=float(expit(x));expected=-p*np.log(p)-(1-p)*np.log1p(-p)
            self.assertAlmostEqual(float(entropy_from_logit(np.array(x))),expected,places=13)
        with self.assertRaises(ValueError):entropy_from_logit(np.array([np.nan]))
    def test_null_boundary_and_direction(self):
        z=np.array([[-3.,0.,3.],[-2.,0.,2.]]);p=expit(z);s=z>=0
        ff=fields(z,p,s);self.assertTrue(np.allclose(ff['MARS_FD'],ff['NULL_FUSION'],atol=1e-16,rtol=0))
        for q in (np.zeros_like(z),np.ones_like(z)):
            a=fields(z,q,s);self.assertEqual(a['CONSISTENCY_ONLY'][0,1],.5)
        self.assertTrue(fields(z,s.astype(float),s)['MASKED_MEAN_DIRECT'].min()>0)
        with self.assertRaises(ValueError):fields(z,p+1,s)
    def test_shift_and_mean_not_variance(self):
        z=np.ones((3,6));q=np.arange(18).reshape(3,6)/18;s=z>0
        a=fields(z,q,s);b=fields(z,np.roll(q,4,1),s)
        self.assertTrue(np.array_equal(a['RESPONSE_SHIFT'],b['MARS_FD']))
        self.assertTrue(np.array_equal(fields(z,(q+1-q)/2,s)['MARS_FD'],fields(z,np.full_like(q,.5),s)['MARS_FD']))
    def test_common_and_all_score_replay(self):
        h=w=64;s=np.zeros((2,h,w),bool);s[0]=True;s[1,:32]=True;b=s.copy();b[1,31,32]=False
        z=np.empty((h,w,2));z[:,:,0]=4;z[:,:,1]=np.where(s[1],3.,-3.)
        f=np.ones((h,w,32));f[:,:,0]=np.arange(w)[None,:]+1
        q=expit(z[:,:,1]);count=np.ones((h,w),np.uint8)*3
        a,c=score_frame(f,z,s,b,q,count);aa,cc=score_frame(f,z,s,b,q,count)
        self.assertEqual(a,aa);self.assertEqual(c,cc);self.assertEqual(c['common'],1)
        self.assertEqual(set(a[0]['scores']),set(NAMES));self.assertEqual(a[0]['query_mask_rate'],3/8)
        self.assertEqual(a[0]['scores']['MARS_FD'],a[0]['scores']['STRATIFIED_ROLL'])
    def test_toy_no_grad_source_and_mask(self):
        import torch
        model=torch.nn.Conv2d(3,2,1).eval().requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()};x=torch.ones(1,3,512,512)
        with torch.no_grad():
            for mask,meta in masks(0,0):
                xm=x.clone();xm[:,:,torch.from_numpy(mask)]=0;out=model(xm)
                self.assertFalse(out.requires_grad);self.assertEqual(int((xm==0).sum()),int(mask.sum())*3)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()));self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertFalse(torch.cuda.is_initialized())
    def test_pidfd_own_identity_signal_zero(self):
        import ctypes,os
        from dpa_ctta.srg_ctta.signal_common import identity
        lib=ctypes.CDLL(None,use_errno=True);before=identity(os.getpid());fd=lib.syscall(434,os.getpid(),0)
        self.assertGreaterEqual(fd,0)
        try:self.assertEqual(lib.syscall(424,fd,0,0,0),0);self.assertEqual(before,identity(os.getpid()))
        finally:os.close(fd)
