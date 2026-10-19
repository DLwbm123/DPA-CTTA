import errno,json,os,subprocess,sys,tempfile,unittest,select
from pathlib import Path
from unittest.mock import patch
from dpa_ctta.srg_ctta.storage_recovery import bytes_used,proc_record,exit_code_from_record,completion_certificate
class TestStorageRecovery(unittest.TestCase):
 def test_missing_directory_and_file_are_tolerated(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'keep').write_bytes(b'123')
   def walk(root,followlinks,onerror):
    onerror(FileNotFoundError(errno.ENOENT,'temporary directory gone'))
    yield d,[],['keep','gone']
   with patch('dpa_ctta.srg_ctta.storage_recovery.os.walk',walk):self.assertEqual(bytes_used(r),3)
 def test_other_os_errors_are_not_hidden(self):
  def walk(root,followlinks,onerror):
   onerror(PermissionError(errno.EACCES,'denied'))
   yield '',[],[]
  with patch('dpa_ctta.srg_ctta.storage_recovery.os.walk',walk):
   with self.assertRaises(PermissionError):bytes_used(Path('.'))
 def test_regular_files_only_no_symlink_recursion(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'one').write_bytes(b'123');(r/'link').symlink_to(r/'one');(r/'loop').symlink_to(r,target_is_directory=True)
   self.assertEqual(bytes_used(r),3)
 def test_proc_parsing_comm_spaces_and_identity(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'42').mkdir();f=['0']*50;f[0]='Z';f[19]='123';f[23]='1';f[49]=str(7<<8)
   (r/'42/stat').write_text('42 (name with spaces) '+' '.join(f));v=proc_record(42,r)
   self.assertEqual(exit_code_from_record(v,123),7);self.assertIsNone(exit_code_from_record(v,124));v['state']='R';self.assertIsNone(exit_code_from_record(v,123));self.assertIsNone(proc_record(1,r))
 def test_zero_is_only_terminal_observation(self):
  v=dict(state='R',start_ticks=5,wait_status=0)
  self.assertIsNone(exit_code_from_record(v,5));v['state']='Z';self.assertEqual(exit_code_from_record(v,5),0)
  v['wait_status']=9;self.assertEqual(exit_code_from_record(v,5),-9);self.assertIsNone(exit_code_from_record(None,5))
 def test_missing_certificate_never_admits(self):
  with tempfile.TemporaryDirectory() as d:self.assertFalse(completion_certificate(Path(d)))
 def test_reconstruction_only_changes_admission_and_receipt_metadata(self):
  import inspect
  from dpa_ctta.srg_ctta import topology_closed_run as original,topology_reconstruction_score as repair
  a=inspect.getsource(original.score);b=inspect.getsource(repair.score)
  new="assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code'] is None\n    admission=read(root/'RECONSTRUCTION_ADMISSION.json');assert admission['status']=='PASS' and admission['pidfd_exit_observed'] and not admission['original_exit0_gate_restored']\n    assert completion_certificate(root)"
  b=b.replace(new,"assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code']==0")
  b=b.replace('RESULTS_RECONSTRUCTED.json','RESULTS.json').replace('RECONSTRUCTED_SCORE_COMPLETE.json','SCORE_COMPLETE.json')
  b=b.replace('execution_exit0_certified=False,independent_scoring_reconstruction=True,','')
  self.assertEqual(a,b)
 def test_nonchild_exit_capture_does_not_fabricate_code(self):
  if not hasattr(os,'pidfd_open'):self.skipTest('pidfd unavailable')
  child="import time,sys;time.sleep(.2);sys.exit(7)"
  launcher="import subprocess,sys;p=subprocess.Popen([sys.executable,'-c',"+repr(child)+"],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(p.pid)"
  pid=int(subprocess.check_output([sys.executable,'-c',launcher],text=True));v=proc_record(pid)
  if v is None:self.skipTest('orphan already exited')
  fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN);self.assertTrue(poll.poll(3000));os.close(fd)
  code=exit_code_from_record(proc_record(pid),v['start_ticks'])
  if code is not None:self.assertEqual(code,7)
if __name__=='__main__':unittest.main()
