"""CPU regressions for atomic-file races and provenance-preserving continuation."""
import copy
import errno
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dpa_ctta.r1.evidence import output_bytes,write
from dpa_ctta.r1.supervise import supervise
from dpa_ctta.r2 import analyze,continuation as c
import test_r1_fixes as process_fixtures
from test_r2 import fixture


class ContinuationChecks(unittest.TestCase):
    def test_atomic_temporary_disappearance_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'records').write_bytes(b'1234');(root/'.write-race').write_bytes(b'xx')
            original=Path.stat
            def race(path,*args,**kwargs):
                if path.name=='.write-race':
                    path.unlink(missing_ok=True)
                    raise FileNotFoundError(errno.ENOENT,'atomic rename',str(path))
                return original(path,*args,**kwargs)
            with patch.object(Path,'stat',race):self.assertEqual(output_bytes(root),4)
            for error in (OSError(errno.EIO,'storage failure'),PermissionError('denied'),FileNotFoundError('ordinary file missing')):
                with patch.object(Path,'stat',side_effect=error),self.assertRaises(type(error)):output_bytes(root)

    def test_supervisor_skips_carried_jobs_preserves_rotation(self):
        case=process_fixtures.ProcessChecks()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);packet=case.packet();completed=['o0a0','o0a2']
            start,created=case.spawner(out,lambda s,n:{})
            result=supervise(out,packet,start,case.caps(),.01,completed=completed)
            expected={a['job_id']:a['worker'] for a in packet['schedule']['assignments'] if a['job_id'] not in completed}
            self.assertEqual({s['key']:s['worker'] for s,p in created if s['phase']=='formal'},expected)
            self.assertEqual(result['status'],'COMPUTE_COMPLETE');self.assertEqual(result['unstarted_jobs'],[])
            self.assertTrue(all(not (out/k).exists() for k in completed))

    def test_cross_run_closeout_original_binding_and_failure_retention(self):
        with tempfile.TemporaryDirectory() as tmp,fixture(Path(tmp)) as (source,assets):
            new=Path(tmp)/'continued';new.mkdir()
            # Change only synthetic run identities, never the procedural R1 control fixture.
            for path in source.rglob('*.json*'):
                path.write_text(path.read_text().replace('c'*32,c.SOURCE_RUN).replace('d'*40,c.SOURCE_CODE))
            original=c.read(source/'receipt.json');packet=c.read(source/'packet.private.json')
            entries=c.read(source/'matrix.processes.json')['processes']
            new_identity=dict(original['binding'],run_id='e'*32,code_sha='f'*40)
            for key in ['device0','device1',*c.PENDING]:
                shutil.copytree(source/key,new/key)
                for path in (new/key).glob('*.json*'):
                    path.write_text(path.read_text().replace(c.SOURCE_RUN,new_identity['run_id']).replace(c.SOURCE_CODE,new_identity['code_sha']))
            for key in c.PENDING[1:]:shutil.rmtree(source/key)
            (source/'o2a4/completion.json').unlink()
            (source/'o2a4/records.jsonl').write_text('{}\n'*576)
            failure=dict(binding=next(e['binding'] for e in entries if e['key']=='o2a4'),status='INCOMPLETE',records=576,
                         reason="No such file or directory: '/output/.write-race'",physical=dict(forwards=4608,backwards=576,base_adam=576,perturb=0,restore=0))
            write(source/'o2a4/failure.json',failure)
            old_entries=[copy.deepcopy(e) for e in entries if e['key'] not in c.PENDING[1:]]
            for e in old_entries:
                if e['key']=='o2a4':e.update(status='INCOMPLETE',exit_code=1)
            write(source/'matrix.processes.json',dict(binding=original['binding'],status='INCOMPLETE',processes=old_entries,exit_codes=[e['exit_code'] for e in old_entries],unstarted_jobs=list(c.PENDING[1:]),active_seconds=2,wall_seconds=1),replace=True)
            write(source/'processes.started.json',dict(binding=original['binding'],processes=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in old_entries]),replace=True)
            write(source/'dispatch.stopped.json',dict(binding=original['binding'],status='INCOMPLETE',reason='worker nonzero exit'))
            auth=dict(packet['authorization'],approved_code_sha=new_identity['code_sha'],continuation_source_binding=original['binding'],continuation_policy='preserve_completed_restart_failed_once')
            with patch.object(c,'matrix',analyze.matrix),patch.object(c,'science',analyze.science):
                continuation=c.inspect_source(source,assets,[6,7],auth)
                with self.assertRaises(PermissionError):c.inspect_source(source,assets,[6,7],packet['authorization'])
                caps=dict(trajectory_seconds=7200,wall_seconds=86000,active_seconds=86000,bytes=2*1024**3-continuation['source_bytes'])
                receipt=dict(original,binding=new_identity,continuation=continuation,runtime_caps=caps)
                write(new/'receipt.json',receipt)
                write(new/'packet.private.json',dict(packet,binding=new_identity,authorization=auth,out=str(new),continuation=continuation,runtime_caps=caps))
                new_entries=[json.loads(json.dumps(e).replace(c.SOURCE_RUN,new_identity['run_id']).replace(c.SOURCE_CODE,new_identity['code_sha'])) for e in entries if e['phase']=='smoke' or e['key'] in c.PENDING]
                write(new/'matrix.processes.json',dict(binding=new_identity,status='COMPUTE_COMPLETE',processes=new_entries,exit_codes=[0]*len(new_entries),unstarted_jobs=[],active_seconds=1,wall_seconds=1))
                write(new/'processes.started.json',dict(binding=new_identity,processes=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in new_entries]))
                result=analyze.recompute(new,assets)
                self.assertEqual(result['physical']['new_records'],240)
                self.assertEqual(result['continuation']['actual_total_physical']['base_adam'],240+576+56)
                self.assertEqual(len(result['continuation']['carried_jobs']),15)
                self.assertTrue((source/'dispatch.stopped.json').is_file())
                self.assertEqual(c.read(source/'o0a0/completion.json')['binding']['code_sha'],c.SOURCE_CODE)
                # A truncated inherited result cannot become complete in the new run.
                (source/'o0a0/records.jsonl').write_text('')
                with self.assertRaises(ValueError):analyze.recompute(new,assets)
                self.assertFalse(c.read(new/'current_result.json')['valid'])


if __name__=='__main__':unittest.main()
