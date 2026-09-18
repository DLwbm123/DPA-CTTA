"""Synthetic fixed-version layouts and failure accounting. No production publisher."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_scalar as scalar
from core import digest, install_guard, read, regular
from pinned_io import Reader, check_layout, layout, record_file, source_mapping
from run import check_snapshot

ROOT=Path(__file__).resolve().parents[2]
ADD=read(ROOT/'analysis/r6d_posthoc_v1/io_addendum/R6D_IO_ADDENDUM.json')


def make_layout(parent):
    source=Path(parent)/'source';source.mkdir()
    version=ADD['pointer']['required_result_directory'];v=source/version;v.mkdir(parents=True)
    raw=b'{"fixed_aggregate": true, "raw_spacing":  2}\n';(v/'public_aggregate.json').write_bytes(raw)
    pointer=dict(valid=True,status='R6A_COMPLETE_NO_ADVANCE',binding=ADD['historical_binding'],result_directory=version)
    (source/'current_result.json').write_text(json.dumps(pointer,indent=2)+'\n')
    (source/'current').symlink_to(version,target_is_directory=True)
    (source/'public_aggregate.json').symlink_to('current/public_aggregate.json')
    return source,v,raw


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.parent=Path(self.temp.name)
        self.source,self.version,self.raw=make_layout(self.parent);self.reader=Reader(self.source)
    def tearDown(self):self.temp.cleanup()

    def change_pointer(self,**changes):
        p=self.source/'current_result.json';x=read(p);x.update(changes);p.write_text(json.dumps(x))

    def test_double_alias_regular_snapshot_and_full_analysis_fixture(self):
        before,raw=layout(self.reader,ADD)
        snapshot=self.parent/'snapshot';snapshot.mkdir()
        info=self.reader.record(ADD['aggregate_input']['source_relative_path'],snapshot/'public_aggregate.json')
        self.assertEqual((snapshot/'public_aggregate.json').read_bytes(),self.raw)
        check_snapshot({'public_aggregate.json':info},snapshot)
        self.assertEqual(check_layout(self.reader,ADD,before),before)
        self.assertEqual(raw,(self.source/'current_result.json').read_bytes())
        scalar.ScalarTests.setUpClass()
        scalar.ScalarTests('test_complete_scalar_export_fixture').test_complete_scalar_export_fixture()

    def test_source_regular_guard_rejects_link_hardlink_fifo_directory(self):
        for kind in ['symlink','hardlink','fifo','directory']:
            p=self.parent/kind
            if kind=='symlink':p.symlink_to(self.version/'public_aggregate.json')
            elif kind=='hardlink':os.link(self.version/'public_aggregate.json',p)
            elif kind=='fifo':os.mkfifo(p)
            else:p.mkdir()
            with self.assertRaises((ValueError,OSError)):record_file(p)
            if p.is_dir():p.rmdir()
            else:p.unlink()

    def test_snapshot_guard_rejects_link_hardlink_fifo_directory(self):
        info=self.reader.record(ADD['aggregate_input']['source_relative_path'])
        snap=self.parent/'snap';snap.mkdir();p=snap/'data'
        for kind in ['symlink','hardlink','fifo','directory']:
            if kind=='symlink':p.symlink_to(self.version/'public_aggregate.json')
            elif kind=='hardlink':os.link(self.version/'public_aggregate.json',p)
            elif kind=='fifo':os.mkfifo(p)
            else:p.mkdir()
            with self.assertRaises((ValueError,OSError)):check_snapshot({'data':info},snap)
            if p.is_dir():p.rmdir()
            else:p.unlink()

    def test_pointer_absolute_parent_extra_component_and_wrong_version(self):
        for path in ['/tmp/out','results/../x','results/'+'0'*32+'/more','results/'+'f'*32]:
            self.change_pointer(result_directory=path)
            with self.assertRaises(ValueError):layout(self.reader,ADD)

    def test_pointer_complete_binding_and_status(self):
        for changes in [dict(valid=False),dict(status='PASS'),dict(binding=dict(ADD['historical_binding'],run_id='wrong')),
                        dict(binding={k:v for k,v in ADD['historical_binding'].items() if k!='production_fingerprint'})]:
            self.change_pointer(valid=True,status='R6A_COMPLETE_NO_ADVANCE',binding=ADD['historical_binding'])
            self.change_pointer(**changes)
            with self.assertRaises(ValueError):layout(self.reader,ADD)

    def test_pointer_itself_symlink_or_hardlink(self):
        p=self.source/'current_result.json';copy_path=self.parent/'pointer_copy';copy_path.write_bytes(p.read_bytes());p.unlink();p.symlink_to(copy_path)
        with self.assertRaises(ValueError):layout(self.reader,ADD)
        p.unlink();os.link(copy_path,p)
        with self.assertRaises(ValueError):layout(self.reader,ADD)

    def test_alias_outside_loop_dangling_mismatch(self):
        p=self.source/'current'
        for text in ['../outside','current','results/'+'a'*32]:
            p.unlink();p.symlink_to(text)
            with self.assertRaises(ValueError):layout(self.reader,ADD)

    def test_aggregate_alias_wrong_text_or_regular(self):
        p=self.source/'public_aggregate.json';p.unlink();p.symlink_to(ADD['aggregate_input']['source_relative_path'])
        with self.assertRaises(ValueError):layout(self.reader,ADD)
        p.unlink();p.write_bytes(b'{}')
        with self.assertRaises(ValueError):layout(self.reader,ADD)

    def test_dangling_bound_target(self):
        (self.version/'public_aggregate.json').unlink()
        with self.assertRaises(FileNotFoundError):layout(self.reader,ADD)

    def test_results_directory_symlink(self):
        p=self.source/'results';moved=self.parent/'moved';p.rename(moved);p.symlink_to(moved,target_is_directory=True)
        with self.assertRaises(OSError):layout(self.reader,ADD)

    def test_version_directory_symlink(self):
        moved=self.parent/'moved';self.version.rename(moved);self.version.symlink_to(moved,target_is_directory=True)
        with self.assertRaises(OSError):layout(self.reader,ADD)

    def test_target_symlink_and_hardlink(self):
        p=self.version/'public_aggregate.json';copy_path=self.parent/'copy';copy_path.write_bytes(self.raw);p.unlink();p.symlink_to(copy_path)
        with self.assertRaises(ValueError):layout(self.reader,ADD)
        p.unlink();os.link(copy_path,p)
        with self.assertRaises(ValueError):layout(self.reader,ADD)

    def test_multiple_versions_never_select_newer(self):
        other=self.source/'results'/('a'*32);other.mkdir();(other/'public_aggregate.json').write_text('{"different":true}')
        os.utime(other,(2000000000,2000000000))
        bound,_=layout(self.reader,ADD);self.assertEqual(bound['pointer']['result_directory'],ADD['pointer']['required_result_directory'])
        info,data=self.reader.record(ADD['aggregate_input']['source_relative_path'],capture=True);self.assertEqual(data,self.raw)

    def test_pointer_changed_since_binding(self):
        before,_=layout(self.reader,ADD);self.change_pointer(extra='new bytes')
        with self.assertRaises(ValueError):check_layout(self.reader,ADD,before)

    def test_link_recreated_same_text_is_detected(self):
        before,_=layout(self.reader,ADD);p=self.source/'current';p.unlink();p.symlink_to(ADD['pointer']['required_result_directory'])
        with self.assertRaises(ValueError):check_layout(self.reader,ADD,before)

    def test_payload_changes_during_copy(self):
        def change():
            with (self.version/'public_aggregate.json').open('ab') as f:f.write(b'changed')
        # One chunk callback only, otherwise a growing stream need not terminate.
        done=[]
        def once():
            if not done:done.append(True);change()
        with self.assertRaises(ValueError):self.reader.record(ADD['aggregate_input']['source_relative_path'],self.parent/'snapshot',on_chunk=once)

    def test_snapshot_pollution_is_detected(self):
        snap=self.parent/'snap';snap.mkdir();p=snap/'public_aggregate.json'
        info=self.reader.record(ADD['aggregate_input']['source_relative_path'],p);p.write_bytes(b'{}')
        with self.assertRaises(ValueError):check_snapshot({'public_aggregate.json':info},snap)

    def test_anchored_open_mapping_and_guard_alias_deny(self):
        code='''import tempfile,pathlib,json,os
from test_io import make_layout,ADD
from pinned_io import Reader,layout,source_mapping
from core import install_guard
p=pathlib.Path(tempfile.mkdtemp());source,version,raw=make_layout(p);out=p/'out';out.mkdir();access={};reader=Reader(source,access)
mapping=source_mapping(ADD,(0,1,4));files=[reader.root/r for r in mapping.values()];dirs={reader.root,reader.root/'results',reader.root/ADD['pointer']['required_result_directory']}
install_guard(files,[],out,dirs,access)
bound,_=layout(reader,ADD)
record,data=reader.record(ADD['aggregate_input']['source_relative_path'],capture=True);assert data==raw
for action in [lambda:(source/'public_aggregate.json').read_bytes(),lambda:(source/'current'/'public_aggregate.json').read_bytes(),lambda:(source/'current_result.json').write_text('bad'),lambda:__import__('torch')]:
 try:action()
 except PermissionError:pass
 else:raise AssertionError('forbidden alias/source/model access permitted')
print('anchored target allowed; aliases/source writes/model import denied')
'''
        r=subprocess.run([sys.executable,'-B','-'],input=code,text=True,capture_output=True,env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent)))
        self.assertEqual(r.returncode,0,r.stderr)

    def test_preflight_and_after_failure_keep_first_error_and_cost(self):
        code='''import tempfile,pathlib,json
from test_io import make_layout,ROOT
import run
p=pathlib.Path(tempfile.mkdtemp());source,_,_=make_layout(p);work=p/'work'
def fail_after(*args):raise RuntimeError('AFTER_SENTINEL')
run.check_layout=fail_after
try:run.main(dict(code=str(ROOT),source=str(source),work=str(work),analysis_sha='fixture'))
except RuntimeError:pass
log=json.loads((work/'public/RUN_LOG.json').read_text())
assert log['analysis_status']=='R6D_INCOMPLETE' and not log['analysis_executed']
assert log['errors'][0]['type']=='FileNotFoundError' and log['errors'][0]['phase']=='primary'
assert log['errors'][1]['type']=='RuntimeError' and log['errors'][1]['phase']=='after_layout'
assert log['wall_seconds']>0 and log['peak_rss_platform_units']>0
assert 'AFTER_SENTINEL' not in (work/'failure_00_primary.private.log').read_text()
assert 'AFTER_SENTINEL' in (work/'failure_01_after_layout.private.log').read_text()
audit=json.loads((work/'public/READONLY_AUDIT.json').read_text());assert audit['source_bytes_unchanged'] is None
print('primary and after error preserved; preflight telemetry measured')
'''
        r=subprocess.run([sys.executable,'-B','-'],input=code,text=True,capture_output=True,env=dict(os.environ,PYTHONPATH=str(Path(__file__).parent)))
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
