"""Old asset identity and new 16k execution must coexist without changing scope."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from test_core import fit_fixture
from dpa_ctta.r9_current_first.assets import validate_metadata
from dpa_ctta.r9_current_first.protocol import SPEC,SCREEN24_ARTIFACT_PROTOCOL_SHA,ROOT
from dpa_ctta.r9_current_first.training import SourceTrainer,MAX_STEPS

class AssetProtocol(unittest.TestCase):
    def test_screen24_metadata_with_full_16k_trainer_and_wrong_identity_rejected(self):
        raw=(ROOT/'src/dpa_ctta/r8_ba/protocol.json').read_bytes()+(ROOT/'docs/review/r8_screen24/SPEC.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),SCREEN24_ARTIFACT_PROTOCOL_SHA)
        canon=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()
            def save(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True))
                return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            configs={'A':dict(rank=32,film_amplitude=.3),'B':dict(rank=64,film_amplitude=.3,observer='global',aux_multiplier=1.)}
            common=dict(code_sha=SPEC['scientific_artifact_sha_from_screen24'],protocol_sha256=SCREEN24_ARTIFACT_PROTOCOL_SHA,refs={},checkpoint_sha256='a'*64)
            oracle=dict(common,amplitude=.3);scaler=dict(common,fold='fit',zero_film=True)
            old=dict(oracle_binding=canon(oracle),scaler_binding=canon(scaler))
            b=dict(configs=configs,refs={},checkpoint_sha256='a'*64,r8_assets=old)
            for kind in ('oracle','bases','scaler'):
                old[kind+'_root']=str(root/kind)
                if kind=='oracle':ident=dict(binding=old['oracle_binding'],amplitude=.3)
                elif kind=='scaler':ident=dict(binding=old['scaler_binding'],fold='fit',zero_film=True)
                else:
                    ident=dict(oracle,oracle_receipt_sha256=b['oracle_receipt_sha256']);old['bases_identity']=ident
                b[kind+'_receipt_sha256']=save(root/kind/(kind+'_complete.json'),{'identity':ident})['sha256']
            index=dict(common,source_jobs={f'{r}_FULL_{seed}':{'config':configs[r]} for r in configs for seed in (20260924,20260925)})
            b['screen24_index']=index;b['screen24_index_ref']=save(root/'index.json',index)
            validate_metadata(b)
            self.assertEqual(MAX_STEPS,16000)
            trainer=SourceTrainer(*fit_fixture(),20260924,{'test':'binding'},recipe='LEGACY')
            trainer.fit_step();self.assertEqual(trainer.steps,1)
            bad=copy.deepcopy(b);bad['r8_assets']['oracle_binding']='0'*64
            with self.assertRaisesRegex(ValueError,'prepared data'):validate_metadata(bad)
            bad=copy.deepcopy(b);bad['screen24_index']['protocol_sha256']='0'*64
            bad['screen24_index_ref']=save(root/'bad-index.json',bad['screen24_index'])
            with self.assertRaisesRegex(ValueError,'index artifact identity'):validate_metadata(bad)
            bad=copy.deepcopy(b);bad['scaler_receipt_sha256']=save(root/'scaler/scaler_complete.json',{'identity':{'binding':'wrong'}})['sha256']
            with self.assertRaisesRegex(ValueError,'receipt identity'):validate_metadata(bad)
