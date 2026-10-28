import numpy as np
import pytest
from scripts.vad_research.m1_v3.calibration import NeighborLibrary, calibrate, normalize_features
from scripts.vad_research.m1_v3.build_audit import partition
from scripts.vad_research.m1_v3.audit_targets import verify_provenance, metrics

def library():
    features=np.eye(10)
    labels=np.tile([0.6,0.4,0.5],(10,1))
    oof=labels-np.tile([0.1,-0.1,0.05],(10,1))
    return NeighborLibrary(tuple(f'fit{i}' for i in range(10)),features,labels,oof)

def test_four_targets_fixed_library():
    lib=library(); base=np.array([[0.5,0.5,0.5]])
    query=np.eye(10)[[0]]
    out={m:calibrate(base,query,['audit'],lib,m)[0] for m in
         ('reference_target','simregmatch_calibration','residual_direct','residual_consistency')}
    np.testing.assert_allclose(out['reference_target'],base)
    np.testing.assert_allclose(out['simregmatch_calibration'],[[0.55,0.45,0.5]])
    np.testing.assert_allclose(out['residual_direct'],[[0.6,0.4,0.55]])
    np.testing.assert_allclose(out['residual_consistency'],out['residual_direct'])
    for value in out.values(): assert np.isfinite(value).all() and ((value>=0)&(value<=1)).all()

def test_direction_gate_rejects_conflict():
    lib=library(); oof=lib.oof_predictions.copy()
    oof[:4,0]=lib.labels[:4,0]-0.1; oof[4:,0]=lib.labels[4:,0]+0.1
    lib=NeighborLibrary(lib.ids,lib.features,lib.labels,oof)
    q=np.ones((1,10)); q=normalize_features(q)
    base=np.array([[0.5,0.5,0.5]])
    direct,_=calibrate(base,q,['audit'],lib,'residual_direct')
    consistent,details=calibrate(base,q,['audit'],lib,'residual_consistency')
    assert details['sign_gate'][0,0]<1e-12
    np.testing.assert_allclose(consistent[0,0],base[0,0])
    np.testing.assert_allclose(direct[0,0],base[0,0],atol=1e-12)

def test_no_query_self_neighbor_and_bad_features():
    lib=library()
    with pytest.raises(ValueError,match='overlap'):
        calibrate([[0.5]*3],np.eye(10)[[0]],['fit0'],lib,'reference_target')
    with pytest.raises(ValueError,match='Zero-norm'):
        normalize_features(np.zeros((1,3)))

def test_partition_and_oof_provenance():
    ids=[f'id{i:03d}' for i in range(500)]
    fit,audit,folds=partition(ids,0)
    fit_ids=[ids[i] for i in fit]; audit_ids=[ids[i] for i in audit]
    manifest={'seed':0,'reference_train_ids':fit_ids,'audit_ids':audit_ids,
              'feature_encoder':'unfitted pretrained RoBERTa, label-blind','folds':[]}
    for j,held in enumerate(folds):
        held_ids=[fit_ids[i] for i in held]
        manifest['folds'].append({'fold':j,'held_out_ids':held_ids,
            'train_ids':[item for item in fit_ids if item not in set(held_ids)]})
    data={'fit_ids':np.asarray(fit_ids),'audit_ids':np.asarray(audit_ids),
          'fit_labels':np.full((400,3),0.5),'audit_labels':np.full((100,3),0.5),
          'oof_predictions':np.full((400,3),0.5),'reference_audit':np.full((100,3),0.5)}
    verify_provenance(data,manifest)
    manifest['folds'][0]['train_ids'].append(manifest['folds'][0]['held_out_ids'][0])
    with pytest.raises(ValueError,match='leakage'):
        verify_provenance(data,manifest)

def test_metric_no_correction_serializable():
    import json
    y=np.full((100,3),0.4); base=np.full((100,3),0.5)
    report=metrics(y,base,base,np.zeros_like(base))
    assert report['direction_accuracy']==[None,None,None]
    json.dumps(report,allow_nan=False)
