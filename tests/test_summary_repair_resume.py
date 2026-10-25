import asyncio
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import summary_pool as summary


def test_completed_bad_summary_rewinds_patient_descendants_without_removing_other_chunks(tmp_path):
    rounds=[[('bad',0,'','','notes'),('good',1,'','','notes')], [('bad',2,'','','later')]]
    for i in range(2):
        (tmp_path/f'round_{i:04d}.parquet').write_bytes(b'archival fixture')
    for i in range(3):
        path=tmp_path/'chunks'/'00000'/f'{i:09d}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}')
    results={0:('r','Wait, I should fix this.','none'),1:('r','Clinical history.','none'),2:('r','Later history.','bad prior')}
    complete,kept=summary.rewind_invalid_summaries(rounds,2,results,tmp_path)
    assert complete==0 and kept=={}
    assert (tmp_path/'chunks/00000/000000001.json').exists()
    assert not (tmp_path/'chunks/00000/000000000.json').exists()
    assert not (tmp_path/'chunks/00000/000000002.json').exists()
    assert len(list((tmp_path/'quality_rejected').glob('*/round_*.parquet')))==2


def test_drafting_summary_is_rejected_after_attempt_trace_is_saved(tmp_path):
    async def base(client,payload):
        return 'original reasoning','Final Polish: bad output',dict(finish_reason='stop',raw_text='raw')
    async def run():
        work=summary.traced_work(base,tmp_path,summary.CompletionSampling(model='synthetic'))
        with pytest.raises(summary.OutputValidationError):
            await work(SimpleNamespace(base_url='synthetic'),dict(chunk_index=0,prompt='original notes'))
    asyncio.run(run())
    saved=json.loads(next((tmp_path/'attempts').glob('*/*.json')).read_text())
    assert saved['reasoning']=='original reasoning' and saved['summary_quality']['flagged']


def test_static_server_retries_bad_summary_and_preserves_both_traces(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('summary_static_test',Path(__file__).parents[1]/'6_summarize_patients.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answers=iter(['Wait, I should rewrite this.','Clinical history.'])
    calls=[]
    async def create(**kwargs):
        calls.append(kwargs['prompt'])
        return SimpleNamespace(choices=[SimpleNamespace(text=next(answers),finish_reason='stop')])
    async def sleep(_):pass
    monkeypatch.setattr(module.asyncio,'sleep',sleep)
    monkeypatch.setattr(module,'postprocess_output',lambda raw,*args:('reasoning',raw))
    client=SimpleNamespace(base_url='synthetic',completions=SimpleNamespace(create=create))
    result=asyncio.run(module.single_inference_request(client,0,'original notes','model',1.0,100,64,'parser',None,
        max_retries=2,trace_dir=str(tmp_path)))
    assert result==(0,'reasoning','Clinical history.')
    assert calls==['original notes','original notes']
    traces=[json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert len(traces)==2 and all(t['reasoning']=='reasoning' for t in traces)
    assert sum(t['summary_quality']['flagged'] for t in traces)==1
