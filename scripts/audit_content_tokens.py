#!/usr/bin/env python3
"""CPU-only Qwen tokenizer checks; no model weight load or inference."""
from pathlib import Path
from tokenizers import Tokenizer
from agent_reliability.io import digest, file_hash, read_json
from prepare_shared_calibration import save
from prepare_four_attempt_protocol import checked


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path,required=True);args=parser.parse_args()
    public=Path('repro/reflexion-20261025');private=Path('runs/reflexion-content-intervention-20261028')
    registry=checked(public/'content-intervention-registry-20261028.json');interface=checked(public/'content-intervention-interface-20261028.json')
    variants=read_json(private/'variants.json');prompts=read_json(private/'interface-prompts.json')
    assert file_hash(private/'variants.json')==registry['private_variants_sha256']
    assert file_hash(private/'interface-prompts.json')==interface['private_prompt_file_sha256']
    tokenizer=Tokenizer.from_file(str(args.model_dir/'tokenizer.json'));rows=[]
    for r,v in zip(registry['entries'],variants):
        sizes={c:len(tokenizer.encode(t,add_special_tokens=False).ids) for c,t in v['variants'].items()}
        first={p['condition']:p for p in prompts if p['index']==r['index']}
        prompt_sizes={c:len(tokenizer.encode(p['wrapped_prompt'],add_special_tokens=False).ids) for c,p in first.items()}
        delta=sizes['neutralized']-sizes['paraphrase']
        assert abs(delta)<=3,(r['index'],sizes)
        assert abs(prompt_sizes['neutralized']-prompt_sizes['paraphrase'])<=3
        assert max(prompt_sizes.values())+256<=32768
        rows.append({'index':r['index'],'native_tokens':sizes,'first_prompt_tokens':prompt_sizes,
            'neutral_minus_paraphrase_native_tokens':delta,'first_prompt_sha256':{c:digest(p['wrapped_prompt']) for c,p in first.items()}})
    result=save(public/'content-intervention-tokens-20261028.json',{'registry_sha256':registry['sha256'],
        'interface_sha256':interface['sha256'],'tokenizer_file_sha256':file_hash(args.model_dir/'tokenizer.json'),
        'tokenizer_config_file_sha256':file_hash(args.model_dir/'tokenizer_config.json'),
        'source_sha256':file_hash(Path(__file__)),'rows':rows,'checks_passed':True,
        'maximum_pair_token_difference':max(abs(r['neutral_minus_paraphrase_native_tokens']) for r in rows),
        'matching':'Neutral/paraphrase native entry and first actor prompt lengths within3Qwen tokens; exact same focal source interval, not exact edit-distance equality',
        'model_calls':0,'model_weight_loads':0})
    print({'tokens':result['sha256'],'cases':len(rows),'max_difference':result['maximum_pair_token_difference'],'new_model_calls':0})


if __name__=='__main__':main()
