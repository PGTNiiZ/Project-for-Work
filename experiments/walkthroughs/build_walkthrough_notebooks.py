"""Generate the R0–R6 and GB teaching notebooks.

Notebooks import the production modules and print their real source code, so
their explanations cannot drift from the implementation that produced results.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent

BOOT = """from pathlib import Path
import sys, json, inspect
import pandas as pd
from IPython.display import Markdown, display

def find_root():
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents,
                  Path(r'D:/66070260-Year3_Term2/Project1/Code')]
    for candidate in candidates:
        if (candidate / 'exp_lib.py').exists(): return candidate
    raise FileNotFoundError('Project root containing exp_lib.py was not found')

ROOT = find_root(); EXP = ROOT / 'experiments'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def source(module, *names):
    for name in names:
        display(Markdown(f'### `{module.__name__}.{name}`'))
        print(inspect.getsource(getattr(module, name)))

def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))

print('Project root:', ROOT)
"""


def md(text: str) -> tuple[str, str]:
    return ('markdown', text)


def code(text: str) -> tuple[str, str]:
    return ('code', text)


SHARED = [
    md('# 00 — Shared protocol\n\nอ่าน notebook นี้ก่อน R0–R6 เพื่อเข้าใจ candidate blocking, entity-aware split, metrics และ evaluation harness ที่ทุก experiment ใช้ร่วมกัน.'),
    code(BOOT),
    md('## 1. Candidate pairs และ fixed errors\n\nระบบสร้าง candidate pairs ก่อน score. คู่จริงที่ไม่เคยผ่าน blocking จะไม่มี probability ให้ model แก้ได้ จึงถูกนับเป็น fixed FN ใน harness.'),
    code("import exp_lib\nsource(exp_lib, 'build_cache', 'split_constants', 'evaluate')\ncache = exp_lib.build_cache(); consts = exp_lib.split_constants(cache)\nprint('Scored cache:', f'{len(cache):,} rows'); print('Test constants:', consts['test'])"),
    md('## 2. Nested entity-aware split\n\nแบ่งตาม entity/person ไม่ใช่แบ่งทีละ pair: `model_train → model_calibration → ga_validation → sealed test`. คู่คร่อม role ถูก drop เพื่อกัน leakage.'),
    code("source(exp_lib, 'build_nested_entity_split', 'nested_split_constants')\nassigned, manifest = exp_lib.build_nested_entity_split(cache, seed=42)\nprint(json.dumps(manifest, ensure_ascii=False, indent=2))"),
    md('## 3. Metrics\n\n`cost = 5×FP + 1×FN + 0.02×REVIEW`; การรวมคนผิดมีน้ำหนักสูงกว่า missed match. REVIEW ไม่ใช่ error แต่เป็นภาระงานคนตรวจ.'),
]

R0 = [
    md('# R0 — Production baseline\n\nR0 คือ 17 handcrafted features และกฎ production เดิม เป็นจุดตั้งต้นสำหรับวัดผลของ MiniLM และ GA.'),
    code(BOOT),
    md('## 1. Features เดิม 17 ตัว\n\nประกอบด้วยความคล้าย username/fullname/bio/location, platform และรูปแบบความยาวของชื่อ.'),
    code("import exp_r2_bert_feature as r2\nfor i, name in enumerate(r2.FEATURE17_COLS, 1): print(f'{i:02d}. {name}')\nsource(r2, 'compute_features17')"),
    md('## 2. Evaluation และผล baseline\n\nHarness จะบวก exact-tier และ blocking-missed FN แบบคงที่ ทำให้เปรียบเทียบ R0–R6 ได้แฟร์.'),
    code("import exp_lib\nr = read_json('experiments/r1_results.json')\ndisplay(pd.DataFrame([r['rules']['test']['R0_production']]))\nsource(exp_lib, 'evaluate')"),
]

R1 = [
    md('# R1 — GA re-decision\n\nR1 ไม่ train model ใหม่ แต่ให้ GA ค้นหากฎ MATCH/REVIEW/NO_MATCH จาก score เดิมของ R0.'),
    code(BOOT),
    md('## 1. Genome และ decision rule\n\nGenome = `t_m, t_r, c_promote, c_demote`. Probability กำหนดช่วงหลัก และ `name_sim` ช่วยตัดสินคู่ก้ำกึ่ง.'),
    code("import exp_r1_ga_redecision as r1\nsource(r1, 'decide_code', 'cost_from_counts', 'run_ga')"),
    md('## 2. Crossover และ mutation\n\nGA สร้างประชากรของ rule, วัด validation cost, เก็บ elite, crossover และ mutate. Test ไม่ถูกใช้เลือก genome.'),
    code("source(r1, 'uniform_crossover', 'mutate')\nr = read_json('experiments/r1_results.json')\ndisplay(pd.DataFrame(r['rules']['test']).T); print('Legacy genome:', r['best_genome'])"),
    md('## 3. Nested automation result\n\nการรันล่าสุดใช้หลาย GA seeds และเลือกจาก validation cost เท่านั้น.'),
    code("s = read_json('experiments/automation/r1_nested_primary_20260809/summary.json')\ndisplay(pd.DataFrame(s['trials'])); print(json.dumps(s['aggregates']['A_current'], ensure_ascii=False, indent=2))"),
    code("# Optional full rerun:\n# import subprocess\n# subprocess.run([sys.executable, str(ROOT/'run_ga_experiments.py'), '--experiment', 'r1', '--seeds', '7', '42', '123', '999', '2025'], check=True)"),
]

R2 = [
    md('# R2 — MiniLM cosine feature\n\nR2 เพิ่ม semantic similarity จาก `all-MiniLM-L6-v2` เป็น feature ที่ 18 แล้ว train IdentityMLP ใหม่ แต่ยังใช้ threshold มือเดิม.'),
    code(BOOT),
    md('## 1. Profile text → embedding → cosine\n\nMiniLM encode `fullName + bio + location` ต่อ profile แล้ว cosine ของ profile pair ถูกเก็บเป็น `bert_cos`.'),
    code("import exp_r2_bert_feature as r2\nsource(r2, 'build_profile_texts', 'encode_profiles', 'compute_bert_cos')\nbert = pd.read_parquet(EXP/'pair_bert_cos.parquet'); display(bert.head()); print(len(bert))"),
    md('## 2. 18 features และ IdentityMLP\n\n17 features เดิม + `bert_cos`; train, calibration และ test แยก role เพื่อกัน leakage.'),
    code("source(r2, 'IdentityMLP', 'train_mlp', 'train_r2_probabilities')"),
    md('## 3. Manual decision\n\nR2 ใช้ `MATCH=0.98`, `REVIEW=0.95`; R3 จะใช้ probability เดียวกัน แต่เปลี่ยน decision layer เป็น GA.'),
    code("source(r2, 'decide_manual', 'run_r2')\nr = read_json('experiments/r2_results.json'); display(pd.DataFrame(r['splits']).T)"),
]

R3 = [
    md('# R3 — MiniLM + IdentityMLP + GA\n\nR3 ใช้ probability จาก R2 เดิม แล้วให้ GA จูนกฎตัดสิน จึงแยกผลของ representation ออกจาก decision layer ได้.'),
    code(BOOT),
    md('## 1. Reuse R2 probability และ optimize บน ga_validation\n\nGA ไม่ train MLP ใหม่ และไม่มีการอ่าน test label ระหว่างการเลือก genome.'),
    code("import exp_r2_bert_feature as r2\nimport exp_r1_ga_redecision as ga\nsource(r2, 'run_r3'); source(ga, 'run_ga', 'decide_code')"),
    md('## 2. Primary result และ GA seeds\n\nค่าเฉลี่ยข้าม seed อาจมีทศนิยมใน TP/REVIEW; ไม่ได้หมายถึงมีคู่ครึ่งคู่.'),
    code("s = read_json('experiments/automation/r3_primary_20260809/summary.json')\ndisplay(pd.DataFrame(s['trials'])); print(json.dumps(s['aggregates']['A_current'], ensure_ascii=False, indent=2))"),
    md('## 3. Robustness ของ model seed\n\nเปรียบเทียบ model seed แยกจาก GA seed เพื่อดูว่าผลขึ้นกับ random initialization มากเกินไปหรือไม่.'),
    code("r = read_json('experiments/automation/r0_r3_report_20260809/r0_r3_report.json')\nprint(json.dumps(r['model_seed_robustness'], ensure_ascii=False, indent=2))"),
]

R4 = [
    md('# R4 — Bloom filter + manual thresholds\n\nR4 แทน similarity ของชื่อ plaintext ด้วย Dice similarity บน Bloom-filter bigrams เพื่อดู privacy/accuracy trade-off.'),
    code(BOOT),
    md('## 1. Bigram → Bloom bit vector → Dice similarity\n\nL เล็กทำให้ collision มากขึ้น: ปกปิดรายละเอียดมากขึ้น แต่อาจลดพลังแยกคู่.'),
    code("import exp_r4_bloom_privacy as r4\nsource(r4, 'bigrams', 'bloom_packed', 'dice_chunked', 'compute_pair_bloom_features')"),
    md('## 2. Train probabilities และ threshold มือ\n\nทดลอง L = 2000, 1000, 500, 250 โดยใช้ threshold เดิมก่อน เพื่อ isolate ผลของ representation.'),
    code("source(r4, 'build_probabilities_for_L', 'eval_manual')\nr = read_json('experiments/r4_privacy_tradeoff.json')\nrows = [{'L': int(L), **x['R4_manual']['test']} for L, x in r['per_L'].items()]\ndisplay(pd.DataFrame(rows).sort_values('L', ascending=False))"),
]

R5 = [
    md('# R5 — Bloom filter + GA\n\nR5 ใช้ probability ของ R4 ในแต่ละ Bloom length แต่ให้ GA แทน threshold มือ.'),
    code(BOOT),
    md('## 1. Privacy rule\n\nGA ใช้ `name_sim_bloom` ไม่ใช่ Jaro-Winkler plaintext; จึงไม่แอบใช้ชื่อจริงใน privacy experiment.'),
    code("import exp_r4_bloom_privacy as r4\nimport exp_r1_ga_redecision as ga\nsource(r4, 'eval_ga'); source(ga, 'run_ga', 'decide_code')"),
    md('## 2. เปรียบเทียบ R4 กับ R5\n\nผลแสดงว่าการจูน decision layer ช่วยกู้ utility หลัง Bloom encoding ได้มากเพียงใด.'),
    code("r = read_json('experiments/r4_privacy_tradeoff.json')\nrows=[]\nfor L,x in r['per_L'].items():\n rows += [{'L':int(L),'experiment':'R4 manual',**x['R4_manual']['test']},{'L':int(L),'experiment':'R5 GA',**x['R5_ga']['test']}]\ndisplay(pd.DataFrame(rows).sort_values(['L','experiment'], ascending=[False,True]))"),
]

R6 = [
    md('# R6 — BERT-ER semantic blocking\n\nR6 พยายามกู้คู่จริงที่หลุดจาก blocking เดิม โดยใช้ frozen DistilBERT → learnable hash → bucket → match head.'),
    code(BOOT),
    md('## 1. Architecture\n\nDistilBERT encode profile ทีละตัว; hash head สร้าง 64-bit code สำหรับ blocking และ match head ให้ probability ของคู่ candidate.'),
    code("import exp_r6_bert_er as r6\nsource(r6, 'BertERModel', 'cosine_contrastive_loss', 'encode_all_profiles')"),
    md('## 2. Train heads และ candidate recovery\n\nBackbone ถูก freeze เพราะ fine-tune บน CPU ช้า. คู่ blocking-missed 3,316 คู่ถูก held out ไม่เข้า training.'),
    code("source(r6, 'build_training_pairs', 'train_heads', 'compute_hash_codes', 'build_buckets', 'blocking_recovery_eval')"),
    md('## 3. แยกผล blocking ออกจาก matching\n\nหาก hash หา candidate ไม่เจอ match head จะทำงานไม่ได้; จึงต้องดู recovery rate ควบคู่ F1.'),
    code("recovery=read_json('experiments/r6_blocking_recovery.json'); results=read_json('experiments/r6_results.json')\nprint(json.dumps(recovery, ensure_ascii=False, indent=2))\ndisplay(pd.DataFrame({'R6 manual':results['splits']['test'], 'R6 GA':results['ga_rules']['test']}).T)"),
]

GB = [
    md('# GB — 17 features → MiniLM → GA\n\nเปรียบเทียบ GB-17, GB-18 และ GB-18-GA บน nested entity-aware split เดียวกับ R3.'),
    code(BOOT),
    md('## 1. Fair inputs\n\nGB-17 และ GB-18 ใช้ sampled rows, labels, seed และ hyperparameters เดียวกัน ต่างเพียง `bert_cos`.'),
    code("import run_gb_transformer_experiments as gb\nsource(gb, 'validate_inputs', 'deterministic_sample_indices', 'train_gb_variant')\ncfg=read_json('experiments/automation/gb_transformer_primary_20260809/config.json'); print(json.dumps(cfg['gb_configuration'], indent=2))"),
    md('## 2. Gradient Boosting และ isotonic calibration\n\nfit ใช้ model_train; calibration ใช้ model_calibration; probability ที่ calibrated ถูกบันทึกเพื่อใช้ได้ทั้ง manual และ GA.'),
    code("m=read_json('experiments/automation/gb_transformer_primary_20260809/training_metadata.json')\ndisplay(pd.DataFrame({k:{'features':len(v['feature_columns']),'train':v['n_model_train_after_undersampling'],'AP':v['calibration_average_precision'],'seconds':v['fit_seconds']} for k,v in m.items() if k.startswith('GB-')}).T)"),
    md('## 3. GA ใช้ probability GB-18 ชุดเดิม\n\nเลือก genome ด้วย ga_validation cost เท่านั้น แล้วจึงเปิด test labels หนึ่งครั้ง.'),
    code("source(gb, 'run_ga_trials', 'build_summary')\ns=read_json('experiments/automation/gb_transformer_primary_20260809/summary.json')\ndisplay(pd.DataFrame(s['comparison'])); print(json.dumps(s['selected_ga'], ensure_ascii=False, indent=2))"),
    md('## 4. Isolated effects และ verification'),
    code("display(pd.DataFrame(s['isolated_effects']).T); print(json.dumps(s['verification'], ensure_ascii=False, indent=2))"),
    code("# Optional full rerun:\n# import subprocess\n# subprocess.run([sys.executable, str(ROOT/'run_gb_transformer_experiments.py')], check=True)"),
]

NOTEBOOKS = {
    '00_shared_protocol/00_shared_protocol.ipynb': SHARED,
    'r0_baseline/R0_baseline.ipynb': R0,
    'r1_ga_redecision/R1_ga_redecision.ipynb': R1,
    'r2_minilm_feature/R2_minilm_feature.ipynb': R2,
    'r3_minilm_ga/R3_minilm_ga.ipynb': R3,
    'r4_bloom_manual/R4_bloom_manual.ipynb': R4,
    'r5_bloom_ga/R5_bloom_ga.ipynb': R5,
    'r6_bert_er/R6_bert_er.ipynb': R6,
    'gb_transformer_ga/GB_transformer_GA.ipynb': GB,
}


def main() -> None:
    for relative, cells in NOTEBOOKS.items():
        path = HERE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        notebook = nbf.v4.new_notebook()
        notebook.cells = [nbf.v4.new_markdown_cell(text) if kind == 'markdown' else nbf.v4.new_code_cell(text) for kind, text in cells]
        notebook.metadata = {'kernelspec': {'display_name': 'Python (.venv)', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}}
        nbf.write(notebook, path)
        print('Wrote', path.relative_to(HERE))


if __name__ == '__main__':
    main()
