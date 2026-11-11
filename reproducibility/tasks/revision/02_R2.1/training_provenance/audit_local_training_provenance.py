"""Local provenance audit only: never imports/runs model code or compares predictions.

Outputs are restricted to this script's directory. Pickles are read using an
allowlist limited to NumPy array reconstruction, not arbitrary model classes.
"""
import os
from pathlib import Path
import csv, gzip, hashlib, json, pickle, re, sys
from collections import Counter
import numpy as np
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(os.environ.get("BE_REPLAY_PROJECT", "."))
OUT = Path(__file__).resolve().parent
MODELS = ['BEDICT-V2','BE-Hive-bystander','BE-Hive-efficiency','FORECasT-BE',
          'Yachie-base-editing-prediction','BE_Endo','DeepBE','EPCNNBE',
          'DeepBaseEditor-Paired-Library','CRISPRon-BE']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def dump_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')

def dump_csv(name, rows, columns=None):
    with (OUT/name).open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=columns or list(rows[0]));w.writeheader();w.writerows(rows)

# A reproducible inventory supports negative claims, bounded to these local roots.
inventory=[]
for model in MODELS:
    for sub in ('repo','resources'):
        for p in sorted((ROOT/model/sub).rglob('*')):
            if not p.is_file() or any(x in p.parts for x in ('.git','__pycache__')): continue
            inventory.append(dict(model=model,path=str(p),bytes=p.stat().st_size))
dump_csv('local_model_file_inventory.csv',inventory)

evidence=[]
def ev(model,key,rel,pattern=None,field=None):
    p=ROOT/rel
    if pattern is not None:
        lines=p.read_text(encoding='utf-8-sig',errors='replace').splitlines()
        found=[(i,s) for i,s in enumerate(lines,1) if re.search(pattern,s,re.I)]
        if not found: raise ValueError((key,rel,pattern))
        for i,s in found: evidence.append(dict(evidence_id=key,model=model,path=str(p),line=i,field='',excerpt=s))
    else:
        evidence.append(dict(evidence_id=key,model=model,path=str(p),line='',field=field or '',excerpt='存在且已读取相应字段/结构'))

ev('all','E01','benchmark_core/config/data_source_relationship_v2.csv',r'^Data-[1-9],')
ev('all','E02','SCI-data/Code/third_party_model_repository_catalog.csv',r'^(BEDICT|BE-Hive|BE_Endo|CRISPRon|DeepBE|DeepBase|EPCNNBE|FORECasT|Yachie)')
ev('BEDICT-V2','B01','BEDICT-V2/repo/README.md',r'example dataset|Pre-trained models|specify the appropriate')
ev('BEDICT-V2','B02','BEDICT-V2/repo/main_py_files/generate_two_stage_model_data.py',r'pd.read_csv|num_splits =|get_train_test_val|data_partitions.pkl')
ev('BEDICT-V2','B03','BEDICT-V2/resources/DOWNLOAD_MANIFEST.md',r'Commit:|30-sequence|full processed/raw|PRJNA|MOESM4')
ev('BE-Hive-efficiency','H01','BE-Hive-efficiency/repo/README.md',r'models were trained|4-mers')
ev('BE-Hive-efficiency','H02','BE-Hive-efficiency/repo/predict.py',r'model_pkl_fn =')
ev('BE-Hive-bystander','H03','BE-Hive-bystander/repo/predict.py',r'log_fn =|model_.*statedict.pt')
ev('BE-Hive-bystander','H04','BE-Hive-bystander/resources/DOWNLOAD_MANIFEST.md',r'Commit:|C8AA|No model')
ev('BE-Hive-bystander','H05','BE-Hive-bystander/repo/params/_log_mES_12kChar_ABE_0_190524_enc_64x2_dec_256x2_nc_r5_d0.05.out',r'train_test_id:')
ev('FORECasT-BE','F01','FORECasT-BE/repo/forecast_be.py',r'MODEL_DIR =|model_paths =|TOTAL_EFFICIENCY_MODELS\[editor\] =')
ev('Yachie-base-editing-prediction','Y01','Yachie-base-editing-prediction/repo/README.md',r'Sakata|models were created')
ev('Yachie-base-editing-prediction','Y02','Yachie-base-editing-prediction/repo/sample_training_codes/README',r'Example code|sample.ini')
ev('BE_Endo','N01','BE_Endo/repo/model_training/Eff_training_main.py',r'X_train,X_train_bio,Y_train|validation_split=|seq=key.split|with open\(train_split_path')
ev('BE_Endo','N02','benchmark_core/adapters/run_be_endo_seq_pilot.py',r'model_data_seq|model_version=')
ev('DeepBE','D01','DeepBE/repo/training_code/Efficiency_train.py',r'PATH_TO_DATASET|1-WT')
ev('DeepBE','D02','DeepBE/repo/training_code/Proportion_train.py',r'PATH_TO_DATASET')
ev('DeepBE','D03','DeepBE/repo/training_code/PAM_train.py',r'PATH_TO_DATASET')
ev('DeepBE','D04','DeepBE/resources/DEEPBE_RELEASE_MODELS.md',r'Release:|Published:')
ev('DeepBE','D05','benchmark_core/adapters/run_deepbe_pilot.py',r'DEFAULT_WEIGHTS|DeepBE_8e_model|model_file.*PAM_variant')
ev('EPCNNBE','P01','EPCNNBE/repo/EPCNNBE_independent_tests.py',r'test_data =|h1_data =|hc_data =')
ev('EPCNNBE','P02','EPCNNBE/repo/EPCNNBE_web.py',r'para = pd.read_csv|weight_path = weight_fold')
ev('EPCNNBE','P03','EPCNNBE/resources/DOWNLOAD_MANIFEST.md',r'87C8|Branch:|Repository:|Google Drive file ID')
ev('DeepBaseEditor-Paired-Library','K01','DeepBaseEditor-Paired-Library/repo/DeepBaseEditor/README.txt',r'Input2:|sample.txt.*Target|Pre-trained Weight Files')
ev('CRISPRon-BE','C01','CRISPRon-BE/repo/data_for_model_development_V1.0/README.md',r'Dataset Version|DATE:|training set|test set|source \(training only\)|partition:|refs:|surro_target:')
ev('CRISPRon-BE','C02','CRISPRon-BE/resources/DOWNLOAD_MANIFEST.md',r'CRISPRonBE_models-1.0|67E9BFBF')
ev('CRISPRon-BE','C03','CRISPRon-BE/repo/README.md',r'trained using 6|CRISPRon_V0|Kissling|SURRO-seq, Song and Arbab')
ev('all','C04','SCI-data/Fig.2/supplementary/training_provenance_output_contract.csv',r'CRISPRon-BE|DeepBE|FORECasT-BE|DeepBaseEditor')

# Source-field counts only; the separate agent owns CRISPRon sequence overlap.
crispron_exposure=[]
for editor in ('ABE','CBE'):
    for role in ('train','test'):
        rel=f'CRISPRon-BE/repo/data_for_model_development_V1.0/{editor}_{role}.tsv.gz'
        with gzip.open(ROOT/rel,'rt',newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f,delimiter='\t')
            counts=Counter(row.get('source','<missing>') for row in reader)
        crispron_exposure.append(dict(path=str(ROOT/rel),role=role,fields=reader.fieldnames,
            outcome_rows_by_source=dict(counts),note='作者test暴露与训练成员必须区分；产物行数不是唯一靶点数。'))
        ev('CRISPRon-BE','C05',rel,field='source/partition; actual values in crispron_author_train_test_source_fields.json')
dump_json('crispron_author_train_test_source_fields.json',crispron_exposure)

# Exact local assets; SHA256 is computed for small assets, large archives are
# identified by saved manifests and paths. A current hash is not an upstream tag.
weight_rows=[]
selectors={
 'BEDICT-V2':lambda p:'model_statedict' in p.parts and p.name!='best_epoch.pkl',
 'BE-Hive-bystander':lambda p:p.suffix=='.pt',
 'BE-Hive-efficiency':lambda p:p.parent.name=='params' and p.suffix=='.pkl',
 'FORECasT-BE':lambda p:p.suffix=='.joblib',
 'Yachie-base-editing-prediction':lambda p:p.parent.name=='sample_models' and p.suffix=='.csv',
 'BE_Endo':lambda p:p.suffix=='.h5',
 'DeepBE':lambda p:('extracted_models' in p.parts and (p.suffix=='.h5' or p.name=='saved_model.pb' or '.data-' in p.name or p.suffix=='.index')),
 'EPCNNBE':lambda p:p.parent.name=='be_weights',
 'DeepBaseEditor-Paired-Library':lambda p:'DeepBaseEditor' in p.parts and any(x.endswith('_Weight') for x in p.parts),
 'CRISPRon-BE':lambda p:False,
}
for row in inventory:
    p=Path(row['path'])
    if selectors[row['model']](p):
        weight_rows.append(dict(model=row['model'],path=str(p),bytes=row['bytes'],
            sha256=sha(p) if row['bytes']<=20*1024*1024 else '',
            hash_scope='computed_current_local_asset' if row['bytes']<=20*1024*1024 else 'not_hashed_large_asset_see_saved_manifest'))
dump_csv('local_weight_asset_inventory.csv',weight_rows)

class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
        if (module,name) in allowed:return allowed[(module,name)]
        raise pickle.UnpicklingError('Non-array global blocked: '+module+'.'+name)

sequence_rows=[]; list_summaries=[]
for editor in ('ABE','CBE'):
    rel=f'BE_Endo/repo/model_training/{editor}_training_pkl/{editor}_all.pkl';p=ROOT/rel
    with p.open('rb') as f:data=ArrayOnlyUnpickler(f).load()
    assert len(data)==6 and all(len(x)==6 for x in data)
    for split,item in enumerate(data):
        for role,idx in [('train',0),('test',3)]:
            vals=item[idx]
            assert isinstance(vals,list) and all(isinstance(x,str) for x in vals)
            seqs=[x.split('_')[0].upper() for x in vals]
            list_summaries.append(dict(model='BE_Endo',source_path=str(p),role=role,split=split,
                raw_rows=len(vals),unique_sequences=len(set(seqs)),sequence_lengths=sorted(set(map(len,seqs))),
                field=f'item[{split}][{idx}]',membership_status='confirmed_for_supplied_endo_training_splits; BE_Seq_weight_link_unknown',sha256=sha(p)))
            for j,(identifier,seq) in enumerate(zip(vals,seqs)):
                sequence_rows.append(dict(model='BE_Endo',editor=editor,role=role,split=split,source_path=str(p),
                    source_field=f'item[{split}][{idx}][{j}]',source_id=identifier,sequence=seq,sequence_length=len(seq),
                    membership_status='confirmed_for_supplied_endo_training_splits; BE_Seq_weight_link_unknown'))
    ev('BE_Endo','N03',rel,field='6 items; each [X_train,X_train_bio,Y_train,X_test,X_test_bio,Y_test]; item[0]/item[3] sequences')

for editor in ('abe','cbe'):
    for role in ('train','test'):
        rel=f'EPCNNBE/repo/tests/{role}_{editor}.csv';p=ROOT/rel
        with p.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        seqmap={}
        for line,row in enumerate(rows,2): seqmap.setdefault(row['target'].upper(),(line,row.get('', '')))
        status='filename_labeled_training_pool_weight_link_unknown' if role=='train' else 'confirmed_independent_test_input_by_code'
        list_summaries.append(dict(model='EPCNNBE',source_path=str(p),role=role,split='',raw_rows=len(rows),
            unique_sequences=len(seqmap),sequence_lengths=sorted(set(map(len,seqmap))),field='target',membership_status=status,sha256=sha(p)))
        for seq,(line,identifier) in sorted(seqmap.items()):
            sequence_rows.append(dict(model='EPCNNBE',editor=editor.upper(),role=role,split='',source_path=str(p),
                source_field=f'CSV line {line}; target',source_id=identifier,sequence=seq,sequence_length=len(seq),membership_status=status))
        ev('EPCNNBE','P04',rel,field='target/outcome/eff/op/por/of; role based on filename; test role also supported by independent_tests.py')
dump_csv('available_sequence_memberships_non_CRISPRon.csv',sequence_rows)
dump_json('available_sequence_list_summary.json',list_summaries)

# Workbook structure is inspected without treating arbitrary supplementary tables
# or the 30-row BEDICT demo as confirmed training membership.
workbooks=[]
wb_paths=[ROOT/'BEDICT-V2/repo/dataset/DeepABE_ABEmax_Train_df_for_BEDICT_pred_df.xlsx']
for base in ['BEDICT-V2/resources/paper_supplements','benchmark_core/data/manual_downloads/be_endo_supplement']:
    wb_paths.extend((ROOT/base).glob('*.xlsx'))
for p in wb_paths:
    w=openpyxl.load_workbook(p,read_only=True,data_only=True)
    sheets=[]
    for s in w:
        top=[list(r) for r in s.iter_rows(min_row=1,max_row=3,values_only=True)]
        sheets.append(dict(sheet=s.title,max_row=s.max_row,max_column=s.max_column,first_three_rows=top))
    workbooks.append(dict(path=str(p),sha256=sha(p),sheets=sheets));w.close()
dump_json('supplement_workbook_structure.json',workbooks)
ev('BEDICT-V2','B04','BEDICT-V2/repo/dataset/DeepABE_ABEmax_Train_df_for_BEDICT_pred_df.xlsx',field='Sheet1!A1:C31; columns ID/refseq/PAM; 30 example sequences')
ev('BEDICT-V2','B05','BEDICT-V2/resources/paper_supplements/13059_2025_3586_MOESM4_ESM.xlsx',field='Overview_Datasets; HEK-/AAV-/Lenti- sheets; see supplement_workbook_structure.json')

# Source-pool fields are useful for later conservative screening, but do not prove
# training membership. No new overlap or performance statistic is computed here.
source_pools=[]
for dirname in ['be_hive_efficiency','forecast_be']:
    for p in sorted((ROOT/'benchmark_core/data/manual_downloads'/dirname).glob('*')):
        if p.suffix not in ('.csv','.tsv'):continue
        with p.open(encoding='utf-8-sig',newline='') as f:
            fields=next(csv.reader(f,delimiter='\t' if p.suffix=='.tsv' else ','))
        source_pools.append(dict(source_family=dirname,path=str(p),fields=fields,
            training_membership='unknown; source pool only; no train/validation/test membership field established'))
dump_json('evaluation_source_pool_fields.json',source_pools)

common_unknown='本次未与评价面板计算重叠；未找到训练清单不等于无重叠；模型来源不同也不等于训练独立。'
matrix=[]
def model_row(model,family,train,develop,validation,lists,weights,version,relation,confirmed,clean,unknown,ids):
    matrix.append(dict(model=model,family=family,training_source=train,development_source=develop,
        validation_test_source=validation,readable_training_ids_sequences=lists,
        local_weight_location=weights,version_evidence=version,evaluation_source_relation=relation,
        confirmed_status=confirmed,known_clean_status=clean,unknown_boundary=unknown,
        evidence_ids=ids,weight_inventory=str(OUT/'local_weight_asset_inventory.csv')))

model_row('BEDICT-V2','BEDICT-V2',
 '代码预期 HEK_BLDLib_10d_* 与 merged_invivo 数据；具体当前权重训练行清单未随 repo 提供。',
 'BEDICT-V2 论文数据源 PRJNA1107439；MOESM4 现已本地存在，含 HEK/AAV/Lenti 多条件源表。',
 '预处理代码 num_splits=3，get_train_test_val(random_state=42)；已保存的实际 data_partitions.pkl 未在 repo/dataset 中找到。',
 str(ROOT/'BEDICT-V2/resources/paper_supplements/13059_2025_3586_MOESM4_ESM.xlsx')+' 是源数据候选；repo/dataset Excel 只有30条示例，非训练全集。',
 'repo/absolute_efficiency_model/output/**/model_statedict/CNN.pkl; repo/proportion_model/output/**/model_statedict/HaplotypeEncoderEncoder.pkl',
 '下载清单/仓库目录记录 commit 646d1dd7b029a323f3e5526f9cf5c4bb62069765；当前资产哈希见清单。',
 'Data-3/4/7 被旧映射标为家族来源关联；其中 Data-7 的旧 BEDICT 来源不能直接等同当前 V2 权重训练成员。',
 'confirmed_source_association_only','unknown',
 '训练处理文件和实际权重—样本划分映射缺失。下载清单中“补充表未下载”已过时，现存表另行核验。'+common_unknown,'B01|B02|B03|B04|B05|E01|E02')
model_row('BE-Hive-bystander','BE-Hive',
 '权重/日志标识 12kChar、editor、celltype；训练日志含 train_test_id=0。',
 '本地 BE-Hive bystander/efficiency 源数据池；exact train/validation/test IDs 尚未建立。',
 '日志有 train/valid loss 和 train_test_id，未找到与该 ID 对应的样本成员文件。',
 'benchmark_core/data/manual_downloads/be_hive_bystander/*.pkl 是源池；be_hive_efficiency/*.csv 有 Name (unique)、gRNA (20nt)、Sequence context (56nt)，不能直接当 bystander 的已确认训练集。',
 'repo/params/model_*_epoch_*_statedict.pt；由 repo/models.csv 和 predict.py 选择。',
 '下载清单 commit 31aadd04a25c604857c7592b226ee987e9e20b31；当前权重 SHA256 见清单。',
 'Data-2/6 及 BE-Hive 来源 T2/T3/T5/T7 属来源关联；不是独立验证。',
 'confirmed_source_association_only','unknown','逐样本 split 和权重训练成员映射缺失。'+common_unknown,'H03|H04|H05|E01')
model_row('BE-Hive-efficiency','BE-Hive',
 'README 明确在覆盖位置1–11底物邻近全部4-mer的 minimally biased library 上训练；权重名 12kChar。',
 '本地 BE-Hive efficiency 源池含设计 ID/20nt guide/56nt context。',
 '本地推理包未给训练/验证/测试成员映射。',
 'benchmark_core/data/manual_downloads/be_hive_efficiency/*.csv；字段 Name (unique), gRNA (20nt), Sequence context (56nt)，仅源池。',
 'repo/params/*_bestmodel.pkl；repo/models.csv 的 Model name 字段选择。',
 '未发现可核验上游提交版本；当前参数文件 SHA256 可定位本地资产。',
 'Data-2/6 同属 BE-Hive 来源关联；各任务应保留该标记。',
 'confirmed_source_association_only','unknown','源数据池可保守筛除但不能把其中每行都称实际训练行。'+common_unknown,'H01|H02|E01|E02')
model_row('FORECasT-BE','FORECasT-BE',
 '本地推理包只有 saved_models，未给训练序列/ID或样本划分。',
 '本地有 forecast_be_processed_data.tsv 源屏幕表。',
 '未知；源表有 Oligo Id/Editor 等字段，但本表未提供序列或实际训练 split。',
 'benchmark_core/data/manual_downloads/forecast_be/forecast_be_processed_data.tsv：可用 Oligo Id 候选，需另外的 ID→sequence 和训练成员映射。',
 'repo/saved_models/{ABE,CBE}/global_efficiency.*.joblib; pos_*.joblib',
 '未发现上游提交/tag；当前18个参数文件哈希见资产清单。',
 'Data-9 来源关联；当前 Fig2 的 BE-Hive 源虽非同来源，训练重叠仍未知。',
 'confirmed_source_association_only','unknown','不允许根据 non-source-associated 宣称 training-independent。'+common_unknown,'F01|E01|E02|C04')
model_row('Yachie-base-editing-prediction','Yachie',
 'README：Sakata/Ishiguro/Mori 2020；用相应编辑器的多个靶点扩增子测序训练转移概率。',
 'sample_models 提供13种方法的概率表；sample_training_codes 仅示例训练流程。',
 '本地未找到当前13个概率表的完整训练/验证/测试靶点清单。',
 'repo/sample_training_codes/sample/{sample.ini,R1.fastq,R2.fastq} 是样例，不能作为完整训练清单。',
 'repo/sample_models/*.csv',
 'README 论文关联；未找到上游提交/tag；13个概率表当前哈希见清单。',
 '旧 source map 无 Yachie 来源专属面板；这不能推出与 BE-Hive/BEDICT/BE_Endo 等独立。',
 'confirmed_paper_level_source_only','unknown','示例 amplicon 数据不覆盖已发布参数的全部开发集。'+common_unknown,'Y01|Y02|E01|E02')
model_row('BE_Endo','BE_Endo',
 'model_training/{ABE,CBE}_training_pkl/*_all.pkl 明确存储6组训练/测试划分；Eff_training_main.py 对 endo 模型使用它们。',
 'BE_Endo 补充表源池及上述 endo 分支开发数据；benchmark 实际调用 BE_Seq 序列分支。',
 '每组 item[0] 是 X_train，item[3] 是 X_test；训练函数 validation_split=0.2；当前 BE_Seq 权重对应哪个 split/全集未知。',
 '已提取到 available_sequence_memberships_non_CRISPRon.csv；原始 *_all.pkl 路径、split、索引、40nt序列均保留。',
 '实际 benchmark：repo/model_data_seq/{ABE,CBE}_Seq/*_Seq.h5；另有 model_data_endo/*/*_endo.h5。',
 '未找到 upstream commit/tag；当前 H5 文件路径/大小/小文件哈希见清单。',
 'Data-8 来源关联；与其它评价源的精确重叠待做。',
 'confirmed_endo_training_splits; benchmark_BE_Seq_link_unknown','unknown',
 '不能把 endo 分支 split 不加限定称为 BE_Seq 权重的确定训练集；可作为已知开发池保守排除。'+common_unknown,'N01|N02|N03|E01|E02')
model_row('DeepBE','DeepBE',
 'Efficiency/Proportion/PAM 三训练脚本均以 PATH_TO_DATASET 占位，未说明实际数据文件和 train/val/test 行。',
 'release version1 的 DeepBE/PAM 模块；完整多阶段训练来源未解决。',
 '未知，不能只审计最终 outcome 模型而漏 PAM 特征模型训练。',
 'repo 中示例 input CSV 非训练清单；未找到真实训练 ID/序列成员表。',
 'resources/extracted_models/DeepBE/DeepBE_8e_model; resources/extracted_models/PAM/PAM_variant_*_model.h5',
 '保存的官方 release version1，发布 2023-07-10；当前文件清单见资产表。',
 'BE-Hive scalar 面板仅 non-source-associated、known-CRISPRon-clean；DeepBE 自身训练重叠未知。',
 'confirmed_release_and_training_code_only','unknown',common_unknown,'D01|D02|D03|D04|D05|C04')
model_row('EPCNNBE','EPCNNBE',
 'tests/train_abe.csv 与 train_cbe.csv 有完整 target/outcome 数值列；角色来自文件名，未找到训练脚本—发布权重的明确绑定。',
 'train_*.csv 可作为候选训练池；compare_datasets.xlsx 是跨工具比较表，不能当该模型的训练清单。',
 'EPCNNBE_independent_tests.py 明确读取 test_*.csv 和 *_endo_h1/hc.csv 作为测试。',
 '已提取4张 train/test CSV 的唯一 target 到 available_sequence_memberships_non_CRISPRon.csv，保留角色可信度、源行和路径。',
 'repo/be_weights/{abe,cbe}_model_top*_{ens_ty}_{seq_ty}; para_*.csv',
 'download manifest: branch main；be_weights.zip SHA256 87C8F27441D556E679BB67A2D9049E856366F7901E17B70D3D27680126A23BA3。',
 '旧 source map 无本模型专属源；训练候选池/测试源与所有评价源的交集尚待核验。',
 'confirmed_readable_candidate_train_pool_and_code_linked_test_pool','unknown',
 '不能仅凭 train 文件名断言所有发布权重都用过这些行；排除候选池后的标签须限定为该已知池。'+common_unknown,'P01|P02|P03|P04|E01|E02')
model_row('DeepBaseEditor-Paired-Library','DeepBaseEditor',
 '本地 README 描述 ABE/CBE Efficiency/Proportion/CA 推理和预训练权重，无训练源或完整成员表。',
 'DeepBaseEditor 子目录；同仓库其它 DeepCas9 工具不能混作 base-editor 训练证据。',
 '样例 *_sample.txt 为推理演示；实际训练/验证/测试划分未知。',
 'repo/DeepBaseEditor/*_sample.txt 不是训练清单；未发现可确认的训练ID/序列表。',
 'repo/DeepBaseEditor/{ABE_Efficiency,ABE_Proportion,CBE_Efficiency,CBE_Efficiency_CA,CBE_Proportion}_Weight/',
 '无可核验上游提交/tag；TF checkpoint 当前资产见清单。',
 '当前 BE-Hive scalar 面板 non-source-associated，仅针对 CRISPRon 已知开发集去重；自身重叠未知。',
 'confirmed_inference_assets_only','unknown',common_unknown,'K01|E02|C04')
model_row('CRISPRon-BE','CRISPRon-BE',
 'README 明确 ABE: SURRO-seq, Song, Arbab, Kissling ABEmax, Kissling ABE8e；CBE: SURRO-seq, Song, Arbab。',
 'data_for_model_development_V1.0，日期2025-08-27；训练/测试 TSV包含 refs、surro_target、partition。',
 'ABE_test.tsv.gz/CBE_test.tsv.gz；实际 source 字段还含 marquart_test（分别9284/10114产物行），不能漏记作者测试暴露；这些行不是训练成员的直接证据。README说明5份训练、剩余1份测试，模型包含6个验证集。',
 'repo/data_for_model_development_V1.0/{ABE,CBE}_{train,test}.tsv.gz；本子任务仅定位，不重复另一代理的重叠审计。',
 'repo/data/CRISPRonBE_models; repo/data/CRISPRon_models_V0',
 'CRISPRonBE_models-1.0.tgz 保存 SHA256 67E9BFBFD4397CC736D568355D2A03E83EFBA1BB98C2D912B4CA8F135F28F339。',
 'Data-1 来源关联；Arbab/BE-Hive 来源跨家族训练重叠已由既存 T1 provenance 确认；Marquart/BEDICT原版来源亦有已确认作者测试暴露，不能作从未接触的独立源。',
 'confirmed_named_training_sources_and_local_memberships',
 '仅既存 T1 V8/V8.1 scalar panel 的 known-CRISPRon-exact-sequence-clean 标签；本次未重算，不外推至其它模型/任务。',
 '精确排除仅针对可见训练/开发/作者测试集及所用序列定义，不能证明近邻序列、辅助模块或未公开数据完全独立。训练与作者测试暴露须分列，不把所有已知暴露统称训练泄漏。','C01|C02|C03|C04|C05|E01')

dump_csv('model_training_provenance_matrix.csv',matrix)
dump_json('model_training_provenance_matrix.json',dict(scope='local-only; no external search, no model execution, no new overlap calculation',
  status_definitions={'confirmed':'证据支持的具体来源/成员或关联事实；不自动等同发生训练重叠',
                      'known-clean':'仅在明确定义的已知成员集合和匹配规则下排除了重叠',
                      'unknown':'缺少完整成员清单、权重映射或尚未计算；绝不可归入 clean'},models=matrix))
dump_csv('provenance_evidence.csv',evidence)
dump_json('audit_summary.json',dict(models=len(matrix),families=len(set(x['family'] for x in matrix)),
 inventory_files=len(inventory),weight_assets=len(weight_rows),evidence_records=len(evidence),
 extracted_sequence_memberships=len(sequence_rows),sequence_list_summaries=list_summaries,
 writes_restricted_to=str(OUT),no_model_calls=True,no_overlap_computation=True))
print(json.dumps({'models':len(matrix),'families':len(set(x['family'] for x in matrix)),
                 'sequence_memberships':len(sequence_rows),'weight_assets':len(weight_rows),'output':str(OUT)},ensure_ascii=False))
