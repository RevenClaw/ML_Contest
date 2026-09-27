import csv,json,re,pathlib,collections
ROOT=pathlib.Path(r'C:\Users\reven\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset')
def rows(p):
    with p.open(encoding='utf-8',newline='') as f: yield from csv.DictReader(f,delimiter='\t')
def norm(s): return ' '.join(re.findall(r'\w+',s.casefold()))
sample={r['entity_id']:r for r in rows(ROOT/'train/train_source1.tsv') if int(r['entity_id'].split('-')[1])%200==0}
names=collections.defaultdict(list)
for sid,r in sample.items(): names[(r['country'],norm(r['business_name']))].append(sid)
labels={r['source1_entity_id']:set(r['matched_entity_ids'].split(',')) for r in rows(ROOT/'train/train_ground_truth.tsv') if r['source1_entity_id'] in sample}
counts=collections.Counter(); examples=[]
for s in [2,3]:
    for r in rows(ROOT/f'train/train_source{s}.tsv'):
        key=(r['country'],norm(r['business_name']))
        for sid in names.get(key,[]):
            true=r['entity_id'] in labels[sid]
            counts['true' if true else 'false']+=1
            counts[r['country']+('_true' if true else '_false')]+=1
            if not true and len(examples)<5: examples.append({'reference':sample[sid],'nonmatch':r})
    print('NAME COLLISIONS',s,dict(counts),flush=True)
test={}
for s in [1,2,3]:
    c=collections.Counter(r['country'] for r in rows(ROOT/f'test/test_source{s}.tsv'))
    test[str(s)]=dict(c)
    print('TEST',s,dict(c),flush=True)
out={'sample_reference_entities':len(sample),'same_country_normalized_exact_name_pairs':dict(counts),'nonmatch_examples':examples,'test_countries':test}
(pathlib.Path(__file__).resolve().parents[1]/'reports/ambiguity_profile.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out),flush=True)
