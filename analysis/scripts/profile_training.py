import csv, json, re, random, collections, pathlib, statistics, time
ROOT=pathlib.Path(r'C:\Users\reven\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset')
OUT=pathlib.Path(__file__).resolve().parents[1]/'reports/training_profile.json'
rng=random.Random(42)
sample={}; needed={}; targets={}; result={'files':{}}
def norm(s): return ' '.join(re.findall(r'\w+',s.casefold()))
def rows(p):
    with p.open(encoding='utf-8',newline='') as f:
        yield from csv.DictReader(f,delimiter='\t')
def summarize(p, sample_s1=False):
    n=0; country=collections.Counter(); blank=collections.Counter(); lengths={k:collections.Counter() for k in ['business_name','business_address']}; flags=collections.Counter()
    for r in rows(p):
        n+=1; country[r['country']]+=1
        for k,v in r.items():
            if not v.strip(): blank[k]+=1
        for k in lengths: lengths[k][len(r[k])]+=1
        if any(ord(c)>127 for c in r['business_name']): flags['non_ascii_name']+=1
        if any(ord(c)>127 for c in r['business_address']): flags['non_ascii_address']+=1
        if not any(c.isdigit() for c in r['business_address']): flags['address_without_digits']+=1
        if sample_s1 and int(r['entity_id'].split('-')[1])%200==0: sample[r['entity_id']]=r
        if r['entity_id'] in needed: targets[r['entity_id']]=r
    lens={}
    for k,h in lengths.items():
        ordered=sorted(h.items()); qs={}
        for q in [0.5,0.9,0.99]:
            acc=0
            for length,count in ordered:
                acc+=count
                if acc>=n*q: qs[str(q)]=length; break
        lens[k]={'mean':sum(k*v for k,v in h.items())/n,'max':max(h),'quantiles':qs}
    d={'rows':n,'countries':dict(country),'blank_fields':dict(blank),'flags':dict(flags),'lengths':lens}
    result['files'][str(p.relative_to(ROOT))]=d
    print(str(p.name),json.dumps(d),flush=True)
summarize(ROOT/'train/train_source1.tsv',True)
hist=collections.Counter(); per_source=collections.Counter(); label_n=0; duplicate_lists=0; sampled_labels={}
for r in rows(ROOT/'train/train_ground_truth.tsv'):
    label_n+=1; ids=r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
    hist[len(ids)]+=1; duplicate_lists+=len(ids)!=len(set(ids))
    for mid in ids: per_source[mid[:2]]+=1
    if r['source1_entity_id'] in sample:
        sampled_labels[r['source1_entity_id']]=ids
        for mid in ids: needed[mid]=r['source1_entity_id']
result['labels']={'rows':label_n,'match_count_histogram':dict(sorted(hist.items())),'pairs_by_source':dict(per_source),'duplicate_lists':duplicate_lists}
print('LABELS',json.dumps(result['labels']),flush=True)
for s in [2,3]: summarize(ROOT/f'train/train_source{s}.tsv')
pair_stats=collections.defaultdict(collections.Counter); examples=[]; js=collections.defaultdict(list); sample_country=collections.Counter(); sample_names=collections.Counter(norm(r['business_name']) for r in sample.values())
for sid,ids in sampled_labels.items():
    a=sample[sid]; country=a['country']; sample_country[country]+=1
    if not ids: pair_stats[country]['singleton_entities']+=1
    for mid in ids:
        if mid not in targets: pair_stats[country]['missing_target']+=1; continue
        b=targets[mid]; c=pair_stats[country]; c['pairs']+=1
        c['same_country']+=a['country']==b['country']
        for col in ['business_name','business_address']:
            c['exact_'+col]+=a[col]==b[col]; c['normalized_exact_'+col]+=norm(a[col])==norm(b[col])
            x=set(norm(a[col]).split()); y=set(norm(b[col]).split()); j=len(x&y)/len(x|y) if x|y else 1
            js[country+'_'+col].append(j)
            c['zero_token_overlap_'+col]+=j==0
        if len(examples)<12 and norm(a['business_name'])!=norm(b['business_name']) and (country=='India' or len(examples)<6): examples.append({'reference':a,'match':b})
result['sample']={'selection':'S1 numeric ID modulo 200 equals zero; deterministic approximately 0.5% sample','entities':len(sample),'countries':dict(sample_country),'pair_stats':{k:dict(v) for k,v in pair_stats.items()},'jaccard':{k:{'mean':statistics.mean(v),'median':statistics.median(v)} for k,v in js.items()},'examples':examples}
OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print('SAMPLE',json.dumps(result['sample'],ensure_ascii=True),flush=True)
