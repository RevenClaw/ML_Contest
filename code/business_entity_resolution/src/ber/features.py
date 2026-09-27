import math
import re
import numpy as np
from .text import forms, grams

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None


def edit_similarity(a,b):
    if not a or not b: return 0.0
    if Levenshtein: return Levenshtein.normalized_similarity(a,b)
    # Exact dependency-free equivalent for small unit-test fixtures only.
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]
        for j,y in enumerate(b,1): curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=curr
    return 1-prev[-1]/max(len(a),len(b))


def overlap(x,y):
    return len(x&y)/len(x|y) if x and y else 0.0


def compare(a,b):
    x,y=set(a.split()),set(b.split())
    gx,gy=grams(a),grams(b)
    return [float(bool(a) and a==b),edit_similarity(a,b),overlap(x,y),
        len(x&y)/min(len(x),len(y)) if x and y else 0.,
        2*len(gx&gy)/(len(gx)+len(gy)) if gx and gy else 0.,
        min(len(a),len(b))/max(len(a),len(b)) if a and b else 0.,
        edit_similarity(' '.join(sorted(x)),' '.join(sorted(y)))]


SUFFIXES=['exact','edit','jaccard','containment','trigram_dice','length_ratio','token_sort_edit']
VIEWS=['name_native','address_native','name_latin','address_latin','name_core']
FEATURE_NAMES=[v+'_'+s for v in VIEWS for s in SUFFIXES]+[
    'name_idf_overlap','address_idf_overlap','name_commonness',
    'address_missing_left','address_missing_right','name_missing_left','name_missing_right',
    'number_jaccard','number_conflict','first_number_equal','first_number_in_range',
    'postal_like_overlap','postal_like_conflict','country_equal','target_source3',
    'name_nonascii_left','name_nonascii_right','name_acronym_equal',
    'rrf','best_rank_inverse','route_count','route_name','route_address','route_joint','route_gram','route_exact_name','route_exact_address',
    'candidate_count_log','name_address_product']


def weighted_overlap(a,b,retriever,column):
    x,y=set(a.split()),set(b.split()); union=x|y
    if not union: return 0.
    weights={t:retriever.idf(t,column) for t in union}
    return sum(weights[t] for t in x&y)/sum(weights.values())


def first_number(s):
    m=re.search(r'\d+',s)
    return int(m.group()) if m else None


def in_range(a,b):
    n=first_number(a)
    m=re.search(r'(\d+)\s*[-–]\s*(\d+)',b)
    return float(n is not None and m is not None and int(m[1])<=n<=int(m[2]))


def pair_features(ref,target,route,candidate_count,retriever):
    left=forms(ref['name'],ref['address']); right=forms(target['name'],target['address'])
    out=[]
    for a,b in zip(left,right): out.extend(compare(a,b))
    name_a,address_a=left[2],left[3]; name_b,address_b=right[2],right[3]
    numbers_a=set(re.findall(r'\d+',address_a)); numbers_b=set(re.findall(r'\d+',address_b))
    post_a={n for n in numbers_a if len(n) in (5,6)}; post_b={n for n in numbers_b if len(n) in (5,6)}
    fa,fb=first_number(address_a),first_number(address_b)
    acronym=lambda s: ''.join(t[0] for t in s.split() if t)
    out += [weighted_overlap(name_a,name_b,retriever,'n'),weighted_overlap(address_a,address_b,retriever,'a'),
        math.log1p(min((retriever.frequency(t,'n') for t in left[4].split()),default=0)),
        float(not address_a),float(not address_b),float(not name_a),float(not name_b),
        overlap(numbers_a,numbers_b),float(bool(numbers_a and numbers_b) and not numbers_a&numbers_b),
        float(fa is not None and fa==fb),max(in_range(ref['address'],target['address']),in_range(target['address'],ref['address'])),
        overlap(post_a,post_b),float(bool(post_a and post_b) and not post_a&post_b),
        float(ref['country']==target['country']),float(target['id'].startswith('S3-')),
        float(not ref['name'].isascii()),float(not target['name'].isascii()),
        float(bool(left[4] and right[4]) and acronym(left[4])==acronym(right[4])),
        route['rrf'],1/route['rank'],len(route['routes'])]
    out += [float(r in route['routes']) for r in ['name','address','joint','gram','exact_name','exact_address']]
    out += [math.log1p(candidate_count),out[2]*out[9]]
    if len(out)!=len(FEATURE_NAMES): raise RuntimeError('Feature schema mismatch')
    return np.asarray(out,dtype=np.float32)
