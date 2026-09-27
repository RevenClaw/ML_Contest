"""Strict streaming validation using the existing disk-backed test ID index."""
import csv
from pathlib import Path
from .common import read_tsv
from .store import connect


def validate(index,output):
    db=connect(index); output=Path(output)
    try:
        split=db.execute("SELECT value FROM meta WHERE key='split'").fetchone()
        if not split or split[0]!='test': raise ValueError('A test index is required')
        matching=read_tsv(output/'matching_results.tsv',['source1_entity_id','matched_entity_ids'])
        candidates=read_tsv(output/'candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids'])
        # Prediction emits sorted IDs. Requiring this here avoids a huge in-memory seen set.
        count=0
        for reference in db.execute('SELECT id FROM refs ORDER BY id'):
            m,c=next(matching,None),next(candidates,None)
            if m is None or c is None or m['source1_entity_id']!=reference[0] or c['source1_entity_id']!=reference[0]:
                raise ValueError(f'Missing, duplicate, extra, or unsorted reference row near {reference[0]}')
            lists=[]
            for row,column in [(m,'matched_entity_ids'),(c,'candidate_entity_ids')]:
                ids=row[column].split(',') if row[column] else []
                if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate ID at {reference[0]}')
                if any(not tid.startswith(('S2-','S3-')) for tid in ids): raise ValueError('Invalid target prefix')
                if ids:
                    q=','.join('?' for _ in ids)
                    found=db.execute(f'SELECT COUNT(*) FROM records WHERE id IN ({q})',ids).fetchone()[0]
                    if found!=len(ids): raise ValueError(f'Nonexistent target at {reference[0]}')
                lists.append(set(ids))
            if not lists[0]<=lists[1]: raise ValueError(f'Final matches outside candidates at {reference[0]}')
            count+=1
        if next(matching,None) is not None or next(candidates,None) is not None: raise ValueError('Extra output rows')
        return {'valid':True,'references':count}
    finally: db.close()

