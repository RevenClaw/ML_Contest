import math
from functools import lru_cache
from .text import core_name, forms, gram_document


def quote(value):
    return '"' + value.replace('"','""') + '"'


class Retriever:
    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.n = db.execute('SELECT COUNT(*) FROM records').fetchone()[0]

    @lru_cache(maxsize=100000)
    def frequency(self, token, column):
        row = self.db.execute('SELECT doc FROM vocabulary WHERE term=? AND col=?', (token,column)).fetchone()
        return row[0] if row else 0

    def idf(self, token, column):
        return math.log1p(self.n / (1+self.frequency(token,column)))

    def terms(self, text, column):
        tokens = sorted(set(text.split()), key=lambda t: (self.frequency(t,column) == 0, self.frequency(t,column), t))
        return tokens[:self.config['query_terms']]

    def get(self, ref, training=False):
        _, _, name, address, core = forms(ref['name'],ref['address'])
        country = ref['country']
        scope = f'country : {quote(country)}'
        if training: scope += ' AND (fold : "train" OR fold : "unowned")'
        routes = {}
        def add(rows, route):
            for rank, row in enumerate(rows, 1):
                rid = row[0]
                item = routes.setdefault(rid, {'rrf':0.0,'routes':set(),'rank':rank})
                item['rrf'] += 1/(60+rank)
                item['routes'].add(route)
                item['rank'] = min(item['rank'],rank)
        def search(expression, limit, route):
            if expression:
                add(self.db.execute('SELECT rowid FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?', (scope+' AND ('+expression+')',limit)),route)
        def expr(column, text):
            terms = self.terms(text,column)
            return f'{column} : ('+' OR '.join(quote(t) for t in terms)+')' if terms else ''
        ne, ae = expr('n', core), expr('a',address)
        search(ne,self.config['name_k'],'name')
        search(ae,self.config['address_k'],'address')
        if ne and ae: search(f'({ne}) AND ({ae})',self.config['joint_k'],'joint')
        search(expr('grams',gram_document(core)),self.config['gram_k'],'gram')
        for column,value,route in [('n',name,'exact_name'),('a',address,'exact_address')]:
            if not value: continue
            sql=f'SELECT rowid FROM records WHERE country=? AND {column}=?'
            if training: sql += " AND fold IN ('train','unowned')"
            sql += ' ORDER BY rowid LIMIT ?'
            add(self.db.execute(sql,(country,value,self.config['exact_k'])),route)
        ranked = sorted(routes, key=lambda rid: (-routes[rid]['rrf'],rid))[:self.config['max_candidates']]
        if not ranked: return []
        placeholders=','.join('?' for _ in ranked)
        rows={r['rowid']:dict(r) for r in self.db.execute(f'SELECT rowid,* FROM records WHERE rowid IN ({placeholders})',ranked)}
        return [(rows[rid],routes[rid]) for rid in ranked]

