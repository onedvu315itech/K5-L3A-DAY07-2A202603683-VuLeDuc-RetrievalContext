"""Convert an exported Day 6 chunk/benchmark JSON array to Day 7 JSON contract."""
import argparse
import json
import re
from pathlib import Path


def convert_chunk(c):
    return {'id': c.get('id', c.get('chunk_id')), 'source_id': c.get('source_id', c.get('document_id')),
            'source': c.get('source'), 'page': c.get('page'), 'text': c.get('text'),
            'metadata': c.get('metadata', {})}


def convert_question(q):
    gold = q.get('gold')
    if gold is None:
        gold = [{'source_id': q.get('gold_source_id'), 'page': q.get('gold_page'),
                 'evidence': q.get('evidence_quote')}]
        for alt in q.get('alternative_gold_evidence', []) or []:
            alt_source = alt.get('source_id') or alt.get('source') or q.get('gold_source_id')
            alt_page = alt.get('page')
            alt_quote = alt.get('evidence') or alt.get('quote')
            if alt_source and alt_page is not None and alt_quote:
                gold.append({'source_id': alt_source, 'page': alt_page, 'evidence': alt_quote})
    result = {'id': q.get('id'), 'split': q.get('split'), 'question': q.get('question'),
              'gold': gold, 'expected_answer': q.get('expected_answer')}
    if 'filter' in q:
        result['filter'] = q['filter']
    return result


def main():
    parser = argparse.ArgumentParser(description='Chuyển JSON export Day 6 sang schema Day 7')
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw_chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    raw_questions = json.loads(args.questions.read_text(encoding='utf-8'))
    if not isinstance(raw_chunks, list) or not isinstance(raw_questions, list):
        raise ValueError('Hai file đầu vào phải là JSON arrays')
    # Conversion does not infer missing metadata or inspect test evidence.
    chunks = [convert_chunk(c) for c in raw_chunks]
    questions = []
    for raw_q in raw_questions:
        q = convert_question(raw_q)
        resolved_gold = []
        for idx, g in enumerate(q['gold']):
            # Keep if exact substring already matches
            if any(c['source_id'] == g['source_id'] and c['page'] == g['page'] and g['evidence'] in c['text'] for c in chunks):
                resolved_gold.append(g)
                continue
            # If whitespace/newline differs in chunk text, align to exact chunk substring
            ev_words = (g['evidence'] or '').split()
            matched = False
            if ev_words:
                pattern = r'\s+'.join(re.escape(w) for w in ev_words)
                for c in chunks:
                    if c['source_id'] == g['source_id'] and c['page'] == g['page']:
                        m = re.search(pattern, c['text'])
                        if m:
                            resolved_gold.append({**g, 'evidence': m.group(0)})
                            matched = True
                            break
            if not matched and idx == 0:
                resolved_gold.append(g)
        q['gold'] = resolved_gold or q['gold'][:1]
        questions.append(q)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (('chunks.json', chunks), ('questions.json', questions)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Đã chuyển đổi; kiểm tra output và chạy baseline trên dev để kiểm chứng.')


if __name__ == '__main__':
    main()
