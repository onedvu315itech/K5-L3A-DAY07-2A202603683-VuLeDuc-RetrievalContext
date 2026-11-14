"""Run Day 7 retrieval, context, and agent benchmarks across splits and configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from agent_metrics import aggregate
from agentic import (
    DenseRetriever,
    HybridRetriever,
    LexicalRetriever,
    OpenAIController,
    RerankRetriever,
    Retriever,
    _load_env_file,
    infer_query_filters,
    run_agent,
)
from baseline import TfidfIndex, hit, pack_context, validate
from embeddings import MyEmbedder
from store import PgVectorStore


BASE_DIR = Path(__file__).resolve().parent


def evaluate_retriever(
    retriever: Retriever,
    questions: list[dict[str, Any]],
    strategy_name: str,
    *,
    top_k: int = 3,
    budget: int = 800,
    use_question_filters: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a Retriever on a list of questions and return (traces, summary)."""
    traces: list[dict[str, Any]] = []
    for q in questions:
        q_filters = q.get('filter') if use_question_filters else infer_query_filters(q['question'])
        start = time.perf_counter()
        # Retrieve top_k for context assembly, and also ensure at least top-3 for Hit@1 / Hit@3 comparison
        eval_k = max(3, top_k)
        raw_results = retriever.search(q['question'], filters=q_filters, top_k=eval_k)
        top3_results = raw_results[:3]
        context_candidates = raw_results[:top_k]
        context, ids = pack_context(context_candidates, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        marked_top3 = [{**c, 'hit': hit(c, q['gold'])} for c in top3_results]
        marked_ctx = [{**c, 'hit': hit(c, q['gold'])} for c in context_candidates]

        traces.append({
            'query_id': q['id'],
            'split': q['split'],
            'question': q['question'],
            'filter': q_filters,
            'strategy': strategy_name,
            'top_k': top_k,
            'budget_chars': budget,
            'gold': q['gold'],
            'results_top3': marked_top3,
            'results': marked_ctx,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked_top3 and marked_top3[0]['hit']),
            'hit3': any(c['hit'] for c in marked_top3),
            'context_hit': any(c['hit'] and c['id'] in ids for c in marked_ctx),
        })

    n = len(traces)
    summary = {
        'strategy': strategy_name,
        'split': questions[0]['split'] if questions else 'dev',
        'top_k': top_k,
        'budget_chars': budget,
        'n': n,
        'hit1': sum(t['hit1'] for t in traces),
        'hit1_rate': round(sum(t['hit1'] for t in traces) / n, 4) if n else 0.0,
        'hit3': sum(t['hit3'] for t in traces),
        'hit3_rate': round(sum(t['hit3'] for t in traces) / n, 4) if n else 0.0,
        'context_hit': sum(t['context_hit'] for t in traces),
        'context_hit_rate': round(sum(t['context_hit'] for t in traces) / n, 4) if n else 0.0,
        'avg_context_chars': round(statistics.mean(t['context_chars'] for t in traces), 2) if n else 0.0,
        'median_ms': round(statistics.median(t['elapsed_ms'] for t in traces), 3) if n else 0.0,
        'service_cost_usd': 0.0,
    }
    return traces, summary


class TfidfRetrieverAdapter:
    """Adapter wrapping baseline TfidfIndex into the Retriever protocol."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        return self.index.search(query, top_k=top_k)


def save_outputs(out_dir: Path, traces: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_retrieval_suite(chunks_path: Path, questions_path: Path, out_root: Path) -> dict[str, Any]:
    _load_env_file()
    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    all_questions = json.loads(questions_path.read_text(encoding='utf-8'))
    dev_questions = [q for q in all_questions if q.get('split') == 'dev']
    test_questions = [q for q in all_questions if q.get('split') == 'test']
    validate(chunks, dev_questions)
    validate(chunks, test_questions)

    db_url = 'postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab'
    embedder = MyEmbedder('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    store = PgVectorStore(db_url, embedder.dimensions)

    # Instantiate retrievers
    tfidf_ret = TfidfRetrieverAdapter(chunks)
    dense_ret = DenseRetriever(store=store, embedder=embedder, strategy='A_900', infer_filters=True)
    bm25_ret = LexicalRetriever(chunks, mode='bm25', infer_filters=True)
    hybrid_ret = HybridRetriever(dense_ret, bm25_ret, fusion='rrf', rrf_k=60, dense_weight=0.45, infer_filters=True)
    rerank_ret = RerankRetriever(hybrid_ret, candidate_k=15, infer_filters=True)

    all_summaries: dict[str, Any] = {}

    # 1. Dev comparisons (Methods + Improvements)
    dev_configs = [
        # Baseline: TF-IDF top_k=3, budget=800
        ('dev-1-baseline-tfidf', tfidf_ret, 'tfidf_baseline', 3, 800),
        # Method 1 (Mandatory): Dense Retrieval on Day 6 pgvector store (top_k=3, budget=800)
        ('dev-2-dense-pgvector', dense_ret, 'dense_pgvector', 3, 800),
        # Method 2: Lexical BM25 with inferred section_id filter (top_k=3, budget=800)
        ('dev-3-lexical-bm25', bm25_ret, 'lexical_bm25', 3, 800),
        # Method 3 / Improvement 1: Hybrid RRF (Dense + BM25) + Context Budget 3200 chars (fixes 800-char truncation error)
        ('dev-4-hybrid-rrf-b3200', hybrid_ret, 'hybrid_rrf_b3200', 3, 3200),
        # Method 4 / Improvement 2: Hybrid + Cross-Feature Reranker (candidate_k=15 -> top_k=4, budget=3600)
        ('dev-5-hybrid-rerank-k4-b3600', rerank_ret, 'hybrid_rerank_k4_b3600', 4, 3600),
    ]

    for folder, ret, name, k, b in dev_configs:
        traces, summary = evaluate_retriever(ret, dev_questions, name, top_k=k, budget=b)
        save_outputs(out_root / folder, traces, summary)
        all_summaries[folder] = summary
        print(f"[DEV] {folder}: Hit@1={summary['hit1']}/{summary['n']} ({summary['hit1_rate']*100:.1f}%) | "
              f"Hit@3={summary['hit3']}/{summary['n']} ({summary['hit3_rate']*100:.1f}%) | "
              f"CtxHit={summary['context_hit']}/{summary['n']} ({summary['context_hit_rate']*100:.1f}%) | "
              f"Median={summary['median_ms']}ms")

    # 2. Test evaluation after freezing chosen config ('hybrid_rerank_k4_b3600') vs Baseline ('tfidf_baseline' & 'dense_pgvector')
    test_configs = [
        ('test-baseline-tfidf', tfidf_ret, 'tfidf_baseline', 3, 800),
        ('test-baseline-dense', dense_ret, 'dense_pgvector', 3, 800),
        ('test-chosen-hybrid-rerank', rerank_ret, 'hybrid_rerank_k4_b3600', 4, 3600),
    ]
    for folder, ret, name, k, b in test_configs:
        traces, summary = evaluate_retriever(ret, test_questions, name, top_k=k, budget=b)
        save_outputs(out_root / folder, traces, summary)
        all_summaries[folder] = summary
        print(f"[TEST] {folder}: Hit@1={summary['hit1']}/{summary['n']} ({summary['hit1_rate']*100:.1f}%) | "
              f"Hit@3={summary['hit3']}/{summary['n']} ({summary['hit3_rate']*100:.1f}%) | "
              f"CtxHit={summary['context_hit']}/{summary['n']} ({summary['context_hit_rate']*100:.1f}%) | "
              f"Median={summary['median_ms']}ms")

    (out_root / 'retrieval_comparison_summary.json').write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return all_summaries


def run_agent_suite(
    chunks_path: Path,
    cases_path: Path,
    out_root: Path,
) -> dict[str, Any]:
    """Run real LLM Agent on Dev and Test for Baseline vs Chosen Configuration."""
    _load_env_file()
    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    cases = json.loads(cases_path.read_text(encoding='utf-8'))

    db_url = 'postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab'
    embedder = MyEmbedder('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    store = PgVectorStore(db_url, embedder.dimensions)

    dense_ret = DenseRetriever(store=store, embedder=embedder, strategy='A_900', infer_filters=True)
    bm25_ret = LexicalRetriever(chunks, mode='bm25', infer_filters=True)
    hybrid_ret = HybridRetriever(dense_ret, bm25_ret, fusion='rrf', rrf_k=60, dense_weight=0.45, infer_filters=True)
    rerank_ret = RerankRetriever(hybrid_ret, candidate_k=15, infer_filters=True)

    runs = [
        # Baseline Agent: Dense retriever, top_k=3, budget=800, max_searches=1
        ('agent-dev-baseline', 'dev', dense_ret, 1, 3, 800),
        # Chosen Agent: Hybrid+Rerank retriever, top_k=4, budget=3600, max_searches=3
        ('agent-dev-chosen', 'dev', rerank_ret, 3, 4, 3600),
        # Test Baseline Agent
        ('agent-test-baseline', 'test', dense_ret, 1, 3, 800),
        # Test Chosen Agent
        ('agent-test-chosen', 'test', rerank_ret, 3, 4, 3600),
    ]

    agent_summaries: dict[str, Any] = {}
    for folder, split, retriever, max_s, top_k, budget in runs:
        split_cases = [c for c in cases if c.get('split') == split]
        controller = OpenAIController(max_searches=max_s, use_cache=True)
        traces: list[dict[str, Any]] = []

        for c in split_cases:
            qid = c['id']
            q_text = c['question']
            q_filters = infer_query_filters(q_text)
            t0 = time.perf_counter()
            tokens_before = (controller.total_prompt_tokens, controller.total_completion_tokens, controller.total_cost_usd)
            try:
                res = run_agent(
                    q_text,
                    controller,
                    retriever,
                    filters=q_filters,
                    max_searches=max_s,
                    top_k=top_k,
                    budget=budget,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                dp = controller.total_prompt_tokens - tokens_before[0]
                dc = controller.total_completion_tokens - tokens_before[1]
                dcost = controller.total_cost_usd - tokens_before[2]
                traces.append({
                    'query_id': qid,
                    'split': split,
                    'question': q_text,
                    'answerable': c['answerable'],
                    'status': res['status'],
                    'answer': res['answer'],
                    'citations': res['citations'],
                    'events': res['events'],
                    'search_calls': sum(1 for e in res['events'] if e.get('event') == 'search'),
                    'elapsed_ms': round(elapsed_ms, 2),
                    'prompt_tokens': dp,
                    'completion_tokens': dc,
                    'cost_usd': round(dcost, 6),
                })
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                traces.append({
                    'query_id': qid,
                    'split': split,
                    'question': q_text,
                    'answerable': c['answerable'],
                    'status': 'error',
                    'error': str(exc),
                    'answer': '',
                    'citations': [],
                    'events': [],
                    'search_calls': 0,
                    'elapsed_ms': round(elapsed_ms, 2),
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'cost_usd': 0.0,
                })

        out_dir = out_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"[AGENT] {folder} ({split}): {len(traces)} cases completed | "
              f"answered={sum(1 for t in traces if t['status']=='answered')} | "
              f"abstain={sum(1 for t in traces if t['status']=='abstain')} | "
              f"errors={sum(1 for t in traces if t['status']=='error')}")
        agent_summaries[folder] = {
            'split': split,
            'total_cases': len(traces),
            'answered': sum(1 for t in traces if t['status'] == 'answered'),
            'abstain': sum(1 for t in traces if t['status'] == 'abstain'),
            'errors': sum(1 for t in traces if t['status'] == 'error'),
            'total_cost_usd': round(controller.total_cost_usd, 6),
        }

    return agent_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Day 7 retrieval and agent experiments')
    parser.add_argument('--chunks', type=Path, default=BASE_DIR / 'data' / 'converted' / 'chunks.json')
    parser.add_argument('--questions', type=Path, default=BASE_DIR / 'data' / 'converted' / 'questions.json')
    parser.add_argument('--cases', type=Path, default=BASE_DIR / 'data' / 'converted' / 'cases.json')
    parser.add_argument('--output', type=Path, default=BASE_DIR / 'results')
    parser.add_argument('--skip-agent', action='store_true')
    args = parser.parse_args()

    run_retrieval_suite(args.chunks, args.questions, args.output)
    if not args.skip_agent:
        run_agent_suite(args.chunks, args.cases, args.output)
        import audit_and_grade
        audit_and_grade.main()


if __name__ == '__main__':
    main()
