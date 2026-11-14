# Retrieval Selection Evidence & Dev Error Analysis

## 1. Summary of All 5 Dev Configurations (Locked Before Test)

| Config ID | Description | Hit@1 (n/23) | Evidence Hit@3 (n/23) | Context Evidence Rate (n/23) | Avg Context Chars | Median Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| `dev-1-baseline-tfidf` | 1. Baseline: TF-IDF (top_k=3, budget=800) | 7/23 (30.4%) | 12/23 (52.2%) | 3/23 (13.0%) | 241 | 0.75 |
| `dev-2-dense-pgvector` | 2. Method 2: Dense pgvector (top_k=3, budget=800) | 8/23 (34.8%) | 12/23 (52.2%) | 2/23 (8.7%) | 95 | 96.93 |
| `dev-3-lexical-bm25` | 3. Method 3: Lexical BM25 + filter (top_k=3, budget=800) | 7/23 (30.4%) | 10/23 (43.5%) | 2/23 (8.7%) | 189 | 0.98 |
| `dev-4-hybrid-rrf-b3200` | 4. Improvement 1: Hybrid RRF Dense+BM25 (top_k=3, budget=3200) | 8/23 (34.8%) | 14/23 (60.9%) | 14/23 (60.9%) | 2935 | 96.23 |
| `dev-5-hybrid-rerank-k4-b3600` | 5. Improvement 2 (Chosen): Hybrid RRF + Rerank (cand_k=15->top_k=4, budget=3600) | 10/23 (43.5%) | 15/23 (65.2%) | 16/23 (69.6%) | 2980 | 100.98 |

## 2. Per-Question Dev Progression (`Q001`–`Q023`)

Legend: `H1` = Hit@1, `H3` = Evidence Hit@3, `Ctx` = Context Evidence Hit (`1` = Hit, `0` = Miss).

| Query ID | Question | 1. TF-IDF (b=800) | 2. Dense (b=800) | 3. BM25 (b=800) | 4. Hybrid RRF (b=3200) | 5. Hybrid+Rerank (b=3600) |
|---|---|:-:|:-:|:-:|:-:|:-:|
| `Q001` | What is a monopoly? | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** |
| `Q002` | What are barriers to entry? | H1:0 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:1** |
| `Q003` | What conditions lead to a natural monopoly? | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:1** | H1:0 H3:1 **Ctx:1** |
| `Q004` | In section 9.1, why were other firms unable to compete with ALCOA in aluminum in the 1930s? | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q005` | What do governments do to create a legal monopoly? | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:1** |
| `Q006` | How do patents let innovative firms recover their research and development costs? | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:1** | H1:0 H3:1 **Ctx:1** |
| `Q007` | What is predatory pricing? | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:1** |
| `Q008` | What demand curve does a monopolist face? | H1:0 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:1** | H1:0 H3:0 **Ctx:0** |
| `Q009` | Why does total revenue for a monopolist first rise and then fall as it sells more output? | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** |
| `Q010` | Why is a monopolist's marginal revenue less than the price of the unit it sells? | H1:0 H3:1 **Ctx:1** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** |
| `Q011` | In section 9.2, what rule does a profit-maximizing monopoly follow to choose its quantity of output? | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:1 **Ctx:1** | H1:0 H3:1 **Ctx:1** |
| `Q012` | Once a monopolist has chosen its output, how does it decide what price to charge? | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** |
| `Q013` | Why is a monopoly considered allocatively inefficient? | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q014` | In what ways can a firm make its products different from those of its competitors? | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q015` | What characterizes monopolistic competition? | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:1** | H1:0 H3:1 **Ctx:1** |
| `Q016` | How does the demand curve facing a monopolistically competitive firm compare with that of a monopoly in terms of elasticity? | H1:0 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q017` | In section 10.1, what happens to monopolistically competitive firms' economic profits in the long run? | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** |
| `Q018` | How can a successful advertising campaign help a monopolistically competitive firm? | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q019` | What defines an oligopoly? | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:0 **Ctx:0** |
| `Q020` | What does mutual interdependence mean in an oligopoly? | H1:0 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q021` | How can the relationship between market demand and economies of scale limit an oligopoly to only a few firms? | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q022` | In section 10.2, what is a cartel? | H1:0 H3:0 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:0 H3:0 **Ctx:0** | H1:0 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
| `Q023` | What is game theory? | H1:1 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:0** | H1:1 H3:1 **Ctx:1** | H1:1 H3:1 **Ctx:1** |
