# Support Integrity Auditor (SIA)
### MARS Open Projects 2026 — Problem Statement 1
#### Models and Robotics Section | AI / Machine Learning / NLP

> **SIA** is a self-supervised, semantics-driven auditor that detects **Priority Mismatch** in enterprise CRM support tickets — cases where the human-assigned priority conflicts with the ticket's true severity — without any pre-annotated mismatch labels.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Architecture Diagram](#architecture-diagram)
3. [Methodology](#methodology)
4. [Signal Fusion Diagram](#signal-fusion-diagram)
5. [Pseudo-Label Generation Diagram](#pseudo-label-generation-diagram)
6. [Ablation Table](#ablation-table)
7. [Metric Results](#metric-results)
8. [Deliverables](#deliverables)
9. [How to Run](#how-to-run)

---

## Problem Statement

In enterprise CRM systems, manual ticket triage suffers from agent fatigue bias, keyword anchoring, and customer favoritism. When critical tickets are mislabeled as "Low" or trivial complaints inflated to "Critical", SLAs are breached and customer churn increases.

**SIA solves this without any pre-annotated mismatch labels** by:
- Inferring true severity from raw ticket data using 4 independent signals
- Generating its own binary mismatch supervision signal (self-supervised)
- Training a fine-tuned LightGBM classifier on pseudo-labeled data
- Producing a structured, hallucination-free Evidence Dossier for every flagged ticket

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW TICKET DATA                          │
│   Subject · Description · Channel · Priority · RT · Sat Score   │
│                     20,000 tickets                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                              │
│  • Lowercase + remove special chars → combined_text_clean       │
│  • Dict-based channel/category encoding (crash-safe)            │
│  • Email domain extraction                                      │
│  • Word count, exclamation count, satisfaction score norm       │
│  • Rows: 20,000 | Columns: 12                                   │
└──────┬──────────────┬────────────────┬──────────────────────────┘
       │              │                │
       ▼              ▼                ▼
┌───────────┐  ┌─────────────┐  ┌──────────────────┐  ┌──────────────┐
│ SIGNAL 1  │  │  SIGNAL 2   │  │    SIGNAL 3      │  │  SIGNAL 4    │
│           │  │             │  │                  │  │              │
│ Rule-Based│  │ Resolution  │  │ Sentence-        │  │ Satisfaction │
│   NLP     │  │    Time     │  │ Transformer      │  │   Score      │
│           │  │ Percentile  │  │ Embeddings +     │  │              │
│ Keywords  │  │             │  │ KMeans(16)       │  │ Inverse      │
│ Escalation│  │ percentile  │  │                  │  │ severity     │
│ Negation  │  │ vs TRAIN    │  │ all-MiniLM-L6-v2 │  │ proxy        │
│           │  │ distribution│  │ 384-dim → norm   │  │              │
│ Score:0-1 │  │ Score: 0-1  │  │ Score: 0-1       │  │ Score: 0-1   │
│ W = 0.30  │  │ W = 0.30    │  │ W = 0.25         │  │ W = 0.15     │
└─────┬─────┘  └──────┬──────┘  └────────┬─────────┘  └──────┬───────┘
      │               │                  │                   │
      └───────────────┼──────────────────┼───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WEIGHTED FUSION                              │
│                                                                 │
│  fused = 0.30×NLP + 0.30×RT + 0.25×Cluster + 0.15×Sat           │
│                                                                 │
│  Q33=0.3847  Q66=0.4949  Q85=0.5622                             │
│                                                                 │
│  Inferred Severity: Low(6594) Medium(6606) High(3800) Crit(3000)│
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               PSEUDO-LABEL GENERATION (Self-Supervised)         │
│                                                                 │
│   Z-Score Strategy (MISMATCH_THRESHOLD = 2.0):                  │
│   • Group by Priority_Level → compute group mean/std of RT      │
│   • Z = (RT - group_mean) / group_std                           │
│   • |Z| ≥ 2.0 → MISMATCH  |  |Z| < 2.0 → CONSISTENT             │
│                                                                 │
│   Result: Consistent=12,487  Hidden Crisis=4,563                |
|     False Alarm=2,950                                           |
│   Total Mismatch: 7,513 (37.57%)                                │
│                                                                 │
│   SMOTE applied → balanced to 8,741 vs 8,741                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                           │
│                                                                 │
│   • PCA(256) on normalized embeddings (explained var: 0.987)    │
│   • Metadata: nlp_score, rt_scaled, cluster_score, sat_score,   │
│               text_len, cap_ratio, channel_enc, category_enc,   │
│               hi_pri_lo_sat, lo_pri_hi_nlp                      │
│   • Final matrix: 20,000 × 272 features                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               LIGHTGBM BINARY CLASSIFIER                        │
│                                                                 │
│   n_estimators=800 · learning_rate=0.02 · num_leaves=127        │
│   class_weight=balanced · early_stopping=50                     │
│   Train: 14,000 (after SMOTE) | Test: 6,000                     │
│   Decision threshold = 0.32                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              EVIDENCE DOSSIER GENERATOR                         │
│                                                                 │
│   2,132 dossiers generated for flagged test tickets             │
│   Every feature_evidence traceable to real input field          │
│   Zero hallucination by design (assert guard enforced)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Methodology

### Stage 1 — Pseudo-Label Generation (Self-Supervised)

Since no pre-annotated mismatch labels exist, SIA generates pseudo-labels using multi-signal severity inference.

A fused severity score is computed using:
- NLP urgency score
- Resolution-time percentile
- Semantic cluster severity
- Customer satisfaction signal

The inferred severity is then compared against the original human-assigned priority level.

Mismatch labels are generated using severity disagreement:

- Positive severity delta - Hidden Crisis
- Negative severity delta - False Alarm
- Small/no delta - Consistent

**Resolution time by priority group (actual data):**

| Priority | Avg RT (hours) |
|----------|---------------|
| Critical | 12.07 |
| High     | 24.52 |
| Medium   | 44.47 |
| Low      | 45.17 |

Four independent signals feed the fused severity score:

| Signal | Method | Weight | Score Range |
|--------|--------|--------|-------------|
| NLP Severity | Keyword density: critical (+3), high (+1.5), low (−1.5) | 0.30 | 0.0 – 0.75 |
| Resolution Time | `percentileofscore` vs training distribution | 0.30 | 0.014 – 0.967 |
| Semantic Cluster | KMeans(16) on MiniLM embeddings, scored by mean RT | 0.25 | 0.0 – 1.0 |
| Satisfaction Score | Inverse customer satisfaction proxy | 0.15 | 0.0 – 1.0 |

**Conflict-aware features** (added to classifier input):

- `hi_pri_lo_sat` — high assigned priority despite strong customer satisfaction
- `lo_pri_hi_nlp` — low assigned priority despite strong urgency language

These features explicitly model disagreement between human-assigned priority and behavioral severity signals.

### Stage 2 — Classifier Training

- **Model:** LightGBM binary classifier
- **Features:** PCA(256) normalized embeddings + 16 metadata + gap features = 272 total
- **Imbalance:** SMOTE (k=5) on training set + `class_weight='balanced'`
- **Threshold:** 0.32 (tuned by scanning validation set)
- **Train/Test:** 14,000 / 6,000 stratified split

### Stage 3 — Evidence Dossier

Every flagged ticket gets a structured JSON dossier traceable to real fields:

```json
{
  "ticket_id": "TKT-118429",
  "assigned_priority": "High",
  "inferred_severity": "Low",
  "mismatch_type": "False Alarm",
  "severity_delta": -2,
  "feature_evidence": [
    {
      "signal": "keyword",
      "value": "none detected",
      "weight": "30%",
      "source_field": "Ticket_Description"
    },
    {
      "signal": "resolution_time",
      "value": "3 hours",
      "interpretation": "Fast resolution time → low/moderate severity signal",
      "source_field": "Resolution_Time_Hours"
    },
    {
      "signal": "semantic_cluster",
      "value": "Cluster 14 (score 0.00) — low-urgency peer group",
      "weight": "25%",
      "source_field": "Ticket_Subject + Ticket_Description (embedding)"
    }
  ],
  "constraint_analysis": "Ticket assigned 'High' but signals indicate 'Low' severity (delta=-2). Routine language, fast resolution of 3h, and placement in a low-urgency semantic cluster suggest the priority is inflated, consuming support bandwidth unnecessarily.",
  "confidence": 0.8821
}
```

---

## Signal Fusion Diagram

```
                    ┌─────────────────────┐
                    │   TICKET TEXT +     │
                    │     METADATA        │
                    └──────────┬──────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
       ▼                       ▼                       ▼
┌─────────────┐      ┌──────────────┐       ┌──────────────────┐
│  NLP Score  │      │   RT Score   │       │  Cluster Score   │
│             │      │              │       │                  │
│  mean=0.095 │      │  mean=0.500  │       │  16 clusters     │
│  max=0.750  │      │  std=0.289   │       │  scored 0.0→1.0  │
│  8919/20000 │      │              │       │  by avg RT       │
│  non-zero   │      │              │       │                  │
└──────┬──────┘      └──────┬───────┘       └────────┬─────────┘
       │                    │                        │
       │ × 0.30             │ × 0.30                 │ × 0.25
       │                    │                        │
       └────────────────────┼────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │ + Sat Score     │
                   │   × 0.15        │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  FUSED SCORE    │
                   │  mean ≈ 0.42    │
                   └────────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       < Q33            Q33–Q85           > Q85
       (0.3847)        (0.3847–          (0.5622)
          │              0.5622)              │
          ▼                 │                 ▼
        Low /               ▼              Critical
       Medium           Medium /
                          High
```

---

## Pseudo-Label Generation Diagram

```
  ALL TICKETS (20,000)
         │
         ▼
  ┌──────────────────────────────────────────┐
  │      GROUP BY Priority_Level             │
  │                                          │
  │  Critical → mean=12.07h,  std=σ₀         │
  │  High     → mean=24.52h,  std=σ₁         │
  │  Medium   → mean=44.47h,  std=σ₂         │
  │  Low      → mean=45.17h,  std=σ₃         │
  └──────────────────┬───────────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────────┐
  │   For each ticket:                       │
  │   Z = (RT - group_mean) / group_std      │
  │   MISMATCH_THRESHOLD = 2.0               │
  └──────────────────┬───────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
     Z ≥ +2.0    -2 < Z < +2   Z ≤ -2.0
        │            │            │
        ▼            ▼            ▼
    MISMATCH    CONSISTENT    MISMATCH
    label=1      label=0      label=1
       │                         │
       ▼                         ▼
  Hidden Crisis             False Alarm
 (under-prioritised)     (over-prioritised)
   4,563 tickets            2,950 tickets

  ┌──────────────────────────────────────────┐
  │         FINAL DISTRIBUTION               │
  │  Consistent  : 12,487  (62.43%)          │
  │  Mismatch    :  7,513  (37.57%)          │
  │    Hidden Crisis: 4,563                  │
  │    False Alarm : 2,950                   │
  └──────────────────────────────────────────┘
                     │
                     ▼
         SMOTE (k=5) on training set
         → balanced: 8,741 vs 8,741
```

---

## Ablation Table

Each row shows Macro F1 when one signal group is removed from the feature matrix. The classifier is retrained identically for each configuration.

| Model Variant | Macro F1 | Δ vs Full | Finding |
|--------------|----------|-----------|---------|
| **Full Model (all signals)** | **0.8904** | — | Baseline |
| Without NLP | 0.8913 | +0.0009 | NLP overlaps with other signals |
| Without Satisfaction | 0.8906 | +0.0002 | Minor additive contribution |
| Without Clustering | 0.8903 | −0.0001 | Marginal but consistent |
| Without Resolution Time | 0.8889 | −0.0015 | RT most important single signal |
| Without Conflict Features | 0.8097 | −0.0807 | **Gap features are critical** |

- **Conflict-aware features** (`hi_pri_lo_sat`, `lo_pri_hi_nlp`) contribute the largest performance gain — removing them drops Macro F1 by 0.08
- **Resolution Time** is the most important base signal (−0.0015 without it)
- **NLP and resolution-time signals** capture different aspects of ticket severity, resulting in low direct correlation but complementary predictive value during fusion
- Full fusion consistently outperforms any single signal

---

## Metric Results

### Final Evaluation (Test Set — 6,000 tickets)

| Metric | Score | Threshold |
|--------|-------|-----------|
| Binary Accuracy | **0.8898** |
| Macro F1 Score | **0.8862** |
| Recall (Consistent) | **0.8561** | 
| Recall (Mismatch) | **0.9459** |
| Cohen's Kappa | **0.7735** | 

### Per-Class Breakdown

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Consistent | 0.96 | 0.86 | 0.91 | 3,746 |
| Mismatch | 0.80 | 0.95 | 0.87 | 2,254 |
| **Accuracy** | | | **0.89** | **6,000** |
| Macro avg | 0.88 | 0.90 | 0.89 | 6,000 |
| Weighted avg | 0.90 | 0.89 | 0.89 | 6,000 |

### Pseudo-Label Quality

| Metric | Value |
|--------|-------|
| Total tickets | 20,000 |
| Mismatch (pseudo-labeled) | 7,513 (37.57%) |
| Hidden Crisis | 4,563 |
| False Alarm | 2,950 |
| Signal Agreement (NLP vs RT) | −0.039 |
| Dossiers generated | 2,132 |

### Adversarial Robustness Test

| Ticket | Type | Result |
|--------|------|--------|
| ADV-001 | Hidden Crisis | Correct |
| ADV-002 | Hidden Crisis | Correct |
| ADV-003 | Hidden Crisis | Correct |
| ADV-004 | Hidden Crisis | Correct |
| ADV-005 | Hidden Crisis | Correct |
| ADV-006 | False Alarm | Incorrect |
| ADV-007 | False Alarm | Correct |
| ADV-008 | False Alarm | Incorrect |
| ADV-009 | False Alarm | Incorrect |
| ADV-010 | False Alarm | Incorrect |

**Score: 6/10** — below bonus threshold (7/10). Model detects Hidden Crisis well but struggles with False Alarms where subjects contain alarm keywords but descriptions are trivial.

---

## Deliverables

```
SIA/
├── notebook.ipynb            # Full reproducible pipeline
├── train_pipeline.py         # Standalone training script
├── predict.py                # Inference: CSV in → predictions + dossiers
├── README.md                 # This file
├── requirements.txt          # Pinned dependencies
├── sia_model.pkl             # Trained model + all preprocessing objects
├── evidence_dossiers.json    # 2,132 dossiers for flagged test tickets
├── sia_predictions.csv       # Test-set predictions
└── adversarial_tickets.csv   # 10 adversarial test cases
|__ streamline web app
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline
```bash
jupyter notebook notebook.ipynb
# Run all cells top to bottom
```

### 3. Inference on new tickets
```python
result_df, dossiers = predict_csv('my_tickets.csv')
```

### 4. Load saved model
```python
import pickle
with open('sia_model.pkl', 'rb') as f:
    model_bundle = pickle.load(f)
```

---

## Dataset

**Customer Support Tickets — CRM Dataset**
Source: [kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset](https://kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset/data)
Size: 20,000 tickets | 12 columns
Key columns: `Ticket_Subject`, `Ticket_Description`, `Priority_Level`, `Ticket_Channel`, `Resolution_Time_Hours`, `Issue_Category`, `Satisfaction_Score`

---

*MARS — Models and Robotics Section | Open Projects 2026*
