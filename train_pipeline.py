
  '''Support Integrity Auditor (SIA) — Standalone Training Script
  Usage: python train_pipeline.py
  Output: sia_model.pkl, sia_predictions.csv, evidence_dossiers.json'''


import os
import re
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, cohen_kappa_score
)
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
import lightgbm as lgb

warnings.filterwarnings("ignore")
#  CONFIG

DATA_PATH        = "enhanced_customer_support_data.csv"
MODEL_OUTPUT     = "sia_model.pkl"
PREDICTIONS_OUT  = "sia_predictions.csv"
DOSSIERS_OUT     = "evidence_dossiers.json"
EMBEDDINGS_CACHE = "ticket_embeddings.npy"

SEED             = 63
N_CLUSTERS       = 16
PCA_DIMS         = 256
MISMATCH_THRESHOLD = 2.0
BEST_THRESH      = 0.32      

W_NLP, W_RT, W_CLUSTER, W_SAT = 0.30, 0.30, 0.25, 0.15

np.random.seed(SEED)


#  KEYWORD LISTS

PRIORITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
INV_PRIORITY_MAP = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}

CRITICAL_KW = [
    'urgent', 'critical', 'immediately', 'emergency', 'asap',
    'data loss', 'breach', 'fraud', 'outage', 'down', 'broken',
    'escalate', 'legal', 'lawsuit', 'refund', 'charge back',
    'cannot access', 'account locked', 'suspended', 'deleted',
    'not received', 'never arrived', 'still waiting', 'no response',
    'threatening', 'dispute', 'unauthorized', 'blocked', 'corrupted',
    'crash', 'crashed', 'loss of data', 'system failure', 'completely down',
    'business impact', 'revenue loss', 'customers affected', 'production down',
    'cannot process', 'payment down', 'pii exposed', 'legal team',
    'chargeback', 'losing revenue', 'all users', 'account deleted'
]
HIGH_KW = [
    'not working', 'error', 'fail', 'failed', 'issue', 'problem',
    'unable', 'incorrect', 'wrong', 'missing', 'delayed',
    'slow', 'timeout', 'not loading', 'keeps failing', 'broken link',
    'not syncing', 'not saving', 'not sending', 'not receiving',
    'frustrating', 'disappointed', 'unacceptable', 'terrible',
    'not responding', 'keeps crashing', 'data missing', 'access denied'
]
LOW_KW = [
    'how to', 'question', 'information', 'curious', 'wondering',
    'where is', 'hours', 'location', 'update', 'just checking',
    'quick question', 'when you get a chance', 'no rush',
    'at your convenience', 'minor', 'small', 'suggestion', 'feedback',
    'slightly slow', 'seems fine now', 'just wanted to mention',
    'forgot my password', 'reset link worked', 'all good now',
    'display name', 'font size', 'visual suggestion'
]

META_COLS = [
    'nlp_score', 'rt_scaled', 'cluster_score', 'fused_score',
    'text_len', 'cap_ratio', 'channel_enc', 'category_enc',
    'sat_score', 'word_count', 'exclamation_count',
    'crit_kw_count', 'low_kw_count',
    'hi_pri_lo_sat', 'lo_pri_hi_nlp', 'rt_score',
]


#  UTILITY FUNCTIONS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def nlp_severity_score(text):
    score = 0.0
    for kw in CRITICAL_KW:
        if kw in text: score += 3.0
    for kw in HIGH_KW:
        if kw in text: score += 1.5
    for kw in LOW_KW:
        if kw in text: score -= 2.0
    return float(np.clip(score / 12.0, 0, 1))

def rt_percentile_score(rt_val, reference):
    return percentileofscore(reference, rt_val, kind='rank') / 100.0

def score_to_severity(s, q33, q66, q85):
    if   s >= q85: return 3
    elif s >= q66: return 2
    elif s >= q33: return 1
    else:          return 0

def get_mismatch_label(delta):
    return 1 if abs(delta) >= MISMATCH_THRESHOLD else 0

def get_mismatch_type(delta):
    if delta >= MISMATCH_THRESHOLD:    return 'Hidden Crisis'
    elif delta <= -MISMATCH_THRESHOLD: return 'False Alarm'
    return 'Consistent'

def build_dossier(row, proba):
    assert row['mismatch_type'] != 'Consistent', \
        f"build_dossier called on Consistent ticket: {row['Ticket_ID']}"

    kw_hits = [kw for kw in CRITICAL_KW + HIGH_KW
               if kw in str(row['combined_text_clean'])]

    if row['Resolution_Time_Hours'] >= 72:
        rt_interp = 'Extremely long resolution time → strong high-severity signal'
    elif row['Resolution_Time_Hours'] >= 36:
        rt_interp = 'Elevated resolution time → moderate/high severity signal'
    else:
        rt_interp = 'Fast resolution time → low/moderate severity signal'

    cluster_interp = (
        f"Cluster {int(row['cluster'])} "
        f"(severity score {row['cluster_score']:.2f}) — "
        + ('high-urgency peer group' if row['cluster_score'] > 0.5 else 'low-urgency peer group')
    )

    sat_interp = (
        f"Satisfaction score {int(row['Satisfaction_Score'])}/5 — "
        + ('low satisfaction may indicate unresolved operational friction'
           if row['Satisfaction_Score'] <= 2
           else 'satisfaction does not strongly indicate unresolved escalation')
    )

    feature_evidence = [
        {'signal': 'keyword',
         'value': ', '.join(kw_hits[:5]) if kw_hits else 'none detected',
         'weight': f'{W_NLP:.0%}', 'source_field': 'Ticket_Description'},
        {'signal': 'resolution_time',
         'value': f"{row['Resolution_Time_Hours']} hours",
         'interpretation': rt_interp, 'source_field': 'Resolution_Time_Hours'},
        {'signal': 'semantic_cluster', 'value': cluster_interp,
         'weight': f'{W_CLUSTER:.0%}',
         'source_field': 'Ticket_Subject + Ticket_Description (embedding)'},
        {'signal': 'satisfaction', 'value': sat_interp,
         'weight': f'{W_SAT:.0%}', 'source_field': 'Satisfaction_Score'}
    ]

    delta  = int(row['severity_delta'])
    assign = str(row['Priority_Level'])
    infer  = str(row['inferred_severity'])
    mtype  = str(row['mismatch_type'])

    strong_kw = ['refund','fraud','crash','down','breach','critical',
                 'emergency','lawsuit','data loss']
    if any(kw in kw_hits for kw in strong_kw):
        lang = 'Strong escalation language detected in the description'
    elif kw_hits:
        lang = 'Moderate urgency indicators detected in the description'
    else:
        lang = 'Behavioral and temporal signals indicate elevated severity'

    if mtype == 'Hidden Crisis':
        analysis = (
            f"Ticket assigned '{assign}' but multi-signal analysis indicates '{infer}' severity "
            f"(delta={delta:+d} levels). {lang} and a resolution time of "
            f"{row['Resolution_Time_Hours']}h suggest the issue is more severe than labeled."
        )
    else:
        cluster_reason = ('low-urgency peer patterns' if row['cluster_score'] < 0.5
                          else 'mixed urgency behavioral patterns')
        rt_phrase = ('fast resolution' if row['Resolution_Time_Hours'] <= 24
                     else ('moderate resolution duration' if row['Resolution_Time_Hours'] <= 72
                           else 'extended resolution duration'))
        analysis = (
            f"Ticket assigned '{assign}' but multi-signal analysis indicates '{infer}' severity "
            f"(delta={delta:+d} levels). Routine language, {rt_phrase} "
            f"({row['Resolution_Time_Hours']}h), and {cluster_reason} suggest "
            f"the assigned priority may be inflated relative to observed behavioral signals."
        )

    return {
        'ticket_id':           str(row['Ticket_ID']),
        'assigned_priority':   assign,
        'inferred_severity':   infer,
        'mismatch_type':       mtype,
        'severity_delta':      delta,
        'feature_evidence':    feature_evidence,
        'constraint_analysis': analysis,
        'confidence':          round(float(proba), 4)
    }


#  STEP 1 — LOAD DATA


print("  SIA Training Pipeline")


print("\n[1/9] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Priority distribution:\n{df['Priority_Level'].value_counts()}")


#  STEP 2 — PREPROCESSING

print("\n[2/9] Preprocessing...")

df['combined_text'] = (df['Ticket_Subject'].fillna('') + ' ' +
                       df['Ticket_Description'].fillna(''))
df['combined_text_clean'] = df['combined_text'].apply(clean_text)
df['priority_num'] = df['Priority_Level'].map(PRIORITY_MAP)
df = df.dropna(subset=['priority_num']).copy()
df['priority_num'] = df['priority_num'].astype(int)

CHANNEL_MAP  = {v:i for i,v in enumerate(sorted(df['Ticket_Channel'].fillna('Unknown').unique()))}
CATEGORY_MAP = {v:i for i,v in enumerate(sorted(df['Issue_Category'].fillna('Unknown').unique()))}
df['channel_enc']  = df['Ticket_Channel'].fillna('Unknown').map(CHANNEL_MAP)
df['category_enc'] = df['Issue_Category'].fillna('Unknown').map(CATEGORY_MAP)

def extract_domain(email):
    return str(email).split('@')[-1].lower()

df['email_domain'] = df['Customer_Email'].apply(extract_domain)
TOP_DOMAINS = df['email_domain'].value_counts().head(20).index
DOMAIN_MAP  = {d:i for i,d in enumerate(TOP_DOMAINS)}
df['domain_enc'] = df['email_domain'].apply(lambda x: DOMAIN_MAP.get(x, -1))

df['word_count']        = df['combined_text_clean'].apply(lambda x: len(str(x).split()))
df['exclamation_count'] = df['combined_text'].apply(lambda x: str(x).count('!'))
df['text_len']          = df['combined_text_clean'].apply(len)
df['cap_ratio']         = df['combined_text'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
df['satisfaction_inv']  = 6 - df['Satisfaction_Score']
df['sat_score']         = (df['satisfaction_inv'] - 1) / 4.0

print(f"  Rows after cleaning: {len(df)}")


#  STEP 3 — NLP SIGNAL

print("\n[3/9] Computing NLP signal...")
tqdm.pandas()
df['nlp_score'] = df['combined_text_clean'].progress_apply(nlp_severity_score)
df['crit_kw_count'] = df['combined_text_clean'].apply(
    lambda t: sum(1 for k in CRITICAL_KW if k in t))
df['low_kw_count']  = df['combined_text_clean'].apply(
    lambda t: sum(1 for k in LOW_KW if k in t))
print(f"  NLP score mean: {df['nlp_score'].mean():.3f}")


#  STEP 4 — RT SIGNAL

print("\n[4/9] Computing Resolution Time signal...")
TRAIN_RT_VALUES = df['Resolution_Time_Hours'].values.copy()
df['rt_score'] = df['Resolution_Time_Hours'].apply(
    lambda x: rt_percentile_score(x, TRAIN_RT_VALUES))
scaler = StandardScaler()
df['rt_scaled'] = scaler.fit_transform(df[['Resolution_Time_Hours']]).flatten()
print(f"  RT score mean: {df['rt_score'].mean():.3f}")


#  STEP 5 — EMBEDDINGS + CLUSTERING

print("\n[5/9] Generating embeddings & clustering...")

if os.path.exists(EMBEDDINGS_CACHE):
    print(f"  Loading cached embeddings from {EMBEDDINGS_CACHE}...")
    embeddings = np.load(EMBEDDINGS_CACHE)
else:
    print("  Encoding with sentence-transformers (2-3 min on CPU)...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = st_model.encode(
        df['combined_text_clean'].tolist(),
        batch_size=256, show_progress_bar=True
    )
    embeddings = normalize(embeddings)
    np.save(EMBEDDINGS_CACHE, embeddings)
    print(f"  Embeddings saved to {EMBEDDINGS_CACHE}")

print(f"  Embeddings shape: {embeddings.shape}")

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
df['cluster'] = kmeans.fit_predict(embeddings)

cluster_rt_mean  = df.groupby('cluster')['rt_score'].mean()
_min, _max       = cluster_rt_mean.min(), cluster_rt_mean.max()
cluster_severity = (cluster_rt_mean - _min) / (_max - _min + 1e-9)
CLUSTER_SEVERITY_MAP   = cluster_severity.to_dict()
DEFAULT_CLUSTER_SCORE  = float(cluster_severity.mean())
df['cluster_score'] = df['cluster'].apply(
    lambda c: CLUSTER_SEVERITY_MAP.get(c, DEFAULT_CLUSTER_SCORE))
print(f"  Clusters: {N_CLUSTERS} | Default score: {DEFAULT_CLUSTER_SCORE:.3f}")


#  STEP 6 — FUSION & SEVERITY INFERENCE

print("\n[6/9] Fusing signals & inferring severity...")

df['fused_score'] = (
    W_NLP     * df['nlp_score']     +
    W_RT      * df['rt_score']      +
    W_CLUSTER * df['cluster_score'] +
    W_SAT     * df['sat_score']
)

Q33 = float(df['fused_score'].quantile(0.33))
Q66 = float(df['fused_score'].quantile(0.66))
Q85 = float(df['fused_score'].quantile(0.85))

df['inferred_severity_num'] = df['fused_score'].apply(
    lambda s: score_to_severity(s, Q33, Q66, Q85))
df['inferred_severity'] = df['inferred_severity_num'].map(INV_PRIORITY_MAP)
print(f"  Thresholds → Q33={Q33:.4f}  Q66={Q66:.4f}  Q85={Q85:.4f}")

# Cross-signal flags
df['hi_pri_lo_sat'] = ((df['priority_num'] >= 2) & (df['Satisfaction_Score'] >= 4)).astype(int)
df['lo_pri_hi_nlp'] = ((df['priority_num'] <= 1) & (df['nlp_score'] >= 0.3)).astype(int)


#  STEP 7 — PSEUDO-LABELS

print("\n[7/9] Generating pseudo-labels...")

df['severity_delta'] = df['inferred_severity_num'] - df['priority_num']
df['mismatch_label'] = df['severity_delta'].apply(get_mismatch_label)
df['mismatch_type']  = df['severity_delta'].apply(get_mismatch_type)

print(f"  Mismatch distribution:\n{df['mismatch_label'].value_counts()}")
print(f"  Mismatch type:\n{df['mismatch_type'].value_counts()}")

signal_agreement = np.corrcoef(df['nlp_score'], df['rt_score'])[0, 1]
print(f"  Signal Agreement (NLP vs RT): {signal_agreement:.3f}")


#  STEP 8 — FEATURE MATRIX

print("\n[8/9] Building feature matrix & training...")

pca = PCA(n_components=PCA_DIMS, random_state=SEED)
emb_reduced = pca.fit_transform(embeddings)
print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

X = np.hstack([emb_reduced, df[META_COLS].values])
y = df['mismatch_label'].values
print(f"  Feature matrix: {X.shape}")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index.values, test_size=0.3, random_state=SEED, stratify=y)

print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# SMOTE oversampling
minority_count = int((y_train == 1).sum())
if minority_count >= 6:
    k = min(5, minority_count - 1)
    X_train_res, y_train_res = SMOTE(
        random_state=SEED, k_neighbors=k).fit_resample(X_train, y_train)
    print(f"  SMOTE(k={k}) → 0: {(y_train_res==0).sum()} | 1: {(y_train_res==1).sum()}")
else:
    X_train_res, y_train_res = X_train, y_train
    print(f"  Skipping SMOTE ({minority_count} minority samples) — using class_weight=balanced")

# LightGBM
clf = lgb.LGBMClassifier(
    objective='binary', metric='binary_logloss',
    n_estimators=800, learning_rate=0.02,
    num_leaves=127, min_child_samples=20,
    subsample=0.85, colsample_bytree=0.7,
    reg_alpha=0.05, reg_lambda=0.1,
    class_weight='balanced', random_state=SEED,
    n_jobs=-1, verbose=-1
)
clf.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)]
)
print("  Training complete!")


#  STEP 9 — EVALUATION

y_proba = clf.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= BEST_THRESH).astype(int)

acc      = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
recalls  = recall_score(y_test, y_pred, average=None)
kappa    = cohen_kappa_score(y_test, y_pred)


print("  EVALUATION RESULTS")

print(f"  Binary Accuracy  : {acc:.4f}")
print(f"  Macro F1         : {macro_f1:.4f}")
print(f"  Recall (class 0) : {recalls[0]:.4f}")
print(f"  Recall (class 1) : {recalls[1]:.4f}")
print(f"  Cohen's Kappa    : {kappa:.4f}")
print(classification_report(y_test, y_pred, target_names=['Consistent', 'Mismatch']))


#  SAVE MODEL

with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump({
        'clf': clf, 'pca': pca, 'scaler': scaler, 'kmeans': kmeans,
        'CHANNEL_MAP': CHANNEL_MAP, 'CATEGORY_MAP': CATEGORY_MAP,
        'CLUSTER_SEVERITY_MAP': CLUSTER_SEVERITY_MAP,
        'DEFAULT_CLUSTER_SCORE': DEFAULT_CLUSTER_SCORE,
        'TRAIN_RT_VALUES': TRAIN_RT_VALUES,
        'Q33': Q33, 'Q66': Q66, 'Q85': Q85,
        'W_NLP': W_NLP, 'W_RT': W_RT, 'W_CLUSTER': W_CLUSTER, 'W_SAT': W_SAT,
        'MISMATCH_THRESHOLD': MISMATCH_THRESHOLD,
        'META_COLS': META_COLS, 'best_thresh': BEST_THRESH,
    }, f)
print(f"\n Model saved → {MODEL_OUTPUT}")

# Save predictions
test_df = df.loc[idx_test].copy()
test_df['pred_label'] = y_pred
test_df['pred_proba'] = y_proba
out_cols = ['Ticket_ID', 'Priority_Level', 'inferred_severity',
            'severity_delta', 'mismatch_type', 'pred_label', 'pred_proba']
test_df[out_cols].to_csv(PREDICTIONS_OUT, index=False)
print(f" Predictions saved → {PREDICTIONS_OUT}")

# Save dossiers
flagged  = test_df[(test_df['pred_label'] == 1) & (test_df['mismatch_type'] != 'Consistent')]
dossiers = [build_dossier(row, row['pred_proba']) for _, row in flagged.iterrows()]
with open(DOSSIERS_OUT, 'w') as f:
    json.dump(dossiers, f, indent=2)
print(f" Dossiers saved  → {DOSSIERS_OUT}  ({len(dossiers)} flagged tickets)")
print("\nTraining complete ")