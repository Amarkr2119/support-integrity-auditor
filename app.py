# Support Integrity Auditor (SIA) — Streamlit Web App
# MARS Open Projects 2026

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import json
import warnings
warnings.filterwarnings('ignore')

# ── Page config

st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🛡️",
    layout="wide"
)

# ── Simple CSS 

st.markdown("""
<style>
    .main-title { font-size:2.5rem; font-weight:bold; color:#1f4e79; text-align:center; }
    .sub-title  { font-size:1.1rem; color:#555; text-align:center; margin-bottom:2rem; }
</style>
""", unsafe_allow_html=True)

# 1. LOAD MODEL

@st.cache_resource
def load_model():
    with open('sia_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

# 2. CONSTANTS & HELPER FUNCTIONS (exact copy from notebook)

PRIORITY_MAP     = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
INV_PRIORITY_MAP = {v: k for k, v in PRIORITY_MAP.items()}

CRITICAL_KW = [
    'urgent','critical','immediately','emergency','asap','data loss','breach',
    'fraud','outage','down','broken','escalate','legal','lawsuit','refund',
    'charge back','cannot access','account locked','suspended','deleted',
    'not received','never arrived','still waiting','no response','threatening',
    'dispute','unauthorized','blocked','corrupted','crash','crashed',
    'loss of data','system failure','completely down','business impact',
    'revenue loss','customers affected','production down','cannot process',
    'payment down','pii exposed','legal team','chargeback','losing revenue',
    'all users','account deleted'
]
HIGH_KW = [
    'not working','error','fail','failed','issue','problem','unable',
    'incorrect','wrong','missing','delayed','slow','timeout','not loading',
    'keeps failing','broken link','not syncing','not saving','not sending',
    'not receiving','frustrating','disappointed','unacceptable','terrible',
    'not responding','keeps crashing','data missing','access denied'
]
LOW_KW = [
    'how to','question','information','curious','wondering','where is',
    'hours','location','update','just checking','quick question',
    'when you get a chance','no rush','at your convenience','minor','small',
    'suggestion','feedback','slightly slow','seems fine now',
    'just wanted to mention','forgot my password','reset link worked',
    'all good now','display name','font size','visual suggestion'
]


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
    from scipy.stats import percentileofscore
    return percentileofscore(reference, rt_val, kind='rank') / 100.0


def score_to_severity(s, Q33, Q66, Q85):
    if   s >= Q85: return 3
    elif s >= Q66: return 2
    elif s >= Q33: return 1
    else:          return 0


def get_mismatch_type(delta, threshold=2.0):
    if   delta >=  threshold: return 'Hidden Crisis'
    elif delta <= -threshold: return 'False Alarm'
    else:                     return 'Consistent'


def build_dossier(row, proba, W_NLP, W_CLUSTER, W_SAT):
    kw_hits = [kw for kw in CRITICAL_KW + HIGH_KW
               if kw in str(row.get('combined_text_clean', ''))]
    rt_h = float(row.get('Resolution_Time_Hours', 0))

    if rt_h >= 72:
        rt_interp = 'Extremely long resolution time → strong high-severity signal'
    elif rt_h >= 36:
        rt_interp = 'Elevated resolution time → moderate/high severity signal'
    else:
        rt_interp = 'Fast resolution time → low/moderate severity signal'

    cluster_score  = float(row.get('cluster_score', 0.5))
    cluster_interp = (
        f"Cluster {int(row.get('cluster', 0))} "
        f"(severity score {cluster_score:.2f}) — "
        + ('high-urgency peer group' if cluster_score > 0.5 else 'low-urgency peer group')
    )
    sat = float(row.get('Satisfaction_Score', 3))
    sat_interp = (
        f"Satisfaction score {int(sat)}/5 — "
        + ('low satisfaction may indicate unresolved operational friction'
           if sat <= 2
           else 'satisfaction level does not indicate unresolved escalation')
    )

    feature_evidence = [
        {'signal': 'keyword',
         'value': ', '.join(kw_hits[:5]) if kw_hits else 'none detected',
         'weight': f'{W_NLP:.0%}', 'source_field': 'Ticket_Description'},
        {'signal': 'resolution_time',
         'value': f"{rt_h} hours",
         'interpretation': rt_interp, 'source_field': 'Resolution_Time_Hours'},
        {'signal': 'semantic_cluster',
         'value': cluster_interp,
         'weight': f'{W_CLUSTER:.0%}',
         'source_field': 'Ticket_Subject + Ticket_Description (embedding)'},
        {'signal': 'satisfaction',
         'value': sat_interp,
         'weight': f'{W_SAT:.0%}', 'source_field': 'Satisfaction_Score'}
    ]

    delta  = int(row.get('severity_delta', 0))
    assign = str(row.get('Priority_Level', ''))
    infer  = str(row.get('inferred_severity', ''))
    mtype  = str(row.get('mismatch_type', ''))

    strong_kw = ['refund','fraud','crash','down','breach',
                 'critical','emergency','lawsuit','data loss']
    if any(kw in kw_hits for kw in strong_kw):
        language_reason = 'Strong escalation language detected in the description'
    elif kw_hits:
        language_reason = 'Moderate urgency indicators detected in the description'
    else:
        language_reason = 'Behavioral and temporal signals indicate elevated severity'

    if mtype == 'Hidden Crisis':
        analysis = (
            f"Ticket assigned '{assign}' but multi-signal analysis indicates '{infer}' "
            f"severity (delta={delta:+d} levels). {language_reason} and a resolution time of "
            f"{rt_h}h suggest the issue is more severe than labeled, creating elevated "
            f"operational and customer-support risk."
        )
    else:
        cluster_reason = ('low-urgency peer patterns' if cluster_score < 0.5
                          else 'mixed urgency behavioral patterns')
        rt_phrase = ('fast resolution' if rt_h <= 24
                     else 'moderate resolution duration' if rt_h <= 72
                     else 'extended resolution duration')
        analysis = (
            f"Ticket assigned '{assign}' but multi-signal analysis indicates '{infer}' "
            f"severity (delta={delta:+d} levels). Routine language, {rt_phrase} "
            f"({rt_h}h), and {cluster_reason} suggest the assigned priority may be "
            f"inflated relative to observed behavioral signals."
        )

    return {
        'ticket_id':           str(row.get('Ticket_ID', 'N/A')),
        'assigned_priority':   assign,
        'inferred_severity':   infer,
        'mismatch_type':       mtype,
        'severity_delta':      delta,
        'feature_evidence':    feature_evidence,
        'constraint_analysis': analysis,
        'confidence':          round(float(proba), 4)
    }

# 3. INFERENCE PIPELINE

def run_inference(df_input, bundle, st_model):
    clf                   = bundle['clf']
    pca                   = bundle['pca']
    scaler                = bundle['scaler']
    kmeans                = bundle['kmeans']
    CHANNEL_MAP           = bundle['CHANNEL_MAP']
    CATEGORY_MAP          = bundle['CATEGORY_MAP']
    CLUSTER_SEVERITY_MAP  = bundle['CLUSTER_SEVERITY_MAP']
    DEFAULT_CLUSTER_SCORE = bundle['DEFAULT_CLUSTER_SCORE']
    TRAIN_RT_VALUES       = bundle['TRAIN_RT_VALUES']
    Q33, Q66, Q85         = bundle['Q33'], bundle['Q66'], bundle['Q85']
    W_NLP                 = bundle['W_NLP']
    W_RT                  = bundle['W_RT']
    W_CLUSTER             = bundle['W_CLUSTER']
    W_SAT                 = bundle['W_SAT']
    META_COLS             = bundle['META_COLS']
    best_thresh           = bundle['best_thresh']
    MISMATCH_THRESHOLD    = bundle['MISMATCH_THRESHOLD']

    new_df = df_input.copy()

    # Text preprocessing
    new_df['combined_text']       = new_df['Ticket_Subject'].fillna('') + ' ' + new_df['Ticket_Description'].fillna('')
    new_df['combined_text_clean'] = new_df['combined_text'].apply(clean_text)

    # Priority encoding
    new_df['priority_num'] = new_df['Priority_Level'].map(PRIORITY_MAP)
    new_df = new_df.dropna(subset=['priority_num']).copy()
    new_df['priority_num'] = new_df['priority_num'].astype(int)

    # Channel / category encoding
    new_df['channel_enc']  = new_df['Ticket_Channel'].fillna('Unknown').apply(lambda x: CHANNEL_MAP.get(x, -1))
    new_df['category_enc'] = new_df['Issue_Category'].fillna('Unknown').apply(lambda x: CATEGORY_MAP.get(x, -1))

    # Signal 1: NLP
    new_df['nlp_score'] = new_df['combined_text_clean'].apply(nlp_severity_score)

    # Signal 2: Resolution time
    new_df['rt_score'] = new_df['Resolution_Time_Hours'].apply(
        lambda x: rt_percentile_score(x, TRAIN_RT_VALUES)
    )

    # Signal 3: Embeddings + clusters
    new_emb = st_model.encode(new_df['combined_text_clean'].tolist(),
                               batch_size=64, show_progress_bar=False)
    new_df['cluster']       = kmeans.predict(new_emb)
    new_df['cluster_score'] = new_df['cluster'].apply(
        lambda c: CLUSTER_SEVERITY_MAP.get(int(c), DEFAULT_CLUSTER_SCORE)
    )

    # Signal 4: Satisfaction
    if 'Satisfaction_Score' not in new_df.columns:
        new_df['Satisfaction_Score'] = 3
    new_df['satisfaction_inv'] = 6 - new_df['Satisfaction_Score'].fillna(3)
    new_df['sat_score']        = (new_df['satisfaction_inv'] - 1) / 4.0

    # Fusion
    new_df['fused_score'] = (
        W_NLP     * new_df['nlp_score']     +
        W_RT      * new_df['rt_score']      +
        W_CLUSTER * new_df['cluster_score'] +
        W_SAT     * new_df['sat_score']
    )
    new_df['inferred_severity_num'] = new_df['fused_score'].apply(
        lambda s: score_to_severity(s, Q33, Q66, Q85)
    )
    new_df['inferred_severity'] = new_df['inferred_severity_num'].map(INV_PRIORITY_MAP)
    new_df['severity_delta']    = new_df['inferred_severity_num'] - new_df['priority_num']
    new_df['mismatch_type']     = new_df['severity_delta'].apply(
        lambda d: get_mismatch_type(d, MISMATCH_THRESHOLD)
    )

    # Feature engineering
    new_df['text_len']          = new_df['combined_text_clean'].apply(len)
    new_df['cap_ratio']         = new_df['combined_text'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    new_df['word_count']        = new_df['combined_text_clean'].apply(lambda x: len(str(x).split()))
    new_df['exclamation_count'] = new_df['combined_text'].apply(lambda x: str(x).count('!'))
    new_df['rt_scaled']         = scaler.transform(new_df[['Resolution_Time_Hours']])
    new_df['crit_kw_count']     = new_df['combined_text_clean'].apply(
        lambda x: sum(kw in x for kw in CRITICAL_KW))
    new_df['low_kw_count']      = new_df['combined_text_clean'].apply(
        lambda x: sum(kw in x for kw in LOW_KW))
    new_df['hi_pri_lo_sat']     = ((new_df['priority_num'] >= 2) & (new_df['sat_score'] < 0.4)).astype(int)
    new_df['lo_pri_hi_nlp']     = ((new_df['priority_num'] <= 1) & (new_df['nlp_score'] > 0.7)).astype(int)
    new_df['rt_priority_gap']   = new_df['rt_score']  - (new_df['priority_num'] / 3.0)
    new_df['sat_priority_gap']  = new_df['sat_score'] - (new_df['priority_num'] / 3.0)
    new_df['nlp_priority_gap']  = new_df['nlp_score'] - (new_df['priority_num'] / 3.0)

    X_new = np.hstack([pca.transform(new_emb), new_df[META_COLS].values])

    new_df['pred_proba'] = clf.predict_proba(X_new)[:, 1]
    new_df['pred_label'] = (new_df['pred_proba'] >= best_thresh).astype(int)

    flagged  = new_df[(new_df['pred_label'] == 1) & (new_df['mismatch_type'] != 'Consistent')]
    dossiers = [build_dossier(row, row['pred_proba'], W_NLP, W_CLUSTER, W_SAT)
                for _, row in flagged.iterrows()]

    return new_df, dossiers

# 4. DOSSIER DISPLAY

def show_dossier(d):
    mtype = d['mismatch_type']
    icon  = '🟠' if mtype == 'Hidden Crisis' else '🔴'
    st.markdown(f"### {icon} `{d['ticket_id']}` — {mtype}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assigned Priority", d['assigned_priority'])
    c2.metric("Inferred Severity", d['inferred_severity'])
    c3.metric("Severity Delta",    f"{d['severity_delta']:+d}")
    c4.metric("Confidence",        f"{d['confidence']*100:.1f}%")

    st.markdown("** Constraint Analysis**")
    st.info(d['constraint_analysis'])

    st.markdown("** Feature Evidence**")
    for ev in d['feature_evidence']:
        sig   = ev['signal'].replace('_', ' ').title()
        val   = ev['value']
        extra = ev.get('interpretation', ev.get('weight', ''))
        st.markdown(f"- **{sig}:** {val} *(source: `{ev['source_field']}`)*"
                    + (f" — {extra}" if extra else ""))

    with st.expander("📄 Raw JSON Dossier"):
        st.json(d)
    st.divider()



# 5. DASHBOARD PAGE

def show_dashboard():
    import matplotlib.pyplot as plt
    import matplotlib

    st.markdown("##  Priority Mismatch Dashboard")

    try:
        pred_df = pd.read_csv('sia_predictions.csv')
    except FileNotFoundError:
        st.warning(" `sia_predictions.csv` not found in the app folder.")
        return

    try:
        full_df = pd.read_csv('enhanced_customer_support_data.csv')
    except FileNotFoundError:
        full_df = None

    # Top metrics
    total       = len(pred_df)
    flagged     = int(pred_df['pred_label'].sum())
    hidden      = int((pred_df['mismatch_type'] == 'Hidden Crisis').sum())
    false_alarm = int((pred_df['mismatch_type'] == 'False Alarm').sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets",      f"{total:,}")
    c2.metric("Flagged Mismatches", f"{flagged:,}", delta=f"{flagged/total*100:.1f}%")
    c3.metric("Hidden Crisis",      f"{hidden:,}")
    c4.metric("False Alarm",        f"{false_alarm:,}")

    st.markdown("---")

    # Row 1: Pie + Bar
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("###  Mismatch Type Distribution")
        type_counts = pred_df['mismatch_type'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        colors = {'Consistent': '#4caf50', 'Hidden Crisis': '#ff9800', 'False Alarm': '#e91e63'}
        ax1.pie(type_counts.values,
                labels=type_counts.index,
                autopct='%1.1f%%',
                colors=[colors.get(l, '#999') for l in type_counts.index],
                startangle=140)
        ax1.set_title('Ticket Classification Breakdown')
        st.pyplot(fig1)
        plt.close()

    with col_b:
        st.markdown("###  Mismatches by Assigned Priority")
        mismatch_only  = pred_df[pred_df['pred_label'] == 1]
        priority_order = ['Low', 'Medium', 'High', 'Critical']
        counts = mismatch_only['Priority_Level'].value_counts().reindex(priority_order, fill_value=0)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.bar(counts.index, counts.values,
                       color=['#42a5f5', '#66bb6a', '#ffa726', '#ef5350'])
        ax2.set_xlabel('Assigned Priority Level')
        ax2.set_ylabel('Number of Mismatches')
        ax2.set_title('Mismatches by Priority Level')
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 2,
                     str(int(bar.get_height())),
                     ha='center', fontsize=10)
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # Row 2: Signal importance + Confidence histogram
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("###  Top Contributing Signals (Ablation)")
        sig_df = pd.DataFrame({
            'Signal': ['Conflict Features (Gap)', 'Resolution Time', 'Satisfaction', 'Clustering', 'NLP Score'],
            'F1 Drop': [0.0807, 0.0015, 0.0002, 0.0001, -0.0009]
        }).sort_values('F1 Drop', ascending=True)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.barh(sig_df['Signal'], sig_df['F1 Drop'],
                 color=['#ef5350' if v > 0 else '#42a5f5' for v in sig_df['F1 Drop']])
        ax3.axvline(0, color='black', linewidth=0.8)
        ax3.set_xlabel('Macro F1 Change When Removed')
        ax3.set_title('Signal Importance (Ablation Study)')
        st.pyplot(fig3)
        plt.close()

    with col_d:
        st.markdown("###  Prediction Confidence Distribution")
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.hist(pred_df['pred_proba'], bins=30, color='#5c6bc0', edgecolor='white')
        ax4.axvline(x=0.32, color='red', linestyle='--', label='Threshold = 0.32')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Number of Tickets')
        ax4.set_title('Model Confidence Distribution')
        ax4.legend()
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")

    # Severity delta heatmap
    st.markdown("###  Severity Delta Heatmap — Issue Category × Ticket Channel")
    st.caption("Average severity delta (inferred − assigned). "
               " Positive = Hidden Crisis risk.  Negative = False Alarm risk.")

    if full_df is not None:
        merged = pred_df.merge(
            full_df[['Ticket_ID', 'Issue_Category', 'Ticket_Channel']],
            on='Ticket_ID', how='left'
        )
        pivot = merged.pivot_table(
            values='severity_delta',
            index='Issue_Category',
            columns='Ticket_Channel',
            aggfunc='mean'
        ).round(2)

        fig5, ax5 = plt.subplots(figsize=(10, 4))
        cmap = matplotlib.colormaps.get_cmap('RdYlGn_r')
        im = ax5.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-2, vmax=2)
        ax5.set_xticks(range(len(pivot.columns)))
        ax5.set_yticks(range(len(pivot.index)))
        ax5.set_xticklabels(pivot.columns, rotation=30, ha='right')
        ax5.set_yticklabels(pivot.index)
        ax5.set_title('Mean Severity Delta by Category and Channel')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax5.text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=9, color='white' if abs(val) > 1 else 'black')
        plt.colorbar(im, ax=ax5, label='Severity Delta')
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()
    else:
        st.info("Place `enhanced_customer_support_data.csv` in the app folder to see the heatmap.")

    st.markdown("---")

    # Flagged tickets table
    st.markdown("###  Flagged Tickets Table")
    flagged_df = pred_df[pred_df['pred_label'] == 1].copy()
    st.dataframe(
        flagged_df[['Ticket_ID', 'Priority_Level', 'inferred_severity',
                    'severity_delta', 'mismatch_type', 'pred_proba']].round(4),
        use_container_width=True, height=350
    )
    st.download_button(
        " Download Flagged Tickets CSV",
        flagged_df.to_csv(index=False).encode('utf-8'),
        file_name='flagged_mismatches.csv',
        mime='text/csv'
    )



# 6. SINGLE TICKET PAGE

def show_single_ticket(bundle, st_model):
    st.markdown("##  Single Ticket Analysis")
    st.markdown("Fill in the ticket details and click **Analyze**.")

    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticket_id    = st.text_input("Ticket ID", value="TKT-TEST-001")
            subject      = st.text_input("Ticket Subject", value="Login failed")
            priority     = st.selectbox("Assigned Priority", ['Low', 'Medium', 'High', 'Critical'])
            channel      = st.selectbox("Ticket Channel", ['Chat', 'Email', 'Web Form'])
        with col2:
            category     = st.selectbox("Issue Category",
                                        ['Technical', 'Billing', 'Account', 'General Inquiry', 'Fraud'])
            resolution_h = st.number_input("Resolution Time (hours)", min_value=1, max_value=120, value=24)
            satisfaction = st.slider("Satisfaction Score (1=Low, 5=High)", 1, 5, 3)
            description  = st.text_area("Ticket Description",
                                        value="Hi support, I cannot access my account. It seems to be locked.",
                                        height=120)
        submitted = st.form_submit_button(" Analyze Ticket", use_container_width=True)

    if submitted:
        ticket_df = pd.DataFrame([{
            'Ticket_ID':             ticket_id,
            'Ticket_Subject':        subject,
            'Ticket_Description':    description,
            'Priority_Level':        priority,
            'Ticket_Channel':        channel,
            'Issue_Category':        category,
            'Resolution_Time_Hours': resolution_h,
            'Satisfaction_Score':    satisfaction
        }])

        with st.spinner("Running SIA pipeline..."):
            result_df, dossiers = run_inference(ticket_df, bundle, st_model)

        row        = result_df.iloc[0]
        pred_label = int(row['pred_label'])
        mtype      = row['mismatch_type']
        confidence = float(row['pred_proba'])

        st.markdown("---")
        st.markdown("###  Result")

        if pred_label == 1 and mtype != 'Consistent':
            if mtype == 'Hidden Crisis':
                st.error(f" **MISMATCH — Hidden Crisis** (confidence: {confidence*100:.1f}%)")
            else:
                st.warning(f" **MISMATCH — False Alarm** (confidence: {confidence*100:.1f}%)")
            if dossiers:
                show_dossier(dossiers[0])
        else:
            st.success(f" **CONSISTENT** — Priority looks correct (confidence: {(1-confidence)*100:.1f}%)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Assigned Priority", row['Priority_Level'])
            c2.metric("Inferred Severity", row['inferred_severity'])
            c3.metric("Severity Delta",    f"{int(row['severity_delta']):+d}")



# 7. BATCH CSV PAGE

def show_batch(bundle, st_model):
    st.markdown("##  Batch CSV Upload")
    st.markdown("Upload a CSV of tickets — the app will flag all mismatches and generate dossiers.")

    st.markdown("**Required columns:**")
    st.code("Ticket_ID, Ticket_Subject, Ticket_Description, Priority_Level, "
            "Ticket_Channel, Issue_Category, Resolution_Time_Hours")
    st.caption("Optional: `Satisfaction_Score` (defaults to 3 if missing)")

    uploaded = st.file_uploader("Upload tickets CSV", type=['csv'])

    if uploaded:
        input_df = pd.read_csv(uploaded)
        st.success(f"Loaded **{len(input_df)}** tickets")
        st.dataframe(input_df.head(5), use_container_width=True)

        if st.button(" Run SIA on all tickets", use_container_width=True):
            with st.spinner(f"Analyzing {len(input_df)} tickets..."):
                result_df, dossiers = run_inference(input_df, bundle, st_model)

            total   = len(result_df)
            flagged = int(result_df['pred_label'].sum())
            hidden  = int((result_df['mismatch_type'] == 'Hidden Crisis').sum())
            false_a = int((result_df['mismatch_type'] == 'False Alarm').sum())

            st.markdown("---")
            st.markdown("###  Results Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets",      total)
            c2.metric("Flagged Mismatches", flagged)
            c3.metric("Hidden Crisis",      hidden)
            c4.metric("False Alarm",        false_a)

            st.markdown("###  All Predictions")
            st.dataframe(
                result_df[['Ticket_ID', 'Priority_Level', 'inferred_severity',
                            'severity_delta', 'mismatch_type',
                            'pred_label', 'pred_proba']].round(4),
                use_container_width=True, height=300
            )

            st.download_button(
                " Download Predictions CSV",
                result_df[['Ticket_ID', 'Priority_Level', 'inferred_severity',
                            'severity_delta', 'mismatch_type',
                            'pred_label', 'pred_proba']].to_csv(index=False).encode('utf-8'),
                file_name='sia_predictions_new.csv', mime='text/csv'
            )

            if dossiers:
                st.markdown("---")
                st.markdown(f"###  Evidence Dossiers ({len(dossiers)} flagged)")
                st.download_button(
                    " Download All Dossiers (JSON)",
                    json.dumps(dossiers, indent=2).encode('utf-8'),
                    file_name='evidence_dossiers.json', mime='application/json'
                )
                for d in dossiers[:20]:
                    show_dossier(d)
                if len(dossiers) > 20:
                    st.info(f"Showing first 20 of {len(dossiers)} dossiers. Download JSON for all.")
            else:
                st.success(" No mismatches detected in this batch!")



# 8. MAIN

def main():
    st.markdown('<div class="main-title">🛡️ Support Integrity Auditor (SIA)</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">MARS Open Projects 2026 · Detecting Priority Mismatch in CRM Tickets</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading SIA model..."):
        try:
            bundle   = load_model()
            st_model = load_sentence_transformer()
        except FileNotFoundError:
            st.error("`sia_model.pkl` not found! Place it in the same folder as app.py.")
            st.stop()

    st.success(" Model loaded!")

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=80)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [" Dashboard", " Single Ticket", " Batch CSV"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Model Info")
    st.sidebar.markdown(f"- **Threshold:** `{bundle['best_thresh']}`")
    st.sidebar.markdown(f"- **Signals:** NLP · RT · Cluster · Satisfaction")
    st.sidebar.markdown(f"- **Weights:** {bundle['W_NLP']} / {bundle['W_RT']} / {bundle['W_CLUSTER']} / {bundle['W_SAT']}")
    st.sidebar.markdown("- **Accuracy:** 0.8898 ")
    st.sidebar.markdown("- **Macro F1:** 0.8862 ")
    st.sidebar.markdown("- **Recall class 0:** 0.8561 ")
    st.sidebar.markdown("- **Recall class 1:** 0.9459 ")

    st.markdown("---")

    if page == " Dashboard":
        show_dashboard()
    elif page == " Single Ticket":
        show_single_ticket(bundle, st_model)
    elif page == " Batch CSV":
        show_batch(bundle, st_model)


if __name__ == "__main__":
    main()

