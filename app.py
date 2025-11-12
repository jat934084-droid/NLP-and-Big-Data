# app.py (final version: dark theme + humor + charts)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import matplotlib.pyplot as plt
import logging
from typing import Optional, Tuple, List
from urllib.parse import urljoin
from ftfy import fix_text
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# API key (update in .streamlit/secrets.toml)
API_KEY = st.secrets.get("FACTCHECK_API_KEY", None)

# Basic config
SCRAPED_DATA_PATH = "politifact_data.csv"
N_SPLITS = 5
MAX_PAGES = 100
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Utility ----------
def clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except Exception:
        pass
    return " ".join(s.split()).strip()

def safe_get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    backoff = REQUEST_BACKOFF
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"Request error ({attempt+1}/{REQUEST_RETRIES}) for {url}: {e}")
            time.sleep(backoff)
            backoff *= 2
    return None

# ---------- Load Spacy ----------
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model missing. Add wheel link in requirements.txt.")
        raise

try:
    NLP_MODEL = load_spacy_model()
except Exception:
    st.stop()

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ---------- Scraper ----------
@st.cache_data(ttl=86400)
def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    seen_urls, rows = set(), []
    page_count = 0

    while current_url and page_count < MAX_PAGES:
        page_count += 1
        if current_url in seen_urls:
            break
        seen_urls.add(current_url)
        resp = safe_get(current_url)
        if resp is None:
            break
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.find_all("li", class_="o-listicle__item")
        if not items:
            break

        stop_if_older = False
        for card in items:
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(" ", strip=True) if date_div else ""
            claim_date = None
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    claim_date = pd.to_datetime(match.group(1), errors='coerce')
            if not claim_date:
                continue
            if claim_date < start_date:
                stop_if_older = True
                break
            if not (start_date <= claim_date <= end_date):
                continue

            statement = clean(card.find("div", class_="m-statement__quote").get_text(" ", strip=True)) if card.find("div", class_="m-statement__quote") else None
            source = clean(card.find("a", class_="m-statement__name").get_text(" ", strip=True)) if card.find("a", class_="m-statement__name") else None
            footer = card.find("footer", class_="m-statement__footer")
            author = None
            if footer:
                text = footer.get_text(" ", strip=True)
                m = re.search(r"By\s+([^•\n\r]+)", text)
                author = clean(m.group(1)) if m else clean(text.split("•")[0].replace("By", ""))
            label_img = card.find("img", alt=True)
            label = clean(label_img['alt'].replace('-', ' ').title()) if label_img else None

            rows.append({"author": author, "statement": statement, "source": source, "date": claim_date.strftime("%Y-%m-%d"), "label": label})
        if stop_if_older:
            break
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        current_url = urljoin(base_url, next_link['href']) if next_link and next_link.get("href") else None

    df = pd.DataFrame(rows).dropna(subset=["statement", "label"])
    if not df.empty:
        df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ---------- Features ----------
def lexical_features_batch(texts, nlp): 
    return [" ".join([t.lemma_.lower() for t in doc if t.is_alpha and t.lemma_.lower() not in stop_words]) for doc in nlp.pipe(texts, disable=["ner", "parser"])]
def syntactic_features_batch(texts, nlp): 
    return [" ".join([t.pos_ for t in doc]) for doc in nlp.pipe(texts, disable=["ner"])]
def semantic_features_batch(texts): 
    return pd.DataFrame([[TextBlob(t).sentiment.polarity, TextBlob(t).sentiment.subjectivity] for t in texts], columns=["polarity","subjectivity"])
def discourse_features_batch(texts, nlp):
    out=[]
    for doc in nlp.pipe(texts, disable=["ner"]):
        sents=[s.text.strip() for s in doc.sents]
        out.append(f"{len(sents)} "+ " ".join([s.split()[0].lower() for s in sents if s.split()]))
    return out
def pragmatic_features_batch(texts):
    return pd.DataFrame([[t.lower().count(w) for w in pragmatic_words] for t in texts], columns=pragmatic_words)

def apply_feature_extraction(X, phase, nlp):
    X_texts = X.astype(str).tolist()
    if phase == "Lexical & Morphological":
        X_proc = lexical_features_batch(X_texts, nlp)
        vect = CountVectorizer(binary=True, ngram_range=(1,2), min_df=2)
        return vect.fit_transform(X_proc), vect
    if phase == "Syntactic":
        X_proc = syntactic_features_batch(X_texts, nlp)
        vect = TfidfVectorizer(max_features=5000)
        return vect.fit_transform(X_proc), vect
    if phase == "Semantic":
        return semantic_features_batch(X_texts).values, None
    if phase == "Discourse":
        X_proc = discourse_features_batch(X_texts, nlp)
        vect = CountVectorizer(ngram_range=(1,2), max_features=5000)
        return vect.fit_transform(X_proc), vect
    if phase == "Pragmatic":
        return pragmatic_features_batch(X_texts).values, None
    raise ValueError("Unknown phase")

# ---------- Models ----------
def get_models_dict():
    return {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    }

def create_binary_target(df):
    REAL = ["True","No Flip","Mostly True","Half Flip","Half True"]
    FAKE = ["False","Barely True","Pants On Fire","Full Flop"]
    def map_label(l):
        if pd.isna(l): return np.nan
        l=str(l).strip()
        if l in REAL: return 1
        if l in FAKE: return 0
        low=l.lower()
        if "true" in low and "mostly" not in low and "half" not in low: return 1
        if "false" in low or "pants" in low or "fire" in low: return 0
        return np.nan
    df=df.copy()
    df["target_label"]=df["label"].apply(map_label)
    return df

def evaluate_models(df, selected_phase, nlp):
    df=create_binary_target(df).dropna(subset=["target_label"])
    df=df[df["statement"].astype(str).str.len()>10]
    X_raw,y_raw=df["statement"].astype(str),df["target_label"].astype(int)
    if len(np.unique(y_raw))<2:
        st.error("Only one class after mapping.")
        return pd.DataFrame()
    X_features, vect = apply_feature_extraction(X_raw, selected_phase, nlp)
    models=get_models_dict()
    results=[]
    skf=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_list=X_raw.tolist()
    for name, model in models.items():
        st.caption(f"Training {name}...")
        fold_acc,fold_f1,fold_prec,fold_rec,train_times,infer_times=[],[],[],[],[],[]
        for tr,te in skf.split(np.zeros(len(y_raw)),y_raw):
            Xtr_raw,Xte_raw=pd.Series([X_list[i] for i in tr]),pd.Series([X_list[i] for i in te])
            ytr,yte=y_raw.values[tr],y_raw.values[te]
            if vect is not None:
                if selected_phase=="Lexical & Morphological":
                    Xtr=vect.transform(lexical_features_batch(Xtr_raw, nlp))
                    Xte=vect.transform(lexical_features_batch(Xte_raw, nlp))
                elif selected_phase=="Syntactic":
                    Xtr=vect.transform(syntactic_features_batch(Xtr_raw, nlp))
                    Xte=vect.transform(syntactic_features_batch(Xte_raw, nlp))
                elif selected_phase=="Discourse":
                    Xtr=vect.transform(discourse_features_batch(Xtr_raw, nlp))
                    Xte=vect.transform(discourse_features_batch(Xte_raw, nlp))
                else:
                    Xtr=vect.transform(Xtr_raw); Xte=vect.transform(Xte_raw)
            else:
                if selected_phase=="Semantic":
                    Xtr=semantic_features_batch(Xtr_raw).values; Xte=semantic_features_batch(Xte_raw).values
                elif selected_phase=="Pragmatic":
                    Xtr=pragmatic_features_batch(Xtr_raw).values; Xte=pragmatic_features_batch(Xte_raw).values
                else:
                    Xtr=Xtr_raw.values.reshape(-1,1); Xte=Xte_raw.values.reshape(-1,1)
            start=time.time()
            try:
                if name=="Naive Bayes":
                    model.fit(np.abs(Xtr).astype(float),ytr); clf=model
                else:
                    clf=ImbPipeline([("smote",SMOTE(random_state=42,k_neighbors=3)),("clf",model)]).fit(Xtr,ytr)
                tr_time=time.time()-start; inf_start=time.time()
                y_pred=clf.predict(Xte)
                inf_time=(time.time()-inf_start)*1000
                fold_acc.append(accuracy_score(yte,y_pred))
                fold_f1.append(f1_score(yte,y_pred,average="weighted",zero_division=0))
                fold_prec.append(precision_score(yte,y_pred,average="weighted",zero_division=0))
                fold_rec.append(recall_score(yte,y_pred,average="weighted",zero_division=0))
                train_times.append(tr_time); infer_times.append(inf_time)
            except Exception as e:
                st.warning(f"{name} failed: {e}")
        results.append({
            "Model":name,"Accuracy":np.mean(fold_acc)*100,"F1-Score":np.mean(fold_f1),
            "Precision":np.mean(fold_prec),"Recall":np.mean(fold_rec),
            "Training Time (s)":round(np.mean(train_times),3),"Inference Latency (ms)":round(np.mean(infer_times),3)
        })
    return pd.DataFrame(results)

# ---------- Humor ----------
def get_phase_critique(phase):
    jokes={"Lexical & Morphological":["Word counter wins again.","Vocabulary is power."],"Syntactic":["Grammar matters!","The syntax sheriff prevails."],"Semantic":["Feelings beat logic!","Vibes > facts."],"Discourse":["Structure rules.","Debate club champion."],"Pragmatic":["Intent detective wins.","Too many exclamations? Liar!"]}
    return random.choice(jokes.get(phase,["Speechless AI."]))
def get_model_critique(model):
    jokes={"Naive Bayes":["Simple genius.","Counts words, wins hearts."],"Decision Tree":["Split until victory.","Judge-y but accurate."],"Logistic Regression":["Straight-line politician.","Reliable veteran."],"SVM":["Margin master.","Hardline hero."]}
    return random.choice(jokes.get(model,["No comment."]))
def generate_humorous_critique(df, phase):
    if df.empty: return "No models to roast yet!"
    best=df.loc[df["F1-Score"].idxmax()]
    return f"### 🤖 Model Roast:\n**{best['Model']}** nailed it on `{phase}` with **{best['Accuracy']:.2f}% Accuracy**.\n\nPhase insight: {get_phase_critique(phase)}\nModel personality: {get_model_critique(best['Model'])}"

# ---------- Google Fact Check ----------
def get_fact_check_results(query):
    if not API_KEY: return []
    url="https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params={"query":query,"key":API_KEY}
    try:
        r=requests.get(url,params=params,timeout=15)
        r.raise_for_status()
        data=r.json().get("claims",[])
        out=[]
        for c in data:
            for r2 in c.get("claimReview",[]):
                out.append({"publisher":r2.get("publisher",{}).get("name","Unknown"),
                            "title":r2.get("title",""),
                            "rating":r2.get("textualRating","No Rating"),
                            "url":r2.get("url","")})
        return out
    except Exception as e:
        return [{"publisher":"Error","title":str(e),"rating":"","url":""}]

# ---------- Streamlit UI ----------
def app():
    st.set_page_config(page_title='AI vs. Fact — Portal', layout='wide')
    st.markdown("""
    <style>
    .stApp {background:linear-gradient(180deg,#05070a,#0b0f12);color:#e9f4ff;}
    .topbar{background:linear-gradient(90deg,#071029,#3b0f34);padding:18px;border-radius:10px;margin-bottom:14px;box-shadow:0 10px 30px rgba(0,0,0,0.6);}
    .panel{background:rgba(255,255,255,0.02);padding:12px;border-radius:10px;margin-bottom:12px;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="topbar"><h1>🧠 AI vs. Fact Portal</h1><p>Scrape • Train • Evaluate • Verify</p></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Home","Scraper","Model Showdown","Fact Check"])

    with tabs[0]:
        st.markdown('<div class="panel"><h3>Welcome!</h3><p>Use this tool to scrape political fact-checks, train NLP models, compare performance, and verify statements.</p></div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Politifact Scraper")
        min_d,max_d=pd.to_datetime('2007-01-01'),pd.to_datetime('today').normalize()
        s_d=st.date_input("Start Date",value=pd.to_datetime('2023-01-01'),min_value=min_d,max_value=max_d)
        e_d=st.date_input("End Date",value=max_d,min_value=min_d,max_value=max_d)
        if st.button("Scrape Politifact Data ⛏️"):
            if s_d>e_d: st.error("Invalid date range.")
            else:
                with st.spinner("Scraping..."):
                    df=scrape_data_by_date_range(pd.to_datetime(s_d),pd.to_datetime(e_d))
                    if df.empty: st.warning("No data scraped.")
                    else:
                        st.session_state['scraped_df']=df
                        st.success(f"Scraped {len(df)} records.")
                        st.download_button("Download CSV",df.to_csv(index=False).encode('utf-8'),"politifact_scraped.csv")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Model Showdown")
        if 'scraped_df' not in st.session_state or st.session_state['scraped_df'].empty:
            st.info("Scrape data first.")
        else:
            df=st.session_state['scraped_df']
            st.write(f"Dataset size: {len(df)}")
            phases=["Lexical & Morphological","Syntactic","Semantic","Discourse","Pragmatic"]
            sel=st.selectbox("Select Feature Phase:",phases)
            if st.button("Run Analysis 🥊"):
                with st.spinner("Training models..."):
                    res=evaluate_models(df,sel,NLP_MODEL)
                    st.session_state['df_results']=res
                    st.session_state['selected_phase_run']=sel
                    if not res.empty: st.success("Done!")
            if 'df_results' in st.session_state and not st.session_state['df_results'].empty:
                res=st.session_state['df_results']
                st.subheader("Results Summary")
                st.dataframe(res,use_container_width=True)
                # Chart
                metric=st.selectbox("Metric to plot:",["Accuracy","F1-Score","Precision","Recall","Training Time (s)","Inference Latency (ms)"])
                dfp=res[['Model',metric]].set_index('Model')
                st.bar_chart(dfp)
                # Scatter
                x=st.selectbox("X (Speed):",["Training Time (s)","Inference Latency (ms)"])
                y=st.selectbox("Y (Quality):",["Accuracy","F1-Score","Precision","Recall"])
                fig,ax=plt.subplots()
                ax.scatter(res[x],res[y],s=120)
                for i,r in res.iterrows(): ax.annotate(r['Model'],(r[x],r[y]))
                ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(True,alpha=0.4)
                st.pyplot(fig)
                # Humor
                st.markdown(generate_humorous_critique(res,st.session_state['selected_phase_run']))
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.header("Cross-Platform Fact Check")
        if not API_KEY:
            st.warning("API key missing in secrets.")
        q=st.text_input("Enter a claim or statement:")
        if st.button("Check Fact Credibility"):
            if not q.strip(): st.warning("Enter a valid claim.")
            else:
                with st.spinner("Checking..."):
                    r=get_fact_check_results(q)
                if not r: st.info("No fact-checks found.")
                else:
                    st.success(f"Found {len(r)} results:")
                    for rr in r[:10]:
                        st.markdown(f"**Source:** {rr['publisher']}<br>**Verdict:** {rr['rating']}<br>[{rr['title']}]({rr['url']})", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
