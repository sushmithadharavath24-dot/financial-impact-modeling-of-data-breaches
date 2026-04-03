import os
import pickle
import requests
import warnings
import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union
from sklearn.preprocessing import StandardScaler

# 1. SETUP
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# Get API Key from Render Environment Variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Path to artifacts (Relative to this file)
CFG_PATH = os.path.join(os.path.dirname(__file__), "artifacts")

# 2. MAPPING LAYERS
MAP_AT = {
    'ransomware': 'Ransomware', 'ddos': 'DDoS', 'iot botnet': 'DDoS',
    'phishing': 'Phishing', 'social engineering': 'Phishing',
    'malware': 'Malware', 'trojan': 'Malware', 'spyware': 'Malware', 'rootkit': 'Malware', 'zero-day': 'Malware',
    'sql injection': 'SQL Injection', 'xss': 'SQL Injection', 'api breach': 'SQL Injection', 
    'data exfiltration': 'SQL Injection',
    'man-in-the-middle': 'Man-in-the-Middle', 'credential stuffing': 'Man-in-the-Middle', 
    'brute force': 'Man-in-the-Middle', 'insider threat': 'Man-in-the-Middle'
}
MAP_TI = {
    'finance': 'Banking', 'insurance': 'Banking', 'legal': 'Banking', 'banking': 'Banking',
    'healthcare': 'Healthcare',
    'technology': 'IT', 'entertainment': 'IT', 'it': 'IT',
    'retail': 'Retail', 'hospitality': 'Retail', 'construction': 'Retail', 'real estate': 'Retail', 
    'logistics': 'Retail',
    'government': 'Government', 'defense': 'Government', 'non-profit': 'Government',
    'telecommunications': 'Telecommunications', 'energy': 'Telecommunications', 
    'automotive': 'Telecommunications', 'aerospace': 'Telecommunications', 'telecom': 'Telecommunications',
    'education': 'Education', 'manufacturing': 'Retail'
}
IND_MEANS = {'Banking': 35e6, 'Healthcare': 25e6, 'IT': 45e6, 'Retail': 20e6, 'Government': 30e6}

def clean_num(v):
    if not v: return 0.0
    if isinstance(v, (int, float)): return float(v)
    s = str(v).lower().replace(',', '').replace('$', '').strip()
    try:
        if 'b' in s: return float(s.replace('b', '')) * 1e9
        if 'm' in s: return float(s.replace('m', '')) * 1e6
        if 'k' in s: return float(s.replace('k', '')) * 1e3
        return float(s)
    except: return 0.0

def smart_map(val, mapping, default="Unknown"):
    v = str(val).lower().strip()
    return mapping.get(v, default) if v not in [x.lower() for x in mapping.values()] else mapping.get(v, val)

# 3. ENGINE CLASS (XGBoost Only Version)
class RE:
    def __init__(self, path):
        self.path = path
        self.e, self.s, self.y, self.tm, self.fm, self.st = {}, None, None, {}, {}, {}
        self.c = ['Country', 'Attack Type', 'Target Industry', 'Attack Source', 'Security Vulnerability Type', 'Defense Mechanism Used', 'Cross_Attack_Industry']
        self.n = ['Year', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
        self.xm = None

    def load(self):
        # Only load XGBoost and Meta (No Keras)
        fp = {k: os.path.join(self.path, f) for k,f in {'x':'xgb.json','d':'meta.pkl'}.items()}
        
        if not all(os.path.exists(v) for v in fp.values()):
            print(f"❌ Error: Could not find model files in {self.path}")
            return False
        
        # Load XGBoost
        self.xm = xgb.XGBRegressor()
        self.xm.load_model(fp['x'])
        
        # Load Meta
        with open(fp['d'], 'rb') as f: 
            d = pickle.load(f)
            self.e, self.s, self.y, self.tm, self.fm, self.st = d['e'], d['s'], d['y'], d['t'], d['f'], d['st']
        print("[SYS] Lightweight Engine Loaded.")
        return True

    def inf(self, d):
        d['Attack Type'] = smart_map(d.get('Attack Type'), MAP_AT, 'Malware')
        d['Target Industry'] = smart_map(d.get('Target Industry'), MAP_TI, 'IT')
        
        c = str(d.get('Country', 'USA')).strip()
        if c.lower() in ['united states', 'america', 'us']: d['Country'] = 'USA'
        elif c.lower() in ['united kingdom', 'england']: d['Country'] = 'UK'
        
        d['Cross_Attack_Industry'] = f"{d['Attack Type']}_{d['Target Industry']}"
        xc, te, fe = {}, [], []
        
        for k in self.c:
            v = str(d.get(k, 'Unknown')).strip()
            cl = k.replace(" ", "_").replace("(", "").replace(")", "")
            val_idx = 0
            if cl in self.e:
                if v in self.e[cl].classes_:
                    val_idx = self.e[cl].transform([v])[0]
                else:
                    for x in self.e[cl].classes_:
                        if str(x).lower() == v.lower():
                            val_idx = self.e[cl].transform([x])[0]; v = x; break
            
            xc[f"in_{cl}"] = np.array([val_idx])
            te.append(self.tm.get(k, {}).get(v, 0))
            fe.append(self.fm.get(k, {}).get(v, 0))

        nv = []
        for n in self.n:
            val = clean_num(d.get(n, 0))
            clamped = max(min(val, self.st.get(f'mx_{n}', 1e9)), self.st.get(f'mn_{n}', 0))
            nv.append(clamped)
        return xc, self.s.transform(np.array([nv + te + fe]))

    def ens(self, xc, xn):
        # XGBoost Only Prediction (Saves 400MB RAM)
        features = np.hstack([xc[k].reshape(-1, 1) for k in xc] + [xn])
        pred_scaled = self.xm.predict(features).reshape(-1, 1)
        return self.y.inverse_transform(pred_scaled).flatten()

# 4. LOGIC
def calc_logic(u, re_obj):
    xc, xn = re_obj.inf(u)
    rm = re_obj.ens(xc, xn)[0] # XGBoost prediction
    
    at = str(u.get('Attack Type','')).lower()
    ind = str(u.get('Target Industry','')).lower()
    src = str(u.get('Attack Source','')).lower()
    vln = str(u.get('Security Vulnerability Type','')).lower()
    usr = clean_num(u.get('Number of Affected Users', 0))
    hrs = clean_num(u.get('Incident Resolution Time (in Hours)', 0))

    if 'ddos' in at:
        br = 25000 if 'fin' in ind or 'retail' in ind or 'bank' in ind else 5000
        return br * (hrs / 2.0) + (usr * 0.5), "Availability-Burn"
    if 'insider' in src:
        bf = 150000; iv = 2e6 if any(x in ind for x in ['tech', 'def', 'gov', 'bank']) else 1e5
        return bf + (iv if 'zero' in vln or 'admin' in vln else 0), "Insider-Vector"
    if 'ransom' in at:
        cpr = 350 if 'health' in ind else (250 if 'fin' in ind or 'bank' in ind else 150)
        dt = 50000 if usr > 1000 else 5000
        return (usr * cpr) + (dt * hrs), "Ransom-Crit"
    
    return max(rm, usr * 145), "Hybrid-ML"

def generate_graph(pred, u, re_obj, nw):
    try:
        users = clean_num(u.get('Number of Affected Users'))
        time_h = clean_num(u.get('Incident Resolution Time (in Hours)'))
        
        sns.set_style("ticks")
        fig = plt.figure(figsize=(18, 6))
        ax1, ax2, ax3 = [fig.add_subplot(1, 3, i+1) for i in range(3)]

        ib = IND_MEANS.get(u.get('Target Industry'), 1e6) 
        sns.barplot(data=pd.DataFrame({'T': ['Ind. Avg', 'Pred'], 'V': [ib, pred]}), x='T', y='V', hue='T', ax=ax1, palette=['grey', 'firebrick'], legend=False)
        ax1.set_yscale('log'); ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
        ax1.set_title('Benchmarking', fontsize=12, fontweight='bold')
        ax1.set_xlabel("Scenario", fontsize=10); ax1.set_ylabel("Loss ($)", fontsize=10)
        
        um, tm = max(users * 1.5, 10000), max(time_h * 1.5, 168)
        pd_df = pd.DataFrame({'M': ['Users', 'Time'], 'P': [min(users/um*100, 100), min(time_h/tm*100, 100)]})
        sns.barplot(data=pd_df, x='P', y='M', hue='M', ax=ax2, orient='h', palette='viridis', legend=False)
        ax2.set_xlim(0, 100); ax2.set_title('Impact Factors', fontsize=12, fontweight='bold')
        ax2.set_xlabel("Severity (%)", fontsize=10); ax2.set_ylabel("Factor", fontsize=10)

        ur = np.logspace(np.log10(max(10, users*0.1)), np.log10(max(1e6, users*10)), 20)
        pr = [pred * (user_val / max(1, users)) for user_val in ur]
        
        sns.lineplot(x=ur, y=pr, ax=ax3, color='navy', lw=2)
        ax3.scatter([users], [pred], color='red', zorder=5)
        if nw > 0:
            ax3.axhline(nw, c='k', ls='--', label='Net Worth')
            ax3.fill_between(ur, nw, pr, where=(np.array(pr) > nw), color='red', alpha=0.1)
        ax3.set_xscale('log'); ax3.set_yscale('log'); ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
        ax3.set_title('Sensitivity Analysis', fontsize=12, fontweight='bold')
        ax3.set_xlabel("Users Affected (Log)", fontsize=10); ax3.set_ylabel("Proj. Loss ($)", fontsize=10)
        
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8'); plt.close(fig)
        return b64
    except: return ""

def ask_ai(l, c):
    if not OPENROUTER_API_KEY: return "AI disabled (Check API Key)"
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                          headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, 
                          json={"model": "mistralai/mistral-7b-instruct", "messages": [{"role": "user", "content": f"Act as CISO. Breach: {c}. Loss: ${l:,.0f}. Give 3 hard technical fixes. No fluff."}]})
        return r.json()['choices'][0]['message']['content'].strip()
    except: return "AI Error"

# 5. INITIALIZATION
re_engine = RE(CFG_PATH)
re_engine.load()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class RiskRequest(BaseModel):
    Country: str = Field(..., example="France")
    Attack_Type: str = Field(..., alias="Attack Type")
    Target_Industry: str = Field(..., alias="Target Industry")
    Attack_Source: str = Field("Unknown", alias="Attack Source")
    Security_Vulnerability_Type: str = Field("Unknown", alias="Security Vulnerability Type")
    Defense_Mechanism_Used: str = Field("Unknown", alias="Defense Mechanism Used")
    Year: Union[str, int] = 2026
    Number_of_Affected_Users: Union[str, int, float] = Field(..., alias="Number of Affected Users")
    Incident_Resolution_Time: Union[str, int, float] = Field(..., alias="Incident Resolution Time (in Hours)")
    Net_Worth: Union[str, int, float] = Field("10M", alias="Net Worth")

@app.post("/calculate_risk")
def calculate_risk(req: RiskRequest):
    u = req.dict(by_alias=True)
    nw = clean_num(u.get('Net Worth'))
    try:
        fl, m = calc_logic(u, re_engine)
        fl = max(min(fl, 1e12), 0)
    except: fl, m = 500000, "Fallback"

    return {
        "loss_formatted": f"${fl:,.2f}",
        "range_formatted": f"${fl*0.85:,.2f} - ${fl*1.15:,.2f}",
        "graph_image": f"data:image/png;base64,{generate_graph(fl, u, re_engine, nw)}",
        "ai_recommendation": ask_ai(fl, u.get('Attack Type')),
        "calculation_method": m
    }
