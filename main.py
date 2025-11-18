import uvicorn
import pandas as pd
import numpy as np
import io
import os
import datetime
import re
from collections import Counter
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, Field
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

from asteval import Interpreter

load_dotenv()

# --- Firebase Initialization ---
try:
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK configured successfully.")
except FileNotFoundError:
    print("="*50); print(f"WARNING: Service account key ('{key_path}') not found."); print("No data saved/fetched from Firestore."); print("="*50)
    db = None
except Exception as e:
    print(f"Error initializing Firebase Admin: {e}"); db = None

# --- Gemini API Configuration ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("="*50); print("WARNING: GOOGLE_API_KEY not set."); print("Agent 3 disabled."); print("="*50)
        gemini_model = None
    else:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini API configured.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}"); gemini_model = None


app = FastAPI(
    title="MediAgent Claims API",
    description="Backend for processing medical claims with trend analysis.",
    version="1.3.0"
)

# --- CORS Configuration ---
origins = ["http://localhost", "http://localhost:8002", "http://127.0.0.1:8002"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# === Agent 1: Validation Agent ===
def agent_1_validation(df: pd.DataFrame, custom_rule: str) -> pd.DataFrame:
    results = []
    interp = Interpreter()
    required_cols = ['claim_id', 'claim_amount', 'procedure_code', 'diagnosis_code']
    all_columns = df.columns.tolist()
    missing_cols = [col for col in required_cols if col not in all_columns]

    if missing_cols:
        for _, row in df.iterrows():
            results.append({**row.to_dict(), 'status': 'Reject', 'reason': f'Missing columns: {", ".join(missing_cols)}'})
        return pd.DataFrame(results)

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        safe_row_vars = {}
        validation_failed = False

        for col_name in all_columns:
            safe_var_name = re.sub(r'\W|^(?=\d)', '_', col_name)
            if not safe_var_name or safe_var_name[0].isdigit():
                safe_var_name = '_' + safe_var_name
            
            value = row_dict.get(col_name)

            if col_name in required_cols and (pd.isna(value) or str(value).strip() == ''):
                results.append({**row_dict, 'status': 'Reject', 'reason': f'Missing data: {col_name}'})
                validation_failed = True
                break
            
            if col_name == 'claim_amount':
                try:
                    safe_row_vars[safe_var_name] = float(value)
                except (ValueError, TypeError):
                    results.append({**row_dict, 'status': 'Reject', 'reason': f'Invalid type (claim_amount): {value}'})
                    validation_failed = True
                    break
            else:
                safe_row_vars[safe_var_name] = value

        if validation_failed:
            continue
        
        try:
            interp.symtable.clear()
            interp.symtable.update(safe_row_vars)
            rule_result = interp.eval(custom_rule)
            
            if interp.error:
                error_msg = interp.error_msg
                raise ValueError(f"Rule error: {error_msg}")
            
            if not rule_result:
                results.append({**row_dict, 'status': 'Reject', 'reason': f'Failed rule: "{custom_rule}"'})
                continue
        except Exception as e:
            results.append({**row_dict, 'status': 'Reject', 'reason': f'Rule error: {e}'})
            continue
        
        results.append({**row_dict, 'status': 'Pending - Agent 2', 'reason': 'Passed validation'})
    
    return pd.DataFrame(results)

# === Agent 2: Anomaly Detection Agent ===
def agent_2_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = df[df['status'] != 'Pending - Agent 2'].copy()
    pending_df = df[df['status'] == 'Pending - Agent 2'].copy()

    if pending_df.empty:
        return processed_df
    
    features = pending_df[['claim_amount']]
    
    if len(features) < 2:
        pending_df['status'] = 'Pending - Agent 3'
        pending_df['reason'] = 'Passed anomaly (skipped)'
        return pd.concat([processed_df, pending_df], ignore_index=True)
    
    model = IsolationForest(contamination='auto', random_state=42)
    pending_df['anomaly_score'] = model.fit_predict(features)
    
    results_list = []
    for _, row in pending_df.iterrows():
        if row['anomaly_score'] == -1:
            row['status'] = 'Flagged for Review'
            row['reason'] = f'Anomaly: Amt ${row["claim_amount"]} outlier.'
        else:
            row['status'] = 'Pending - Agent 3'
            row['reason'] = 'Passed anomaly check'
        results_list.append(row)
        
    final_df = pd.concat([processed_df, pd.DataFrame(results_list)], ignore_index=True)
    return final_df.drop(columns=['anomaly_score'], errors='ignore')


# === Agent 3: Adjudication Agent (Gemini) ===
async def agent_3_adjudication(df: pd.DataFrame) -> pd.DataFrame:
    if gemini_model is None:
        print("WARN: Agent 3 skipped.")
        df['status'] = df['status'].replace('Pending - Agent 3', 'Flagged for Review')
        df['reason'] = df.apply(lambda r: 'AI Adjudication (No Model)' if r['status'] == 'Flagged for Review' and r['reason'] == 'Passed anomaly check' else r['reason'], axis=1)
        return df
    
    proc_df = df[df['status'] != 'Pending - Agent 3'].copy()
    pend_df = df[df['status'] == 'Pending - Agent 3'].copy()
    
    if pend_df.empty:
        return proc_df
        
    sys_prompt = """Expert adjudicator (HIPAA/ICD-10). Analyze procedure_code, diagnosis_code, claim_amount. Respond JSON list: [{"claim_id":"...", "status":"Accept/Reject", "reason":"..."}]. Accept simple/low claims ('99213'/'J06.9'<$500). Reject mismatch ('99213'/'S02.0XXA'). Reject complex/high claims (need docs)."""
    claims = pend_df[['claim_id','procedure_code','diagnosis_code','claim_amount']].to_dict('records')
    
    if not claims:
        return proc_df
        
    prompt = f"Adjudicate:\n{str(claims)}"
    
    try:
        resp = await gemini_model.generate_content_async([sys_prompt, prompt], generation_config={"response_mime_type":"application/json"})
        ai_txt = resp.text
        ai_res: List[Dict[str,str]] = pd.read_json(io.StringIO(ai_txt)).to_dict('records')
        ai_decs = {res['claim_id']: {'status': res['status'], 'reason': res['reason']} for res in ai_res}

        def apply_dec(row):
            d = ai_decs.get(row['claim_id'])
            if d:
                row['status'] = d['status']
                row['reason'] = d['reason']
            else:
                row['status'] = 'Flagged for Review'
                row['reason'] = 'AI no decision.'
            return row
            
        adj_df = pend_df.apply(apply_dec, axis=1)
        final_df = pd.concat([proc_df, adj_df], ignore_index=True)
        return final_df
        
    except Exception as e:
        print(f"ERR Gemini: {e}")
        pend_df['status'] = 'Flagged for Review'
        pend_df['reason'] = f'AI Error: {e}. Review needed.'
        final_df = pd.concat([proc_df, pend_df], ignore_index=True)
        return final_df

# === Firestore Save Function ===
def save_job_to_firestore(job_data: dict, file_name: str, custom_rule: str) -> str:
    if db is None:
        print("Firestore skip save.")
        return None
    try:
        doc_ref = db.collection('jobs').document()
        total = len(job_data)
        accepted = len([r for r in job_data if r['status']=='Accept'])
        rejected = len([r for r in job_data if r['status']=='Reject'])
        flagged = len([r for r in job_data if r['status']=='Flagged for Review'])
        job_summary = {
            'jobId': doc_ref.id,
            'fileName': file_name,
            'customRule': custom_rule,
            'createdAt': datetime.datetime.now(datetime.timezone.utc),
            'summary': { 'total': total, 'accepted': accepted, 'rejected': rejected, 'flagged': flagged },
            'results': job_data
        }
        doc_ref.set(job_summary)
        print(f"Job saved: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        print(f"ERR Firestore save: {e}")
        return None

# === Main API Endpoint ===
@app.post("/upload-and-process")
async def upload_and_process(file: UploadFile = File(...), custom_rule: str = Form(...)):
     if not file.filename.endswith('.csv'):
         raise HTTPException(400, "Upload .csv")
     
     try:
         contents = await file.read()
     except Exception as e:
         raise HTTPException(400, f"Read Error: {e}")

     try:
         df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
     except UnicodeDecodeError:
         try:
             df = pd.read_csv(io.StringIO(contents.decode('latin-1')))
         except Exception as de:
             raise HTTPException(400, f"Decode Error: {de}")
     except pd.errors.ParserError as pe:
         raise HTTPException(400, f"CSV Parse Error: {pe}")
         
     if df.empty:
         raise HTTPException(400, "CSV empty.")
         
     try:
        df1 = agent_1_validation(df, custom_rule)
        df2 = agent_2_anomaly(df1)
        df_final = await agent_3_adjudication(df2)
        
        results = df_final.replace({np.nan: None}).to_dict('records')
        
        save_job_to_firestore(results, file.filename, custom_rule)
        
        return results # Return only results array
     except Exception as e:
        print(f"ERR processing: {e}") # This will print the UnboundLocalError
        raise HTTPException(500, f"Internal error.") # This is what you see


# === Job History Endpoint ===
@app.get("/jobs")
async def get_job_history():
     if db is None:
         raise HTTPException(500, "Firestore disconnected.")
         
     try:
         jobs_ref = db.collection('jobs').order_by('createdAt', direction=firestore.Query.DESCENDING).limit(20)
         jobs = jobs_ref.stream()
         job_list = []
     except Exception as e:
         print(f"ERR fetch history query: {e}")
         raise HTTPException(500, "DB Query Error.")
         
     try:
         for job in jobs:
             job_data = job.to_dict()
             job_data['jobId'] = job.id

             if 'createdAt' in job_data and isinstance(job_data['createdAt'], datetime.datetime):
                 job_data['createdAt'] = job_data['createdAt'].isoformat()
             elif 'createdAt' not in job_data:
                 job_data['createdAt'] = datetime.datetime.min.isoformat()

             if 'results' in job_data:
                 del job_data['results']

             job_list.append(job_data)
         return job_list
     except Exception as e:
         print(f"ERR processing history results: {e}")
         raise HTTPException(500, "Error processing history.")

# === Job Details Endpoint ===
@app.get("/job/{job_id}")
async def get_job_details(job_id: str):
     if db is None:
         raise HTTPException(500, "Firestore disconnected.")
         
     try:
         doc_ref = db.collection('jobs').document(job_id)
         doc = doc_ref.get()
     except Exception as e:
         print(f"ERR fetching job doc {job_id}: {e}")
         raise HTTPException(500, "DB Get Error.")
         
     if not doc.exists:
         raise HTTPException(404, "Job not found.")
         
     try:
         job_data = doc.to_dict()
         if 'createdAt' in job_data and isinstance(job_data['createdAt'], datetime.datetime):
             job_data['createdAt'] = job_data['createdAt'].isoformat()
         
         job_data['results'] = job_data.get('results', [])
         if not isinstance(job_data['results'], list):
             print(f"WARN: Job {job_id} results not list. Resetting.")
             job_data['results'] = []
             
         return job_data
     except Exception as e:
         print(f"ERR processing job doc {job_id}: {e}")
         raise HTTPException(500, "Error processing job data.")

# === Trend Analysis Endpoint ===
@app.get("/trends/rejection_reasons")
async def get_rejection_reason_trends(limit: int = 50, top_n: int = 10):
    if db is None:
        raise HTTPException(500, "Firestore disconnected.")
        
    rejection_reasons = Counter()
    processed_job_count = 0
    
    try:
        jobs_ref = db.collection('jobs').order_by('createdAt', direction=firestore.Query.DESCENDING).limit(limit)
        jobs = jobs_ref.stream()
        
        for job in jobs:
            processed_job_count += 1
            job_data = job.to_dict()
            results = job_data.get('results', [])
            
            if isinstance(results, list):
                for claim in results:
                    if isinstance(claim, dict) and claim.get('status') == 'Reject' and claim.get('reason'):
                        reason = str(claim['reason']).strip()
                        if reason.startswith("Missing data"): reason = "Missing required data"
                        elif reason.startswith("Failed rule"): reason = "Failed custom rule"
                        elif reason.startswith("Invalid type"): reason = "Invalid data type"
                        elif reason.startswith("AI Error"): reason = "AI Adjudication Error"
                        elif reason.startswith("Rule error"): reason = "Custom rule syntax error"
                        rejection_reasons[reason] += 1
            else:
                print(f"WARN: Job {job.id} invalid 'results'.")
                
        top_reasons = rejection_reasons.most_common(top_n)
        
        colors = ['#FF6384','#36A2EB','#FFCE56','#4BC0C0','#9966FF','#FF9F40','#C9CBCF','#7C8AFF','#8AFF8A','#FF8A8A']
        
        chart_data = {
            "labels": [r for r, c in top_reasons],
            "datasets": [{
                "label": f"Top {top_n} Reasons (Last {processed_job_count} jobs)",
                "data": [c for r, c in top_reasons],
                "backgroundColor": [c+'B3' for c in colors[:top_n]], # Add transparency
                "borderColor": [c for c in colors[:top_n]],
                "borderWidth": 1
            }]
        }
        
        return {
            "processed_job_count": processed_job_count,
            "total_rejections_analyzed": sum(rejection_reasons.values()),
            "top_reasons_data": chart_data
        }
    except Exception as e:
        print(f"ERR trends: {e}")
        raise HTTPException(500, f"Error generating trends: {e}")

@app.get("/")
def read_root():
    return {"message": "MediAgent API running."}