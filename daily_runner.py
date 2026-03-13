"""
Darvas AI — Daily Runner with Google Drive
Runs every weekday 9:15 AM IST via GitHub Actions
Saves results directly to Google Drive/DarvasAI/
"""
import os, json, warnings, base64
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

TODAY = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── GOOGLE DRIVE SETUP ─────────────────────────────────────
def setup_drive():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds_json = os.environ.get('GDRIVE_CREDENTIALS', '')
        if not creds_json:
            print('⚠️ No Google Drive credentials found')
            return None
        creds_dict = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        print('✅ Google Drive connected!')
        return service
    except Exception as e:
        print(f'⚠️ Drive setup failed: {e}')
        return None

def get_folder_id(service, folder_name='DarvasAI'):
    try:
        results = service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        if files:
            print(f'✅ Found folder: {folder_name} ({files[0]["id"]})')
            return files[0]['id']
        print(f'⚠️ Folder {folder_name} not found in Drive')
        return None
    except Exception as e:
        print(f'⚠️ Folder search failed: {e}')
        return None

def upload_to_drive(service, folder_id, filename, content, mime='application/json'):
    try:
        from googleapiclient.http import MediaIoBaseUpload
        import io
        # Check if file exists
        results = service.files().list(
            q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
            fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        data = content.encode('utf-8') if isinstance(content, str) else content
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime)
        if files:
            # Update existing file
            service.files().update(
                fileId=files[0]['id'],
                media_body=media,
                supportsAllDrives=True
            ).execute()
            print(f'✅ Updated: {filename}')
        else:
            # Create new file
            file_metadata = {'name': filename, 'parents': [folder_id]}
            service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            print(f'✅ Created: {filename}')
        return True
    except Exception as e:
        print(f'⚠️ Upload failed for {filename}: {e}')
        return False

# ── NSE SYMBOLS ────────────────────────────────────────────
SYMBOLS = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
    'HINDUNILVR.NS','ITC.NS','SBIN.NS','BHARTIARTL.NS','KOTAKBANK.NS',
    'LT.NS','AXISBANK.NS','ASIANPAINT.NS','MARUTI.NS','TITAN.NS',
    'SUNPHARMA.NS','ULTRACEMCO.NS','BAJFINANCE.NS','WIPRO.NS','NESTLEIND.NS',
    'TECHM.NS','HCLTECH.NS','POWERGRID.NS','NTPC.NS','ONGC.NS',
    'TATAMOTORS.NS','JSWSTEEL.NS','TATASTEEL.NS','ADANIENT.NS','BAJAJFINSV.NS',
    'FEDERALBNK.NS','BANDHANBNK.NS','IDFCFIRSTB.NS','RBLBANK.NS','CANBK.NS',
    'BANKBARODA.NS','PNB.NS','DIVISLAB.NS','DRREDDY.NS','CIPLA.NS',
    'LUPIN.NS','TORNTPHARM.NS','ALKEM.NS','AUROPHARMA.NS','BIOCON.NS',
    'APOLLOHOSP.NS','DABUR.NS','MARICO.NS','GODREJCP.NS','COLPAL.NS',
    'HEROMOTOCO.NS','BAJAJ-AUTO.NS','EICHERMOT.NS','BALKRISIND.NS',
    'APOLLOTYRE.NS','MRF.NS','DIXON.NS','HAVELLS.NS','VOLTAS.NS',
    'JKCEMENT.NS','ACC.NS','SHREECEM.NS','DLF.NS','GODREJPROP.NS',
    'OBEROIRLTY.NS','COALINDIA.NS','VEDL.NS','HINDZINC.NS','NMDC.NS',
    'ZOMATO.NS','IRCTC.NS','LTIM.NS','MPHASIS.NS','COFORGE.NS',
    'PERSISTENT.NS','KPITTECH.NS','HDFCLIFE.NS','SBILIFE.NS','CDSL.NS',
    'MCX.NS','TATAPOWER.NS','ADANIGREEN.NS','ADANIPORTS.NS',
    'PIDILITIND.NS','SRF.NS','JUBLFOOD.NS','DMART.NS','TRENT.NS',
    'BAJAJHLDNG.NS','BRITANNIA.NS','MCDOWELL-N.NS','UNITDSPR.NS',
    'BERGEPAINT.NS','KANSAINER.NS','INDIGO.NS','INTERGLOBE.NS',
    'CHOLAFIN.NS','MUTHOOTFIN.NS','MANAPPURAM.NS','PEL.NS',
]

TIMEFRAMES = {
    'weekly':     {'rule':'W',   'min_bars':4,'max_wait':52,'tolerance':0.03},
    'monthly':    {'rule':'ME',  'min_bars':3,'max_wait':18,'tolerance':0.04},
    'quarterly':  {'rule':'QE',  'min_bars':3,'max_wait':8, 'tolerance':0.05},
    'halfyearly': {'rule':'2QE', 'min_bars':2,'max_wait':5, 'tolerance':0.06},
    'yearly':     {'rule':'YE',  'min_bars':2,'max_wait':3, 'tolerance':0.07},
}

# ── DARVAS BOX ENGINE ──────────────────────────────────────
def resample_df(df, rule):
    try:
        r = df.set_index('date').resample(rule).agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
        ).dropna().reset_index()
        return r
    except:
        return None

def detect_boxes(df_tf, tf_cfg):
    if df_tf is None or len(df_tf) < 10:
        return []
    tol = tf_cfg['tolerance']
    min_b = tf_cfg['min_bars']
    max_w = tf_cfg['max_wait']
    boxes = []
    i = min_b
    while i < len(df_tf) - min_b:
        bh = df_tf.iloc[i]['close']
        ceil_ok = all(df_tf.iloc[j]['close'] <= bh*(1+tol) for j in range(i+1, min(i+min_b+1, len(df_tf))))
        if not ceil_ok: i+=1; continue
        bl = df_tf.iloc[max(0,i-1):i+min_b+1]['close'].min()
        fl_ok = all(df_tf.iloc[j]['close'] >= bl*(1-tol) for j in range(i+1, min(i+min_b+1, len(df_tf))))
        if not fl_ok: i+=1; continue
        br = bh - bl
        if br/max(bl,1) < 0.01: i+=1; continue
        avg_vol = df_tf.iloc[max(0,i-10):i]['volume'].mean()
        bo = -1; box_end = min(i+min_b, len(df_tf)-1); false_bos = 0
        for k in range(i+min_b, min(i+max_w, len(df_tf))):
            cur = df_tf.iloc[k]
            if cur['high'] > bh*(1+tol) and cur['close'] <= bh*(1+tol/2):
                false_bos+=1; box_end=k; continue
            if cur['close'] > bh*1.002: bo=k; box_end=k-1; break
            if cur['close'] < bl*(1-tol): box_end=k; break
            bl = min(bl, cur['close']*1.005); box_end=k
        h52 = df_tf.iloc[max(0,i-52):i+1]['high'].max()
        vol_bo = df_tf.iloc[bo]['volume'] > avg_vol*1.5 if bo != -1 else False
        status = 'BREAKOUT' if bo != -1 else ('ACTIVE' if box_end == len(df_tf)-1 else 'FAILED')
        bd = {
            'start_idx':i,'end_idx':box_end,'breakout_idx':bo,
            'start_date':str(df_tf.iloc[i]['date'])[:10],
            'box_high':round(bh,2),'box_low':round(bl,2),
            'box_range_pct':round(br/max(bl,1)*100,2),
            'stop_loss':round(bl*0.985,2),
            'target1':round(bh+br,2),'target2':round(bh+br*2,2),
            'is_52wk_high':bh >= h52*0.95,
            'false_breakouts':false_bos,'bars_in_box':box_end-i,
            'status':status,
            'breakout_date':str(df_tf.iloc[bo]['date'])[:10] if bo!=-1 else None,
            'breakout_price':round(df_tf.iloc[bo]['close'],2) if bo!=-1 else None,
            'vol_surge':vol_bo,
        }
        if bo != -1:
            entry = bd['breakout_price']
            fut = df_tf.iloc[bo:min(bo+max_w, len(df_tf))]
            mg = (fut['high'].max()-entry)/entry*100
            bd['max_gain_pct'] = round(mg,2); bd['success'] = mg > 8
            i = bo+1
        else:
            bd['max_gain_pct'] = 0; bd['success'] = False; i+=1
        boxes.append(bd)
    return boxes

def analyze_stock(df):
    results = {}
    for tf_name, tf_cfg in TIMEFRAMES.items():
        df_tf = resample_df(df, tf_cfg['rule'])
        if df_tf is None or len(df_tf) < 10: continue
        boxes = detect_boxes(df_tf, tf_cfg)
        if not boxes: continue
        br = [b for b in boxes if b['status']=='BREAKOUT']
        sc = [b for b in br if b.get('success',False)]
        # Include last 60 candles for chart
        candles = df_tf.tail(60).copy()
        candles['date'] = candles['date'].astype(str).str[:10]
        results[tf_name] = {
            'boxes':boxes,
            'candles':candles.to_dict('records'),
            'total':len(boxes),'breakouts':len(br),'success':len(sc),
            'success_rate':round(len(sc)/len(br)*100,1) if br else 0,
            'avg_gain':round(np.mean([b['max_gain_pct'] for b in sc]),2) if sc else 0,
            'best':max([b['max_gain_pct'] for b in br],default=0),
        }
    return results

def best_tf(tf_res):
    if not tf_res: return None
    scores = {tf: d['success_rate']*0.4 + min(d['avg_gain'],50)*0.3 + min(d['breakouts'],10)*2 for tf,d in tf_res.items()}
    return max(scores, key=scores.get)

def mtf_score(tf_res):
    score=0; sigs={}
    for tf,data in tf_res.items():
        if not data['boxes']: continue
        lb = data['boxes'][-1]; blen=len(data['candles'])
        if lb['status']=='ACTIVE': sigs[tf]='ACTIVE_BOX'; score+=0.5
        elif lb['status']=='BREAKOUT':
            bi = lb.get('breakout_idx',0) or 0; bsince=blen-bi
            if bsince<=3: sigs[tf]='FRESH_BREAKOUT'; score+=1
            elif bsince<=8: sigs[tf]='RECENT_BREAKOUT'; score+=0.7
            else: sigs[tf]='OLD_BREAKOUT'
        else: sigs[tf]='FAILED'
    return round(score/max(len(tf_res),1)*100), sigs

def generate_signals(stock_data):
    print(f'🔍 Scanning {len(stock_data)} stocks...')
    signals = []
    for sym, df in stock_data.items():
        try:
            tf_res = analyze_stock(df)
            if not tf_res: continue
            bt = best_tf(tf_res); conf, sigs = mtf_score(tf_res)
            td = tf_res.get(bt, {}); boxes = td.get('boxes', [])
            lb = boxes[-1] if boxes else None
            last = df.iloc[-1]; prev = df.iloc[-2]
            chg = (last['close']-prev['close'])/prev['close']*100
            h52 = df.iloc[-252:]['high'].max() if len(df)>=252 else df['high'].max()
            ai_signal='HOLD'; confidence=0; reasons=[]; warnings_list=[]

            if lb:
                blen = len(td.get('candles',[]))
                if lb['status']=='BREAKOUT':
                    bsince = blen-(lb.get('breakout_idx') or 0)
                    if bsince<=3:
                        ai_signal='BUY'; confidence=min(int(conf*0.4+td.get('success_rate',0)*0.6),100)
                        reasons.append(f"✅ {bt.upper()} Fresh Breakout @ ₹{lb.get('breakout_price','?')}")
                        if lb.get('vol_surge'): reasons.append("✅ Volume surge confirmed (2x+ average)")
                        else: warnings_list.append("⚠️ Volume low at breakout")
                        if lb.get('false_breakouts',0)>0: reasons.append(f"✅ {lb['false_breakouts']} false spikes filtered")
                    elif bsince<=8:
                        ai_signal='WATCH'; confidence=50
                        reasons.append(f"📦 {bt.upper()} Recent Breakout — monitor closely")
                elif lb['status']=='ACTIVE':
                    ai_signal='WATCH'; confidence=40
                    reasons.append(f"📦 {bt.upper()} Box forming: ₹{lb['box_low']}–₹{lb['box_high']}")

            if lb and lb.get('is_52wk_high'): reasons.append("✅ Near 52-week high")
            if lb and td.get('success_rate',0)>60: reasons.append(f"✅ High success rate: {td.get('success_rate')}%")
            reasons.append(f"📊 MTF Confluence: {conf}%")
            for tf, sig in list(sigs.items())[:4]:
                e='✅' if 'BREAKOUT' in sig else '📦' if 'ACTIVE' in sig else '❌'
                reasons.append(f"   {e} {tf}: {sig}")

            # TF summary for chart
            tf_summary = {}
            for tf, d in tf_res.items():
                if not d['boxes']: continue
                lb2 = d['boxes'][-1]
                tf_summary[tf] = {
                    'success_rate': d['success_rate'],
                    'avg_gain': d['avg_gain'],
                    'latest_status': lb2['status'],
                    'box_high': lb2['box_high'],
                    'box_low': lb2['box_low'],
                }

            signals.append({
                'symbol':sym,'date':TODAY,
                'price':round(last['close'],2),'change_pct':round(chg,2),
                'ai_signal':ai_signal,'confidence':confidence,
                'best_tf':bt,'mtf_confluence':conf,
                'near_52wk_high':last['close']>=h52*0.95,
                'success_rate':td.get('success_rate',0),'avg_gain':td.get('avg_gain',0),
                'box_high':lb['box_high'] if lb else None,
                'box_low':lb['box_low'] if lb else None,
                'stop_loss':lb['stop_loss'] if lb else None,
                'target1':lb['target1'] if lb else None,
                'target2':lb['target2'] if lb else None,
                'box_status':lb['status'] if lb else None,
                'candles':td.get('candles',[]),
                'boxes':[{k:v for k,v in b.items() if k not in ['start_idx','end_idx','breakout_idx']} for b in boxes],
                'tf_summary':tf_summary,
                'reasons':reasons,'warnings':warnings_list,
            })
        except Exception as e:
            continue

    signals.sort(key=lambda x:(0 if x['ai_signal']=='BUY' else 1 if x['ai_signal']=='WATCH' else 2,-x['confidence']))
    return signals

def main():
    print(f'🚀 Darvas AI — {TODAY}')

    # Setup Google Drive
    drive = setup_drive()
    folder_id = get_folder_id(drive) if drive else None

    # Download stock data
    print('📥 Downloading latest data...')
    end = datetime.now(); start = end - timedelta(days=365*3)
    stock_data = {}
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) < 100: continue
            df = df.reset_index()
            df.columns = ['date','open','high','low','close','volume']
            df['date'] = pd.to_datetime(df['date'])
            stock_data[sym] = df.dropna()
            print(f'  ✅ {sym}')
        except:
            print(f'  ❌ {sym}'); continue

    print(f'\n✅ {len(stock_data)} stocks loaded')
    signals = generate_signals(stock_data)

    buys = [s for s in signals if s['ai_signal']=='BUY']
    watches = [s for s in signals if s['ai_signal']=='WATCH']
    print(f'\n🔥 BUY: {len(buys)} | 👀 WATCH: {len(watches)}')
    for s in buys:
        print(f"  🔥 {s['symbol']} @ ₹{s['price']} | Conf: {s['confidence']}% | MTF: {s['mtf_confluence']}%")

    # Prepare output
    output = {
        'date':TODAY,
        'total_scanned':len(stock_data),
        'buy_count':len(buys),
        'watch_count':len(watches),
        'signals':signals
    }
    output_json = json.dumps(output, indent=2, default=str)

    # Save locally
    with open(f'{OUTPUT_DIR}/signals_{TODAY}.json','w') as f: f.write(output_json)
    with open(f'{OUTPUT_DIR}/latest_signals.json','w') as f: f.write(output_json)
    print(f'💾 Saved locally: output/latest_signals.json')

    # Upload to Google Drive
    if drive and folder_id:
        print('\n📤 Uploading to Google Drive...')
        upload_to_drive(drive, folder_id, 'latest_signals.json', output_json)
        upload_to_drive(drive, folder_id, f'signals_{TODAY}.json', output_json)
        print('✅ Google Drive updated!')
    else:
        print('⚠️ Skipping Drive upload — check credentials')

    print(f'\n✅ DONE! {TODAY}')

if __name__ == '__main__':
    main()
