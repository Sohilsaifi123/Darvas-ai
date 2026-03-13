"""
Darvas AI — Daily Runner
Runs every weekday 9:15 AM IST via GitHub Actions
"""
import os, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

TODAY = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    'APOLLOHOSP.NS','MAXHEALTH.NS','METROPOLIS.NS','LALPATHLAB.NS',
    'DABUR.NS','MARICO.NS','GODREJCP.NS','COLPAL.NS','EMAMILTD.NS',
    'HEROMOTOCO.NS','BAJAJ-AUTO.NS','EICHERMOT.NS','TVSMOTORS.NS',
    'BALKRISIND.NS','APOLLOTYRE.NS','MRF.NS','CEATLTD.NS',
    'DIXON.NS','HAVELLS.NS','VOLTAS.NS','CROMPTON.NS',
    'JKCEMENT.NS','ACC.NS','SHREECEM.NS',
    'DLF.NS','GODREJPROP.NS','OBEROIRLTY.NS','PRESTIGE.NS',
    'COALINDIA.NS','VEDL.NS','HINDZINC.NS','NMDC.NS',
    'ZOMATO.NS','IRCTC.NS','INDIAMART.NS',
    'LTIM.NS','MPHASIS.NS','COFORGE.NS','PERSISTENT.NS','KPITTECH.NS',
    'HDFCLIFE.NS','SBILIFE.NS','CDSL.NS','MCX.NS',
    'TATAPOWER.NS','ADANIGREEN.NS','ADANIPORTS.NS',
    'PIDILITIND.NS','SRF.NS','JUBLFOOD.NS','DMART.NS','TRENT.NS',
]

TIMEFRAMES = {
    'weekly':     {'rule': 'W',   'min_bars': 4, 'max_wait': 52, 'tolerance': 0.03},
    'monthly':    {'rule': 'ME',  'min_bars': 3, 'max_wait': 18, 'tolerance': 0.04},
    'quarterly':  {'rule': 'QE',  'min_bars': 3, 'max_wait': 8,  'tolerance': 0.05},
    'halfyearly': {'rule': '2QE', 'min_bars': 2, 'max_wait': 5,  'tolerance': 0.06},
    'yearly':     {'rule': 'YE',  'min_bars': 2, 'max_wait': 3,  'tolerance': 0.07},
}

def resample_df(df, rule):
    try:
        r = df.set_index('date').resample(rule).agg({
            'open':'first','high':'max','low':'min',
            'close':'last','volume':'sum'
        }).dropna().reset_index()
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
        ceil_ok = all(
            df_tf.iloc[j]['close'] <= bh*(1+tol)
            for j in range(i+1, min(i+min_b+1, len(df_tf)))
        )
        if not ceil_ok:
            i += 1
            continue
        bl = df_tf.iloc[max(0,i-1):i+min_b+1]['close'].min()
        fl_ok = all(
            df_tf.iloc[j]['close'] >= bl*(1-tol)
            for j in range(i+1, min(i+min_b+1, len(df_tf)))
        )
        if not fl_ok:
            i += 1
            continue
        br = bh - bl
        if br/max(bl,1) < 0.01:
            i += 1
            continue
        avg_vol = df_tf.iloc[max(0,i-10):i]['volume'].mean()
        bo = -1
        box_end = min(i+min_b, len(df_tf)-1)
        false_bos = 0
        for k in range(i+min_b, min(i+max_w, len(df_tf))):
            cur = df_tf.iloc[k]
            if cur['high'] > bh*(1+tol) and cur['close'] <= bh*(1+tol/2):
                false_bos += 1
                box_end = k
                continue
            if cur['close'] > bh*1.002:
                bo = k
                box_end = k-1
                break
            if cur['close'] < bl*(1-tol):
                box_end = k
                break
            bl = min(bl, cur['close']*1.005)
            br = bh - bl
            box_end = k
        h52 = df_tf.iloc[max(0,i-52):i+1]['high'].max()
        vol_bo = df_tf.iloc[bo]['volume'] > avg_vol*1.5 if bo != -1 else False
        status = 'BREAKOUT' if bo != -1 else ('ACTIVE' if box_end == len(df_tf)-1 else 'FAILED')
        bd = {
            'start_idx': i, 'end_idx': box_end, 'breakout_idx': bo,
            'start_date': str(df_tf.iloc[i]['date'])[:10],
            'box_high': round(bh, 2), 'box_low': round(bl, 2),
            'box_range_pct': round(br/max(bl,1)*100, 2),
            'stop_loss': round(bl*0.985, 2),
            'target1': round(bh+br, 2), 'target2': round(bh+br*2, 2),
            'is_52wk_high': bh >= h52*0.95,
            'vol_declining': True,
            'false_breakouts': false_bos,
            'bars_in_box': box_end - i,
            'status': status,
            'breakout_date': str(df_tf.iloc[bo]['date'])[:10] if bo != -1 else None,
            'breakout_price': round(df_tf.iloc[bo]['close'], 2) if bo != -1 else None,
            'vol_surge': vol_bo,
        }
        if bo != -1:
            entry = bd['breakout_price']
            fut = df_tf.iloc[bo:min(bo+max_w, len(df_tf))]
            mg = (fut['high'].max() - entry) / entry * 100
            bd['max_gain_pct'] = round(mg, 2)
            bd['success'] = mg > 8
            i = bo + 1
        else:
            bd['max_gain_pct'] = 0
            bd['success'] = False
            i += 1
        boxes.append(bd)
    return boxes

def analyze_stock(df):
    results = {}
    for tf_name, tf_cfg in TIMEFRAMES.items():
        df_tf = resample_df(df, tf_cfg['rule'])
        if df_tf is None or len(df_tf) < 10:
            continue
        boxes = detect_boxes(df_tf, tf_cfg)
        if not boxes:
            continue
        br = [b for b in boxes if b['status'] == 'BREAKOUT']
        sc = [b for b in br if b.get('success', False)]
        results[tf_name] = {
            'df': df_tf, 'boxes': boxes,
            'total': len(boxes), 'breakouts': len(br), 'success': len(sc),
            'success_rate': round(len(sc)/len(br)*100, 1) if br else 0,
            'avg_gain': round(np.mean([b['max_gain_pct'] for b in sc]), 2) if sc else 0,
            'best': max([b['max_gain_pct'] for b in br], default=0),
        }
    return results

def best_tf(tf_res):
    if not tf_res:
        return None
    scores = {
        tf: d['success_rate']*0.4 + min(d['avg_gain'],50)*0.3 + min(d['breakouts'],10)*2
        for tf, d in tf_res.items()
    }
    return max(scores, key=scores.get)

def mtf_score(tf_res):
    score = 0
    sigs = {}
    for tf, data in tf_res.items():
        if not data['boxes']:
            continue
        lb = data['boxes'][-1]
        blen = len(data['df'])
        if lb['status'] == 'ACTIVE':
            sigs[tf] = 'ACTIVE_BOX'
            score += 0.5
        elif lb['status'] == 'BREAKOUT':
            bi = lb.get('breakout_idx', 0) or 0
            bsince = blen - bi
            if bsince <= 3:
                sigs[tf] = 'FRESH_BREAKOUT'
                score += 1
            elif bsince <= 8:
                sigs[tf] = 'RECENT_BREAKOUT'
                score += 0.7
            else:
                sigs[tf] = 'OLD_BREAKOUT'
        else:
            sigs[tf] = 'FAILED'
    confluence = round(score/max(len(tf_res),1)*100)
    return confluence, sigs

def generate_signals(stock_data):
    print(f'🔍 Scanning {len(stock_data)} stocks...')
    signals = []
    for sym, df in stock_data.items():
        try:
            tf_res = analyze_stock(df)
            if not tf_res:
                continue
            bt = best_tf(tf_res)
            conf, sigs = mtf_score(tf_res)
            td = tf_res.get(bt, {})
            boxes = td.get('boxes', [])
            lb = boxes[-1] if boxes else None
            last = df.iloc[-1]
            prev = df.iloc[-2]
            chg = (last['close'] - prev['close']) / prev['close'] * 100
            h52 = df.iloc[-252:]['high'].max() if len(df) >= 252 else df['high'].max()
            near_52wk = last['close'] >= h52 * 0.95

            # Signal logic
            ai_signal = 'HOLD'
            confidence = 0
            reasons = []
            warnings_list = []

            if lb:
                if lb['status'] == 'BREAKOUT':
                    bsince = len(td['df']) - (lb.get('breakout_idx') or 0)
                    if bsince <= 3:
                        ai_signal = 'BUY'
                        confidence = min(int(conf*0.4 + td.get('success_rate',0)*0.6), 100)
                        reasons.append(f"✅ {bt.upper()} Fresh Breakout @ ₹{lb.get('breakout_price','?')}")
                        if lb.get('vol_surge'):
                            reasons.append("✅ Volume surge confirmed (2x+ average)")
                        else:
                            warnings_list.append("⚠️ Volume low at breakout")
                        if lb.get('false_breakouts', 0) > 0:
                            reasons.append(f"✅ {lb['false_breakouts']} false spikes filtered")
                    elif bsince <= 8:
                        ai_signal = 'WATCH'
                        confidence = 50
                        reasons.append(f"📦 {bt.upper()} Recent Breakout — monitor closely")
                elif lb['status'] == 'ACTIVE':
                    ai_signal = 'WATCH'
                    confidence = 40
                    reasons.append(f"📦 {bt.upper()} Box forming: ₹{lb['box_low']}–₹{lb['box_high']}")
                    reasons.append(f"   Range: {lb['box_range_pct']}% | Bars: {lb['bars_in_box']}")

            if lb and lb.get('is_52wk_high'):
                reasons.append("✅ Near 52-week high")
            if lb and td.get('success_rate', 0) > 60:
                reasons.append(f"✅ High success rate: {td.get('success_rate')}%")

            reasons.append(f"📊 MTF Confluence: {conf}%")
            for tf, sig in list(sigs.items())[:4]:
                e = '✅' if 'BREAKOUT' in sig else '📦' if 'ACTIVE' in sig else '❌'
                reasons.append(f"   {e} {tf}: {sig}")

            signals.append({
                'symbol': sym,
                'date': TODAY,
                'price': round(last['close'], 2),
                'change_pct': round(chg, 2),
                'ai_signal': ai_signal,
                'confidence': confidence,
                'best_tf': bt,
                'mtf_confluence': conf,
                'near_52wk_high': near_52wk,
                'success_rate': td.get('success_rate', 0),
                'avg_gain': td.get('avg_gain', 0),
                'box_high': lb['box_high'] if lb else None,
                'box_low': lb['box_low'] if lb else None,
                'stop_loss': lb['stop_loss'] if lb else None,
                'target1': lb['target1'] if lb else None,
                'target2': lb['target2'] if lb else None,
                'box_status': lb['status'] if lb else None,
                'reasons': reasons,
                'warnings': warnings_list,
            })
        except Exception as e:
            continue

    signals.sort(key=lambda x: (
        0 if x['ai_signal']=='BUY' else 1 if x['ai_signal']=='WATCH' else 2,
        -x['confidence']
    ))
    return signals

def print_report(signals):
    buys = [s for s in signals if s['ai_signal'] == 'BUY']
    watches = [s for s in signals if s['ai_signal'] == 'WATCH']
    print(f'\n{"="*60}')
    print(f'🤖 DARVAS AI — Daily Report — {TODAY}')
    print(f'{"="*60}')
    print(f'\n🟢 BUY SIGNALS ({len(buys)} stocks):')
    print(f'{"Symbol":<20} {"Price":>8} {"Chg%":>7} {"Conf":>6} {"MTF":>6} {"SR":>6} {"T1":>10}')
    print('-'*68)
    for s in buys:
        print(f"🔥 {s['symbol']:<18} {s['price']:>8.1f} {s['change_pct']:>+6.1f}% {s['confidence']:>5}% {s['mtf_confluence']:>5}% {s['success_rate']:>5}% {str(s['target1'] or '—'):>10}")
        for r in s['reasons'][:3]:
            print(f"   {r}")
        for w in s['warnings']:
            print(f"   {w}")
        print()
    print(f'\n👀 WATCH LIST ({len(watches)} stocks):')
    for s in watches[:8]:
        print(f"  📌 {s['symbol']} — ₹{s['price']:.1f} | Box: ₹{s['box_low'] or '?'}–{s['box_high'] or '?'} | MTF: {s['mtf_confluence']}%")
    print(f'\n{"="*60}')

def main():
    print(f'🚀 Darvas AI — {TODAY}')
    print('📥 Downloading latest data...')
    end = datetime.now()
    start = end - timedelta(days=365*3)
    stock_data = {}
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) < 100:
                continue
            df = df.reset_index()
            df.columns = ['date','open','high','low','close','volume']
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()
            stock_data[sym] = df
            print(f'  ✅ {sym}')
        except:
            print(f'  ❌ {sym}')
            continue
    print(f'\n✅ {len(stock_data)} stocks loaded')
    signals = generate_signals(stock_data)
    print_report(signals)

    # Save results
    output = {
        'date': TODAY,
        'total_scanned': len(stock_data),
        'buy_count': len([s for s in signals if s['ai_signal']=='BUY']),
        'watch_count': len([s for s in signals if s['ai_signal']=='WATCH']),
        'signals': signals
    }
    out_file = f'{OUTPUT_DIR}/signals_{TODAY}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(f'{OUTPUT_DIR}/latest_signals.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'💾 Saved: {out_file}')
    print('✅ Done!')

if __name__ == '__main__':
    main()
