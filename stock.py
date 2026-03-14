import os
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import requests
from datetime import datetime
import time

# ==========================================
# 1. 設定區 (從 GitHub Secrets 讀取)
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

# 完整觀察清單
TARGET_LIST = [
    "2330.TW", "2454.TW", "0050.TW", "2301.TW", "3481.TW", 
    "3324.TWO", "3017.TW", "2344.TW", "2308.TW", "2317.TW",
    "3711.TW", "5289.TWO", "8299.TWO", "2327.TW", "2382.TW",
    "3289.TWO", "3260.TWO", "8039.TW", "1101.TW", "1301.TW",
    "2408.TW", "2449.TW", "3037.TW", "5469.TW", "6213.TW",
    "2603.TW", "3231.TW", "2421.TW", "3653.TW", "6805.TW",
    "8996.TW", "6125.TWO", "1587.TW", "6230.TW", "3533.TW",
    "4566.TW", "4551.TW", "2233.TW", "6197.TW", "2228.TW",
    "4569.TW", "3484.TWO", "3013.TW", "3162.TWO", "6982.TWO"
]

genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 2. 功能函數
# ==========================================
def send_line_message(text):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {LINE_CHANNEL_ACCESS_TOKEN}'}
    data = {"to": LINE_USER_ID, "messages": [{"type": "text", "text": text}]}
    try:
        requests.post(url, headers=headers, json=data)
    except Exception as e:
        print(f"LINE 發送異常: {e}")

def get_stock_report(symbol):
    try:
        # 下載資料
        df = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 35: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 計算指標
        last_close = float(df['Close'].iloc[-1])
        support = float(df['Low'].tail(20).min())
        resistance = float(df['High'].tail(20).max())
        
        # RSI 計算
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        last_rsi = float((100 - (100 / (1 + (gain / loss)))).iloc[-1])

        # 判斷是否為強勢警報 (RSI < 35)
        is_alert = last_rsi < 35
        alert_tag = "🚨【觸發低檔警報】" if is_alert else "📊【例行診斷】"

        # 呼叫 AI 診斷
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (f"你是操盤手。{symbol}現價{last_close:.2f}，RSI {last_rsi:.1f}，"
                  f"支撐{support:.2f}/壓力{resistance:.2f}。請用20字內給出操作建議。")
        response = model.generate_content(prompt)
        
        return f"{alert_tag}{symbol}: {last_close:.2f}\n💡{response.text.strip()}\n"
    except Exception as e:
        return f"❌ {symbol} 錯誤: {str(e)}\n"

# ==========================================
# 3. 主程式
# ==========================================
def main():
    if not all([GOOGLE_API_KEY, LINE_CHANNEL_ACCESS_TOKEN, LINE_USER_ID]):
        print("缺少 Secrets 設定。")
        return

    print(f"[{datetime.now()}] 啟動全清單掃描...")
    
    # 因為 45 檔太多，我們分批發送，每則訊息包含 15 檔，避免 LINE 拒收
    batch_size = 15
    for i in range(0, len(TARGET_LIST), batch_size):
        batch = TARGET_LIST[i:i + batch_size]
        batch_reports = []
        
        for symbol in batch:
            report = get_stock_report(symbol)
            if report:
                batch_reports.append(report)
            time.sleep(1) # 稍微停頓避免 AI 限制頻率
        
        if batch_reports:
            full_msg = f"📉 股市診斷 (第 {i//batch_size + 1} 組)\n" + "\n".join(batch_reports)
            send_line_message(full_msg)
            
    print("✅ 掃描並發送完畢。")

if __name__ == "__main__":
    main()
