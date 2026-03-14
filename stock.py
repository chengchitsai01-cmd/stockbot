import os
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import requests
from datetime import datetime

# ==========================================
# 1. 設定區 (從 GitHub Secrets 讀取)
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID")

# 觀察清單
TARGET_LIST = [
    "2330.TW", "2454.TW", "0050.TW", "2301.TW", "3481.TW", 
    "3324.TW", "3017.TW", "2344.TW", "2308.TW", "2317.TW",
    "3711.TW", "5289.TW", "8299.TW", "2327.TW", "2382.TW",
    "3289.TW", "3260.TW", "8039.TW", "1101.TW", "1301.TW",
    "2408.TW", "2449.TW", "3037.TW", "5469.TW", "6213.TW",
    "2603.TW", "3231.TW", "2421.TW", "3653.TW", "6805.TW",
    "8996.TW", "6125.TW", "1587.TW", "6230.TW", "3533.TW",
    "4566.TW", "4551.TW", "2233.TW", "6197.TW", "2228.TW",
    "4569.TW", "3484.TW", "3013.TW", "3162.TW", "6982.TW"
]

genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 2. 功能函數
# ==========================================
def send_line_message(text):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_CHANNEL_ACCESS_TOKEN}'
    }
    data = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": text}]
    }
    requests.post(url, headers=headers, json=data)

def analyze_stock(symbol):
    try:
        # 修正點 1: 加上 progress=False 且確保資料抓取穩定
        df = yf.download(symbol, period="60d", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty or len(df) < 35: 
            return None

        # 修正點 2: 處理 yfinance 可能產生的多層索引 (Multi-Index)
        # 強制轉換為單層欄位名稱，避免 "The truth value of a Series is ambiguous" 錯誤
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- 技術指標計算 ---
        # 1. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # --- 判斷訊號 ---
        # 修正點 3: 使用 .iloc[-1].item() 確保取得的是單一數值而不是一個數列
        last_close = df['Close'].iloc[-1]
        last_rsi = df['RSI'].iloc[-1]
        last_macd = df['MACD'].iloc[-1]
        last_signal = df['Signal_Line'].iloc[-1]
        
        prev_macd = df['MACD'].iloc[-2]
        prev_signal = df['Signal_Line'].iloc[-2]
        
        avg_vol = df['Volume'].tail(5).mean()
        last_vol = df['Volume'].iloc[-1]

        # 條件：RSI < 35 且 MACD 金叉
        is_rsi_low = float(last_rsi) < 35
        is_macd_cross = (float(prev_macd) < float(prev_signal)) and (float(last_macd) > float(last_signal))
        is_vol_up = float(last_vol) > float(avg_vol)

        if is_rsi_low and is_macd_cross:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"你是台股操盤專家。股票 {symbol} RSI為 {float(last_rsi):.2f} 且 MACD 剛金叉。請用 50 字內建議進場策略與風險。"
            response = model.generate_content(prompt)
            
            return (f"🚨【{symbol} 買入訊號】🚨\n"
                    f"現價: {float(last_close):.2f}\n"
                    f"RSI: {float(last_rsi):.2f}\n"
                    f"量能: {'放量' if is_vol_up else '量縮'}\n\n"
                    f"🤖 AI 分析：\n{response.text}")
        
        return None
    except Exception as e:
        print(f"分析 {symbol} 失敗: {str(e)}")
        return None
# ==========================================
# 3. 主程式執行
# ==========================================
def main():
    if not all([GOOGLE_API_KEY, LINE_CHANNEL_ACCESS_TOKEN, LINE_USER_ID]):
        print("缺少環境變數，請檢查 GitHub Secrets 設定。")
        return

    print(f"[{datetime.now()}] 開始掃描...")
    all_alerts = []
    
    for symbol in TARGET_LIST:
        report = analyze_stock(symbol)
        if report:
            all_alerts.append(report)
    
    if all_alerts:
        full_msg = "\n" + "="*15 + "\n".join(all_alerts)
        send_line_message(full_msg)
        print("警報已發送。")
    else:
        print("目前無符合條件之標的。")

if __name__ == "__main__":
    main()
