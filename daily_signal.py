import os
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime
from xgboost import XGBClassifier
import smtplib
from email.mime.text import MIMEText

# 설정
FINAL_TOPN = 5
FINAL_THRESHOLD = 0.60
TOTAL_CAPITAL = 10_000_000
MAX_WEIGHT = 0.3

PAPER_LOG_FILE = "paper_trading_log.csv"

# 이메일
def send_email(subject, body):
    user = os.getenv("EMAIL_USER")
    pw = os.getenv("EMAIL_APP_PASSWORD")
    to = os.getenv("EMAIL_TO")

    if not user:
        print(body)
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, pw)
        server.send_message(msg)

# 데이터 수집
def load_data():
    kosdaq = fdr.StockListing("KOSDAQ")
    top = kosdaq.sort_values("Marcap", ascending=False).head(150)

    all_data = []

    for _, row in top.iterrows():
        try:
            df = fdr.DataReader(row["Code"], "2022-01-01")
            df = df.reset_index()
            df["ticker"] = row["Code"]
            df["name"] = row["Name"]
            all_data.append(df)
        except:
            pass

    df = pd.concat(all_data)

    df = df.rename(columns={
        "Date": "날짜",
        "Open": "시가",
        "High": "고가",
        "Low": "저가",
        "Close": "종가",
        "Volume": "거래량",
        "Change": "등락률"
    })

    df["날짜"] = pd.to_datetime(df["날짜"])
    return df

# feature 생성
def make_features(df):
    df = df.sort_values(["ticker", "날짜"])
    g = df.groupby("ticker")

    df["거래대금"] = df["종가"] * df["거래량"]
    df["거래대금_log"] = np.log1p(df["거래대금"])

    df["return_1d"] = g["종가"].pct_change()
    df["return_5d"] = g["종가"].pct_change(5)
    df["return_20d"] = g["종가"].pct_change(20)

    df["ma5"] = g["종가"].transform(lambda x: x.rolling(5).mean())
    df["ma20"] = g["종가"].transform(lambda x: x.rolling(20).mean())
    df["ma60"] = g["종가"].transform(lambda x: x.rolling(60).mean())

    df["ma5_gap"] = df["종가"] / df["ma5"] - 1
    df["ma20_gap"] = df["종가"] / df["ma20"] - 1
    df["ma60_gap"] = df["종가"] / df["ma60"] - 1

    df["volatility_20"] = g["return_1d"].transform(lambda x: x.rolling(20).std())

    df["volume_change"] = g["거래량"].pct_change()
    df["volume_ma20"] = g["거래량"].transform(lambda x: x.rolling(20).mean())
    df["volume_ratio"] = df["거래량"] / df["volume_ma20"]

    df["high_low_gap"] = (df["고가"] - df["저가"]) / df["종가"]
    df["open_close_gap"] = (df["종가"] - df["시가"]) / df["시가"]

    df["next_open"] = g["시가"].shift(-1)
    df["next_close"] = g["종가"].shift(-1)

    df["target"] = (df["next_close"] > df["next_open"]).astype(int)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

return df

# 모델
def train_model(df):
    features = [
        "시가","고가","저가","종가","거래량","거래대금_log","등락률",
        "return_1d","return_5d","return_20d",
        "ma5_gap","ma20_gap","ma60_gap",
        "volatility_20",
        "volume_change","volume_ratio",
        "high_low_gap","open_close_gap"
    ]

    model = XGBClassifier(n_estimators=200, max_depth=5)

    model.fit(df[features], df["target"])

    return model, features

# 신호 생성
def generate_signal(df, model, features):
    latest = df["날짜"].max()
    today = df[df["날짜"] == latest].copy()

    today["proba"] = model.predict_proba(today[features])[:,1]
    today["rank"] = today["proba"].rank(ascending=False)

    sig = today[
        (today["rank"] <= FINAL_TOPN) &
        (today["proba"] >= FINAL_THRESHOLD)
    ]

    return sig.sort_values("rank"), latest

# 포지션 계산
def allocate(sig):
    if len(sig) == 0:
        return sig

    sig["weight"] = (1/len(sig)).clip(upper=MAX_WEIGHT)
    sig["target_amount"] = TOTAL_CAPITAL * sig["weight"]

    sig["shares"] = (sig["target_amount"] // sig["종가"]).astype(int)
    sig["buy_amount"] = sig["shares"] * sig["종가"]

    return sig

# 저장
def save(sig, date):
    if len(sig) == 0:
        return

    sig["signal_date"] = date
    sig["status"] = "OPEN"

    if os.path.exists(PAPER_LOG_FILE):
        old = pd.read_csv(PAPER_LOG_FILE)
        sig = pd.concat([old, sig])

    sig.to_csv(PAPER_LOG_FILE, index=False)

# 실행
def main():
    df = load_data()
    df = make_features(df)
    model, features = train_model(df)

    sig, date = generate_signal(df, model, features)
    sig = allocate(sig)

    save(sig, date)

    msg = f"{date}\n"
    for _, r in sig.iterrows():
        msg += f"{r['name']} {r['shares']}주\n"

    send_email("Daily Signal", msg)

if __name__ == "__main__":
    main()
