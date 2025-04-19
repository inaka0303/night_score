import fugashi, unidic_lite

dic_dir = unidic_lite.DICDIR               # ← パスが文字列で取れる
tagger = fugashi.Tagger(f"-d {unidic_lite.DICDIR}")# 明示的に辞書を指定
from sentence_transformers import SentenceTransformer, util
import numpy as np, streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "pfnet/plamo-embedding-1b",
        trust_remote_code=True
    )

model = load_model()
tagger = fugashi.Tagger()                  # UniDic Lite 内蔵

# プロトタイプ
night_terms = ["夜", "夜空", "星", "月", "闇", "深夜",
               "ネオン", "暗闇", "夜風", "夜明け前"]
v_night = model.encode(night_terms, normalize_embeddings=True).mean(axis=0)

def tokenize(text: str):
    """歌詞を surface 文字列のリストで返す"""
    return [tok.surface for tok in tagger(text) if tok.surface.strip()]

# --- 単語スコア関数 -------------------------------
def token_night_score(token: str) -> float:
    vec = model.encode(token, normalize_embeddings=True)
    return float(util.dot_score(vec, v_night))   # [-1,1]

# --- 歌詞全体スコア -------------------------------
def score_lyrics_token_sum(lyrics: str) -> float:
    tokens = tokenize(lyrics)
    scores = [max(0, token_night_score(t)) for t in tokens]
    if not scores:
        return 0.0
    return float(np.mean(scores) * 100)          # 平均→0‑100
# ---------- Streamlit UI ----------
st.title("🌙 夜度チェッカー – token 合計版")
lyrics = st.text_area("歌詞を貼り付けてください")

if st.button("判定"):
    if lyrics.strip():
        s = score_lyrics_token_sum(lyrics)
        st.metric("夜度", f"{s:.2f}")
        st.progress(s / 100)

        # ★ Top10 単語夜度表示 (文字列で処理)
        st.subheader("トークン別夜度トップ10")
        uniq = set(tokenize(lyrics))
        token_scores = sorted(
            [(t, token_night_score(t)) for t in uniq],
            key=lambda x: x[1], reverse=True
        )[:10]
        for t, sc in token_scores:
            st.write(f"{t:<6}  {sc:.2f}")
