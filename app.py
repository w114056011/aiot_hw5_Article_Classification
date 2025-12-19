"""
AI vs Human 文章分類工具
使用 Streamlit 建立的 AI 內容偵測器
"""

import streamlit as st
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer


def is_chinese_text(text):
    """判斷文本是否主要為中文"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.sub(r'\s', '', text))
    return chinese_chars / total_chars > 0.3 if total_chars > 0 else False


def tokenize_text(text):
    """根據文本語言進行分詞"""
    if is_chinese_text(text):
        # 中文：按字符分詞，同時保留標點符號分隔
        # 簡單分詞：每個中文字為一個 token，英文單詞保持完整
        tokens = []
        current_word = ""
        for char in text:
            if re.match(r'[\u4e00-\u9fff]', char):
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                tokens.append(char)
            elif re.match(r'[a-zA-Z0-9]', char):
                current_word += char
            else:
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
        if current_word:
            tokens.append(current_word)
        return tokens
    else:
        # 英文：按空格分詞
        return text.split()

# 頁面設定
st.set_page_config(
    page_title="AI vs Human 文章分類器",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .ai-result {
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
    }
    .human-result {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


class TextFeatureExtractor:
    """文本特徵提取器 - 使用自建特徵法"""
    
    @staticmethod
    def extract_features(text):
        """提取文本的多維特徵"""
        features = {}
        
        # 判斷語言
        is_chinese = is_chinese_text(text)
        features['is_chinese'] = is_chinese
        
        # 基本統計特徵
        words = tokenize_text(text)
        
        # 中文句子分割
        if is_chinese:
            sentences = re.split(r'[.!?。！？，,;；]+', text)
        else:
            sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['sentence_count'] = len(sentences) if sentences else 1
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        
        # 詞彙多樣性
        unique_words = set(w.lower() for w in words)
        features['vocabulary_richness'] = len(unique_words) / len(words) if words else 0
        
        # 標點符號統計（包含中文標點）
        all_punctuation = string.punctuation + '，。！？、；：「」『』【】（）《》〈〉'
        punctuation_count = sum(1 for c in text if c in all_punctuation)
        features['punctuation_ratio'] = punctuation_count / len(text) if text else 0
        
        # 特殊字符統計
        features['comma_ratio'] = (text.count(',') + text.count('，')) / len(words) if words else 0
        features['semicolon_ratio'] = (text.count(';') + text.count('；')) / len(words) if words else 0
        
        # 段落分析
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        
        # 句子長度變異性（針對中文調整）
        if is_chinese:
            sentence_lengths = [len(tokenize_text(s)) for s in sentences if s]
        else:
            sentence_lengths = [len(s.split()) for s in sentences if s]
        features['sentence_length_std'] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # 重複詞比例（AI 傾向重複特定模式）
        word_freq = Counter(w.lower() for w in words)
        if words:
            most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
            features['repetition_score'] = most_common_freq / len(words)
        else:
            features['repetition_score'] = 0
        
        # 過渡詞比例（AI 常用過渡詞）- 擴展中文過渡詞
        transition_words_en = ['however', 'therefore', 'furthermore', 'moreover', 
                              'additionally', 'consequently', 'nevertheless', 'thus',
                              'hence', 'accordingly', 'meanwhile', 'subsequently']
        transition_words_zh = ['然而', '因此', '此外', '而且', '另外', '總之', '首先', '其次',
                              '綜上所述', '總而言之', '換句話說', '也就是說', '進一步', '具體來說',
                              '一方面', '另一方面', '與此同時', '值得注意的是', '不僅如此']
        
        if is_chinese:
            # 檢查中文過渡詞（在原文中搜索）
            transition_count = sum(1 for tw in transition_words_zh if tw in text)
            features['transition_ratio'] = transition_count / features['sentence_count']
        else:
            transition_count = sum(1 for w in words if w.lower() in transition_words_en)
            features['transition_ratio'] = transition_count / len(words) if words else 0
        
        # 口語化表達偵測（人類特徵）
        colloquial_zh = ['啦', '嘛', '呢', '吧', '喔', '哦', '耶', '欸', '誒', '嗯', '唉',
                        '說實話', '老實說', '其實', '反正', '不過', '話說', '對了',
                        '超', '很', '蠻', '挺', '還不錯', '普普', '還好', '有點']
        colloquial_en = ["i'm", "i've", "don't", "can't", "won't", "it's", "that's",
                        "gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "nope",
                        "well", "anyway", "actually", "basically", "honestly"]
        
        if is_chinese:
            colloquial_count = sum(1 for cw in colloquial_zh if cw in text)
        else:
            colloquial_count = sum(1 for w in words if w.lower() in colloquial_en)
        features['colloquial_ratio'] = colloquial_count / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # 第一人稱使用（人類特徵）
        first_person_zh = ['我', '我們', '我的', '我覺得', '我認為', '我想']
        first_person_en = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        
        if is_chinese:
            first_person_count = sum(text.count(fp) for fp in first_person_zh)
        else:
            first_person_count = sum(1 for w in words if w.lower() in first_person_en)
        features['first_person_ratio'] = first_person_count / len(words) if words else 0
        
        # 被動語態標記詞（AI 傾向使用）
        passive_markers = ['is', 'are', 'was', 'were', 'been', 'being', 'be']
        if not is_chinese:
            passive_count = sum(1 for w in words if w.lower() in passive_markers)
            features['passive_ratio'] = passive_count / len(words) if words else 0
        else:
            features['passive_ratio'] = 0
        
        # 數字使用比例
        digit_count = sum(1 for c in text if c.isdigit())
        features['digit_ratio'] = digit_count / len(text) if text else 0
        
        # 大寫字母比例
        upper_count = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = upper_count / len(text) if text else 0
        
        # Burstiness（詞彙突發性）- AI 文本通常更均勻
        if len(words) > 10:
            word_positions = {}
            for i, w in enumerate(words):
                w_lower = w.lower()
                if w_lower not in word_positions:
                    word_positions[w_lower] = []
                word_positions[w_lower].append(i)
            
            bursts = []
            for positions in word_positions.values():
                if len(positions) > 1:
                    gaps = np.diff(positions)
                    bursts.append(np.std(gaps) if len(gaps) > 0 else 0)
            features['burstiness'] = np.mean(bursts) if bursts else 0
        else:
            features['burstiness'] = 0
        
        return features


class AIDetector:
    """AI 偵測器主類別"""
    
    def __init__(self):
        self.feature_extractor = TextFeatureExtractor()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化預訓練模型（模擬）"""
        # 在實際應用中，這裡會載入真正的預訓練模型
        # 這裡使用啟發式規則來模擬
        pass
    
    def analyze_text(self, text, method='ensemble'):
        """
        分析文本並返回 AI/Human 機率
        
        Parameters:
        - text: 輸入文本
        - method: 分析方法 ('feature', 'statistical', 'ensemble')
        
        Returns:
        - dict: 包含分析結果
        """
        if not text or len(text.strip()) < 50:
            return {
                'ai_probability': 0.5,
                'human_probability': 0.5,
                'confidence': 'low',
                'features': {},
                'warning': '文本太短，建議輸入至少 50 個字符以獲得更準確的結果'
            }
        
        # 提取特徵
        features = self.feature_extractor.extract_features(text)
        
        # 根據不同方法計算 AI 機率
        if method == 'feature':
            ai_prob = self._feature_based_detection(features)
        elif method == 'statistical':
            ai_prob = self._statistical_detection(text, features)
        else:  # ensemble
            ai_prob = self._ensemble_detection(text, features)
        
        # 確定置信度
        if abs(ai_prob - 0.5) > 0.3:
            confidence = 'high'
        elif abs(ai_prob - 0.5) > 0.15:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'ai_probability': ai_prob,
            'human_probability': 1 - ai_prob,
            'confidence': confidence,
            'features': features,
            'method': method
        }
    
    def _feature_based_detection(self, features):
        """基於特徵的偵測"""
        ai_score = 0.5
        is_chinese = features.get('is_chinese', False)
        
        # === 人類特徵（降低 AI 分數）===
        
        # 1. 口語化表達（強烈的人類特徵）
        colloquial_ratio = features.get('colloquial_ratio', 0)
        if colloquial_ratio > 0.5:
            ai_score -= 0.25
        elif colloquial_ratio > 0.2:
            ai_score -= 0.15
        elif colloquial_ratio > 0:
            ai_score -= 0.1
        
        # 2. 第一人稱使用（人類特徵）
        first_person_ratio = features.get('first_person_ratio', 0)
        if first_person_ratio > 0.05:
            ai_score -= 0.15
        elif first_person_ratio > 0.02:
            ai_score -= 0.08
        
        # 3. 句子長度變異性高（人類寫作更不規則）
        sentence_std = features.get('sentence_length_std', 0)
        if is_chinese:
            if sentence_std > 8:
                ai_score -= 0.1
            elif sentence_std < 2:
                ai_score += 0.1
        else:
            if sentence_std > 15:
                ai_score -= 0.1
            elif sentence_std < 5:
                ai_score += 0.1
        
        # 4. 突發性高（人類特徵）
        burstiness = features.get('burstiness', 0)
        if burstiness > 10:
            ai_score -= 0.1
        elif burstiness < 2:
            ai_score += 0.08
        
        # === AI 特徵（增加 AI 分數）===
        
        # 5. 過渡詞使用率高（AI 特徵）
        transition_ratio = features.get('transition_ratio', 0)
        if is_chinese:
            if transition_ratio > 0.3:
                ai_score += 0.2
            elif transition_ratio > 0.1:
                ai_score += 0.1
        else:
            if transition_ratio > 0.05:
                ai_score += 0.15
            elif transition_ratio > 0.02:
                ai_score += 0.08
        
        # 6. 詞彙多樣性（AI 傾向較高）
        vocab_richness = features.get('vocabulary_richness', 0)
        if not is_chinese:  # 英文適用
            if vocab_richness > 0.8:
                ai_score += 0.1
            elif vocab_richness < 0.5:
                ai_score -= 0.05
        
        # 7. 被動語態（AI 傾向使用，僅英文）
        if not is_chinese:
            passive_ratio = features.get('passive_ratio', 0)
            if passive_ratio > 0.1:
                ai_score += 0.1
        
        # 8. 標點符號使用（中文口語常用更多標點）
        punctuation_ratio = features.get('punctuation_ratio', 0)
        if is_chinese:
            if punctuation_ratio > 0.08:
                ai_score -= 0.05  # 較多標點可能是口語化
        
        return np.clip(ai_score, 0.05, 0.95)
    
    def _statistical_detection(self, text, features):
        """基於統計的偵測"""
        ai_score = 0.5
        is_chinese = features.get('is_chinese', False)
        
        # 使用正確的分詞方法
        words = tokenize_text(text)
        words_lower = [w.lower() for w in words]
        
        # 計算詞頻分布的 Zipf 定律偏離度
        word_freq = Counter(words_lower)
        freqs = sorted(word_freq.values(), reverse=True)
        
        if len(freqs) > 10:
            # 理想 Zipf: freq(rank) ∝ 1/rank
            ranks = np.arange(1, min(len(freqs), 50) + 1)
            actual_freqs = np.array(freqs[:50]) if len(freqs) >= 50 else np.array(freqs)
            actual_freqs = actual_freqs[:len(ranks)]
            
            # 計算與理想分布的偏離
            ideal_freqs = actual_freqs[0] / ranks[:len(actual_freqs)]
            deviation = np.mean(np.abs(actual_freqs - ideal_freqs) / (ideal_freqs + 1))
            
            # AI 文本通常偏離較小（更符合理想分布）
            if deviation < 0.3:
                ai_score += 0.1
            elif deviation > 0.6:
                ai_score -= 0.08
        
        # 考慮口語化和第一人稱（統計層面的人類特徵）
        colloquial_ratio = features.get('colloquial_ratio', 0)
        first_person_ratio = features.get('first_person_ratio', 0)
        
        if colloquial_ratio > 0.1 or first_person_ratio > 0.03:
            ai_score -= 0.15
        
        # N-gram 重複模式（僅對非中文有效，因為中文字符級別重複很常見）
        if not is_chinese and len(words) > 5:
            bigrams = [' '.join(words_lower[i:i+2]) for i in range(len(words_lower)-1)]
            trigrams = [' '.join(words_lower[i:i+3]) for i in range(len(words_lower)-2)]
            
            bigram_repetition = len(bigrams) - len(set(bigrams))
            trigram_repetition = len(trigrams) - len(set(trigrams))
            
            repetition_score = (bigram_repetition + trigram_repetition * 2) / len(words) if words else 0
            
            # AI 傾向有更多的短語重複
            if repetition_score > 0.1:
                ai_score += 0.08
        
        return np.clip(ai_score, 0.05, 0.95)
    
    def _ensemble_detection(self, text, features):
        """整合多種方法的偵測"""
        feature_score = self._feature_based_detection(features)
        statistical_score = self._statistical_detection(text, features)
        
        # 加權平均
        ensemble_score = 0.6 * feature_score + 0.4 * statistical_score
        
        return np.clip(ensemble_score, 0.05, 0.95)


def create_gauge_chart(ai_prob, human_prob):
    """建立儀表板圖表"""
    fig = go.Figure()
    
    # AI 機率儀表
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=ai_prob * 100,
        title={'text': "AI 生成機率", 'font': {'size': 20}},
        domain={'x': [0, 0.45], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#ff6b6b"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#51cf66'},
                {'range': [30, 70], 'color': '#fcc419'},
                {'range': [70, 100], 'color': '#ff6b6b'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': ai_prob * 100
            }
        }
    ))
    
    # Human 機率儀表
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=human_prob * 100,
        title={'text': "人類撰寫機率", 'font': {'size': 20}},
        domain={'x': [0.55, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#51cf66"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ff6b6b'},
                {'range': [30, 70], 'color': '#fcc419'},
                {'range': [70, 100], 'color': '#51cf66'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': human_prob * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial"}
    )
    
    return fig


def create_feature_radar_chart(features):
    """建立特徵雷達圖"""
    # 選擇關鍵特徵並正規化
    feature_names = [
        '詞彙豐富度', '句子變異性', '過渡詞使用', 
        '突發性', '標點符號', '重複度'
    ]
    
    values = [
        min(features.get('vocabulary_richness', 0) * 100, 100),
        min(features.get('sentence_length_std', 0) * 5, 100),
        min(features.get('transition_ratio', 0) * 1000, 100),
        min(features.get('burstiness', 0) * 10, 100),
        min(features.get('punctuation_ratio', 0) * 1000, 100),
        min(features.get('repetition_score', 0) * 500, 100)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # 閉合圖形
        theta=feature_names + [feature_names[0]],
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.3)',
        line=dict(color='#1E88E5', width=2),
        name='文本特徵'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title=dict(text='文本特徵分析', x=0.5),
        height=400,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig


def create_statistics_table(features):
    """建立統計資訊表格"""
    stats_data = {
        '指標': [
            '總字數', '總字符數', '句子數量', '段落數量',
            '平均詞長', '平均句長', '詞彙豐富度', '標點符號比例'
        ],
        '數值': [
            f"{features.get('word_count', 0):,}",
            f"{features.get('char_count', 0):,}",
            f"{features.get('sentence_count', 0):,}",
            f"{features.get('paragraph_count', 0):,}",
            f"{features.get('avg_word_length', 0):.2f}",
            f"{features.get('avg_sentence_length', 0):.2f}",
            f"{features.get('vocabulary_richness', 0):.2%}",
            f"{features.get('punctuation_ratio', 0):.2%}"
        ]
    }
    return pd.DataFrame(stats_data)


def main():
    """主程式"""
    
    # 標題
    st.markdown('<h1 class="main-header">🤖 AI vs Human 文章分類器</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">使用機器學習技術偵測文章是否由 AI 生成</p>', unsafe_allow_html=True)
    
    # 初始化偵測器
    detector = AIDetector()
    
    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 選擇分析方法
        method = st.selectbox(
            "選擇分析方法",
            options=['ensemble', 'feature', 'statistical'],
            format_func=lambda x: {
                'ensemble': '🔄 整合分析 (推薦)',
                'feature': '📊 特徵分析法',
                'statistical': '📈 統計分析法'
            }.get(x, x),
            help="ensemble: 結合多種方法 | feature: 基於文本特徵 | statistical: 基於統計分布"
        )
        
        st.markdown("---")
        
        st.header("📖 使用說明")
        st.markdown("""
        1. 在文本框中貼上要分析的文章
        2. 點擊「開始分析」按鈕
        3. 查看 AI/Human 機率結果
        4. 檢視詳細特徵分析
        
        **提示：** 建議輸入至少 100 字以獲得更準確的結果
        """)
        
        st.markdown("---")
        
        st.header("ℹ️ 關於")
        st.markdown("""
        此工具使用多種自然語言處理技術來判斷文本是否由 AI 生成：
        
        - **詞彙分析**: 檢測詞彙多樣性與使用模式
        - **句法分析**: 分析句子結構與長度變化
        - **統計分析**: 評估文本的統計特徵
        - **模式識別**: 識別 AI 生成文本的典型模式
        """)
    
    # 主要內容區
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 輸入文本")
        
        # 範例文本
        sample_texts = {
            "請選擇...": "",
            "AI 生成範例": """Artificial intelligence has revolutionized the way we interact with technology. Furthermore, it has transformed various industries, including healthcare, finance, and education. The implementation of machine learning algorithms has enabled systems to learn from data and make intelligent decisions. Moreover, natural language processing has made it possible for computers to understand and generate human language with remarkable accuracy. Consequently, businesses are increasingly adopting AI solutions to improve efficiency and productivity. Additionally, the continuous advancement in deep learning techniques has opened new possibilities for innovation and discovery.""",
            "人類撰寫範例": """昨天我去了一家新開的咖啡店，說實話，有點失望。店面裝潢還不錯啦，很有文青風，但咖啡味道普普。我點了一杯拿鐵，結果等了快二十分鐘才送來，而且溫度不夠熱。不過他們的甜點倒是挺好吃的，那個提拉米蘇入口即化。下次可能會再去試試其他品項，但純喝咖啡的話，我還是會選擇老店。"""
        }
        
        selected_sample = st.selectbox("選擇範例文本", list(sample_texts.keys()))
        
        text_input = st.text_area(
            "貼上您要分析的文章",
            value=sample_texts[selected_sample],
            height=300,
            placeholder="請在此輸入或貼上要分析的文本...\n\n建議至少 100 字以獲得更準確的結果。"
        )
        
        analyze_button = st.button("🔍 開始分析", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 快速統計")
        if text_input:
            word_count = len(text_input.split())
            char_count = len(text_input)
            sentence_count = len(re.split(r'[.!?。！？]+', text_input))
            
            st.metric("字數", f"{word_count:,}")
            st.metric("字符數", f"{char_count:,}")
            st.metric("句子數", f"{sentence_count:,}")
            
            # 文本長度警告
            if char_count < 100:
                st.warning("⚠️ 文本較短，結果可能不夠準確")
            elif char_count > 5000:
                st.info("ℹ️ 較長的文本通常能獲得更準確的結果")
        else:
            st.info("請輸入文本以查看統計資訊")
    
    # 分析結果
    if analyze_button and text_input:
        with st.spinner("正在分析中..."):
            result = detector.analyze_text(text_input, method=method)
        
        st.markdown("---")
        st.subheader("📈 分析結果")
        
        # 顯示警告（如果有）
        if 'warning' in result:
            st.warning(result['warning'])
        
        # 儀表板圖表
        gauge_fig = create_gauge_chart(result['ai_probability'], result['human_probability'])
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # 結果摘要
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            ai_class = "ai-result" if result['ai_probability'] > 0.5 else ""
            st.markdown(f"""
            <div class="metric-card">
                <h3>🤖 AI 機率</h3>
                <h1 style="color: #ff6b6b;">{result['ai_probability']:.1%}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_result2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👤 人類機率</h3>
                <h1 style="color: #51cf66;">{result['human_probability']:.1%}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_result3:
            confidence_color = {'high': '#51cf66', 'medium': '#fcc419', 'low': '#ff6b6b'}
            confidence_text = {'high': '高', 'medium': '中', 'low': '低'}
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 置信度</h3>
                <h1 style="color: {confidence_color[result['confidence']]};">{confidence_text[result['confidence']]}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # 判斷結果
        st.markdown("---")
        if result['ai_probability'] > 0.7:
            st.error("🤖 **判斷結果**: 此文本很可能由 AI 生成")
        elif result['ai_probability'] > 0.5:
            st.warning("⚠️ **判斷結果**: 此文本可能包含 AI 生成的內容")
        elif result['ai_probability'] > 0.3:
            st.info("ℹ️ **判斷結果**: 此文本可能大部分由人類撰寫")
        else:
            st.success("👤 **判斷結果**: 此文本很可能由人類撰寫")
        
        # 詳細分析
        st.markdown("---")
        st.subheader("🔬 詳細分析")
        
        tab1, tab2, tab3 = st.tabs(["📊 特徵雷達圖", "📋 統計資訊", "🔍 特徵詳情"])
        
        with tab1:
            radar_fig = create_feature_radar_chart(result['features'])
            st.plotly_chart(radar_fig, use_container_width=True)
            
            st.markdown("""
            **圖表說明：**
            - **詞彙豐富度**: 使用獨特詞彙的比例
            - **句子變異性**: 句子長度的變化程度
            - **過渡詞使用**: 使用連接詞和過渡詞的頻率
            - **突發性**: 詞彙出現的不規律程度
            - **標點符號**: 標點符號使用比例
            - **重複度**: 詞彙重複出現的程度
            """)
        
        with tab2:
            stats_df = create_statistics_table(result['features'])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # 分布圖
            if result['features'].get('word_count', 0) > 0:
                words = text_input.split()
                word_lengths = [len(w) for w in words]
                
                fig_dist = px.histogram(
                    x=word_lengths, 
                    nbins=20,
                    title="詞長分布",
                    labels={'x': '詞長', 'count': '頻率'}
                )
                fig_dist.update_layout(
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab3:
            st.markdown("#### 所有提取的特徵值")
            
            feature_df = pd.DataFrame([
                {'特徵名稱': k, '數值': f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in result['features'].items()
            ])
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
            
            st.markdown("""
            **AI 文本的典型特徵：**
            - ✓ 較高的詞彙多樣性
            - ✓ 較一致的句子長度（低變異性）
            - ✓ 較多使用過渡詞和連接詞
            - ✓ 較低的突發性（詞彙分布均勻）
            - ✓ 結構化的段落安排
            """)
    
    # 頁尾
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>💡 此工具僅供參考，結果不能作為絕對判斷依據</p>
        <p>Made with ❤️ using Streamlit | AIOT HW5</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
