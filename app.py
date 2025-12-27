import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import math
import time
from PIL import Image

# --- 1. الهوية والتبويب ---
LOGO_URL = "https://i.postimg.cc/R0cQyjrR/logo-png.png" 

st.set_page_config(
    page_title="بصيرة | Smart Sign Translator", 
    page_icon=LOGO_URL, 
    layout="wide"
)

# --- 2. تهيئة الذاكرة (إصلاح AttributeError) ---
if 'auth' not in st.session_state:
    st.session_state.auth = {'in': False, 'user': None, 'role': None}
if 'live_code' not in st.session_state:
    st.session_state.live_code = None
if 'last_time' not in st.session_state:
    st.session_state.last_time = time.time()
if 'stab_count' not in st.session_state:
    st.session_state.stab_count = 0
if 'last_s' not in st.session_state:
    st.session_state.last_s = ""

# --- 3. الاتصال بالسحابة ---
@st.cache_resource
def init_system():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
        client = gspread.authorize(creds)
        db = client.open("Basira_DB")
        
        mp_hands = mp.solutions.hands
        engine = mp_hands.Hands(max_num_hands=1, model_complexity=1, min_detection_confidence=0.7)
        return db.worksheet("Signs_DB"), db.worksheet("Users_Admin"), engine, mp.solutions.drawing_utils
    except Exception as e:
        st.error(f"⚠️ خطأ اتصال: {e}")
        st.stop()

signs_sheet, auth_sheet, hands_engine, mp_draw = init_system()

# --- 4. المحرك الرياضي (3D Euclidean Distance) ---
def calculate_math(hl):
    lm = hl.landmark
    # $d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + (z_2-z_1)^2}$
    palm = math.sqrt((lm[0].x-lm[9].x)**2 + (lm[0].y-lm[9].y)**2 + (lm[0].z-lm[9].z)**2)
    tips = [4, 8, 12, 16, 20]
    return ",".join([str(round(math.sqrt((lm[t].x-lm[0].x)**2 + (lm[t].y-lm[0].y)**2 + (lm[t].z-lm[0].z)**2)/palm, 1)) for t in tips])

# --- 5. معالج الفيديو (WebRTC) ---
class SignProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands_engine.process(rgb)
        
        if res.multi_hand_landmarks:
            st.session_state.last_time = time.time()
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hl, mp.solutions.hands.HAND_CONNECTIONS)
            st.session_state.live_code = calculate_math(hl)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. الواجهة وتوزيع الصلاحيات ---
st.markdown("<style> * { font-family: 'Cairo', sans-serif; text-align: right; } </style>", unsafe_allow_html=True)

if not st.session_state.auth['in']:
    st.title("🔒 دخول منصة بصيرة")
    u, p = st.text_input("المستخدم"), st.text_input("السر", type="password")
    if st.button("دخول"):
        df = pd.DataFrame(auth_sheet.get_all_records())
        found = df[(df['Username'].astype(str)==u) & (df['Password'].astype(str)==p)]
        if not found.empty:
            st.session_state.auth = {'in': True, 'user': u, 'role': found.iloc[0]['Role'].strip()}
            st.rerun()
else:
    st.sidebar.image(LOGO_URL, use_container_width=True)
    role = st.session_state.auth['role']
    
    if role == "User":
        st.header("📸 المترجم السحابي")
        # حل مشكلة الكاميرا: إضافة ICE Servers
        webrtc_streamer(
            key="basira-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=SignProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # منطق الترجمة (كما سبق)
        if time.time() - st.session_state.last_time < 2.0:
            if st.session_state.live_code:
                # (البحث عن المطابقة هنا...)
                st.write(f"بصمة اليد الحالية: {st.session_state.live_code}")

    elif role == "Admin":
        st.header("⚙️ لوحة تحكم المدير (كاملة)")
        tab1, tab2 = st.tabs(["➕ إضافة إشارة جديدة", "📋 قاعدة البيانات"])
        
        with tab1:
            name = st.text_input("اسم الإشارة")
            file = st.file_uploader("ارفع صورة للتحليل", type=['jpg','png','jpeg'])
            if file:
                img = Image.open(file)
                st.image(img, width=250)
                if st.button("تحليل وحفظ"):
                    with mp.solutions.hands.Hands(static_image_mode=True) as static_hands:
                        res = static_hands.process(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
                        if res.multi_hand_landmarks:
                            code = calculate_math(res.multi_hand_landmarks[0])
                            signs_sheet.append_row([name, code])
                            st.success(f"تم حفظ {name} ببصمة {code}")
                        else: st.error("لم يتم رصد يد في الصورة")
        
        with tab2:
            st.dataframe(pd.DataFrame(signs_sheet.get_all_records()), use_container_width=True)

    if st.sidebar.button("خروج"):
        st.session_state.auth['in'] = False; st.rerun()