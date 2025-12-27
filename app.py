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

# --- 1. إعدادات الهوية والتبويب العلوي ---
LOGO_URL = "https://i.postimg.cc/R0cQyjrR/logo-png.png" 

st.set_page_config(
    page_title="بصيرة | Smart Sign Translator", 
    page_icon=LOGO_URL, 
    layout="wide"
)

# --- 2. تهيئة الذاكرة المؤقتة (Session State) ---
keys = {
    'auth': {'in': False, 'user': None, 'role': None},
    'stab_count': 0, 'last_s': "", 'final_s': "", 'last_time': time.time(),
    'live_code': None
}
for key, val in keys.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 3. محركات النظام (السحابة والذكاء الاصطناعي) ---
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
        st.error(f"⚠️ خطأ في التهيئة: {e}")
        st.stop()

signs_sheet, auth_sheet, hands_engine, mp_draw = init_system()

# --- 4. واجهة المستخدم الاحترافية (CSS & JS) ---
def apply_ui():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    * {{ font-family: 'Cairo', sans-serif; text-align: right; }}
    .stApp {{ animation: fadeIn 1.5s; }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    .stButton>button {{ border-radius: 12px; background: linear-gradient(45deg, #1e3a8a, #3b82f6); color: white; transition: 0.3s; width: 100%; }}
    [data-testid="stSidebar"] {{ background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. معالج الفيديو السحابي (WebRTC Processor) ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands_engine.process(rgb)
        
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hl, mp.solutions.hands.HAND_CONNECTIONS)
            
            # حساب البصمة الرياضية
            lm = hl.landmark
            palm = math.sqrt((lm[0].x-lm[9].x)**2 + (lm[0].y-lm[9].y)**2 + (lm[0].z-lm[9].z)**2)
            tips = [4, 8, 12, 16, 20]
            code = ",".join([str(round(math.sqrt((lm[t].x-lm[0].x)**2 + (lm[t].y-lm[0].y)**2 + (lm[t].z-lm[0].z)**2)/palm, 1)) for t in tips])
            st.session_state["live_code"] = code
            st.session_state["last_time"] = time.time()
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. منطق التطبيق الرئيسي ---
apply_ui()

if not st.session_state.auth['in']:
    st.title("🔒 دخول منصة بصيرة")
    u = st.text_input("اسم المستخدم")
    p = st.text_input("كلمة السر", type="password")
    if st.button("دخول"):
        df = pd.DataFrame(auth_sheet.get_all_records())
        found = df[(df['Username'].astype(str)==u) & (df['Password'].astype(str)==p)]
        if not found.empty:
            st.session_state.auth = {'in': True, 'user': u, 'role': found.iloc[0]['Role'].strip()}
            st.rerun()
else:
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.success(f"👤 مرحباً: {st.session_state.auth['user']}")
    
    signs_df = pd.DataFrame(signs_sheet.get_all_records())
    
    if st.session_state.auth['role'] == "User":
        st.header("📸 المترجم السحابي المستقر")
        
        webrtc_ctx = webrtc_streamer(
            key="basira-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # منطق المعالجة والاستقرار في الواجهة
        if webrtc_ctx.video_processor:
            if time.time() - st.session_state.last_time < 2.0:
                live_code = st.session_state.get("live_code")
                if live_code:
                    # البحث عن أقرب إشارة (Threshold 0.3)
                    best_match = None
                    min_err = 100
                    live_vals = np.array([float(x) for x in live_code.split(',')])
                    
                    for _, row in signs_df.iterrows():
                        db_vals = np.array([float(x) for x in str(row['Finger_Code']).split(',')])
                        err = np.mean(np.abs(live_vals - db_vals))
                        if err < min_err and err < 0.3:
                            min_err, best_match = err, row['Sign_Name']
                    
                    # فلتر الاستقرار (10 إطارات)
                    if best_match == st.session_state.last_s:
                        st.session_state.stab_count += 1
                    else:
                        st.session_state.stab_count = 0
                        st.session_state.last_s = best_match
                    
                    if st.session_state.stab_count >= 10:
                        st.session_state.final_s = best_match
                        st.title(f"✨ الترجمة: {best_match}")
            else:
                st.session_state.final_s = "" # المسح التلقائي
    
    elif st.session_state.auth['role'] == "Admin":
        st.header("⚙️ لوحة الإدارة")
        # (كود إضافة الإشارات كما في النسخ السابقة)
        st.info("يمكنك إدارة قاعدة البيانات من هنا أو عبر Google Sheets مباشرة.")