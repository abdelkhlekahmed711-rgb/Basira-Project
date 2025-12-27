import streamlit as st
import cv2
import mediapipe as mp
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from PIL import Image
from gtts import gTTS
import base64
import math
import time

# --- 1. إعدادات الهوية واللوجو في التبويب (Favicon) ---
# ملاحظة: استخدم رابط مباشر للصورة لضمان ظهورها دائماً
LOGO_URL = "https://i.postimg.cc/R0cQyjrR/logo-png.png" 

st.set_page_config(
    page_title="بصيرة | Smart Sign Translator", 
    page_icon=LOGO_URL, # إظهار اللوجو في تبويب المتصفح
    layout="wide"
)

# --- 2. محركات النظام (السحابة والذكاء الاصطناعي) ---
@st.cache_resource
def init_system():
    # الربط مع Google Sheets
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    db = client.open("Basira_DB")
    
    # محرك MediaPipe
    mp_hands = mp.solutions.hands
    engine = mp_hands.Hands(max_num_hands=1, model_complexity=1, min_detection_confidence=0.7)
    return db.worksheet("Signs_DB"), db.worksheet("Users_Admin"), engine, mp.solutions.drawing_utils

signs_sheet, auth_sheet, hands_engine, mp_draw = init_system()

# --- 3. الواجهة الاحترافية (CSS & JS) ---
def apply_ui():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    * {{ font-family: 'Cairo', sans-serif; text-align: right; }}
    .stApp {{ animation: fadeIn 1s; }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    @media (prefers-color-scheme: dark) {{ .stApp {{ background-color: #0E1117; color: white; }} }}
    .stButton>button {{ border-radius: 12px; background: linear-gradient(45deg, #1e3a8a, #3b82f6); color: white; transition: 0.3s; }}
    </style>
    """, unsafe_allow_html=True)

    # جافا سكريبت للتتبع التلقائي (Auto-Zoom)
    st.components.v1.html("""
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
    window.parent.updateZoom = function(x, y, active) {
        const video = window.parent.document.querySelector('video');
        if (!video) return;
        video.style.transition = "transform 0.6s ease-out";
        if (active) {
            video.style.transformOrigin = `${x*100}% ${y*100}%`;
            video.style.transform = "scale(1.8)";
        } else { video.style.transform = "scale(1)"; }
    }
    </script>
    """, height=0)

# --- 4. المحرك الرياضي ومنطق المطابقة المرنة ---
def get_finger_math(hl):
    lm = hl.landmark
    # حساب المسافة الإقليدية 3D
    palm = math.sqrt((lm[0].x-lm[9].x)**2 + (lm[0].y-lm[9].y)**2 + (lm[0].z-lm[9].z)**2)
    tips = [4, 8, 12, 16, 20]
    return np.array([math.sqrt((lm[t].x-lm[0].x)**2 + (lm[t].y-lm[0].y)**2 + (lm[t].z-lm[0].z)**2)/palm for t in tips])

def match_sign(live_vector, db_df, threshold=0.3):
    best_name, min_err = None, float('inf')
    for _, row in db_df.iterrows():
        try:
            db_vec = np.array([float(x) for x in str(row['Finger_Code']).split(',')])
            err = np.mean(np.abs(live_vector - db_vec))
            if err < min_err and err < threshold:
                min_err, best_name = err, row['Sign_Name']
        except: continue
    return best_name

# --- 5. تهيئة الذاكرة المؤقتة (إصلاح خطأ AttributeError) ---
# تم إضافة 'auth' هنا لضمان عدم حدوث الخطأ
for key in ['auth', 'stab_count', 'last_s', 'final_s', 'last_time']:
    if key not in st.session_state:
        if key == 'auth': st.session_state[key] = {'in': False, 'user': None, 'role': None}
        elif key == 'stab_count': st.session_state[key] = 0
        elif key == 'last_time': st.session_state[key] = time.time()
        else: st.session_state[key] = ""

# --- 6. تشغيل التطبيق ---
apply_ui()

if not st.session_state.auth['in']:
    st.title("🔒 تسجيل الدخول - بصيرة")
    u = st.text_input("اسم المستخدم")
    p = st.text_input("كلمة السر", type="password")
    if st.button("دخول"):
        df = pd.DataFrame(auth_sheet.get_all_records())
        found = df[(df['Username'].astype(str)==u) & (df['Password'].astype(str)==p)]
        if not found.empty:
            st.session_state.auth = {'in': True, 'user': u, 'role': found.iloc[0]['Role']}
            st.rerun()
        else: st.error("بيانات غير صحيحة")
else:
    # القائمة الجانبية (Sidebar)
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.success(f"👤 مرحباً: {st.session_state.auth['user']}")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.auth['in'] = False; st.rerun()

    # واجهة المستخدم (المترجم الفوري)
    st.header("📸 المترجم الذكي المطور (بصيرة)")
    signs_df = pd.DataFrame(signs_sheet.get_all_records())
    run = st.toggle("تفعيل الكاميرا والترجمة")
    win = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        res = hands_engine.process(rgb)

        if res.multi_hand_landmarks:
            st.session_state.last_time = time.time()
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(rgb, hl, mp.solutions.hands.HAND_CONNECTIONS)
            
            # 1. التتبع التلقائي (Zoom) بالجافا سكريبت
            st.components.v1.html(f"<script>window.parent.updateZoom({hl.landmark[9].x}, {hl.landmark[9].y}, true);</script>", height=0)
            
            # 2. المطابقة المرنة (Flexible Matching)
            live_vec = get_finger_math(hl)
            current_sign = match_sign(live_vec, signs_df)
            
            # 3. نظام الاستقرار (Stability Counter - 10 frames)
            if current_sign and current_sign == st.session_state.last_s:
                st.session_state.stab_count += 1
            else:
                st.session_state.stab_count = 0
                st.session_state.last_s = current_sign

            if st.session_state.stab_count >= 10:
                st.session_state.final_s = current_sign
                st.title(f"✨ الترجمة المؤكدة: {current_sign}")
        else:
            # إعادة الزوم للوضع الطبيعي
            st.components.v1.html("<script>window.parent.updateZoom(0,0,false);</script>", height=0)
            
            # 4. نظام المسح التلقائي (Auto-Clear بعد 2 ثانية)
            if time.time() - st.session_state.last_time > 2.0:
                st.session_state.final_s = ""
                st.session_state.stab_count = 0
                st.session_state.last_s = ""

        win.image(rgb)
    cap.release()