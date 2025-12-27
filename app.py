import streamlit as st
import cv2
import mediapipe as mp
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from PIL import Image
import math
import time

# --- 1. إعدادات الهوية (Favicon) لتجنب خطأ الشعار ---
LOGO_URL = "https://i.postimg.cc/R0cQyjrR/logo-png.png" 

st.set_page_config(
    page_title="بصيرة | Smart Sign Translator", 
    page_icon=LOGO_URL, 
    layout="wide"
)

# --- 2. تهيئة الذاكرة فوراً (حل خطأ AttributeError) ---
# يجب أن يكون هذا الجزء في بداية الكود تماماً لضمان وجود المتغيرات
keys = {
    'auth': {'in': False, 'user': None, 'role': None},
    'stab_count': 0,
    'last_s': "",
    'final_s': "",
    'last_time': time.time(),
    'temp_code': None
}
for key, val in keys.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 3. محركات النظام مع معالجة الأخطاء ---
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

# --- 4. الواجهة الاحترافية (CSS & JS) ---
def apply_ui():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    * {{ font-family: 'Cairo', sans-serif; text-align: right; }}
    .stApp {{ animation: fadeIn 1.5s; }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    .stButton>button {{ border-radius: 12px; background: linear-gradient(45deg, #1e3a8a, #3b82f6); color: white; transition: 0.3s; width: 100%; }}
    </style>
    """, unsafe_allow_html=True)

    st.components.v1.html("""
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

# --- 5. المحرك الرياضي ---
def get_finger_math(hl):
    lm = hl.landmark
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

# --- 6. تشغيل التطبيق ---
apply_ui()

if not st.session_state.auth['in']:
    st.title("🔒 تسجيل الدخول - بصيرة")
    u = st.text_input("اسم المستخدم").strip()
    p = st.text_input("كلمة السر", type="password").strip()
    if st.button("دخول المنصة"):
        users = pd.DataFrame(auth_sheet.get_all_records())
        found = users[(users['Username'].astype(str)==u) & (users['Password'].astype(str)==p)]
        if not found.empty:
            st.session_state.auth = {'in': True, 'user': u, 'role': found.iloc[0]['Role'].strip()}
            st.rerun()
        else: st.error("بيانات الدخول غير صحيحة")
else:
    # القائمة الجانبية (Sidebar) - نستخدم الرابط لتجنب خطأ الملف المفقود
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.success(f"👤 مرحباً: {st.session_state.auth['user']}")
    
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.auth['in'] = False; st.rerun()

    # واجهة المستخدم (المترجم الفوري)
    st.header("📸 المترجم الذكي (بصيرة)")
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
            st.components.v1.html(f"<script>window.parent.updateZoom({hl.landmark[9].x}, {hl.landmark[9].y}, true);</script>", height=0)
            
            current_sign = match_sign(get_finger_math(hl), signs_df)
            
            if current_sign == st.session_state.last_s:
                st.session_state.stab_count += 1
            else:
                st.session_state.stab_count = 0
                st.session_state.last_s = current_sign

            if st.session_state.stab_count >= 10:
                st.session_state.final_s = current_sign
                st.subheader(f"✨ الترجمة المؤكدة: {current_sign}")
        else:
            st.components.v1.html("<script>window.parent.updateZoom(0,0,false);</script>", height=0)
            if time.time() - st.session_state.last_time > 2.0:
                st.session_state.final_s = ""; st.session_state.stab_count = 0; st.session_state.last_s = ""

        win.image(rgb)
    cap.release()