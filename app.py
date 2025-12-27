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
LOGO_URL = "https://i.postimg.cc/R0cQyjrR/logo-png.png" 

st.set_page_config(
    page_title="بصيرة | Smart Sign Translator", 
    page_icon=LOGO_URL, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. محركات النظام (السحابة والذكاء الاصطناعي) ---
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
        st.error(f"خطأ في الاتصال بالسحابة: {e}")
        st.stop()

signs_sheet, auth_sheet, hands_engine, mp_draw = init_system()

# --- 3. الواجهة الاحترافية (CSS & JS المطور) ---
def apply_ui():
    # تصميم Glassmorphism وخط Cairo ودعم الوضع الليلي
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    * {{ font-family: 'Cairo', sans-serif; text-align: right; }}
    .stApp {{ animation: fadeIn 1.5s; }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    
    /* تنسيق القائمة الجانبية الزجاجي */
    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }}
    
    .stButton>button {{ 
        border-radius: 12px; 
        background: linear-gradient(45deg, #1e3a8a, #3b82f6); 
        color: white; 
        transition: 0.3s; 
        width: 100%;
    }}
    .stButton>button:hover {{ transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
    </style>
    """, unsafe_allow_html=True)

    # جافا سكريبت متطور للتنبيهات والتتبع
    st.components.v1.html("""
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
    window.parent.updateZoom = function(x, y, active) {
        const video = window.parent.document.querySelector('video');
        if (!video) return;
        video.style.transition = "transform 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)";
        if (active) {
            video.style.transformOrigin = `${x*100}% ${y*100}%`;
            video.style.transform = "scale(1.8)";
        } else { video.style.transform = "scale(1)"; }
    }
    window.parent.successToast = function(msg) {
        Swal.fire({ icon: 'success', title: msg, toast: true, position: 'top-end', timer: 3000, showConfirmButton: false });
    }
    </script>
    """, height=0)

# --- 4. المحرك الرياضي ومنطق المطابقة ---
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

# --- 5. تهيئة الذاكرة المؤقتة (إصلاح AttributeError) ---
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
    # القائمة الجانبية (Sidebar) مع الإحصائيات
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.markdown(f"### 👤 مرحباً: {st.session_state.auth['user']}")
    
    # حساب إحصائيات قاعدة البيانات للمدير
    all_signs = pd.DataFrame(signs_sheet.get_all_records())
    total_count = len(all_signs)
    st.sidebar.metric("الإشارات المسجلة", f"{total_count} / 20")
    st.sidebar.progress(min(total_count/20, 1.0))
    
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.auth['in'] = False; st.rerun()

    role = st.session_state.auth['role']

    if role == "User":
        st.header("📸 المترجم الذكي المطور (بصيرة)")
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
                
                # 1. التتبع التلقائي (Zoom)
                st.components.v1.html(f"<script>window.parent.updateZoom({hl.landmark[9].x}, {hl.landmark[9].y}, true);</script>", height=0)
                
                # 2. المطابقة والاستقرار (10 إطارات)
                live_vec = get_finger_math(hl)
                current_sign = match_sign(live_vec, all_signs)
                
                if current_sign and current_sign == st.session_state.last_s:
                    st.session_state.stab_count += 1
                else:
                    st.session_state.stab_count = 0
                    st.session_state.last_s = current_sign

                if st.session_state.stab_count >= 10:
                    if current_sign != st.session_state.final_s:
                        st.session_state.final_s = current_sign
                        st.subheader(f"✨ الترجمة المؤكدة: {current_sign}")
                        # نطق الكلمة مرة واحدة فقط عند الاستقرار
            else:
                # إعادة الزوم والمسح التلقائي بعد 2 ثانية
                st.components.v1.html("<script>window.parent.updateZoom(0,0,false);</script>", height=0)
                if time.time() - st.session_state.last_time > 2.0:
                    st.session_state.final_s = ""; st.session_state.stab_count = 0; st.session_state.last_s = ""

            win.image(rgb)
        cap.release()

    elif role == "Admin":
        st.header("⚙️ لوحة تحكم الإدارة")
        tab1, tab2 = st.tabs(["➕ إضافة إشارة", "📋 سجل البيانات"])
        
        with tab1:
            name = st.text_input("اسم الإشارة الجديدة")
            # دمج مؤشر التحميل الذكي عند المعالجة
            up = st.file_uploader("ارفع صورة الإشارة للتحليل", type=["jpg", "png", "jpeg"])
            if up:
                img = Image.open(up)
                st.image(img, width=250)
                with st.spinner('جاري تحليل البصمة الرياضية...'):
                    with mp.solutions.hands.Hands(static_image_mode=True) as det:
                        r = det.process(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
                        if r.multi_hand_landmarks:
                            code = ",".join([str(round(x,1)) for x in get_finger_math(r.multi_hand_landmarks[0])])
                            st.session_state.temp_code = code
                            st.success(f"✅ تم استخراج البصمة: {code}")
                        else: st.error("لم يتم رصد يد واضحة")

            if st.button("💾 حفظ في السحابة"):
                if name and st.session_state.temp_code:
                    signs_sheet.append_row([name, st.session_state.temp_code])
                    st.components.v1.html(f"<script>window.parent.successToast('تم حفظ {name} بنجاح! 🚀');</script>", height=0)
                    st.session_state.temp_code = None
                else: st.warning("اكتب الاسم والتقط البصمة أولاً")
        
        with tab2:
            st.dataframe(all_signs, use_container_width=True)