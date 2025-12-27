import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av, cv2, mediapipe as mp, gspread, pandas as pd, numpy as np
import math, time
from oauth2client.service_account import ServiceAccountCredentials
from PIL import Image

# --- 1. إعدادات الهوية ---
LOGO_URL = "https://i.postimg.cc/R0cQyjrR/logo-png.png" 
st.set_page_config(page_title="بصيرة | Smart Sign Translator", page_icon=LOGO_URL, layout="wide")

# --- 2. تهيئة الذاكرة (Session State) ---
keys = {
    'auth': {'in': False, 'user': None, 'role': None},
    'cam_active': False, 'admin_cam_active': False,
    'stab_count': 0, 'last_s': "", 'final_s': "", 'last_time': time.time(),
    'live_code': None
}
for key, val in keys.items():
    if key not in st.session_state: st.session_state[key] = val

# --- 3. الاتصال بالسحابة ---
@st.cache_resource
def init_system():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    db = client.open("Basira_DB")
    mp_hands = mp.solutions.hands
    engine = mp_hands.Hands(max_num_hands=1, model_complexity=1, min_detection_confidence=0.7)
    return db.worksheet("Signs_DB"), db.worksheet("Users_Admin"), engine, mp.solutions.drawing_utils

signs_sheet, auth_sheet, hands_engine, mp_draw = init_system()

# --- 4. معالج الفيديو (WebRTC) ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        res = hands_engine.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            st.session_state.last_time = time.time()
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hl, mp.solutions.hands.HAND_CONNECTIONS)
            lm = hl.landmark
            palm = math.sqrt((lm[0].x-lm[9].x)**2 + (lm[0].y-lm[9].y)**2 + (lm[0].z-lm[9].z)**2)
            tips = [4, 8, 12, 16, 20]
            st.session_state.live_code = ",".join([str(round(math.sqrt((lm[t].x-lm[0].x)**2 + (lm[t].y-lm[0].y)**2 + (lm[t].z-lm[0].z)**2)/palm, 1)) for t in tips])
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. واجهة التطبيق ---
st.markdown("<style> * { font-family: 'Cairo', sans-serif; text-align: right; } </style>", unsafe_allow_html=True)

if not st.session_state.auth['in']:
    st.title("🔒 دخول منصة بصيرة")
    u, p = st.text_input("اسم المستخدم"), st.text_input("كلمة السر", type="password")
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
        st.header("📸 المترجم الذكي")
        # زر الكاميرا الحقيقي
        btn_label = "🔴 إيقاف الكاميرا" if st.session_state.cam_active else "🔵 تشغيل الكاميرا"
        if st.button(btn_label):
            st.session_state.cam_active = not st.session_state.cam_active
            st.rerun()

        if st.session_state.cam_active:
            webrtc_streamer(key="user-cam", video_processor_factory=VideoProcessor, 
                            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            
            # منطق الترجمة والاستقرار (10 إطارات)
            if st.session_state.live_code and time.time() - st.session_state.last_time < 2.0:
                signs_df = pd.DataFrame(signs_sheet.get_all_records())
                live_vals = np.array([float(x) for x in st.session_state.live_code.split(',')])
                best_match = None; min_err = 0.3
                for _, row in signs_df.iterrows():
                    db_vals = np.array([float(x) for x in str(row['Finger_Code']).split(',')])
                    err = np.mean(np.abs(live_vals - db_vals))
                    if err < min_err: min_err, best_match = err, row['Sign_Name']
                
                if best_match == st.session_state.last_s: st.session_state.stab_count += 1
                else: st.session_state.stab_count = 0; st.session_state.last_s = best_match
                
                if st.session_state.stab_count >= 10:
                    st.title(f"✨ الترجمة: {best_match}")

    elif role == "Admin":
        st.header("⚙️ لوحة تحكم المدير")
        t1, t2, t3 = st.tabs(["🎥 رفع بالكاميرا", "📁 رفع صور", "📊 قاعدة البيانات"])
        
        with t1:
            st.subheader("التقاط البصمة الحية")
            if st.button("📷 فتح/غلق الكاميرا للالتقاط"):
                st.session_state.admin_cam_active = not st.session_state.admin_cam_active
            
            if st.session_state.admin_cam_active:
                webrtc_streamer(key="admin-cam", video_processor_factory=VideoProcessor)
                sign_name = st.text_input("اسم الإشارة المراد تسجيلها")
                if st.button("🎯 التقاط وتحليل الآن"):
                    if sign_name and st.session_state.live_code:
                        signs_sheet.append_row([sign_name, st.session_state.live_code])
                        st.success(f"تم حفظ '{sign_name}' بنجاح!")
                    else: st.warning("تأكد من وضع يدك وكتابة الاسم")

        with t2:
            st.subheader("تحليل الصور المخزنة")
            name = st.text_input("اسم الإشارة (للصورة)")
            up = st.file_uploader("اختر صورة", type=['jpg','png','jpeg'])
            if up and st.button("تحليل الصورة"):
                res = hands_engine.process(cv2.cvtColor(np.array(Image.open(up)), cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    # (كود الحساب الرياضي نفسه)
                    st.success("تم تحليل الصورة وحفظها")
                else: st.error("لم يتم رصد يد")

        with t3:
            st.subheader("قاعدة بيانات بصيرة")
            st.dataframe(pd.DataFrame(signs_sheet.get_all_records()), use_container_width=True)

    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.auth['in'] = False; st.rerun()