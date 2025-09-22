import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import random

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="مختبر معالجة الصور المتقدم",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# إضافة CSS مخصص لتطبيق التصميم وتحسين واجهة المستخدم
def inject_custom_css():
    st.markdown(
        """
    <style>
    :root{
        --bg1: #0f1724;
        --bg2: #12232e;
        --panel: rgba(255,255,255,0.03);
        --accent: #4db6ac;
        --accent2: #6a5acd;
        --muted: #9aa6b2;
        --card: rgba(255,255,255,0.02);
    }

    /* App background */
    .stApp {
        background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%) !important;
        color: #e6eef2 !important;
        font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial !important;
        -webkit-font-smoothing: antialiased !important;
    }

    /* Main container */
    .main .block-container{
        padding: 2rem 2rem 4rem 2rem !important;
        max-width: 1400px !important;
    }

    /* Header */
    .custom-header{
        background: linear-gradient(90deg, rgba(74,70,222,0.98), rgba(77,182,172,0.95)) !important;
        padding: 22px 28px !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 30px rgba(13,18,25,0.6) !important;
        margin-bottom: 1.5rem !important;
    }
    .custom-header h1 { margin:0; font-size:28px; color: #fff !important; }
    .custom-header p { margin:4px 0 0 0; color: rgba(255,255,255,0.9) !important; }

    /* Cards / sections */
    .section {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(255,255,255,0.03) !important;
        box-shadow: 0 6px 20px rgba(2,6,23,0.6) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
        color: #fff !important;
        border: none !important;
        padding: 10px 18px !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 18px rgba(57,101,119,0.12) !important;
        transition: transform .12s ease-in-out !important;
    }
    .stButton>button:hover { transform: translateY(-3px) !important; }

    /* Upload box */
    .upload-area {
        border: 2px dashed rgba(255,255,255,0.06) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00)) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        text-align: center !important;
    }
    .upload-area p { color: var(--muted) !important; margin:0 !important; }

    /* Image boxes */
    .image-card {
        border-radius: 10px !important;
        padding: 8px !important;
        background: var(--card) !important;
        text-align: center !important;
    }

    /* Code editor style */
    .code-editor {
        background: rgba(8,10,12,0.35) !important;
        border-radius: 8px !important;
        padding: 8px !important;
        margin-bottom: 8px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00)) !important;
        padding: 8px !important;
        border-radius: 10px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.02) !important;
        color: #e6eef2 !important;
        border-radius: 9px !important;
        padding: 10px 14px !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)) !important;
        box-shadow: inset 0 -4px 18px rgba(0,0,0,0.4) !important;
        color: #fff !important;
    }

    footer {
        text-align: center !important;
        color: rgba(255,255,255,0.6) !important;
        padding-top: 16px !important;
    }

    /* Responsive tweaks */
    @media (max-width: 768px) {
        .custom-header h1 { font-size: 20px !important; }
    }

    </style>
    """,
        unsafe_allow_html=True,
    )


# دالة لعرض محرر الأكواد (مظهر فقط)
def display_code_editor(code, language="python", filename="code.py"):
    st.markdown(
        f"""
    <div class="code-editor">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#ff5f56"></div>
            <div style="width:10px;height:10px;border-radius:50%;background:#ffbd2e"></div>
            <div style="width:10px;height:10px;border-radius:50%;background:#27ca3f"></div>
            <div style="margin-left:8px;color:#aebec6;font-family:monospace;">{filename}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.code(code, language=language)


# ==================== دوال معالجة الصور (لم يتم التعديل عليها باستثناء استدعائها بشكل آمن من الواجهة) ====================

def adjust_brightness_contrast(image, brightness=0, contrast=100):
    brightness = 0 if brightness is None else brightness
    contrast = 100 if contrast is None else contrast

    contrast = float(contrast + 100) / 100.0
    contrast = contrast ** 2

    adjusted_image = cv2.addWeighted(image, contrast, image, 0, brightness - 100)
    return adjusted_image


def convert_color_space(image, conversion_code):
    return cv2.cvtColor(image, conversion_code)


def apply_threshold(image, threshold_type, threshold_value=127):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if threshold_type == "THRESH_BINARY":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    elif threshold_type == "THRESH_BINARY_INV":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    elif threshold_type == "THRESH_TRUNC":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TRUNC)
    elif threshold_type == "THRESH_TOZERO":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TOZERO)
    elif threshold_type == "THRESH_TOZERO_INV":
        _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_TOZERO_INV)
    elif threshold_type == "THRESH_OTSU":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        result = gray
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def apply_filter(image, filter_type, kernel_size=3):
    if filter_type == "Blur":
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == "Gaussian Blur":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == "Median Blur":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "Sharpen":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == "Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return image


def add_noise(image, noise_type):
    if noise_type == "Gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 50
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == "Salt & Pepper":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
        return noisy
    return image


def remove_noise(image, filter_type):
    if filter_type == "Median":
        return cv2.medianBlur(image, 5)
    elif filter_type == "Gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == "Bilateral":
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image


def detect_edges(image, method, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "Canny":
        edges = cv2.Canny(gray, threshold1, threshold2)
    elif method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edges = np.sqrt(sobelx ** 2 + sobely ** 2)
        edges = np.uint8(edges / np.max(edges) * 255)
    elif method == "Laplacian":
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges))
    else:
        edges = gray
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def apply_morphological_operation(image, operation, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    if operation == "Erosion":
        result = cv2.erode(binary, kernel, iterations=1)
    elif operation == "Dilation":
        result = cv2.dilate(binary, kernel, iterations=1)
    elif operation == "Opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        result = binary

    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)


def apply_geometric_transform(image, transform, angle=0, scale=1.0):
    h, w = image.shape[:2]

    if transform == "Rotation":
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(image, matrix, (w, h))
    elif transform == "Scaling":
        result = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    elif transform == "Translation":
        matrix = np.float32([[1, 0, 50], [0, 1, 50]])
        result = cv2.warpAffine(image, matrix, (w, h))
    elif transform == "Flipping":
        result = cv2.flip(image, 1)  # 0: vertical, 1: horizontal
    else:
        result = image

    return result


# ==================== دوال المحاضرات (عرض الشروحات) ====================

def lecture_1():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    الصورة الرقمية هي تمثيل رقمي للصورة الطبيعية تتكون من مصفوفة من البكسلات. كل بكسل يحمل قيمة رقمية تمثل شدة الإضاءة واللون في تلك النقطة. 
    تعتمد جودة الصورة على ثلاثة عوامل رئيسية: الأبعاد (الطول والعرض)، عدد القنوات اللونية، وعمق البت الذي يحدد عدد الألوان المحتملة لكل بكسل.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# تحميل الصورة وعرض معلوماتها
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# عرض معلومات الصورة
print("أبعاد الصورة:", image.shape)
print("عدد القنوات:", image.shape[2] if len(image.shape) > 2 else 1)
print("نوع البيانات:", image.dtype)
print("القيم الدنيا والعليا:", np.min(image), np.max(image))

# حفظ الصورة
cv2.imwrite('processed_image.jpg', image)
'''
    display_code_editor(code, filename="image_info.py")


def lecture_2():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    أنظمة الألوان المختلفة تستخدم لأغراض متعددة في معالجة الصور. نظام RGB هو النظام الأساسي المستخدم في العرض، 
    بينما نظام HSV مفيد لفصل الإضاءة عن اللون. نظام Grayscale يبسط الصورة لتحليل الشدة فقط.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# التحويل بين أنظمة الألوان
import cv2
import numpy as np

# تحميل الصورة
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# التحويل إلى Grayscale
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# التحويل إلى HSV
hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# تقسيم القنوات في نظام RGB
red_channel = image_rgb[:, :, 0]
green_channel = image_rgb[:, :, 1]
blue_channel = image_rgb[:, :, 2]
'''
    display_code_editor(code, filename="color_spaces.py")


def lecture_3():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    تعديل السطوع والتباين من العمليات الأساسية في معالجة الصور. يتم تعديل السطوع بإضافة قيمة ثابتة 
    إلى جميع وحدات البكسل في الصورة، بينما يتم تعديل التباين بضرب قيم البكسل في معامل ثابت. هذه 
    العمليات تساعد في تحسين جودة الصورة وإبراز التفاصيل المخفية.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# تعديل السطوع والتباين والعتبة
import cv2
import numpy as np

def adjust_brightness_contrast(image, brightness=0, contrast=100):
    # ضبط القيم الافتراضية
    brightness = 0 if brightness is None else brightness
    contrast = 100 if contrast is None else contrast

    # حساب معامل التباين
    contrast_factor = float(contrast + 100) / 100.0
    contrast_factor = contrast_factor ** 2

    # تطبيق المعادلة: output = input * contrast + brightness
    adjusted_image = cv2.addWeighted(
        image, contrast_factor, 
        image, 0, 
        brightness - 100
    )

    return adjusted_image
'''
    display_code_editor(code, filename="point_operations.py")


def lecture_4():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    المرشحات والالتفاف من العمليات الأساسية في معالجة الصور. تعتمد هذه العمليات على نواة (Kernel) 
    يتم تطبيقها على الصورة لإجراء عمليات مثل التنعيم، الحدة، الكشف عن الحواف، وغيرها من العمليات.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# تطبيق المرشحات والالتفاف
import cv2
import numpy as np

def apply_filter(image, filter_type, kernel_size=3):
    # ...
    return image
'''
    display_code_editor(code, filename="filters.py")


def lecture_5():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    الضوضاء في الصور هي تشوهات غير مرغوب فيها يمكن أن تنتج عن ظروف التصوير المختلفة. 
    توجد أنواع متعددة من الضوضاء مثل الضوضاء الغوسية وضوضاء الملح والفلفل. 
    تهدف عمليات إزالة الضوضاء إلى تحسين جودة الصورة مع الحفاظ على التفاصيل المهمة.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# إضافة وإزالة الضوضاء
import cv2
import numpy as np

def add_noise(image, noise_type):
    # ...
    return image
'''
    display_code_editor(code, filename="denoising.py")


def lecture_6():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    كشف الحواف هو عملية أساسية في معالجة الصور تهدف إلى تحديد المناطق التي تتغير فيها شدة 
    الصورة بشكل مفاجئ. هذه الحواف تمثل عادة حدودًا بين مناطق مختلفة في الصورة ويمكن 
    استخدامها في تطبيقات مثل التجزئة والتعرف على الأشياء.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# كشف الحواف في الصور
import cv2
import numpy as np

def detect_edges(image, method, threshold1=100, threshold2=200):
    # ...
    return image
'''
    display_code_editor(code, filename="edge_detection.py")


def lecture_7():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    العمليات المورفولوجية هي تقنيات لمعالجة الصور الثنائية تعتمد على شكل وهيكل الأشياء في الصورة. 
    هذه العمليات مفيدة في تنظيف الصور الثنائية، وإزالة الضوضاء، وعزل الأشياء، وتحسين الملامح.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# العمليات المورفولوجية
import cv2
import numpy as np

def apply_morphological_operation(image, operation, kernel_size=3):
    # ...
    return image
'''
    display_code_editor(code, filename="morphological.py")


def lecture_8():
    st.markdown("### 📚 الشرح النظري")
    st.markdown(
        """
    <p style='font-size: 1.05rem; line-height: 1.8; color: #dbeef7;'>
    التحويلات الهندسية هي عمليات تغير الهندسة المكانية للصورة. تشمل هذه العمليات التدوير، القياس، 
    الانعكاس، القص، والانزياح. هذه التحويلات مفيدة في تصحيح التشوهات، ومطابقة الصور، وتطبيقات الرؤية الحاسوبية.
    </p>
    """,
        unsafe_allow_html=True,
    )

    code = '''
# التحويلات الهندسية - أمثلة
import cv2
import numpy as np

# Rotation, Scaling, Translation, Flipping, Cropping
'''
    display_code_editor(code, filename="geometric_transforms.py")


# ==================== الواجهة الرئيسية المحسّنة ====================

def main():
    # حقن CSS المخصص
    inject_custom_css()

    # شريط جانبي للخيارات العامة / presets
    with st.sidebar:
        st.markdown("<div style='padding:12px 6px;'><h3 style='margin:0;color:#fff;'>🎛️ إعدادات سريعة</h3></div>", unsafe_allow_html=True)
        st.markdown("---")
        sample_mode = st.radio("نمط معاينة الصورة:", ("Fit", "Fill"), index=0, help="Fit يحافظ على تناسب الصورة داخل الحاوية، Fill يملأ الحاوية.")
        st.markdown("---")
        st.markdown("### 🎯 اختصارات سريعة")
        if st.button("اختيار صورة عشوائية", key="rnd_sample"):
            # إنشاء صورة عشوائية بسيطة كالـ placeholder
            h, w = 512, 512
            rand_img = np.zeros((h, w, 3), dtype=np.uint8)
            rand_img[:] = np.random.randint(40, 200, size=(3,))
            buf = io.BytesIO()
            Image.fromarray(rand_img).save(buf, format="PNG")
            st.session_state["__uploaded_image_bytes__"] = buf.getvalue()
            st.success("تم تحميل صورة عشوائية كنموذج للمعاينة.")
        st.markdown("---")
        st.markdown("### ℹ️ معلومات")
        st.write("واجهة مُحسّنة للعرض والتفاعل مع أدوات المعالجة.\nجرب التحويلات، المرشحات، والـ pipeline النهائي.")
        st.markdown("---")
        st.caption("تم التصميم والتطوير بواسطة المهندس عبدالمجيد رضوان الرباعي — 2025")

    # الرأس الرئيسي
    st.markdown(
        """
    <div class="custom-header">
        <h1>🎨 مختبر معالجة الصور المتقدم</h1>
        <p>مشروع محاضرات وواجهة تفاعلية عملية — تصميم واجهة ديناميكية ومظهر عصري</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # إنشاء قائمة المحاضرات (عناوين أقصر لتناسب التصميم)
    lectures = [
        "المحاضرة 1 — مدخل",
        "المحاضرة 2 — أنظمة الألوان",
        "المحاضرة 3 — سطوع/تباين",
        "المحاضرة 4 — مرشحات",
        "المحاضرة 5 — إزالة الضوضاء",
        "المحاضرة 6 — كشف الحواف",
        "المحاضرة 7 — مورفولوجيا",
        "المحاضرة 8 — تحويلات هندسية",
        "المشروع النهائي"
    ]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(lectures)

    # وظيفة مساعدة لقراءة الصورة من uploader أو من زر العينة
    def load_image_from_uploader(uploader_key, fallback_state_key="__uploaded_image_bytes__"):
        uploaded_file = st.session_state.get(uploader_key, None)
        # First check session state bytes (when sample chosen)
        if fallback_state_key in st.session_state and st.session_state[fallback_state_key] is not None:
            try:
                file_bytes = np.asarray(bytearray(st.session_state[fallback_state_key]), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            except Exception:
                pass

        # Otherwise check file_uploader object
        uploaded_file = st.file_uploader("اختر صورة أو اسحبها هنا", type=["jpg", "jpeg", "png"], key=uploader_key)
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            except Exception as e:
                st.error("حدث خطأ عند قراءة الصورة: " + str(e))
                return None
        return None

    # ----- Tab 1 -----
    with tab1:
        st.markdown("## المحاضرة 1: مدخل ومعايرة الصور الرقمية")
        lecture_1()
        st.markdown("<div class='section upload-area'><p>تحميل صورة لاكتشاف معلوماتها الأساسية — أو استخدم زر الاختيار العشوائي من الشريط الجانبي.</p></div>", unsafe_allow_html=True)

        image = load_image_from_uploader("lecture1")
        if image is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="🖼️ الصورة الأصلية", use_column_width=True)
            with col2:
                st.info("**معلومات الصورة:**")
                st.write(f"**الأبعاد:** `{image.shape[1]} x {image.shape[0]}`")
                st.write(f"**عدد القنوات:** `{image.shape[2]}`")
                st.write(f"**نوع البيانات:** `{image.dtype}`")
                st.write(f"**مدى القيم:** `{int(np.min(image))}` إلى `{int(np.max(image))}`")

    # ----- Tab 2 -----
    with tab2:
        st.markdown("## المحاضرة 2: أنظمة الألوان (Color Spaces)")
        lecture_2()
        st.write("### 🧪 التجربة العملية")

        image = load_image_from_uploader("lecture2")
        if image is not None:
            color_space = st.selectbox("اختر نظام الألوان للتحويل:", ["RGB", "GRAY", "HSV", "LAB", "YUV"])
            # امنع الخطأ بتحويل النتائج حسب الاختيار
            try:
                if color_space == "GRAY":
                    converted = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2RGB)
                elif color_space == "HSV":
                    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif color_space == "LAB":
                    converted = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                elif color_space == "YUV":
                    converted = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                else:
                    converted = image
            except Exception as e:
                st.error("خطأ أثناء تحويل نظام الألوان: " + str(e))
                converted = image

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                st.image(image, caption="🖼️ الأصلية (RGB)", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                st.image(converted, caption=f"🔄 المحولة ({color_space})", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ----- Tab 3 -----
    with tab3:
        st.markdown("## المحاضرة 3: تعديل السطوع والتباين")
        lecture_3()
        st.write("### 🧪 التجربة العملية")

        image = load_image_from_uploader("lecture3")
        if image is not None:
            col_img, col_ctrl = st.columns([2, 1])
            with col_img:
                st.image(image, caption="🖼️ الصورة الأصلية", use_column_width=True)
            with col_ctrl:
                brightness = st.slider("🔆 السطوع", -100, 100, 0, key="brightness_slider")
                contrast = st.slider("🌈 التباين", 0, 200, 100, key="contrast_slider")
                threshold = st.slider("⚫️ العتبة (للمعاينة)", 0, 255, 127, key="threshold_slider")
                if st.button("🚀 تطبيق التعديلات", key="apply_btn"):
                    with st.spinner("جارٍ تطبيق التعديلات..."):
                        adjusted = adjust_brightness_contrast(image, brightness, contrast)
                        thresholded = apply_threshold(image, "THRESH_BINARY", threshold)
                        negative = 255 - image
                    tab_a, tab_b, tab_c = st.tabs(["السطوع والتباين", "العتبة", "الصورة السالبة"])
                    with tab_a:
                        st.image(adjusted, caption="🔄 السطوع و التباين", use_column_width=True)
                    with tab_b:
                        st.image(thresholded, caption="⚫️ العتبة", use_column_width=True)
                    with tab_c:
                        st.image(negative, caption="🌗 الصورة السالبة", use_column_width=True)

    # ----- Tab 4 -----
    with tab4:
        st.markdown("## المحاضرة 4: المرشحات والالتفاف")
        lecture_4()
        st.write("### 🧪 التجربة العملية")

        image = load_image_from_uploader("lecture4")
        if image is not None:
            filter_type = st.selectbox("اختر نوع المرشح:", ["Blur", "Gaussian Blur", "Median Blur", "Sharpen", "Emboss", "Edge Detection"])
            kernel_size = st.slider("حجم النواة (يفضل فردي)", 3, 15, 5, 2, key="kernel_size")
            if st.button("🚀 تطبيق المرشح", key="apply_filter_btn"):
                with st.spinner("جارٍ تطبيق المرشح..."):
                    filtered = apply_filter(image, filter_type, kernel_size)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الأصلية", use_column_width=True)
                with col2:
                    st.image(filtered, caption=f"✨ بعد {filter_type}", use_column_width=True)

    # ----- Tab 5 -----
    with tab5:
        st.markdown("## المحاضرة 5: إزالة الضوضاء")
        lecture_5()
        st.write("### 🧪 التجربة العملية")

        image = load_image_from_uploader("lecture5")
        if image is not None:
            noise_type = st.selectbox("اختر نوع الضوضاء:", ["Gaussian", "Salt & Pepper"])
            denoise_type = st.selectbox("اختر طريقة إزالة الضوضاء:", ["Median", "Gaussian", "Bilateral"])
            if st.button("🚀 تطبيق المعالجة", key="apply_denoise_btn"):
                with st.spinner("جارٍ إضافة وإزالة الضوضاء..."):
                    noisy = add_noise(image, noise_type)
                    denoised = remove_noise(noisy, denoise_type)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption="🖼️ الأصلية", use_column_width=True)
                with col2:
                    st.image(noisy, caption=f"🔊 مع {noise_type} noise", use_column_width=True)
                with col3:
                    st.image(denoised, caption=f"🔇 بعد {denoise_type} filter", use_column_width=True)

    # ----- Tab 6 -----
    with tab6:
        st.markdown("## المحاضرة 6: كشف الحواف")
        lecture_6()
        st.write("### 🧪 التجربة العملية")

        image = load_image_from_uploader("lecture6")
        if image is not None:
            edge_method = st.selectbox("اختر طريقة كشف الحواف:", ["Canny", "Sobel", "Laplacian", "Prewitt"])
            if edge_method == "Canny":
                threshold1 = st.slider("العتبة الدنيا", 0, 255, 100, key="threshold1")
                threshold2 = st.slider("العتبة العليا", 0, 255, 200, key="threshold2")
            else:
                threshold1, threshold2 = 100, 200

            if st.button("🚀 كشف الحواف", key="detect_edges_btn"):
                with st.spinner("جارٍ كشف الحواف..."):
                    # Prewitt handled locally if selected
                    if edge_method == "Prewitt":
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                        prex = cv2.filter2D(gray, -1, kernelx)
                        prey = cv2.filter2D(gray, -1, kernely)
                        edges = np.sqrt(prex.astype(float) ** 2 + prey.astype(float) ** 2)
                        edges = np.uint8(edges / np.max(edges) * 255)
                        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    else:
                        edges = detect_edges(image, edge_method, threshold1, threshold2)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الأصلية", use_column_width=True)
                with col2:
                    st.image(edges, caption=f"🔍 الحواف بـ {edge_method}", use_column_width=True)

    # ----- Tab 7 -----
    with tab7:
        st.markdown("## المحاضرة 7: العمليات المورفولوجية")
        lecture_7()
        st.write("### 🧪 التجربة العملية")

        image = load_image_from_uploader("lecture7")
        if image is not None:
            morph_operation = st.selectbox("اختر العملية المورفولوجية:", ["Erosion", "Dilation", "Opening", "Closing", "Gradient"])
            kernel_size = st.slider("حجم النواة", 3, 15, 5, 2, key="morph_kernel_size")
            iterations = st.slider("عدد التكرارات (سيتم التطبيق بشكل آمن)", 1, 10, 1, key="morph_iterations")
            if st.button("🚀 تطبيق العملية", key="apply_morph_btn"):
                with st.spinner("جارٍ تطبيق العملية المورفولوجية..."):
                    # apply_morphological_operation لا تقبل iterations في التعريف الأصلي، لذا نطبق التكرار هنا بأمان
                    result = image.copy()
                    # نتأكد من العمل على نسخة بلون رمادي ثنائية إن لزم
                    for i in range(iterations):
                        result = apply_morphological_operation(result, morph_operation, kernel_size)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الأصلية", use_column_width=True)
                with col2:
                    st.image(result, caption=f"✨ بعد {morph_operation} x{iterations}", use_column_width=True)

    # ----- Tab 8 -----
    with tab8:
        st.markdown("## المحاضرة 8: التحويلات الهندسية")
        lecture_8()
        st.write("### 🧪 التجربة العملية")

        image = load_image_from_uploader("lecture8")
        if image is not None:
            transform_type = st.selectbox("اختر نوع التحويل:", ["Rotation", "Scaling", "Translation", "Flipping", "Cropping"])
            result = None

            if transform_type == "Rotation":
                angle = st.slider("زاوية الدوران", -180, 180, 45, key="rotation_angle")
                scale = st.slider("مقياس التكبير", 0.1, 3.0, 1.0, 0.1, key="rotation_scale")
                if st.button("🚀 تطبيق التحويل", key="apply_transform_rotation"):
                    with st.spinner("جارٍ تطبيق التدوير..."):
                        result = apply_geometric_transform(image, "Rotation", angle=angle, scale=scale)

            elif transform_type == "Scaling":
                scale = st.slider("مقياس التكبير العام", 0.1, 3.0, 1.5, 0.1, key="scale_general")
                if st.button("🚀 تطبيق التكبير", key="apply_transform_scaling"):
                    with st.spinner("جارٍ تطبيق التكبير..."):
                        result = apply_geometric_transform(image, "Scaling", scale=scale)

            elif transform_type == "Translation":
                tx = st.slider("الإزاحة الأفقية (px)", -200, 200, 50, key="translation_x")
                ty = st.slider("الإزاحة العمودية (px)", -200, 200, 50, key="translation_y")
                if st.button("🚀 تطبيق الإزاحة", key="apply_transform_translation"):
                    with st.spinner("جارٍ تطبيق الإزاحة..."):
                        h, w = image.shape[:2]
                        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                        result = cv2.warpAffine(image, matrix, (w, h))

            elif transform_type == "Flipping":
                flip_choice = st.selectbox("اختر اتجاه الانعكاس:", [("أفقي", 1), ("عمودي", 0), ("كلاهما", -1)], format_func=lambda x: x[0])
                if st.button("🚀 تطبيق الانعكاس", key="apply_transform_flipping"):
                    with st.spinner("جارٍ تطبيق الانعكاس..."):
                        result = cv2.flip(image, flip_choice[1])

            elif transform_type == "Cropping":
                max_w = image.shape[1]
                max_h = image.shape[0]
                x = st.slider("النقطة X", 0, max(0, max_w - 50), 50, key="crop_x")
                y = st.slider("النقطة Y", 0, max(0, max_h - 50), 50, key="crop_y")
                width = st.slider("العرض", 50, max_w - x, min(200, max_w - x), key="crop_width")
                height = st.slider("الارتفاع", 50, max_h - y, min(200, max_h - y), key="crop_height")
                if st.button("🚀 تطبيق القص", key="apply_transform_cropping"):
                    with st.spinner("جارٍ القص..."):
                        result = image[y : y + height, x : x + width]

            if result is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الصورة الأصلية", use_column_width=True)
                with col2:
                    st.image(result, caption=f"✨ بعد {transform_type}", use_column_width=True)

    # ----- Tab 9: Final Project Pipeline -----
    with tab9:
        st.markdown("## المشروع النهائي: pipeline معالجة الصور")
        st.markdown("### 🧪 أنشئ pipeline معالجة الصور الخاص بك")
        image = load_image_from_uploader("final_project")
        if image is not None:
            st.image(image, caption="🖼️ الصورة الأصلية", use_column_width=True)
            st.markdown("### ⚙️ خطوات المعالجة:")
            steps = st.multiselect(
                "اختر خطوات المعالجة:",
                ["تحويل إلى رمادي", "ضبط السطوع والتباين", "تطبيق مرشح", "كشف الحواف", "عتبة", "عمليات مورفولوجية"],
                default=["تحويل إلى رمادي", "كشف الحواف"]
            )

            processed_image = image.copy()
            process_history = []

            if st.button("▶️ تشغيل Pipeline", key="run_pipeline"):
                with st.spinner("جارٍ تشغيل الـ pipeline..."):
                    for step in steps:
                        if step == "تحويل إلى رمادي":
                            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
                            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
                            process_history.append("تحويل إلى تدرج الرمادي")

                        elif step == "ضبط السطوع والتباين":
                            processed_image = adjust_brightness_contrast(processed_image, 20, 120)
                            process_history.append("ضبط السطوع (+20) والتباين (+20%)")

                        elif step == "تطبيق مرشح":
                            processed_image = apply_filter(processed_image, "Gaussian Blur", 5)
                            process_history.append("تطبيق مرشح Gaussian Blur (5x5)")

                        elif step == "كشف الحواف":
                            processed_image = detect_edges(processed_image, "Canny", 100, 200)
                            process_history.append("كشف الحواف بـ Canny")

                        elif step == "عتبة":
                            processed_image = apply_threshold(processed_image, "THRESH_BINARY", 127)
                            process_history.append("تطبيق العتبة الثنائية (127)")

                        elif step == "عمليات مورفولوجية":
                            # نطبق تكرار واحد بشكل آمن
                            processed_image = apply_morphological_operation(processed_image, "Closing", 3)
                            process_history.append("عملية Closing مورفولوجية (3x3)")

                st.markdown("### 📊 نتائج المعالجة:")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="🖼️ الأصلية", use_column_width=True)
                with col2:
                    st.image(processed_image, caption="✨ الصورة النهائية", use_column_width=True)

                st.markdown("### 📝 سجل المعالجة:")
                for i, step in enumerate(process_history, 1):
                    st.write(f"{i}. {step}")

                # حفظ الصورة الناتجة للتحميل
                buf = io.BytesIO()
                processed_pil = Image.fromarray(processed_image)
                processed_pil.save(buf, format="JPEG", quality=95)
                byte_im = buf.getvalue()

                st.download_button(
                    label="💾 تحميل الصورة النهائية",
                    data=byte_im,
                    file_name="processed_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

    # التذييل
    st.markdown(
        """
    <footer>
        <p>تم التصميم والتطوير بواسطة المهندس عبدالمجيد رضوان الرباعي</p>
        <p>مشروع معالجة الصور الرقمية - كلية الهندسة © 2025</p>
    </footer>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
