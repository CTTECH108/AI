# app.py
# Machine Alignment Monitor — fixed version (no Streamlit calls inside worker)
# Requirements:
# pip install streamlit streamlit-webrtc opencv-python-headless numpy av

import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
import math
import tempfile
import os
import time
from typing import List

st.set_page_config(page_title="Machine Alignment Monitor", layout="wide")

# -------------------- CONFIG --------------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

DEFAULT_TRANSLATION_PX = 30
DEFAULT_ROTATION_DEG = 5
DEFAULT_SCALE_DIFF = 0.10

# -------------------- HELPERS (main thread safe) --------------------
def orb_keypoints_and_descriptors(img_gray, n_features=1000):
    orb = cv2.ORB_create(n_features)
    kps, des = orb.detectAndCompute(img_gray, None)
    return kps, des

def match_orb(des1, des2):
    if des1 is None or des2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return sorted(matches, key=lambda x: x.distance)

def estimate_transform(kp1, kp2, matches):
    # return homography H or (None,...)
    if matches is None or len(matches) < 6:
        return None, None, None
    ptsA = np.float32([kp1[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)
    return H, ptsA, ptsB

def decompose_homography(H):
    if H is None:
        return None
    try:
        H = H / H[2, 2]
        A = H[0:2, 0:2]
        U, s, Vt = np.linalg.svd(A)
        R = np.dot(U, Vt)
        scale = float(np.mean(s))
        angle = float(math.degrees(math.atan2(R[1, 0], R[0, 0])))
        tx = float(H[0, 2])
        ty = float(H[1, 2])
        return {"angle_deg": angle, "scale": scale, "tx": tx, "ty": ty}
    except Exception:
        return None

def draw_overlay(frame, lines: List[str], bbox=None, color=(0,255,0)):
    y0 = 25
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y0 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if bbox is not None:
        try:
            cv2.polylines(frame, [np.int32(bbox)], True, color, 2)
        except Exception:
            pass
    return frame

# -------------------- UI --------------------
st.title("📸 Machine Alignment Monitor — Live & Upload")

col1, col2 = st.columns([2, 1])
with col1:
    st.header("1) Reference (ideal) capture")
    ref_upload = st.file_uploader("Upload reference image (jpg/png)", type=["jpg","jpeg","png"])
    capture_request = st.button("Request capture reference from camera")
    ref_preview = st.empty()
with col2:
    st.header("2) Settings")
    translation_px = st.number_input("Translation threshold (px)", value=DEFAULT_TRANSLATION_PX, min_value=1)
    rotation_deg = st.number_input("Rotation threshold (deg)", value=DEFAULT_ROTATION_DEG, min_value=0)
    scale_diff = st.number_input("Scale diff threshold (fraction)", value=DEFAULT_SCALE_DIFF, step=0.01, min_value=0.0)
    process_every = st.number_input("Process every N frames (performance)", value=2, min_value=1, max_value=10)
    auto_record = st.checkbox("Save frames when alert occurs", value=True)
    visualize_matches = st.checkbox("Show feature matches (slow)", value=False)

# store uploaded reference (main thread)
uploaded_ref = None
if ref_upload:
    file_bytes = np.asarray(bytearray(ref_upload.read()), dtype=np.uint8)
    uploaded_ref = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    ref_preview.image(cv2.cvtColor(uploaded_ref, cv2.COLOR_BGR2RGB), caption="Uploaded reference")
    st.success("Reference uploaded (you still need to 'Capture current frame as reference' or set it to processor)")

# -------------------- Video Processor class --------------------
class AlignmentProcessor:
    """
    Lightweight processor class used by streamlit-webrtc.
    Note: we DO NOT call st.* inside this class.
    The main thread will call set_reference() when needed.
    """
    def __init__(self, n_features=800):
        self.reference = None           # BGR image
        self.ref_kp = None
        self.ref_des = None
        self.n_features = n_features
        self.last_frame = None          # last received frame (BGR)
        self.frame_counter = 0
        self.last_alert_time = 0

    def set_reference(self, img_bgr):
        if img_bgr is None:
            return
        # standardize reference size to speed up matching (keep aspect ratio)
        ref = img_bgr.copy()
        if max(ref.shape[:2]) > 800:
            scale = 800.0 / max(ref.shape[:2])
            ref = cv2.resize(ref, (int(ref.shape[1]*scale), int(ref.shape[0]*scale)))
        gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        kp, des = orb_keypoints_and_descriptors(gray, n_features=self.n_features)
        # set local reference and descriptors
        self.reference = ref
        self.ref_kp = kp
        self.ref_des = des

    def process_frame(self, frame_bgr, translation_px_local, rotation_deg_local, scale_diff_local,
                      visualize_matches_local=False, process_every_local=2, auto_record_local=True):
        """
        Runs alignment computation on the provided frame_bgr.
        Returns annotated frame (BGR) and info dict (or None).
        """
        self.last_frame = frame_bgr.copy()
        self.frame_counter += 1

        # Resize input for speed (maintain aspect)
        h, w = frame_bgr.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            frame_proc = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))
        else:
            frame_proc = frame_bgr.copy()

        # Only run heavy matching every N frames
        if self.reference is None:
            out = draw_overlay(frame_proc, ["No reference set — upload or capture first"], color=(0,0,255))
            return out, None

        if (self.frame_counter % process_every_local) != 0:
            # show last computed overlay text if available - here we just return the current frame
            return frame_proc, None

        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        kp, des = orb_keypoints_and_descriptors(gray, n_features=400)

        matches = match_orb(self.ref_des, des)
        H, ptsA, ptsB = estimate_transform(self.ref_kp, kp, matches)
        info = decompose_homography(H)

        lines = []
        alert = False
        if info is None:
            lines.append("Not enough matches to compute alignment")
        else:
            tx = info["tx"]; ty = info["ty"]; ang = info["angle_deg"]; scale = info["scale"]
            lines.append(f"tx={tx:.1f}px ty={ty:.1f}px rot={ang:.2f}° scale={scale:.3f}")
            if abs(tx) > translation_px_local or abs(ty) > translation_px_local:
                alert = True
                lines.append("⚠ Translation exceeded")
            if abs(ang) > rotation_deg_local:
                alert = True
                lines.append("⚠ Rotation exceeded")
            if abs(scale - 1.0) > scale_diff_local:
                alert = True
                lines.append("⚠ Scale exceeded")

        color = (0,255,0) if not alert else (0,0,255)
        out = draw_overlay(frame_proc, lines, color=color)

        # Draw homography bbox if available
        if H is not None and self.reference is not None:
            try:
                h_ref, w_ref = self.reference.shape[:2]
                corners = np.array([[0,0],[w_ref,0],[w_ref,h_ref],[0,h_ref]], dtype='float32').reshape(-1,1,2)
                dst = cv2.perspectiveTransform(corners, H)
                bbox = dst.reshape(-1,2)
                cv2.polylines(out, [np.int32(bbox)], True, color, 2)
            except Exception:
                pass

        # optional: draw matches (very slow) - skip heavy drawing by default
        if visualize_matches_local and matches:
            try:
                matches_to_draw = matches[:40]
                out = cv2.drawMatches(self.reference, self.ref_kp, out, kp, matches_to_draw, None, flags=2)
            except Exception:
                pass

        # Save alert frame (debounced)
        if alert and auto_record_local and (time.time() - self.last_alert_time > 3):
            folder = os.path.join(tempfile.gettempdir(), "alignment_alerts")
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"alert_{int(time.time())}.jpg")
            try:
                cv2.imwrite(fname, out)
                self.last_alert_time = time.time()
            except Exception:
                pass

        return out, info

# -------------------- Streamlit-WebRTC integration --------------------
from streamlit_webrtc import VideoProcessorBase

class WebRTCProcessor(VideoProcessorBase):
    """
    Adapter class for streamlit-webrtc's video_processor_factory.
    This class wraps AlignmentProcessor and implements recv().
    DO NOT call st.* in this class.
    """
    def __init__(self):
        self.aligner = AlignmentProcessor(n_features=800)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # convert to ndarray
        frm = frame.to_ndarray(format="bgr24")
        # process and get annotated frame
        out_frame, info = self.aligner.process_frame(
            frm,
            translation_px_local=translation_px,
            rotation_deg_local=rotation_deg,
            scale_diff_local=scale_diff,
            visualize_matches_local=visualize_matches,
            process_every_local=process_every,
            auto_record_local=auto_record
        )
        # ensure out_frame is BGR ndarray
        if out_frame is None:
            out_frame = frm
        return av.VideoFrame.from_ndarray(out_frame, format="bgr24")

# webrtc streamer (SENDRECV so browser sends camera)
webrtc_ctx = webrtc_streamer(
    key="alignment_monitor",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=WebRTCProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# -------------------- Main thread interactions (safe) --------------------
# get processor instance (works across versions)
def get_processor_instance(ctx):
    if ctx is None:
        return None
    proc = getattr(ctx, "video_processor", None)
    if proc is None:
        proc = getattr(ctx, "video_transformer", None)
    return proc

proc = get_processor_instance(webrtc_ctx)

# If user uploaded a reference image, set it to processor (main thread)
if uploaded_ref is not None and proc is not None and hasattr(proc, "aligner"):
    try:
        proc.aligner.set_reference(uploaded_ref)
        st.success("Uploaded reference set to live processor.")
    except Exception as e:
        st.error(f"Failed to set uploaded reference: {e}")

# If user clicked "Request capture", we set a small prompt: wait one second then attempt capture
if capture_request:
    if proc is None:
        st.warning("Camera not started yet - wait for the camera feed and try again.")
    else:
        # Attempt to grab last frame from processor
        last = getattr(proc.aligner, "last_frame", None)
        if last is not None:
            proc.aligner.set_reference(last)
            st.success("Reference captured from live camera.")
        else:
            st.warning("No camera frame available yet. Wait a second for the feed to initialize and press the button again.")

# Manual capture button (explicit)
if proc is not None and st.button("Capture current frame as reference"):
    last = getattr(proc.aligner, "last_frame", None)
    if last is not None:
        proc.aligner.set_reference(last)
        st.success("Reference set from live camera frame.")
    else:
        st.warning("No camera frame available yet. Wait for the feed to start.")

# Show simple camera status (main thread only)
camera_status = "connected" if (proc is not None and getattr(proc.aligner, "last_frame", None) is not None) else "not connected"
st.markdown(f"**Camera status:** {camera_status}")

# Preview stored reference if available
if proc is not None and getattr(proc.aligner, "reference", None) is not None:
    try:
        ref_img = proc.aligner.reference
        st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption="Reference (in processor)")
    except Exception:
        pass
elif uploaded_ref is not None:
    st.image(cv2.cvtColor(uploaded_ref, cv2.COLOR_BGR2RGB), caption="Uploaded reference (not yet set)")

# -------------------- Offline analysis (optional) --------------------
st.header("Offline: upload image/video for batch analysis")
upload_off = st.file_uploader("Upload image or short video", type=["jpg","jpeg","png","mp4"])
if upload_off is not None:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(upload_off.read())
    tmp.flush()
    path = tmp.name
    ext = os.path.splitext(upload_off.name)[1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(path)
        if proc is None or getattr(proc.aligner, "reference", None) is None:
            st.warning("Set a reference first (live or upload).")
        else:
            out, info = proc.aligner.process_frame(img, translation_px, rotation_deg, scale_diff,
                                                   visualize_matches, process_every, auto_record)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            st.json(info)
    else:
        st.info("Video analysis: analyzing first 200 frames (may be slow).")
        cap = cv2.VideoCapture(path)
        count = 0
        alerts = 0
        while cap.isOpened() and count < 200:
            ret, frame = cap.read()
            if not ret: break
            out, info = proc.aligner.process_frame(frame, translation_px, rotation_deg, scale_diff,
                                                  visualize_matches, process_every, auto_record)
            if info:
                if abs(info["tx"]) > translation_px or abs(info["ty"]) > translation_px or abs(info["angle_deg"]) > rotation_deg or abs(info["scale"]-1.0) > scale_diff:
                    alerts += 1
            count += 1
        cap.release()
        st.write(f"Frames checked: {count}, alerts found: {alerts}")

st.markdown("---")
st.info("If camera feed stays blank: refresh the page, ensure you allowed camera permissions in the browser, and make sure no other app (Zoom, Teams) is using the camera.")
