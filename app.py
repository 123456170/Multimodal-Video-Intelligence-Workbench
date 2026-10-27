import io
import os
import json
import time
import math
import tempfile
import wave

import numpy as np
import pandas as pd
import cv2
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Multimodal Video Intelligence Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------
# Constants and synthetic ground truth
# ------------------------------------------------------------------------------

W, H = 640, 360
FPS = 10
DUR = 24.0
SR = 22050

BASE_EVENTS = [
    {
        "id": 1,
        "start": 0.0,
        "end": 3.0,
        "label": "scene_start",
        "caption": "Camera view of a room with a table and two chairs.",
        "entity": "scene_1",
        "modality": "visual",
        "confidence": 0.99,
    },
    {
        "id": 2,
        "start": 2.0,
        "end": 7.0,
        "label": "person_enters",
        "caption": "A person enters the room and walks toward the table.",
        "entity": "person_1",
        "modality": "visual",
        "confidence": 0.95,
    },
    {
        "id": 3,
        "start": 3.0,
        "end": 8.0,
        "label": "carries_box",
        "caption": "The person carries a box or parcel across the room.",
        "entity": "box_1",
        "modality": "visual",
        "confidence": 0.93,
    },
    {
        "id": 4,
        "start": 7.5,
        "end": 11.5,
        "label": "puts_box_on_table",
        "caption": "A person puts a box on a table.",
        "entity": "box_1",
        "modality": "visual",
        "confidence": 0.96,
    },
    {
        "id": 5,
        "start": 11.5,
        "end": 14.0,
        "label": "stands_near_table",
        "caption": "The person stands near the table after placing the parcel.",
        "entity": "person_1",
        "modality": "visual",
        "confidence": 0.90,
    },
    {
        "id": 6,
        "start": 14.0,
        "end": 18.0,
        "label": "alarm_sound",
        "caption": "An alarm sounds and a red warning light flashes.",
        "entity": "alarm_1",
        "modality": "audio_visual",
        "confidence": 0.94,
    },
    {
        "id": 7,
        "start": 16.0,
        "end": 20.0,
        "label": "missing_ppe",
        "caption": "A person is visible without a safety helmet during the alarm.",
        "entity": "person_1",
        "modality": "visual",
        "confidence": 0.88,
    },
    {
        "id": 8,
        "start": 20.0,
        "end": 22.0,
        "label": "person_falls",
        "caption": "A person falls down near the right side of the room.",
        "entity": "person_1",
        "modality": "visual",
        "confidence": 0.91,
    },
    {
        "id": 9,
        "start": 21.2,
        "end": 22.2,
        "label": "crash_sound",
        "caption": "A crash impact sound occurs as the person hits the floor.",
        "entity": "person_1",
        "modality": "audio",
        "confidence": 0.89,
    },
    {
        "id": 10,
        "start": 22.0,
        "end": 24.0,
        "label": "incident_end",
        "caption": "The scene remains still after the fall and crash.",
        "entity": "scene_1",
        "modality": "visual",
        "confidence": 0.97,
    },
]

AUDIO_EVENTS_BASE = [
    {
        "start": 14.0,
        "end": 14.25,
        "label": "beep",
        "type": "beep",
        "freq": 1200.0,
        "confidence": 0.92,
    },
    {
        "start": 14.25,
        "end": 18.0,
        "label": "alarm",
        "type": "siren",
        "freq": 800.0,
        "confidence": 0.96,
    },
    {
        "start": 21.2,
        "end": 22.0,
        "label": "crash",
        "type": "impact",
        "freq": 0.0,
        "confidence": 0.91,
    },
]

EVAL_QUERIES = [
    ("a person puts a box on a table", ["puts_box_on_table", "carries_box"]),
    ("alarm sounds", ["alarm_sound"]),
    ("person falls", ["person_falls", "crash_sound"]),
    ("missing helmet", ["missing_ppe"]),
    ("parcel on table", ["puts_box_on_table", "carries_box"]),
]


# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ease(x):
    x = clamp(float(x), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a + (b - a) * clamp(float(t), 0.0, 1.0)


def iter_event_records(events):
    if isinstance(events, pd.DataFrame):
        return events.to_dict("records")
    return list(events)


def safe_float(value, default=0.0):
    try:
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


# ------------------------------------------------------------------------------
# Synthetic scene renderer
# ------------------------------------------------------------------------------

def person_pose(t):
    t = float(t)

    if t < 2.0:
        px = 50
    elif t < 7.0:
        px = int(lerp(50, 330, ease((t - 2.0) / 5.0)))
    elif t < 13.0:
        px = 330
    elif t < 18.0:
        px = int(lerp(330, 560, ease((t - 13.0) / 5.0)))
    else:
        px = 560

    fall = clamp((t - 20.0) / 1.5, 0.0, 1.0)
    e = ease(fall)

    head_y = int(205 + 75 * e)
    body_top = int(225 + 55 * e)
    body_h = int(70 - 35 * e)

    return px, head_y, body_top, body_h, fall


def box_pose(t, overrides=None):
    ov = overrides or {}
    if ov.get("box_removed", False):
        return None, None, False, "removed"

    px, head_y, _, _, _ = person_pose(t)

    if t < 3.0:
        bx, by = 80, 260
        loc = "shelf"
    elif t < 7.0:
        bx = int(px + 22)
        by = int(head_y + 8)
        loc = "carried"
    elif t < 8.0:
        alpha = ease(t - 7.0)
        bx = int(lerp(px + 22, 430, alpha))
        by = int(lerp(head_y + 8, 195, alpha))
        loc = "placing"
    else:
        bx, by = 430, 195
        loc = "table"

    return bx, by, True, loc


def draw_chair(img, cx, cy):
    color = (80, 130, 190)
    cv2.rectangle(img, (cx - 22, cy - 45), (cx + 22, cy - 35), color, -1)
    cv2.rectangle(img, (cx - 25, cy), (cx + 25, cy + 10), color, -1)
    cv2.rectangle(img, (cx - 22, cy + 10), (cx - 17, cy + 50), color, -1)
    cv2.rectangle(img, (cx + 17, cy + 10), (cx + 22, cy + 50), color, -1)


def render_frame_bgr(t, overrides=None):
    ov = overrides or {}
    t = float(t)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (42, 42, 48)

    # Room
    cv2.rectangle(img, (0, 0), (W, H), (38, 38, 44), -1)
    cv2.rectangle(img, (20, 30), (W - 20, 120), (50, 50, 58), -1)
    cv2.line(img, (0, 300), (W, 300), (105, 105, 105), 2)

    # Furniture
    draw_chair(img, 210, 250)
    draw_chair(img, 500, 250)

    cv2.rectangle(img, (350, 220), (590, 235), (34, 60, 120), -1)
    cv2.rectangle(img, (365, 235), (375, 300), (34, 60, 120), -1)
    cv2.rectangle(img, (565, 235), (575, 300), (34, 60, 120), -1)

    # Person
    px, head_y, body_top, body_h, fall = person_pose(t)

    if ov.get("person_fallen", False):
        head_y = 280
        body_top = 270
        body_h = 25
        fall = 1.0

    cv2.rectangle(img, (px - 18, body_top), (px + 18, body_top + body_h), (70, 140, 210), -1)
    cv2.circle(img, (px, head_y), 18, (90, 170, 230), -1)

    # Helmet / PPE
    helmet = ov.get("helmet", t < 16.0)
    if helmet:
        cv2.ellipse(img, (px, head_y - 8), (22, 12), 0, 180, 360, (0, 200, 0), -1)

    # Box / parcel
    bx, by, visible, _ = box_pose(t, ov)
    if visible:
        cv2.rectangle(img, (bx, by), (bx + 45, by + 35), (0, 90, 200), -1)
        cv2.putText(
            img,
            "BOX",
            (bx + 6, by + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    # Alarm lamp
    alarm = ov.get("alarm", 14.0 <= t < 18.0)
    if alarm:
        flash = int(t * 6) % 2
        color = (0, 0, 255) if flash else (0, 0, 120)
        cv2.circle(img, (600, 40), 18, color, -1)
        cv2.putText(
            img,
            "ALARM",
            (560, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.circle(img, (600, 40), 18, (70, 70, 70), -1)

    # Overlay
    cv2.rectangle(img, (0, 0), (185, 30), (0, 0, 0), -1)
    cv2.putText(
        img,
        f"T={t:05.1f}s",
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    highlight = ov.get("highlight")
    if highlight:
        boxes = get_track_boxes(t, ov)
        if highlight in boxes:
            x1, y1, x2, y2 = boxes[highlight]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

    return img


def render_frame_rgb(t, overrides=None):
    return cv2.cvtColor(render_frame_bgr(t, overrides), cv2.COLOR_BGR2RGB)


def get_track_boxes(t, overrides=None):
    ov = overrides or {}
    px, head_y, body_top, body_h, _ = person_pose(t)

    if ov.get("person_fallen", False):
        head_y = 280
        body_top = 270
        body_h = 25

    boxes = {
        "person_1": (px - 30, head_y - 25, px + 30, body_top + body_h + 10),
        "table_1": (350, 220, 590, 300),
        "alarm_1": (580, 20, 620, 60),
        "chair_1": (185, 205, 235, 300),
        "chair_2": (475, 205, 525, 300),
        "scene_1": (0, 0, W, H),
    }

    bx, by, visible, _ = box_pose(t, ov)
    if visible:
        boxes["box_1"] = (bx, by, bx + 45, by + 35)

    helmet = ov.get("helmet", t < 16.0)
    if helmet:
        boxes["helmet_1"] = (px - 22, head_y - 20, px + 22, head_y + 2)

    return boxes


# ------------------------------------------------------------------------------
# Audio synthesis
# ------------------------------------------------------------------------------

def audio_array_to_wav_bytes(y):
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak * 0.90

    pcm = (y * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    wf = wave.open(buf, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    wf.writeframes(pcm.tobytes())
    wf.close()

    buf.seek(0)
    return buf.read()


def synthesize_audio_array(events):
    events = iter_event_records(events)
    n = int(SR * DUR)
    y = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(42)

    for ev in events:
        start = int(safe_float(ev.get("start", 0.0)) * SR)
        end = int(safe_float(ev.get("end", 0.0)) * SR)
        start = clamp(start, 0, n)
        end = clamp(end, 0, n)
        d = end - start
        if d <= 0:
            continue

        tt = np.arange(d, dtype=np.float32) / float(SR)
        typ = str(ev.get("type", "beep")).lower()

        if typ == "siren":
            freq = 700.0 + 350.0 * np.sin(2.0 * np.pi * 1.8 * tt)
            phase = 2.0 * np.pi * np.cumsum(freq) / float(SR)
            sig = np.sin(phase) + 0.30 * np.sin(2.0 * phase)
            env = np.clip(np.minimum(tt * 10.0, ((end - start) / float(SR) - tt) * 10.0), 0.0, 1.0)
            y[start:end] += 0.28 * sig * env

        elif typ == "impact":
            noise = rng.normal(0.0, 1.0, size=d).astype(np.float32)
            kernel = np.ones(8, dtype=np.float32) / 8.0
            noise = np.convolve(noise, kernel, mode="same")
            env = np.exp(-tt * 10.0)
            y[start:end] += 0.65 * noise * env

        else:
            freq = safe_float(ev.get("freq", 1000.0), 1000.0)
            env = np.exp(-tt * 4.0)
            y[start:end] += 0.25 * np.sin(2.0 * np.pi * freq * tt) * env

    return y


def synthesize_audio_bytes(events):
    y = synthesize_audio_array(events)
    return audio_array_to_wav_bytes(y), y


def make_tone_bytes(kind="alarm", duration=1.0):
    duration = clamp(float(duration), 0.1, 3.0)
    typ = "siren" if kind == "alarm" else "impact" if kind == "crash" else "beep"
    events = [
        {
            "start": 0.0,
            "end": duration,
            "label": kind,
            "type": typ,
            "freq": 1000.0,
            "confidence": 0.90,
        }
    ]

    n = int(SR * duration)
    y = np.zeros(n, dtype=np.float32)
    tt = np.arange(n, dtype=np.float32) / float(SR)

    if typ == "siren":
        freq = 700.0 + 350.0 * np.sin(2.0 * np.pi * 1.8 * tt)
        phase = 2.0 * np.pi * np.cumsum(freq) / float(SR)
        y += 0.30 * (np.sin(phase) + 0.3 * np.sin(2 * phase))
    elif typ == "impact":
        rng = np.random.default_rng(abs(hash(kind)) % (2**32))
        noise = rng.normal(0.0, 1.0, size=n).astype(np.float32)
        kernel = np.ones(8, dtype=np.float32) / 8.0
        noise = np.convolve(noise, kernel, mode="same")
        y += 0.65 * noise * np.exp(-tt * 10.0)
    else:
        y += 0.30 * np.sin(2.0 * np.pi * 1000.0 * tt) * np.exp(-tt * 4.0)

    return audio_array_to_wav_bytes(y)


# ------------------------------------------------------------------------------
# Video writing
# ------------------------------------------------------------------------------

def write_mp4_from_times(times, overrides=None):
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        times = np.array([0.0], dtype=float)

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            path,
            fps=FPS,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            for t in times:
                writer.append_data(render_frame_rgb(float(t), overrides))

    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass

        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, FPS, (W, H))
        for t in times:
            out.write(render_frame_bgr(float(t), overrides))
        out.release()

    with open(path, "rb") as f:
        data = f.read()

    try:
        os.unlink(path)
    except Exception:
        pass

    return data


def generate_video_bytes():
    times = np.arange(0.0, DUR, 1.0 / FPS, dtype=float)
    return write_mp4_from_times(times)


# ------------------------------------------------------------------------------
# Cached demo assets and indexes
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=1)
def build_demo_assets():
    video_bytes = generate_video_bytes()
    audio_bytes, audio_np = synthesize_audio_bytes(AUDIO_EVENTS_BASE)

    events_df = pd.DataFrame(BASE_EVENTS)
    audio_events_df = pd.DataFrame(AUDIO_EVENTS_BASE)

    return {
        "video": video_bytes,
        "audio": audio_bytes,
        "audio_np": audio_np,
        "events": events_df,
        "audio_events": audio_events_df,
    }


@st.cache_resource(show_spinner=False)
def get_search_engine():
    df = pd.DataFrame(BASE_EVENTS)
    texts = (df["caption"] + " " + df["label"] + " " + df["entity"]).str.lower()

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)

    return vec, X, df


@st.cache_resource(show_spinner=False)
def get_entity_linker():
    df = pd.DataFrame(BASE_EVENTS)
    profiles = build_track_profiles(df)

    keys = list(profiles.keys())
    texts = [profiles[k] for k in keys]

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)

    return vec, X, keys, profiles, df


@st.cache_data(show_spinner=False, max_entries=32)
def make_clip_bytes(start, end, overrides_json="{}"):
    start = clamp(float(start), 0.0, DUR)
    end = clamp(float(end), 0.0, DUR)
    if end <= start:
        end = min(DUR, start + 1.0)

    overrides = json.loads(overrides_json) if overrides_json else {}
    times = np.arange(start, end + 1e-6, 1.0 / FPS, dtype=float)

    return write_mp4_from_times(times, overrides)


# ------------------------------------------------------------------------------
# Retrieval, correlation, summarization, QA, commands, safety, memory
# ------------------------------------------------------------------------------

def semantic_search(query, top_k=5, min_score=0.0):
    vec, X, df = get_search_engine()
    q = vec.transform([str(query).lower()])
    scores = cosine_similarity(q, X)[0]

    out = df.copy()
    out["score"] = scores
    out = out[out["score"] >= min_score]
    out = out.sort_values("score", ascending=False).head(int(top_k))
    return out.reset_index(drop=True)


def evaluate_search(top_k=5):
    recalls = []
    aps = []
    latencies = []

    for query, relevant in EVAL_QUERIES:
        t0 = time.perf_counter()
        ranked = semantic_search(query, top_k=max(20, len(BASE_EVENTS)))
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        top = ranked.head(top_k)
        top_labels = set(top["label"].tolist())
        recall = len(top_labels & set(relevant)) / max(1, len(relevant))
        recalls.append(recall)

        hits = [1 if label in relevant else 0 for label in ranked["label"].tolist()]
        num_rel = 0
        score = 0.0
        for i, hit in enumerate(hits, start=1):
            if hit:
                num_rel += 1
                score += num_rel / i

        ap = score / max(1, len(relevant)) if hits else 0.0
        aps.append(ap)

    return float(np.mean(recalls)), float(np.mean(aps)), float(np.mean(latencies))


def build_track_profiles(events_df):
    profiles = {}

    for _, r in events_df.iterrows():
        ent = r["entity"]
        profiles.setdefault(ent, [])
        profiles[ent].append(r["caption"])

    profiles.setdefault("table_1", []).append("A wooden table in the room.")
    profiles.setdefault("chair_1", []).append("A chair near the left side of the table.")
    profiles.setdefault("chair_2", []).append("A chair near the right side of the table.")
    profiles.setdefault("helmet_1", []).append("A green safety helmet worn by the person.")
    profiles.setdefault("scene_1", []).append("The room, furniture, and overall camera view.")

    return {k: " ".join(v) for k, v in profiles.items()}


def link_entity(mention, top_k=5):
    vec, X, keys, profiles, df = get_entity_linker()
    q = vec.transform([str(mention).lower()])
    scores = cosine_similarity(q, X)[0]

    rows = []
    for key, score in zip(keys, scores):
        ev = df[df["entity"] == key]
        if ev.empty:
            time_ranges = "static object"
        else:
            time_ranges = ", ".join(
                f"{r.start:.1f}-{r.end:.1f}" for r in ev.itertuples()
            )

        rows.append(
            {
                "track": key,
                "score": float(score),
                "uncertainty": float(max(0.0, 1.0 - score)),
                "description": profiles[key][:180],
                "time_ranges": time_ranges,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=False).head(int(top_k)).reset_index(drop=True)
    return out


def answer_vqa(question, t, overrides=None):
    t0 = time.perf_counter()
    ov = overrides or {}
    q = str(question).lower()

    helmet = ov.get("helmet", t < 16.0)
    alarm = ov.get("alarm", 14.0 <= t < 18.0)
    box_removed = ov.get("box_removed", False)

    px, _, _, _, fall = person_pose(t)
    if ov.get("person_fallen", False):
        fall = 1.0

    answer = None
    confidence = 0.0
    evidence = []
    abstain = False

    if "chair" in q and ("how many" in q or "count" in q):
        answer = 2
        confidence = 0.98
        evidence = ["scene_graph.chairs"]

    elif "person" in q and ("how many" in q or "count" in q):
        answer = 1
        confidence = 0.96
        evidence = ["tracker.person_1"]

    elif ("box" in q or "parcel" in q or "package" in q) and ("how many" in q or "count" in q):
        answer = 0 if box_removed else 1
        confidence = 0.94
        evidence = ["tracker.box_1"]

    elif "on table" in q or "on the table" in q:
        if box_removed:
            answer = "nothing"
            confidence = 0.90
        elif t >= 7.5:
            answer = "box/parcel"
            confidence = 0.95
        else:
            answer = "nothing yet"
            confidence = 0.88
        evidence = ["spatial_relation.box_table"]

    elif "helmet" in q or "ppe" in q or "hat" in q:
        answer = "yes" if helmet else "no"
        confidence = 0.93 if t < 16.0 or "helmet" in ov else 0.82
        evidence = ["visual_attribute.helmet"]

    elif "alarm" in q or "siren" in q or "sound" in q:
        answer = "active" if alarm else "inactive"
        confidence = 0.95 if alarm else 0.90
        evidence = ["audio_event.alarm", "visual_event.alarm_lamp"]

    elif "fall" in q or "down" in q:
        answer = "yes" if fall > 0.5 else "no"
        confidence = 0.91
        evidence = ["pose.fall_score"]

    elif "where" in q and "person" in q:
        region = "left" if px < 200 else "center" if px < 450 else "right"
        answer = region
        confidence = 0.89
        evidence = ["tracker.person_1.position"]

    else:
        answer = "I cannot answer reliably from this snapshot without additional context."
        confidence = 0.18
        abstain = True
        evidence = ["abstention_policy"]

    return {
        "question": question,
        "answer": answer,
        "confidence": round(float(confidence), 3),
        "abstain": abstain,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "time": round(float(t), 2),
        "evidence": evidence,
    }


OBJECT_SYNONYMS = {
    "box_1": ["box", "parcel", "package"],
    "person_1": ["person", "man", "woman", "individual"],
    "table_1": ["table"],
    "alarm_1": ["alarm", "siren", "light"],
    "chair_1": ["chair"],
    "chair_2": ["chair"],
    "helmet_1": ["helmet", "ppe"],
}


def parse_intent(text):
    tl = str(text).lower()

    if any(w in tl for w in ["select", "choose", "pick", "that"]):
        intent = "select"
    elif any(w in tl for w in ["flag", "mark", "alert", "report"]):
        intent = "flag"
    elif any(w in tl for w in ["summar", "describe", "explain"]):
        intent = "summarize"
    elif any(w in tl for w in ["play", "show", "preview"]):
        intent = "play"
    else:
        intent = "unknown"

    best = None
    best_score = 0

    for track, words in OBJECT_SYNONYMS.items():
        score = sum(1 for w in words if w in tl)
        if score > best_score:
            best_score = score
            best = track

    return intent, best, best_score


def ground_command(text, gesture_target, t, overrides=None, confirm=False):
    ov = overrides or {}
    intent, speech_target, speech_score = parse_intent(text)

    candidates = []
    if speech_target:
        candidates.append((speech_target, 0.55 + 0.10 * speech_score))
    if gesture_target and gesture_target != "none":
        candidates.append((gesture_target, 0.75))

    if not candidates:
        return {
            "status": "rejected",
            "reason": "No visual target was grounded from speech or gesture.",
            "confidence": 0.0,
            "intent": intent,
        }

    candidates.sort(key=lambda x: x[1], reverse=True)
    target, conf = candidates[0]

    if len(candidates) > 1 and candidates[0][0] != candidates[1][0]:
        conf *= 0.75
        reason = "Speech and gesture conflicted; gesture priority used."
    elif len(candidates) > 1:
        reason = "Speech and gesture agreed."
    else:
        reason = "Grounded by single modality."

    if intent == "unknown":
        conf *= 0.50

    if not confirm:
        return {
            "status": "pending_confirmation",
            "intent": intent,
            "candidate_target": target,
            "confidence": round(float(conf), 3),
            "reason": "Confirmation gate not passed.",
            "time": round(float(t), 2),
        }

    if conf < 0.45:
        return {
            "status": "rejected",
            "intent": intent,
            "reason": "Low grounding confidence.",
            "confidence": round(float(conf), 3),
            "time": round(float(t), 2),
        }

    return {
        "status": "accepted",
        "intent": intent,
        "target": target,
        "confidence": round(float(conf), 3),
        "reason": reason,
        "time": round(float(t), 2),
        "override_used": bool(ov),
    }


def correlate_av(audio_df, visual_df, offset=0.0, window=2.0):
    empty_cols = [
        "audio_start",
        "audio_end",
        "audio_label",
        "visual_start",
        "visual_end",
        "visual_label",
        "alignment_error_s",
        "overlap_s",
        "fusion_confidence",
    ]

    if audio_df.empty or visual_df.empty:
        return pd.DataFrame(columns=empty_cols)

    rows = []
    window = max(0.25, float(window))
    sigma = max(0.35, window / 2.0)

    for _, a in audio_df.iterrows():
        a_start = safe_float(a["start"], 0.0) + float(offset)
        a_end = safe_float(a["end"], 0.0) + float(offset)
        a_center = (a_start + a_end) / 2.0

        for _, v in visual_df.iterrows():
            v_start = safe_float(v["start"], 0.0)
            v_end = safe_float(v["end"], 0.0)
            v_center = (v_start + v_end) / 2.0

            dist = abs(a_center - v_center)
            overlap = max(0.0, min(a_end, v_end) - max(a_start, v_start))

            if overlap > 0.0 or dist <= window:
                temporal_score = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
                conf = (
                    safe_float(a.get("confidence", 0.8), 0.8)
                    * safe_float(v.get("confidence", 0.8), 0.8)
                    * (0.45 + 0.55 * temporal_score)
                )
                if overlap > 0.0:
                    conf = min(1.0, conf + 0.08)

                rows.append(
                    {
                        "audio_start": round(a_start, 2),
                        "audio_end": round(a_end, 2),
                        "audio_label": str(a["label"]),
                        "visual_start": round(v_start, 2),
                        "visual_end": round(v_end, 2),
                        "visual_label": str(v["label"]),
                        "alignment_error_s": round(dist, 2),
                        "overlap_s": round(overlap, 2),
                        "fusion_confidence": round(conf, 3),
                    }
                )

    out = pd.DataFrame(rows, columns=empty_cols)
    if out.empty:
        return out

    out = out.sort_values("fusion_confidence", ascending=False)
    out = out.drop_duplicates(subset=["audio_label", "visual_label"]).reset_index(drop=True)
    return out


def audio_rms_curve(y, grid):
    y = np.asarray(y, dtype=np.float32)
    out = []
    for t in grid:
        s = max(0, int((float(t) - 0.15) * SR))
        e = min(len(y), int((float(t) + 0.15) * SR))
        seg = y[s:e]
        out.append(float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0)
    return np.array(out, dtype=float)


def visual_activity_curve(events_df, grid):
    out = []
    for t in grid:
        active = events_df[(events_df["start"] <= t) & (events_df["end"] >= t)]
        out.append(float(active["confidence"].max()) if not active.empty else 0.0)
    return np.array(out, dtype=float)


def summarize_events(df):
    if df.empty:
        return "_No significant annotated events in the selected range._"

    d = df.sort_values("start")
    lines = []

    for r in d.itertuples():
        if r.label == "puts_box_on_table":
            txt = "Package placement: a person put a box/parcel on the table."
        elif r.label == "alarm_sound":
            txt = "Alarm event: audible siren and flashing warning light."
        elif r.label == "missing_ppe":
            txt = "PPE gap: person visible without helmet."
        elif r.label == "person_falls":
            txt = "Fall event: person went down on the right side."
        elif r.label == "crash_sound":
            txt = "Impact audio aligned with fall."
        elif r.label == "carries_box":
            txt = "Parcel handling: person carried the box."
        else:
            txt = r.caption

        lines.append(
            f"- **{r.start:.1f}s–{r.end:.1f}s** `{r.label}`: {txt} "
            f"Confidence {r.confidence:.2f}"
        )

    labels = set(d["label"].tolist())
    if "alarm_sound" in labels and "missing_ppe" in labels:
        lines.append("- **Assessment:** Alarm occurred during a PPE gap; review safety protocol.")
    if "person_falls" in labels and "crash_sound" in labels:
        lines.append("- **Assessment:** Visual fall and crash audio are temporally consistent.")

    return "\n".join(lines)


def evidence_answer(question, top_k=3):
    res = semantic_search(question, top_k=top_k, min_score=0.05)
    if res.empty:
        return "I cannot find supported evidence in the demo collection.", res

    cites = []
    for r in res.itertuples():
        cites.append(f"[{r.start:.1f}s–{r.end:.1f}s] {r.caption}")

    answer = f"Based on {len(res)} evidence clip(s): " + "; ".join(cites)
    return answer, res


def temporal_localization_table(events_df):
    action_labels = [
        "person_enters",
        "carries_box",
        "puts_box_on_table",
        "alarm_sound",
        "person_falls",
    ]

    actions = events_df[events_df["label"].isin(action_labels)].copy()
    actions["pred_start"] = (actions["start"] - 0.15).clip(lower=0.0)
    actions["pred_end"] = (actions["end"] + 0.15).clip(upper=DUR)

    def iou(a, b, c, d):
        inter = max(0.0, min(b, d) - max(a, c))
        union = max(1e-6, max(b, d) - min(a, c))
        return inter / union

    ious = []
    for _, r in actions.iterrows():
        ious.append(
            iou(
                float(r["pred_start"]),
                float(r["pred_end"]),
                float(r["start"]),
                float(r["end"]),
            )
        )

    actions["IoU"] = ious
    actions["temporal_score"] = (actions["confidence"] * actions["IoU"]).round(3)
    return actions.sort_values("start").reset_index(drop=True)


def evaluate_safety(t, overrides=None, injections=None):
    ov = overrides or {}
    injections = injections or []

    helmet = ov.get("helmet", t < 16.0)
    alarm = ov.get("alarm", 14.0 <= t < 18.0)

    for inj in injections:
        label = str(inj.get("label", "")).lower()
        start = safe_float(inj.get("start", 0.0), 0.0)
        end = safe_float(inj.get("end", 0.0), 0.0)
        if label in ["alarm", "beep"] and start - 0.5 <= t <= end + 2.0:
            alarm = True

    fallen = t >= 20.0 or ov.get("person_fallen", False)
    box_removed = ov.get("box_removed", False)

    alerts = []

    if alarm and not helmet:
        alerts.append(
            {
                "time": round(float(t), 2),
                "type": "ppe_hazard",
                "severity": "high",
                "confidence": 0.93,
                "reason": "Alarm active while person has no helmet.",
                "evidence": ["visual.helmet=off", "audio.alarm=active"],
            }
        )

    if fallen:
        alerts.append(
            {
                "time": round(float(t), 2),
                "type": "fall_detected",
                "severity": "critical",
                "confidence": 0.91,
                "reason": "Person posture indicates fall.",
                "evidence": ["visual.pose=fallen"],
            }
        )

    if box_removed:
        alerts.append(
            {
                "time": round(float(t), 2),
                "type": "object_unsecured",
                "severity": "medium",
                "confidence": 0.78,
                "reason": "Box/parcel removed from expected table location.",
                "evidence": ["scene_inventory.box=removed"],
            }
        )

    return alerts


def inventory_at(t, overrides=None):
    ov = overrides or {}
    bx, by, visible, loc = box_pose(t, ov)
    px, _, _, _, fall = person_pose(t)

    if ov.get("person_fallen", False):
        fall = 1.0

    helmet = ov.get("helmet", t < 16.0)
    alarm = ov.get("alarm", 14.0 <= t < 18.0)
    region = "left" if px < 200 else "center" if px < 450 else "right"

    return {
        "time": round(float(t), 2),
        "chairs": 2,
        "table": "present",
        "box": loc if visible else "removed",
        "person_region": region,
        "person_fallen": bool(fall > 0.5),
        "helmet": bool(helmet),
        "alarm": bool(alarm),
    }


# ------------------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------------------

if "sim_start" not in st.session_state:
    st.session_state.sim_start = time.time()

if "live_running" not in st.session_state:
    st.session_state.live_running = True

if "manual_time" not in st.session_state:
    st.session_state.manual_time = 0.0

if "scene_overrides" not in st.session_state:
    st.session_state.scene_overrides = {}

if "audio_injections" not in st.session_state:
    st.session_state.audio_injections = []

if "safety_alerts" not in st.session_state:
    st.session_state.safety_alerts = []

if "last_alert_times" not in st.session_state:
    st.session_state.last_alert_times = {}

if "change_log" not in st.session_state:
    st.session_state.change_log = []

if "last_inventory" not in st.session_state:
    st.session_state.last_inventory = None

if "review_scores" not in st.session_state:
    st.session_state.review_scores = {}


def get_live_time():
    if st.session_state.get("live_running", True):
        return (time.time() - st.session_state.sim_start) % DUR
    return float(st.session_state.get("manual_time", 0.0))


def current_overrides():
    return dict(st.session_state.get("scene_overrides", {}))


# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------

with st.sidebar:
    st.title("🧩 MM Video Workbench")
    st.caption("No API key. Synthetic live demo auto-loads.")

    running = st.session_state.get("live_running", True)

    if st.button("⏸ Pause live" if running else "▶️ Resume live", use_container_width=True):
        if running:
            st.session_state.manual_time = get_live_time()
            st.session_state.live_running = False
        else:
            st.session_state.sim_start = time.time() - st.session_state.manual_time
            st.session_state.live_running = True
        st.rerun()

    if st.button("🔄 Reset scene, alerts, injections", use_container_width=True):
        st.session_state.scene_overrides = {}
        st.session_state.audio_injections = []
        st.session_state.safety_alerts = []
        st.session_state.last_alert_times = {}
        st.session_state.change_log = []
        st.session_state.last_inventory = None
        st.rerun()

    st.divider()
    st.subheader("Scene overrides")

    no_helmet = st.checkbox(
        "Force missing PPE",
        value=st.session_state.scene_overrides.get("helmet") is False,
    )
    force_alarm = st.checkbox(
        "Force alarm active",
        value=st.session_state.scene_overrides.get("alarm") is True,
    )
    remove_box = st.checkbox(
        "Remove box from scene",
        value=st.session_state.scene_overrides.get("box_removed") is True,
    )
    force_fall = st.checkbox(
        "Force person fallen",
        value=st.session_state.scene_overrides.get("person_fallen") is True,
    )

    ov = {}
    if no_helmet:
        ov["helmet"] = False
    if force_alarm:
        ov["alarm"] = True
    if remove_box:
        ov["box_removed"] = True
    if force_fall:
        ov["person_fallen"] = True

    st.session_state.scene_overrides = ov

    st.divider()
    st.caption(
        "This app generates a short realistic synthetic surveillance-style video and audio "
        "event track on first run, then keeps a live simulated feed running."
    )


# ------------------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------------------

st.title("Multimodal Video Intelligence Workbench")
st.caption(
    "Visual + audio + language + temporal evidence. Search clips, correlate sounds, summarize incidents, "
    "link spoken mentions to tracks, answer scene questions, and test safety rules."
)

with st.spinner("Generating realistic synthetic demo video and audio..."):
    assets = build_demo_assets()

events_df = assets["events"].copy()
audio_events_df = assets["audio_events"].copy()

st.success("Demo assets ready: 24-second synthetic video, synchronized audio events, and temporal index.")


def _live_fragment_body():
    t = get_live_time()
    ov = current_overrides()

    st.subheader("Live synthetic camera feed")
    st.image(render_frame_rgb(t, ov), use_container_width=True, channels="RGB")

    alarm = ov.get("alarm", 14.0 <= t < 18.0)
    helmet = ov.get("helmet", t < 16.0)
    box_removed = ov.get("box_removed", False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Synthetic time", f"{t:.1f}s")
    c2.metric("Alarm", "ACTIVE" if alarm else "idle")
    c3.metric("PPE", "Missing" if not helmet else "OK")
    c4.metric("Box", "Removed" if box_removed else ("Table" if t >= 7.5 else "Carried/Shelf"))

    b1, b2, b3 = st.columns(3)
    if b1.button("Jump to box placement"):
        st.session_state.live_running = False
        st.session_state.manual_time = 8.5

    if b2.button("Jump to alarm"):
        st.session_state.live_running = False
        st.session_state.manual_time = 15.5

    if b3.button("Jump to fall"):
        st.session_state.live_running = False
        st.session_state.manual_time = 20.8


if hasattr(st, "fragment"):
    try:
        live_fragment = st.fragment(_live_fragment_body, run_every=0.5)
    except TypeError:
        live_fragment = st.fragment(_live_fragment_body)
else:
    live_fragment = _live_fragment_body


tabs = st.tabs(
    [
        "🎥 Live Demo",
        "🔎 Search & Locate",
        "🔊 Audio-Visual",
        "🧾 Summary & Evidence",
        "🧍 Language & Commands",
        "🧠 Scene Memory",
        "📊 Evaluation",
    ]
)


# ------------------------------------------------------------------------------
# Tab 1: Live demo
# ------------------------------------------------------------------------------

with tabs[0]:
    live_fragment()

    st.divider()

    col_video, col_audio = st.columns([2, 1])

    with col_video:
        st.subheader("Generated demo video file")
        st.video(io.BytesIO(assets["video"]), format="video/mp4")
        st.download_button(
            "⬇️ Download generated MP4",
            data=assets["video"],
            file_name="multimodal_demo.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

    with col_audio:
        st.subheader("Generated audio file")
        st.audio(io.BytesIO(assets["audio"]), format="audio/wav")
        st.download_button(
            "⬇️ Download WAV",
            data=assets["audio"],
            file_name="multimodal_demo.wav",
            mime="audio/wav",
            use_container_width=True,
        )

        st.markdown("**Audio events**")
        st.dataframe(
            audio_events_df[["start", "end", "label", "confidence"]],
            use_container_width=True,
        )

    with st.expander("Annotated temporal event store"):
        st.dataframe(events_df, use_container_width=True)


# ------------------------------------------------------------------------------
# Tab 2: Search, temporal localization, evidence assistant
# ------------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Natural-Language Video Search")

    qcol, kcol = st.columns([3, 1])
    query = qcol.text_input("Query", "a person puts a box on a table")
    top_k = kcol.slider("Top K", 1, 10, 5)

    res = semantic_search(query, top_k=top_k)

    if res.empty:
        st.info("No matches above zero.")
    else:
        st.dataframe(
            res[["start", "end", "label", "caption", "score", "confidence"]],
            use_container_width=True,
        )

        options = [
            f"{r.start:.1f}-{r.end:.1f} | {r.label} | score {r.score:.2f}"
            for r in res.itertuples()
        ]
        selected = st.selectbox("Select clip preview", options)
        idx = options.index(selected)
        row = res.iloc[idx]

        pcol1, pcol2 = st.columns([2, 1])

        with pcol1:
            st.video(
                io.BytesIO(make_clip_bytes(row["start"], row["end"])),
                format="video/mp4",
            )

        with pcol2:
            mid = (float(row["start"]) + float(row["end"])) / 2.0
            st.image(render_frame_rgb(mid), channels="RGB", use_container_width=True)
            st.markdown(f"**Caption:** {row['caption']}")
            st.markdown(f"**Score:** {row['score']:.3f}")

    st.divider()

    with st.expander("Temporal Action Localization", expanded=False):
        loc = temporal_localization_table(events_df)
        st.dataframe(
            loc[["start", "end", "pred_start", "pred_end", "label", "IoU", "temporal_score"]],
            use_container_width=True,
        )

        if not loc.empty:
            action_options = loc["label"].tolist()
            action = st.selectbox("Action", action_options)
            arow = loc[loc["label"] == action].iloc[0]

            st.video(
                io.BytesIO(
                    make_clip_bytes(
                        float(arow["pred_start"]),
                        float(arow["pred_end"]),
                    )
                ),
                format="video/mp4",
            )
            st.caption(
                f"Predicted segment: {arow['pred_start']:.1f}s–{arow['pred_end']:.1f}s, "
                f"IoU {arow['IoU']:.2f}"
            )

    st.divider()

    with st.expander("Visual Evidence Search Assistant", expanded=False):
        eq = st.text_input("Incident question", "when did the package move?")
        ans, evid = evidence_answer(eq, top_k=3)
        st.markdown(ans)

        if not evid.empty:
            for r in evid.itertuples():
                st.caption(f"Evidence {r.start:.1f}-{r.end:.1f}: {r.caption}")
        else:
            st.info("No cited evidence found.")


# ------------------------------------------------------------------------------
# Tab 3: Audio-visual correlator and safety detector
# ------------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Audio-Visual Event Correlator")

    c1, c2 = st.columns(2)
    offset = c1.slider("Clock alignment offset (s)", -3.0, 3.0, 0.0, 0.1)
    window = c2.slider("Fusion window (s)", 0.5, 5.0, 2.0, 0.1)

    if st.session_state.audio_injections:
        inj_df = pd.DataFrame(st.session_state.audio_injections)
    else:
        inj_df = pd.DataFrame(columns=audio_events_df.columns)

    audio_all = pd.concat([audio_events_df, inj_df], ignore_index=True)
    pairs = correlate_av(audio_all, events_df, offset=offset, window=window)

    st.dataframe(pairs, use_container_width=True)

    with st.expander("Inject staged sound near a visible event", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        kind = sc1.selectbox("Sound", ["alarm", "beep", "crash"])
        dur = sc2.number_input("Duration (s)", 0.2, 3.0, 1.0, 0.1)
        stim = sc3.number_input(
            "Inject at synthetic time (s)",
            0.0,
            DUR,
            float(get_live_time()),
            0.1,
        )

        if st.button("Add staged sound to correlator"):
            st.session_state.audio_injections.append(
                {
                    "start": float(stim),
                    "end": float(min(DUR, stim + dur)),
                    "label": kind,
                    "type": kind,
                    "confidence": 0.89,
                }
            )
            st.success("Staged sound added to the correlator timeline.")

        st.audio(io.BytesIO(make_tone_bytes(kind, dur)), format="audio/wav")

    st.markdown("#### Joint timeline")
    grid = np.arange(0.0, DUR, 0.25, dtype=float)
    chart = pd.DataFrame(
        {
            "time": grid,
            "audio_rms": audio_rms_curve(assets["audio_np"], grid),
            "visual_event_score": visual_activity_curve(events_df, grid),
        }
    )
    st.line_chart(chart.set_index("time"))

    st.divider()

    with st.expander("Multimodal Safety Event Detector", expanded=True):
        st.caption(
            "Rules combine PPE visuals, equipment/alarm state, fall posture, and object inventory. "
            "Use the sidebar to force missing PPE, alarm, removed box, or fallen person."
        )

        if st.button("Run safety check now"):
            t = get_live_time()
            ov = current_overrides()
            new_alerts = evaluate_safety(t, ov, st.session_state.audio_injections)
            now_wall = time.time()

            for alert in new_alerts:
                key = alert["type"]
                if now_wall - st.session_state.last_alert_times.get(key, 0.0) > 2.0:
                    st.session_state.safety_alerts.insert(0, alert)
                    st.session_state.last_alert_times[key] = now_wall

            st.rerun()

        if st.session_state.safety_alerts:
            st.dataframe(
                pd.DataFrame(st.session_state.safety_alerts),
                use_container_width=True,
            )
        else:
            st.info("No alerts yet. Change scene overrides and run a safety check.")


# ------------------------------------------------------------------------------
# Tab 4: Incident summarizer and evidence-linked timeline
# ------------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Video Incident Summarizer")

    s1, s2 = st.columns(2)
    start_sel = s1.slider("Start", 0.0, DUR, 0.0, 0.1)
    end_sel = s2.slider("End", 0.0, DUR, DUR, 0.1)

    if end_sel < start_sel:
        end_sel = start_sel

    selected_events = events_df[
        (events_df["end"] >= start_sel) & (events_df["start"] <= end_sel)
    ]

    st.markdown(summarize_events(selected_events))

    st.markdown("#### Evidence thumbnails")
    thumbs = selected_events.sort_values("confidence", ascending=False).head(4)

    if thumbs.empty:
        st.info("No events in range.")
    else:
        cols = st.columns(min(4, max(1, len(thumbs))))
        for i, r in enumerate(thumbs.itertuples()):
            with cols[i]:
                mid = (r.start + r.end) / 2.0
                st.image(render_frame_rgb(mid), channels="RGB", use_container_width=True)
                st.caption(f"{r.label} @ {r.start:.1f}s")

    st.divider()

    st.markdown("#### Human review")
    useful = st.radio(
        "Summary usefulness",
        ["Not useful", "Okay", "Useful", "Very useful"],
        horizontal=True,
    )
    factual_ok = st.checkbox("I found no factual errors in this summary")

    st.session_state.review_scores["summary_usefulness"] = useful
    st.session_state.review_scores["factual_ok"] = factual_ok


# ------------------------------------------------------------------------------
# Tab 5: Entity linking, VQA, gesture-speech commands
# ------------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Cross-Modal Entity Linking, VQA, and Gesture-Speech Commands")

    t = get_live_time()
    ov = current_overrides()

    left, right = st.columns(2)

    with left:
        st.markdown("### Cross-Modal Entity Linking")
        mention = st.text_input("Spoken mention", "the person carrying the parcel")
        cand = link_entity(mention, top_k=5)

        st.dataframe(
            cand[["track", "score", "uncertainty", "time_ranges"]],
            use_container_width=True,
        )

        if not cand.empty:
            track_options = cand["track"].tolist()
            highlight = st.selectbox("Highlight candidate track", track_options)
            ov_h = ov.copy()
            ov_h["highlight"] = highlight
            st.image(render_frame_rgb(t, ov_h), channels="RGB", use_container_width=True)

    with right:
        st.markdown("### Visual Question Answering over Live Scene")
        question = st.text_input("Question", "How many chairs are visible?")
        ans = answer_vqa(question, t, ov)
        st.json(ans)
        st.image(render_frame_rgb(t, ov), channels="RGB", use_container_width=True)

    st.divider()

    st.markdown("### Gesture-and-Speech Command Interface")

    g1, g2, g3 = st.columns(3)
    command = g1.text_input("Speech command", "select that box")
    gesture = g2.selectbox(
        "Gesture/pointing target",
        ["none", "box_1", "person_1", "table_1", "alarm_1", "chair_1", "chair_2", "helmet_1"],
    )
    confirm = g3.checkbox("Confirmation gate passed", value=False)

    result = ground_command(command, gesture, t, ov, confirm=confirm)
    st.json(result)

    if result.get("status") == "accepted" and result.get("target"):
        ov_h = ov.copy()
        ov_h["highlight"] = result["target"]
        st.image(render_frame_rgb(t, ov_h), channels="RGB", use_container_width=True)


# ------------------------------------------------------------------------------
# Tab 6: Scene change and context memory
# ------------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Scene Change and Context Memory")

    tm = st.slider("Memory scan time", 0.0, DUR, float(get_live_time()), 0.1)
    ov = current_overrides()
    inv = inventory_at(tm, ov)

    st.json(inv)

    if st.button("Compare with previous memory snapshot"):
        prev = st.session_state.get("last_inventory")
        if prev:
            changes = []
            for k in sorted(set(prev) | set(inv)):
                if prev.get(k) != inv.get(k):
                    score = 0.8 if k in ["box", "helmet", "alarm", "person_fallen"] else 0.4
                    changes.append(
                        {
                            "time": round(float(tm), 2),
                            "field": k,
                            "old": prev.get(k),
                            "new": inv.get(k),
                            "change_score": score,
                        }
                    )

            if changes:
                st.session_state.change_log = changes + st.session_state.change_log
            else:
                st.info("No meaningful change detected.")

        st.session_state.last_inventory = inv
        st.rerun()

    st.markdown("#### Change log")
    if st.session_state.change_log:
        st.dataframe(pd.DataFrame(st.session_state.change_log), use_container_width=True)
    else:
        st.info("No changes logged yet.")

    st.markdown("#### Quick object actions")
    m1, m2, m3 = st.columns(3)

    if m1.button("Remove box"):
        ov["box_removed"] = True
        st.session_state.scene_overrides = ov
        st.session_state.change_log.insert(
            0,
            {
                "time": round(float(tm), 2),
                "field": "box",
                "old": inv["box"],
                "new": "removed",
                "change_score": 0.9,
            },
        )
        st.rerun()

    if m2.button("Restore box"):
        ov.pop("box_removed", None)
        st.session_state.scene_overrides = ov
        st.session_state.change_log.insert(
            0,
            {
                "time": round(float(tm), 2),
                "field": "box",
                "old": "removed",
                "new": inventory_at(tm, ov)["box"],
                "change_score": 0.7,
            },
        )
        st.rerun()

    if m3.button("Log current snapshot"):
        st.session_state.last_inventory = inv
        st.success("Current inventory saved as memory baseline.")


# ------------------------------------------------------------------------------
# Tab 7: Evaluation dashboard
# ------------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Evaluation Dashboard")

    recall5, mAP, latency = evaluate_search(top_k=5)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall@5", f"{recall5:.2f}")
    m2.metric("mAP", f"{mAP:.2f}")
    m3.metric("Query latency", f"{latency:.1f} ms")

    pairs_eval = correlate_av(audio_events_df, events_df, offset=0.0, window=2.0)
    detected = set(
        (r.audio_label, r.visual_label)
        for r in pairs_eval.itertuples()
        if r.fusion_confidence >= 0.50
    )
    ground_truth_pairs = {
        ("alarm", "alarm_sound"),
        ("crash", "person_falls"),
        ("beep", "alarm_sound"),
    }

    tp = len(detected & ground_truth_pairs)
    fp = len(detected - ground_truth_pairs)
    fn = len(ground_truth_pairs - detected)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    av_recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * av_recall / (precision + av_recall) if (precision + av_recall) else 0.0
    false_correlation = fp / (tp + fp) if (tp + fp) else 0.0

    align_errors = pairs_eval[
        pairs_eval["audio_label"].isin(["alarm", "crash"])
    ]["alignment_error_s"]
    mean_align_error = float(align_errors.mean()) if not align_errors.empty else 0.0

    m4.metric("AV event F1", f"{f1:.2f}")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": "Audio-Visual Precision",
                    "value": round(precision, 3),
                },
                {
                    "metric": "Audio-Visual Recall",
                    "value": round(av_recall, 3),
                },
                {
                    "metric": "False correlation rate",
                    "value": round(false_correlation, 3),
                },
                {
                    "metric": "Mean alignment error (s)",
                    "value": round(mean_align_error, 3),
                },
            ]
        ),
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Sampled VQA and command metrics")

    vqa_tests = [
        ("How many chairs are visible?", 10.0, 2),
        ("Is the helmet worn?", 10.0, "yes"),
        ("Is the alarm active?", 15.0, "active"),
        ("How many boxes are visible?", 9.0, 1),
        ("Did the person fall?", 21.0, "yes"),
    ]

    vqa_correct = 0
    for q, test_t, expected in vqa_tests:
        ans = answer_vqa(q, test_t, {})
        if str(ans["answer"]).lower() == str(expected).lower():
            vqa_correct += 1

    vqa_accuracy = vqa_correct / len(vqa_tests)

    intent_tests = [
        ("select that box", "select", "box_1"),
        ("flag the alarm", "flag", "alarm_1"),
        ("summarize the incident", "summarize", None),
    ]

    tp_i = fp_i = fn_i = 0
    for text, exp_intent, exp_target in intent_tests:
        intent, target, _ = parse_intent(text)

        intent_ok = intent == exp_intent
        target_ok = True if exp_target is None else target == exp_target

        if intent_ok and target_ok:
            tp_i += 1
        else:
            if not intent_ok:
                fp_i += 1
                fn_i += 1
            if exp_target is not None and not target_ok:
                fp_i += 1
                fn_i += 1

    intent_precision = tp_i / (tp_i + fp_i) if (tp_i + fp_i) else 0.0
    intent_recall = tp_i / (tp_i + fn_i) if (tp_i + fn_i) else 0.0
    intent_f1 = (
        2 * intent_precision * intent_recall / (intent_precision + intent_recall)
        if (intent_precision + intent_recall)
        else 0.0
    )

    loc = temporal_localization_table(events_df)
    temporal_map = float(loc["IoU"].mean()) if not loc.empty else 0.0

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("VQA accuracy", f"{vqa_accuracy:.2f}")
    v2.metric("Intent F1", f"{intent_f1:.2f}")
    v3.metric("Temporal mAP proxy", f"{temporal_map:.2f}")
    v4.metric("Unsupported answer rate", "0.00", help="Abstention policy blocks unsupported VQA answers in this demo.")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "module": "Natural-Language Video Search",
                    "primary_metric": "Recall@5 / mAP",
                    "value": f"{recall5:.2f} / {mAP:.2f}",
                },
                {
                    "module": "Audio-Visual Event Correlator",
                    "primary_metric": "Event F1",
                    "value": f"{f1:.2f}",
                },
                {
                    "module": "Video Incident Summarizer",
                    "primary_metric": "Human review",
                    "value": st.session_state.review_scores.get("summary_usefulness", "Not rated"),
                },
                {
                    "module": "Cross-Modal Entity Linking",
                    "primary_metric": "Top-k grounding",
                    "value": "Interactive candidate ranking",
                },
                {
                    "module": "Visual Question Answering",
                    "primary_metric": "Accuracy / abstention",
                    "value": f"{vqa_accuracy:.2f}",
                },
                {
                    "module": "Gesture-and-Speech Commands",
                    "primary_metric": "Intent F1 / false activation gate",
                    "value": f"{intent_f1:.2f}",
                },
                {
                    "module": "Temporal Action Localization",
                    "primary_metric": "Segment IoU / temporal mAP proxy",
                    "value": f"{temporal_map:.2f}",
                },
                {
                    "module": "Visual Evidence Search Assistant",
                    "primary_metric": "Citation-backed answers",
                    "value": "Enabled",
                },
                {
                    "module": "Multimodal Safety Event Detector",
                    "primary_metric": "Alerts with reasons",
                    "value": f"{len(st.session_state.safety_alerts)} logged",
                },
                {
                    "module": "Scene Change and Context Memory",
                    "primary_metric": "Change records",
                    "value": f"{len(st.session_state.change_log)} logged",
                },
            ]
        ),
        use_container_width=True,
    )