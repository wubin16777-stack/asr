# -*- coding: utf-8 -*-
"""字幕工作室  (需要系统里装好 ffmpeg, 导出MP4用)
  (DJ混音歌曲 -> 人声分离 -> 唱歌分类 -> 注入歌词的字级ASR)
安装: pip install PySide6 requests numpy soundfile torch torchaudio demucs faster-whisper panns-inference librosa pypinyin
运行: python subtitle_studio.py
"""
import sys, os, re, glob, math, html, json, subprocess, tempfile, difflib
import requests
from dataclasses import dataclass, field
import numpy as np
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


@dataclass
class Cue:
    start: float
    end: float
    text: str
    track: int = 0
    words: list = field(default_factory=list)  # [(字, 开始, 结束)] 字级时间窗


STYLE = dict(font="Microsoft YaHei", size=56, color="#FFFFFF", color2="#FFD54A", fill_mode="纯色", cycle_speed=1.0,
             stroke="#000000", sw=3, bold=True, italic=False, x=0.5, y=0.85, anim_in="淡入", anim_out="淡出", dur=0.4,
             bg="#000000", bg_img="", karaoke=True, hl="#FFD54A")
PINYIN_FIX_THRESHOLD = 0.60  # 逐句独立匹配参考歌词的最低综合相似度
STYLE_FILE = os.path.join(os.path.expanduser("~"), ".subtitle_studio_style.json")  # 样式自动保存, 批量时直接用
try: STYLE.update(json.load(open(STYLE_FILE, encoding="utf-8")))
except Exception: pass
STYLE.setdefault("color2", "#FFD54A"); STYLE.setdefault("fill_mode", "纯色"); STYLE.setdefault("cycle_speed", 1.0); STYLE.setdefault("italic", False)
STYLE.setdefault("title_text", "")
STYLE.setdefault("title_style", {"font":"Microsoft YaHei", "size":42, "color":"#FFFFFF", "color2":"#FFD54A", "fill_mode":"纯色", "cycle_speed":1.0, "stroke":"#000000", "sw":2, "bold":True, "italic":False, "x":0.5, "y":0.12})
STYLE.setdefault("gradient_angle", 45.0)
STYLE.setdefault("letter_spacing", 0.0)
STYLE["title_style"].setdefault("gradient_angle",45.0); STYLE["title_style"].setdefault("layout_mode","横排")
STYLE["title_style"].setdefault("letter_spacing",0.0)
STYLE["title_style"].setdefault("loop_anim","无"); STYLE["title_style"].setdefault("loop_speed",1.0); STYLE["title_style"].setdefault("stay",0.0); STYLE["title_style"].setdefault("interval",0.0)
STYLE.setdefault("theme", "专业深色")
STYLE.setdefault("layout_mode", "横排")
STYLE.setdefault("current_scale", 30); STYLE.setdefault("other_opacity", 65)
STYLE.setdefault("in_anim", "淡入"); STYLE.setdefault("in_speed", 1.0)
STYLE.setdefault("out_anim", "淡出"); STYLE.setdefault("out_speed", 1.0)
STYLE.setdefault("loop_anim", "无"); STYLE.setdefault("loop_speed", 1.0)
STYLE.setdefault("current_row", "循环")
STYLE.setdefault("lyric_slots", [{"enabled": i < 3, "x": 0.5, "y": 0.68+i*0.09, "angle": 0.0} for i in range(5)])
while len(STYLE["lyric_slots"]) < 5: STYLE["lyric_slots"].append({"enabled":False,"x":0.5,"y":0.5,"angle":0.0})
for _slot in STYLE["lyric_slots"]: _slot.pop("current", None)
ANIM_IN = ["无", "淡入", "上滑入", "下滑入", "左滑入", "右滑入", "放大入", "缩小入", "旋转入", "弹跳入", "翻转入", "打字机"]
ANIM_OUT = ["无", "淡出", "上滑出", "下滑出", "左滑出", "右滑出", "放大出", "缩小出", "旋转出", "翻转出", "溶解出"]
ANIM_LOOP = ["无", "呼吸", "上下浮动", "左右摆动", "轻微旋转", "闪烁"]
PRESETS = {"淡入淡出": ("淡入", "淡出"), "综艺弹跳": ("弹跳入", "缩小出"), "上滑下滑": ("上滑入", "下滑出"),
           "放大冲击": ("放大入", "放大出"), "卡拉OK打字": ("打字机", "无")}
THEMES = {
    "专业深色": """QWidget{background:#171923;color:#e9ecf5;font-size:13px} QGroupBox,QTabWidget::pane{border:1px solid #34384a;border-radius:8px} QLineEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QTableWidget{background:#202331;color:#f4f5fa;border:1px solid #3c4257;border-radius:5px;padding:5px} QPushButton{background:#30364b;color:#f4f5fa;border:1px solid #4a5270;border-radius:5px;padding:6px 10px} QPushButton:hover{background:#7257d9} QTabBar::tab{background:#24283a;padding:9px 14px} QTabBar::tab:selected{background:#7657e8;color:white} QProgressBar{border:1px solid #41475d;border-radius:5px;text-align:center} QProgressBar::chunk{background:#7657e8}""",
    "紫色工作室": """QWidget{background:#201b2e;color:#f5efff;font-size:13px} QLineEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QTableWidget{background:#2c2540;color:#fff;border:1px solid #5e4b86;border-radius:6px;padding:5px} QPushButton{background:#3d2e61;color:#fff;border:1px solid #7455bd;border-radius:6px;padding:6px 10px} QPushButton:hover{background:#8b5cf6} QTabBar::tab{background:#302543;padding:9px 14px} QTabBar::tab:selected{background:#8b5cf6;color:white} QProgressBar::chunk{background:#a855f7}""",
    "浅色简洁": """QWidget{background:#f3f5f8;color:#202532;font-size:13px} QLineEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QTableWidget{background:white;color:#202532;border:1px solid #cbd2df;border-radius:5px;padding:5px} QPushButton{background:#ffffff;color:#283247;border:1px solid #c4ccda;border-radius:5px;padding:6px 10px} QPushButton:hover{background:#e7eaff} QTabBar::tab{background:#e2e6ef;padding:9px 14px} QTabBar::tab:selected{background:#536dfe;color:white} QProgressBar::chunk{background:#536dfe}""",
}
COLOR_PRESETS = {"白金":("#FFFFFF","#FFD54A"),"樱花":("#FF8FB3","#FFF0F6"),"海蓝":("#42D9FF","#4263FF"),"紫霞":("#C084FC","#6366F1"),"青柠":("#D9F99D","#22C55E"),"火焰":("#FFF7AD","#F97316"),"彩虹":("#FF4DDE","#40C9FF"),"黑金":("#FFD166","#8B5E00")}


def fmt(t):
    m = int(t // 60)
    return f"{m:02d}:{t - m * 60:05.2f}"


def parse(s):
    if ":" in s:
        m, x = s.split(":")
        return int(m) * 60 + float(x)
    return float(s)


def anim_state(st, c, t):
    """返回 (透明度, 垂直偏移px, 缩放, 显示字符比例)"""
    a, oy, sc, ch = 1.0, 0.0, 1.0, 1.0
    d = max(st["dur"], 0.01)
    p = min(1, max(0, (t - c.start) / d))
    q = min(1, max(0, (c.end - t) / d))
    e, e2 = 1 - (1 - p) ** 3, 1 - (1 - q) ** 3
    ai, ao = st["anim_in"], st["anim_out"]
    if ai == "淡入": a *= e
    elif ai == "上滑入": a *= e; oy += (1 - e) * 60
    elif ai == "下滑入": a *= e; oy -= (1 - e) * 60
    elif ai == "左滑入": a *= e
    elif ai == "右滑入": a *= e
    elif ai == "放大入": a *= e; sc *= 0.3 + 0.7 * e
    elif ai == "缩小入": a *= e; sc *= 1.8 - 0.8 * e
    elif ai == "旋转入": a *= e; sc *= .8+.2*e
    elif ai == "翻转入": a *= e; sc *= .2+.8*e
    elif ai == "弹跳入": a *= min(1, p * 3); sc *= 1 - (1 - p) * abs(math.cos(p * math.pi * 2.5))
    elif ai == "打字机": ch = p
    if ao == "淡出": a *= e2
    elif ao == "上滑出": a *= e2; oy -= (1 - e2) * 60
    elif ao == "下滑出": a *= e2; oy += (1 - e2) * 60
    elif ao in ("左滑出","右滑出"): a *= e2
    elif ao == "放大出": a *= e2; sc *= 1 + (1 - e2) * 0.8
    elif ao == "缩小出": a *= e2; sc *= 0.2 + 0.8 * e2
    elif ao in ("旋转出","翻转出","溶解出"): a *= e2
    return a, oy, sc, ch


def text_brush(st, rect, t):
    """根据纯色/渐变/循环跑色模式生成字幕填充画刷。"""
    mode=st.get("fill_mode", "纯色")
    c1=QColor(st.get("color", "#FFFFFF")); c2=QColor(st.get("color2", "#FFD54A"))
    ang=math.radians(float(st.get("gradient_angle",45.0))); cx=rect.center().x(); cy=rect.center().y(); dx=math.cos(ang)*rect.width()/2; dy=math.sin(ang)*rect.height()/2
    if mode=="渐变色":
        g=QLinearGradient(cx-dx,cy-dy,cx+dx,cy+dy); g.setColorAt(0,c1); g.setColorAt(1,c2)
        return QBrush(g)
    if mode=="循环跑色":
        speed=float(st.get("cycle_speed",1.0)); h=(c1.hue() if c1.hue()>=0 else 0)
        c1.setHsv((h+int(t*speed*180))%360, c1.saturation(), c1.value(), c1.alpha())
        c2.setHsv((h+int(t*speed*180)+100)%360, max(150,c2.saturation()), c2.value(), c2.alpha())
        g=QLinearGradient(cx-dx,cy-dy,cx+dx,cy+dy); g.setColorAt(0,c1); g.setColorAt(1,c2)
        return QBrush(g)
    return QBrush(c1)


def make_text_path(text, font, mode="横排", spacing=0.0):
    path=QPainterPath(); fm=QFontMetricsF(font); pos=0.0
    if mode=="竖排":
        for ch in text:
            path.addText(0,pos,font,ch); pos += fm.height()+spacing
    else:
        for ch in text:
            path.addText(pos,0,font,ch); pos += fm.horizontalAdvance(ch)+spacing
    return path


MIN_LINES, MIN_CHARS = 8, 60  # 歌词少于这个就认为有问题, 换下一个来源
# 只检测唱歌：优先保留最多两段较长、连续的人声演唱。
SING_THRESHOLD = 0.15  # PANNs Singing 分数阈值；可在此调高/调低
SING_MAX_GAP_SEC = 10.0  # 两个唱歌窗口之间小于此值时合并
SING_MAX_BLOCKS = 2  # 每首歌最多保留的连续唱歌大段数
# 下载歌词和 ASR 中常见的片尾水印/字幕署名，永不输出。
BANNED_ASR_PHRASES = (
    "中文字幕提供", "中文字幕志愿者", "优优独播剧场", "YoYo Television Series Exclusive",
    "字幕志愿者", "字幕组", "YoYo Television", "欢迎", "感谢收看", "请不吝点赞", "订阅", "转发", "打赏", "栏目",
)
UA = {"User-Agent": "Mozilla/5.0"}
CREDIT = re.compile(r"^\s*(作词|作曲|编曲|演唱|原唱|歌手|制作人?|监制|混音|母带|录音|和声|配唱|出品|发行|词|曲|OP|SP|ISRC|"
                    r"Lyrics?|Lyricist|Composer|Composed|Music|Arranger|Arranged|Producer|Produced|Mixed|Mixing|"
                    r"Mastered|Mastering|Vocals?|Written|Words|By)\s*(by)?\s*[:：]", re.I)


def clean_lyrics(t):
    """去掉时间窗/标签/作词作曲演唱等信息, 一句一行"""
    t = html.unescape(t or "")
    t = re.sub(r"\[[^\]]*\]|<\d+:\d+(?:[.:]\d+)?>", "", t)
    out = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln or CREDIT.match(ln) or re.fullmatch(r"[\W_]+", ln): continue
        if re.search(r"纯音乐|暂无歌词|没有填词|版权所有|未经.{0,6}许可", ln): continue
        out.append(ln)
    if out and re.match(r"^.{1,40}\s[-–—]\s.{1,40}$", out[0]): out.pop(0)  # 首行"歌名 - 歌手"
    return "\n".join(out)


def strip_banned_asr(text):
    """删除水印/署名行，避免歌词提示词和最终 ASR 输出污染字幕。"""
    kept=[]
    for ln in (text or "").splitlines():
        # 当前目标是只保留中文唱歌：含英文/拉丁字母的片段一律不显示。
        if re.search(r"[A-Za-z]", ln):
            continue
        if any(bad.lower() in ln.lower() for bad in BANNED_ASR_PHRASES):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _han(s):
    return "".join(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", s or ""))


def _pinyin_seq(s):
    from pypinyin import lazy_pinyin
    return lazy_pinyin(_han(s))


def _phonetic_score(a, b):
    pa, pb = _pinyin_seq(a), _pinyin_seq(b)
    if not pa or not pb:
        return 0.0
    return difflib.SequenceMatcher(None, pa, pb).ratio()


def _char_score(a, b):
    return difflib.SequenceMatcher(None, _han(a), _han(b)).ratio()


def _retime_corrected_text(cue, old_text, new_text):
    """用拼音对齐旧 ASR 字符和纠正后的歌词字符，为新增字符插入时间。"""
    old_chars=list(_han(old_text)); new_chars=list(_han(new_text))
    if not new_chars: return []
    old_words=[]
    for w, a, b in (cue.words or []):
        for ch in _han(w): old_words.append((ch, a, b))
    if len(old_words)!=len(old_chars):
        dt=max(0.01,(cue.end-cue.start)/max(1,len(old_chars)))
        old_words=[(ch,cue.start+i*dt,min(cue.end,cue.start+(i+1)*dt)) for i,ch in enumerate(old_chars)]
    if not old_words:
        dt=max(0.01,(cue.end-cue.start)/len(new_chars))
        return [(ch,cue.start+i*dt,min(cue.end,cue.start+(i+1)*dt)) for i,ch in enumerate(new_chars)]
    pa=_pinyin_seq(old_text); pb=_pinyin_seq(new_text)
    sm=difflib.SequenceMatcher(None,pa,pb); spans=[None]*len(new_chars)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if j1==j2: continue
        if tag in ('equal','replace'):
            a=old_words[min(i1,len(old_words)-1)][1] if i1<len(old_words) else cue.start
            b=old_words[max(i1,min(i2,len(old_words)))-1][2] if i2>0 else cue.end
            step=(b-a)/max(1,j2-j1)
            for j in range(j1,j2): spans[j]=(a+(j-j1)*step,a+(j-j1+1)*step)
    # 插入的新字：在最近的左右锚点之间均匀插入，避免全部挤在句末。
    for j in range(len(spans)):
        if spans[j] is not None: continue
        l=j-1
        while l>=0 and spans[l] is None: l-=1
        r=j+1
        while r<len(spans) and spans[r] is None: r+=1
        left=spans[l][1] if l>=0 else cue.start
        right=spans[r][0] if r<len(spans) else cue.end
        count=(r-l-1) if r<len(spans) else (len(spans)-l-1)
        step=(right-left)/max(1,count)
        spans[j]=(left+(j-l-1)*step,left+(j-l)*step)
    # 将锚点转成严格递增的边界，避免新增字得到 0 秒窗口。
    total=max(0.01, cue.end-cue.start)
    if len(new_chars)!=len(old_chars):
        # 多/少字时没有真实的新增字级声学边界：以原字时长作锚，新增字使用邻近平均时长，最后归一化到整句时长。
        weights=[total/max(1,len(new_chars))]*len(new_chars)
        for tag,i1,i2,j1,j2 in sm.get_opcodes():
            if j1==j2: continue
            if tag=='equal':
                for j,i in zip(range(j1,j2),range(i1,i2)):
                    if i<len(old_words): weights[j]=max(0.01,old_words[i][2]-old_words[i][1])
            elif tag=='replace' and i2>i1:
                span=sum(max(0.01,old_words[i][2]-old_words[i][1]) for i in range(i1,min(i2,len(old_words))))
                for j in range(j1,j2): weights[j]=max(0.01,span/max(1,j2-j1))
        scale=total/sum(weights)
        out=[]; t=cue.start
        for ch,w in zip(new_chars,weights):
            nt=t+w*scale; out.append((ch,t,min(cue.end,nt))); t=nt
        out[-1]=(out[-1][0],out[-1][1],cue.end)
        return out
    centers=[(a+b)/2 for a,b in spans]
    min_gap=max(0.01, total/(len(centers)*20))
    if total < min_gap*len(centers)*1.1:
        step=total/len(centers)
        return [(ch,cue.start+i*step,cue.start+(i+1)*step) for i,ch in enumerate(new_chars)]
    for i in range(1,len(centers)):
        centers[i]=max(centers[i], centers[i-1]+min_gap)
    if centers[-1] > cue.end-min_gap/2:
        shift=centers[-1]-(cue.end-min_gap/2)
        centers=[x-shift for x in centers]
    bounds=[cue.start]+[(centers[i]+centers[i+1])/2 for i in range(len(centers)-1)]+[cue.end]
    return [(ch,bounds[i],bounds[i+1]) for i,ch in enumerate(new_chars)]


def correct_cues_by_pinyin(cues, lyrics):
    """以 ASR 为时间来源，把识别段独立匹配到参考歌词。
    Whisper 一个 segment 可能包含多句歌词，命中相邻参考行时拆成单行 Cue，
    这样左侧字幕表、字幕轨和预览都不会显示一条超长合并句。
    """
    lines=[x.strip() for x in clean_lyrics(lyrics).splitlines() if _han(x)]
    if not lines or not cues: return cues, 0
    changed=0; out=[]
    for cue in sorted(cues, key=lambda c:c.start):
        raw=strip_banned_asr(cue.text)
        if not _han(raw): continue
        best=None; raw_len=max(1,len(_han(raw)))
        # 每个识别段独立遍历整份歌词，不要求识别顺序与歌词文件顺序一致。
        # n>1 仅用于识别 segment 合并多句；输出仍然是一行一个 Cue。
        for i in range(len(lines)):
            for n in range(1,min(4,len(lines)-i)+1):
                cand="".join(lines[i:i+n]); cand_len=len(_han(cand))
                if cand_len > raw_len*1.35 or raw_len > cand_len*1.8: continue
                score=.78*_phonetic_score(raw,cand)+.22*_char_score(raw,cand)
                score-=min(.18,abs(raw_len-cand_len)/max(raw_len,cand_len,1)*.22)
                if best is None or score>best[0]: best=(score,i,n,cand)
        if not best or best[0] < PINYIN_FIX_THRESHOLD:
            out.append(cue); continue
        _,i,n,_=best; rows=lines[i:i+n]; total=max(.01,cue.end-cue.start)
        weights=[max(1,len(_han(x))) for x in rows]; sw=sum(weights); tt=cue.start
        for j,(row,weight) in enumerate(zip(rows,weights)):
            ee=cue.end if j==len(rows)-1 else min(cue.end,tt+total*weight/sw)
            nc=Cue(tt,ee,row,cue.track,[]); nc.words=_retime_corrected_text(nc,row,row); out.append(nc); tt=ee
        if "".join(rows)!=raw or n>1: changed+=len(rows)
    return out, changed


def good(t): return len(t.split("\n")) >= MIN_LINES and len(t.replace("\n", "")) >= MIN_CHARS


def _netease(q):
    h = dict(UA, Referer="https://music.163.com/")
    r = requests.get("https://music.163.com/api/search/get/web", params={"s": q, "type": 1, "limit": 5, "offset": 0}, headers=h, timeout=10)
    for sg in ((r.json().get("result") or {}).get("songs") or [])[:5]:
        j = requests.get("https://music.163.com/api/song/lyric", params={"id": sg["id"], "lv": 1}, headers=h, timeout=10).json()
        yield sg["name"] + " - " + sg["artists"][0]["name"], (j.get("lrc") or {}).get("lyric", "")


def _qq(q):
    h = dict(UA, Referer="https://y.qq.com/")
    r = requests.get("https://c.y.qq.com/soso/fcgi-bin/client_search_cp", params={"w": q, "format": "json", "n": 5, "p": 1}, headers=h, timeout=10)
    for sg in r.json()["data"]["song"]["list"][:5]:
        j = requests.get("https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg",
                         params={"songmid": sg["songmid"], "format": "json", "nobase64": 1}, headers=h, timeout=10).json()
        yield sg["songname"] + " - " + sg["singer"][0]["name"], j.get("lyric", "")


def _lrclib(q):
    r = requests.get("https://lrclib.net/api/search", params={"q": q}, headers=UA, timeout=10)
    r.raise_for_status()
    for it in r.json()[:5]:
        yield it.get("trackName", "") + " - " + it.get("artistName", ""), it.get("plainLyrics") or it.get("syncedLyrics") or ""


def fetch_lyrics(q):
    """依次试 网易云 / QQ音乐 / lrclib, 每个来源试前5个结果; 清洗后太少就换下一个. 返回 (来源, 歌名, 歌词)"""
    q = re.sub(r"[\(（\[【][^\)）\]】]*[\)）\]】]|\b(DJ|Remix|Mix|Edit|Official|MV)\b|混音|伴奏", " ", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip()
    for name, fn in (("网易云", _netease), ("QQ音乐", _qq), ("lrclib", _lrclib)):
        try:
            for title, raw in fn(q):
                t = clean_lyrics(raw)
                if good(t): return name, title, t
        except Exception:
            continue
    return None, "", ""


_BG = {}


def draw_subs(p, r, st, cues, t):
    """画一帧: 背景 + 当前时刻的字幕(含入场/出场动画). 预览和导出MP4共用, 所见即所得"""
    p.fillRect(r, QColor(st["bg"]))
    if st.get("bg_img"):
        key = (st["bg_img"], int(r.width()), int(r.height()))
        if key not in _BG:
            if len(_BG) > 6: _BG.clear()
            im = QImage(st["bg_img"])
            _BG[key] = None if im.isNull() else im.scaled(key[1], key[2], Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        bi = _BG[key]
        if bi: p.drawImage(r.topLeft(), bi, QRectF((bi.width() - key[1]) / 2, (bi.height() - key[2]) / 2, key[1], key[2]))
    w, h, k, hit = r.width(), r.height(), r.width() / 1920, None
    title=(st.get("title_text") or "").strip(); ts=st.get("title_style", {})
    if title:
        speed=max(.1,float(ts.get("loop_speed",1.0))); anim_cycle=1.0/speed; stay=float(ts.get("stay",0.0)); visible_for=stay if stay>0 else anim_cycle; period=visible_for+max(0.0,float(ts.get("interval",0.0))); phase=t%max(.01,period)
        title_visible=phase<visible_for
        lp=ts.get("loop_anim","无"); ta=1.0; tox=0.0; toy=0.0; trot=0.0; tsc=1.0
        if title_visible:
            u=(phase/anim_cycle)%1.0
            if lp=="呼吸": tsc=1+.06*math.sin(u*math.pi*2)
            elif lp=="上下浮动": toy=8*math.sin(u*math.pi*2)
            elif lp=="左右摆动": tox=8*math.sin(u*math.pi*2)
            elif lp=="轻微旋转": trot=3*math.sin(u*math.pi*2)
            elif lp=="闪烁": ta=.65+.35*(.5+.5*math.sin(u*math.pi*4))
        else: title_visible=False
        if not title_visible: title=""
    if title:
        tf=QFont(ts.get("font", "Microsoft YaHei")); tf.setPixelSize(max(4, int(ts.get("size",42)*k))); tf.setBold(ts.get("bold",True)); tf.setItalic(ts.get("italic",False))
        tp=make_text_path(title,tf,ts.get("layout_mode","横排"),float(ts.get("letter_spacing",0))*k)
        tr=tp.boundingRect(); tcx=r.x()+float(ts.get("x",0.5))*w+tox; tcy=r.y()+float(ts.get("y",0.12))*h+toy*k
        p.save(); p.setOpacity(ta); p.translate(tcx,tcy); p.rotate(trot); p.scale(tsc,tsc); p.translate(-tr.center().x(),-tr.center().y())
        if ts.get("sw",2)>0: p.strokePath(tp,QPen(QColor(ts.get("stroke","#000000")),float(ts.get("sw",2))*k*2,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
        p.fillPath(tp,text_brush(ts,tr,t)); p.restore()
    ordered=sorted(cues,key=lambda c:c.start); active=next((i for i,c in enumerate(ordered) if c.start<=t<c.end),-1)
    slots=[x for x in st.get("lyric_slots",[]) if x.get("enabled",False)][:5]
    if not slots: slots=[{"enabled":True,"x":st.get("x",.5),"y":st.get("y",.85),"angle":0,"current":"循环"}]
    if active>=0 and slots:
        rule=str(st.get("current_row","循环"))
        current_slot=(active % len(slots)) if rule=="循环" else min(len(slots)-1,max(0,int(rule)-1))
    else: current_slot=0
    for si,slot in enumerate(slots):
        if active<0: continue
        # 当前行在全局指定槽位；其余槽位按当前行前后顺序显示邻近歌词。
        idx=active + (si-current_slot)
        if idx<0 or idx>=len(ordered): continue
        c=ordered[idx]; is_current=(idx==active)
        local=dict(st); local["anim_in"]=st.get("in_anim",st.get("anim_in","淡入")); local["anim_out"]=st.get("out_anim",st.get("anim_out","淡出")); local["dur"]=max(.01,st.get("dur",.4)/max(.1,float(st.get("in_speed",1))))
        a,oy,sc,ch=anim_state(local,c,t)
        sc*=1.0+(float(st.get("current_scale",30))/100 if is_current else 0)
        a*=1.0 if is_current else float(st.get("other_opacity",65))/100
        lp=st.get("loop_anim","无"); ls=float(st.get("loop_speed",1.0)); phase=(t-c.start)*ls
        if lp=="呼吸": a*=.78+.22*(.5+.5*math.sin(phase*math.pi*2))
        elif lp=="上下浮动": oy+=math.sin(phase*math.pi*2)*8
        elif lp=="左右摆动": slot=dict(slot); slot["x"]=float(slot.get("x",.5))+math.sin(phase*math.pi*2)*.008
        elif lp=="轻微旋转": slot=dict(slot); slot["angle"]=float(slot.get("angle",0))+math.sin(phase*math.pi*2)*3
        elif lp=="闪烁": a*=.55+.45*(.5+.5*math.sin(phase*math.pi*4))
        txt=c.text[:max(1,int(len(c.text)*ch+.999))]
        f=QFont(st["font"]); f.setPixelSize(max(4,int(st["size"]*k))); f.setBold(st.get("bold",False)); f.setItalic(st.get("italic",False))
        path=make_text_path(txt,f,st.get("layout_mode","横排"),float(st.get("letter_spacing",0))*k)
        br=path.boundingRect(); cx=r.x()+float(slot.get("x",.5))*w; cy=r.y()+float(slot.get("y",.85))*h+oy*k
        p.save(); p.setOpacity(max(0,min(1,a))); p.translate(cx,cy); p.rotate(float(slot.get("angle",0))); p.scale(sc,sc); p.translate(-br.center().x(),-br.center().y())
        if st["sw"]>0: p.strokePath(path,QPen(QColor(st["stroke"]),st["sw"]*k*2,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
        p.fillPath(path,text_brush(st,br,t))
        if st.get("karaoke") and c.words and is_current:
            tot=sum(len(w[0].strip()) for w in c.words) or 1; done=sum(len(w.strip())*min(1,max(0,(t-ws)/max(we-ws,.05))) for w,ws,we in c.words)
            p.save(); p.setClipRect(QRectF(br.left()-4,br.top()-6,br.width()*min(1,done/tot)+4,br.height()+12)); p.fillPath(path,QColor(st["hl"])); p.restore()
        p.restore(); hit=(QRectF(cx-br.width()*sc/2,cy-br.height()*sc/2,br.width()*sc,br.height()*sc),c) if is_current else hit
    return hit


def audio_len(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", path]).decode().strip())


def render_mp4(audio, cues, st, out, prog=None, W=1280, H=720, fps=30):
    """逐帧画字幕, 通过管道交给ffmpeg, 和原音频合成MP4"""
    dur = audio_len(audio); n = int(math.ceil(dur * fps))
    pr = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast",
                           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", out],
                          stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    img = QImage(W, H, QImage.Format_ARGB32)
    try:
        for i in range(n):
            p = QPainter(img); p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
            draw_subs(p, QRectF(0, 0, W, H), st, cues, i / fps); p.end()
            pr.stdin.write(bytes(img.constBits()))
            if prog and i % 15 == 0: prog(int(100 * i / n))
        pr.stdin.close()
    except BrokenPipeError:
        pass
    pr.wait()
    if pr.returncode: raise RuntimeError("ffmpeg失败, 请确认已安装ffmpeg并在PATH里")


def viterbi(S, order, pen=3.0):
    """S: (N,2) 每个窗口[唱歌,说唱]的对数得分; order: 如 ['rap','sing','rap','sing'].
    只能按给定顺序往后走, 不能回头, 所以整首歌只会切出len(order)段."""
    idx, K, N = {"sing": 0, "rap": 1}, len(order), len(S)
    if N < K: return [order[0]] * N
    best = np.full((N, K), -1e18); back = np.zeros((N, K), int)
    best[0, 0] = S[0, idx[order[0]]]
    for i in range(1, N):
        for k in range(K):
            stay, adv = best[i - 1, k], (best[i - 1, k - 1] - pen if k else -1e18)
            back[i, k] = k if stay >= adv else k - 1
            best[i, k] = max(stay, adv) + S[i, idx[order[k]]]
    k, out = K - 1, []
    for i in range(N - 1, -1, -1):
        out.append(order[k]); k = back[i, k]
    return out[::-1]


class Worker(QThread):
    prog = Signal(int)
    lyr = Signal(str)
    done = Signal(list)
    err = Signal(str)
    lab = Signal(list)
    found = Signal(float, float)

    def __init__(s, path, lyrics, rap, order, bpm=0.0, off=0.0, grid=8, pinyin_fix=True):
        super().__init__()
        s.path, s.lyrics, s.rap, s.order = path, lyrics, rap, order
        s.bpm, s.off, s.grid, s.pinyin_fix = bpm, off, grid, pinyin_fix

    def run(s):
        try:
            s.done.emit(s.work())
        except Exception as ex:
            s.err.emit(repr(ex))

    def work(s):
        import soundfile as sf, torch, torchaudio
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # 0) 自动下载歌词
        if not s.lyrics.strip():
            try:
                name = re.sub(r"[_\-\.]+", " ", os.path.splitext(os.path.basename(s.path))[0])
                _, _, s.lyrics = fetch_lyrics(name)
                if s.lyrics: s.lyr.emit(s.lyrics)
            except Exception:
                pass
        s.prog.emit(3)
        # 1) 人声分离 (demucs)
        tmp = tempfile.mkdtemp()
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", tmp, s.path],
                       check=True, capture_output=True)
        voc = glob.glob(os.path.join(tmp, "**", "vocals.wav"), recursive=True)[0]
        y, sr = sf.read(voc, always_2d=True)
        y = torch.from_numpy(y.mean(1).astype(np.float32))
        y16 = torchaudio.functional.resample(y, sr, 16000).numpy()
        y32 = torchaudio.functional.resample(y, sr, 32000).numpy()
        s.prog.emit(35)
        # 2) 分类: 唱歌 / 说唱 / 说话 / 静音  (PANNs, AudioSet标签)
        from panns_inference import AudioTagging, labels
        at = AudioTagging(checkpoint_path=None, device=dev)
        groups = {"sing": ["Singing", "Male singing", "Female singing"],
                  "rap": ["Rapping", "Hip hop music"],
                  "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking"],
                  "silence": ["Silence"]}
        gi = {k: [labels.index(n) for n in v if n in labels] for k, v in groups.items()}
        # 按BPM网格切成"每格N拍"的单元(没设BPM就按2秒), 每个单元整体分类, 边界自然落在网格上
        if s.bpm <= 0:  # BPM填0=自动检测BPM和首拍
            try:
                import librosa
                yy, sr0 = librosa.load(s.path, sr=22050, mono=True)
                tempo, beats = librosa.beat.beat_track(y=yy, sr=sr0)
                s.bpm = float(np.atleast_1d(tempo)[0])
                s.off = float(librosa.frames_to_time(beats[0], sr=sr0)) if len(beats) else 0.0
                s.found.emit(s.bpm, s.off)
            except Exception:
                s.bpm = 0.0
        total = len(y32) / 32000
        if s.bpm > 0:
            cell = s.grid * 60 / s.bpm
            edges, e = [0.0], s.off + math.ceil((0.5 - s.off) / cell) * cell
            while e < total - 1:
                if e - edges[-1] >= 1: edges.append(e)
                e += cell
            edges.append(total)
        else:
            edges = [float(x) for x in np.arange(0, total, 2.0)] + [total]
        cells = list(zip(edges[:-1], edges[1:]))
        # 只检测 Singing，不再强制 rap -> sing -> rap -> sing。
        # 先记录每个窗口的 Singing 分数，再合并相邻窗口，最多保留两段最长连续演唱。
        sing_scores, silent = [], []
        for a, b in cells:
            ch = y32[int(a * 32000):int(b * 32000)]
            if len(ch) < 32000: ch = np.pad(ch, (0, 32000 - len(ch)))
            out, _ = at.inference(ch[None])
            sc = {k: max([out[0][j] for j in v] or [1e-4]) for k, v in gi.items()}
            sing_scores.append(float(sc["sing"]))
            silent.append(float(np.sqrt(np.mean(ch ** 2))) < 0.005)
        sing_scores = np.array(sing_scores)
        candidate = (sing_scores >= SING_THRESHOLD) & (~np.array(silent))

        # 窗口级标签只用于时间轴显示；静音/伴奏不再被贴成 rap。
        kinds = ["sing" if ok else "other" for ok in candidate]
        s.lab.emit([(float(a), float(b), k, sl) for (a, b), k, sl in zip(cells, kinds, silent)])

        # 形成候选连续段，并把短间隙合并，避免一句歌被切碎。
        runs = []
        for i, ok in enumerate(candidate):
            if ok:
                if not runs or i > runs[-1][1] + 1:
                    runs.append([i, i])
                else:
                    runs[-1][1] = i
        merged = []
        for a_i, b_i in runs:
            if merged and cells[a_i][0] - cells[merged[-1][1]][1] <= SING_MAX_GAP_SEC:
                merged[-1][1] = b_i
            else:
                merged.append([a_i, b_i])
        # 只保留最长的两段；按时间顺序返回，方便字幕自然排列。
        merged = sorted(merged, key=lambda r: cells[r[1]][1] - cells[r[0]][0], reverse=True)[:SING_MAX_BLOCKS]
        merged.sort()
        segs = []
        for a_i, b_i in merged:
            segs.append([max(0, cells[a_i][0] - 0.3), min(total, cells[b_i][1] + 0.3)])
        s.prog.emit(50)
        # 3) 字级ASR: faster-whisper large-v3, 歌词注入 initial_prompt
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device=dev, compute_type="float16" if dev == "cuda" else "int8")
        prompt = " ".join(dict.fromkeys(clean_lyrics(s.lyrics).split("\n")))[:400] or None
        cues = []
        for n, (a, b) in enumerate(segs):
            chunk = y16[int(a * 16000):int(b * 16000)]
            res, _ = model.transcribe(chunk, word_timestamps=True, initial_prompt=prompt,
                                      condition_on_previous_text=False, beam_size=5)
            for sg in res:
                ws = [(w.word, a + w.start, a + w.end) for w in (sg.words or [])]
                cleaned = strip_banned_asr(sg.text.strip())
                if cleaned:
                    cues.append(Cue(a + sg.start, a + sg.end, cleaned, 0, ws))
            s.prog.emit(50 + int(50 * (n + 1) / len(segs)))
        s.fixed_count=0
        if s.pinyin_fix and s.lyrics.strip():
            cues, s.fixed_count = correct_cues_by_pinyin(cues, s.lyrics)
        return cues


class RenderWorker(QThread):
    prog = Signal(int)
    done = Signal(str)
    err = Signal(str)

    def __init__(s, audio, cues, st, out):
        super().__init__(); s.a = (audio, cues, st, out)

    def run(s):
        try:
            render_mp4(*s.a, prog=s.prog.emit); s.done.emit("已生成:\n" + s.a[3])
        except Exception as ex:
            s.err.emit(repr(ex))


class BatchWorker(QThread):
    """批量: 每首歌 自动下歌词 -> 人声分离 -> 分类 -> 识别 -> 用当前样式导出MP4"""
    prog = Signal(int)
    done = Signal(list)

    def __init__(s, files, outdir, st, rap, order):
        super().__init__(); s.files, s.outdir, s.st, s.rap, s.order = files, outdir, st, rap, order

    def run(s):
        bad, n = [], len(s.files)
        for i, f in enumerate(s.files):
            try:
                w = Worker(f, "", s.rap, s.order, pinyin_fix=True)
                w.prog.connect(lambda v, i=i: s.prog.emit(int((i + v / 100 * 0.7) / n * 100)))
                cues = w.work()
                out = os.path.join(s.outdir, os.path.splitext(os.path.basename(f))[0] + ".mp4")
                render_mp4(f, cues, s.st, out, prog=lambda v, i=i: s.prog.emit(int((i + 0.7 + v / 100 * 0.3) / n * 100)))
            except Exception as ex:
                bad.append(os.path.basename(f) + ": " + repr(ex))
        s.prog.emit(100); s.done.emit(bad)


class Preview(QWidget):
    def __init__(s, app):
        super().__init__()
        s.app, s.hit, s.dragging, s.r = app, None, False, QRectF()
        s.setMinimumSize(420, 236); s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def hasHeightForWidth(s): return True
    def heightForWidth(s, w): return int(w*9/16)+18
    def sizeHint(s): return QSize(960, 558)

    def paintEvent(s, e):
        p = QPainter(s)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        p.fillRect(s.rect(), QColor("#111"))
        W, H = s.width(), s.height()
        w = min(W, H * 16 / 9)
        h = w * 9 / 16
        s.r = QRectF((W - w) / 2, (H - h) / 2, w, h)
        p.fillRect(s.r, QColor("#000"))
        draw_cues=s.app.demo_cues if getattr(s.app,"demo_mode",False) and s.app.demo_cues is not None else s.app.cues
        s.hit = draw_subs(p, s.r, s.app.st, draw_cues, s.app.t)
        # 预览辅助坐标只显示在编辑器中，不会导出到视频。
        p.save(); p.setPen(QPen(QColor(255,255,255,70),1,Qt.DashLine))
        for q in (0.25,0.5,0.75):
            p.drawLine(QPointF(s.r.left()+s.r.width()*q,s.r.top()),QPointF(s.r.left()+s.r.width()*q,s.r.bottom()))
            p.drawLine(QPointF(s.r.left(),s.r.top()+s.r.height()*q),QPointF(s.r.right(),s.r.top()+s.r.height()*q))
        p.setPen(QColor("#d9dce8")); p.drawText(s.r.left()+6,s.r.top()+16,"1920 × 1080")
        p.drawText(s.r.left()+6,s.r.bottom()-8,f"歌词 X={s.app.st.get('x',0.5):.3f}  Y={s.app.st.get('y',0.85):.3f} | 歌名 X={s.app.st.get('title_style',{}).get('x',0.5):.3f}  Y={s.app.st.get('title_style',{}).get('y',0.12):.3f}")
        p.restore()

    def mousePressEvent(s, e):  # 拖动字幕位置
        if s.hit and s.hit[0].adjusted(-20, -20, 20, 20).contains(e.position()): s.dragging = True

    def mouseMoveEvent(s, e):
        if s.dragging and s.r.width() > 0:
            st = s.app.st
            st["x"] = min(1, max(0, (e.position().x() - s.r.x()) / s.r.width()))
            st["y"] = min(1, max(0, (e.position().y() - s.r.y()) / s.r.height()))
            s.update()

    def mouseReleaseEvent(s, e): s.dragging = False

    def mouseDoubleClickEvent(s, e):  # 双击直接改字
        if s.hit and s.hit[0].contains(e.position()):
            c = s.hit[1]
            t, ok = QInputDialog.getText(s, "编辑字幕", "文本", text=c.text)
            if ok: c.text = t; s.app.refill()


class Timeline(QWidget):
    LW, RH, TOP = 70, 96, 22

    def __init__(s, app):
        super().__init__()
        s.app, s.pps, s.drag, s.active, s.auto_fit = app, 50.0, None, 0, True
        s.fit()

    def fit(s):
        content_w=int(s.LW + max(s.app.dur, 10) * s.pps + 100)
        # 时间轴是下方独立的整宽区域：内容较短时也铺满窗口，避免右侧出现空白块。
        viewport_w=0
        if hasattr(s.app,"sc") and s.app.sc is not None:
            viewport_w=s.app.sc.viewport().width()
        if s.auto_fit and viewport_w > s.LW + 40 and s.app.dur > 0:
            s.pps=max(2.0,min(50.0,(viewport_w-s.LW-20)/max(s.app.dur,10.0)))
        s.setFixedSize(max(content_w,viewport_w), s.TOP + s.RH * len(s.app.tracks) + 10)
        s.update()

    def X(s, t): return s.LW + t * s.pps
    def T(s, x): return (x - s.LW) / s.pps

    def paintEvent(s, e):
        a, p = s.app, QPainter(s)
        p.fillRect(s.rect(), QColor("#1e1e1e"))
        p.setPen(QColor("#999"))
        step = 1 if s.pps >= 40 else 5 if s.pps >= 10 else 30
        for sec in range(0, int(a.dur) + 1, step):
            x = int(s.X(sec)); p.drawLine(x, s.TOP - 6, x, s.TOP); p.drawText(x + 2, s.TOP - 8, fmt(sec)[:5])
        for i, (name, kind) in enumerate(a.tracks):
            y = s.TOP + i * s.RH
            p.fillRect(0, y, s.width(), s.RH - 2, QColor("#33384a" if i == s.active else "#2a2a2a"))
            p.setPen(QColor("#ddd")); p.drawText(6, y + 27, name)
            if kind == "audio":  # 每格分类结果: 绿=唱歌 橙=说唱 灰=静音
                for c0, c1, k, sl in a.cells:
                    p.fillRect(QRectF(s.X(c0), y + 1, (c1 - c0) * s.pps - 1, 6), QColor("#555" if sl else "#3ecf6e" if k == "sing" else "#ff9f43"))
            if kind == "audio" and getattr(a,"path",""):
                ar=QRectF(s.X(0),y+8,max(8,s.app.dur*s.pps),s.RH-18)
                p.fillRect(ar,QColor(53,75,102,210)); p.setPen(QColor("#d7e7ff")); p.drawText(ar.adjusted(6,0,0,0),Qt.AlignVCenter,os.path.basename(a.path))
            if kind == "audio" and a.env is not None:
                p.setPen(QColor("#4aa3ff")); mid = y + s.RH // 2 - 1
                for x in range(s.LW, s.width() - 100):
                    i0 = int((x - s.LW) / s.pps / 0.01); seg = a.env[i0:i0 + max(1, int(0.01 ** -1 / s.pps))]
                    if len(seg): v = int(min(1, seg.max() * 1.5) * (s.RH / 2 - 4)); p.drawLine(x, mid - v, x, mid + v)
            if kind == "sub":
                timeline_cues=a.demo_cues if getattr(a,"demo_mode",False) and a.demo_cues is not None else a.cues
                for c in timeline_cues:
                    if c.track == i:
                        r = QRectF(s.X(c.start), y + 6, max(4, (c.end - c.start) * s.pps), s.RH - 14)
                        p.fillRect(r, QColor("#c9812b" if c is a.cur else "#8a5a1e"))
                        p.setPen(QColor("#fff")); p.drawText(r.adjusted(3, 0, 0, 0), Qt.AlignVCenter, c.text)
            if kind == "video" and getattr(a,"video_path",""):
                r=QRectF(s.X(0),y+7,max(8,(a.dur)*s.pps),s.RH-16); p.fillRect(r,QColor("#435b78")); p.setPen(QColor("#fff")); p.drawText(r.adjusted(5,0,0,0),Qt.AlignVCenter,os.path.basename(a.video_path))
        bpm, off, g = a.bpm_sp.value(), a.off_sp.value(), int(a.grid_cb.currentText())
        if bpm > 0:  # BPM网格: 黄线=标注点, 淡线=每8拍
            beat, n = 60 / bpm, max(0, math.ceil(-off / (60 / bpm)))
            while off + n * beat < a.dur:
                x = int(s.X(off + n * beat))
                if n % g == 0:
                    p.setPen(QPen(QColor("#ffd54a"), 2)); p.drawLine(x, s.TOP, x, s.height()); p.drawText(x + 3, s.height() - 4, str(n))
                n += 1
        x = int(s.X(a.t)); p.setPen(QPen(QColor("#ff4d4d"), 2)); p.drawLine(x, 0, x, s.height())

    def mousePressEvent(s, e):
        a, x, y = s.app, e.position().x(), e.position().y()
        r = int((y - s.TOP) // s.RH) if y >= s.TOP else -1
        if 0 <= r < len(a.tracks):
            s.active = r
            kind=a.tracks[r][1]
            if x < s.LW:
                if kind == "sub": a.import_subtitle()
                elif kind == "video": a.import_video()
                elif kind == "audio": a.open_audio()
                return
            if kind == "sub":
                for c in a.cues:
                    if c.track == r and s.X(c.start) <= x <= s.X(c.end):
                        s.drag = (c, s.T(x) - c.start); a.sel(c); s.update(); return
        a.seek(s.T(x))

    def mouseMoveEvent(s, e):
        if s.drag:
            c, off = s.drag
            ns = max(0, s.T(e.position().x()) - off); dt = ns - c.start
            c.words = [(w, x + dt, y + dt) for w, x, y in c.words]
            c.end += dt; c.start = ns; s.update(); s.app.preview.update()

    def mouseReleaseEvent(s, e):
        if s.drag: s.drag = None; s.app.refill()

    def wheelEvent(s, e):
        if e.modifiers() & Qt.ControlModifier:
            s.auto_fit=False
            s.pps = min(400, max(5, s.pps * (1.2 if e.angleDelta().y() > 0 else 1 / 1.2))); s.fit()
        else:
            e.ignore()


class Main(QMainWindow):
    def __init__(s):
        super().__init__()
        s.setWindowTitle("字幕工作室"); s.resize(1500, 900)
        s.st, s.cues, s.t, s.dur, s.env, s.cur, s.blk, s.path, s.worker = dict(STYLE), [], 0.0, 60.0, None, None, False, "", None
        s.tracks = [["导入字幕", "sub"], ["导入视频", "video"], ["导入音频", "audio"]]
        s.video_path = ""
        s.cells = []
        s.player = QMediaPlayer(); s.ao = QAudioOutput(); s.player.setAudioOutput(s.ao)
        s.player.positionChanged.connect(lambda ms: s.set_t(ms / 1000, True))
        s.player.durationChanged.connect(s.on_dur)
        s.preview, s.tl = Preview(s), Timeline(s)
        s.demo_timer=QTimer(s); s.demo_timer.setInterval(33); s.demo_timer.timeout.connect(s.demo_tick)
        s.demo_mode=False; s.demo_cues=None; s.demo_original_cues=None

        # ---- 左: 音频输入 + 字幕内容 | 歌词内容 ----
        left = QWidget(); L = QVBoxLayout(left); L.setContentsMargins(6,4,6,4); L.setSpacing(4)
        s.song = QLabel("未选择音频"); s.song.setWordWrap(True)
        L.addWidget(s.song)
        s.btn_asr = QPushButton("AI识别并生成字幕"); s.btn_asr.setMinimumHeight(46)
        s.btn_asr.setStyleSheet("background:#7c4dff;color:white;font-weight:bold;border-radius:6px")
        s.btn_asr.clicked.connect(s.run_asr); L.addWidget(s.btn_asr)
        s.bar = QProgressBar(); s.bar.setRange(0, 100); L.addWidget(s.bar)
        sub = QGroupBox("字幕（AI识别结果）"); sub.setStyleSheet("QGroupBox{background:#202d46;border:1px solid #4b78b8;border-radius:6px;margin-top:8px;padding-top:10px;} QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 6px;color:#8fc5ff;font-weight:bold;}"); sl = QVBoxLayout(sub); sl.setContentsMargins(6,8,6,6); sl.setSpacing(4)
        s.tb = QTableWidget(0, 3); s.tb.setMinimumHeight(150); s.tb.setHorizontalHeaderLabels(["识别字幕", "开始", "结束"])
        s.tb.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); s.tb.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents); s.tb.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        s.tb.itemChanged.connect(lambda it: s.cell(it.row(), it.column()))
        s.tb.cellClicked.connect(lambda r, c: s.seek(s.cues[r].start) if r < len(s.cues) else None)
        sl.addWidget(s.tb)
        ab = QPushButton("在播放位置添加字幕"); ab.clicked.connect(s.add_cue); sl.addWidget(ab)
        lyr = QGroupBox("歌词（下载/参考歌词）"); lyr.setStyleSheet("QGroupBox{background:#382b43;border:1px solid #a76ac4;border-radius:6px;margin-top:8px;padding-top:10px;} QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 6px;color:#e2aaff;font-weight:bold;}"); ll = QVBoxLayout(lyr); ll.setContentsMargins(6,8,6,6); ll.setSpacing(4)
        s.q = QLineEdit(); s.q.setPlaceholderText("歌名 歌手（用于下载歌词）"); ll.addWidget(s.q)
        db = QPushButton("下载歌词"); db.clicked.connect(s.dl_lyrics); ll.addWidget(db)
        s.lyrics = QPlainTextEdit(); s.lyrics.setPlaceholderText("下载的歌词将按一句一行显示，可上下滚动查看"); ll.addWidget(s.lyrics,1)
        L.addWidget(sub,2); L.addWidget(lyr,2)
        s.info = QLabel(""); s.info.setWordWrap(True); L.addWidget(s.info)
        L.addStretch(1)
        def panel_scroll(widget):
            sc=QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); sc.setMinimumSize(0,0); sc.setWidget(widget); return sc
        s.tabs = QTabWidget(); s.tabs.addTab(s.general_settings(), "设置"); s.tabs.addTab(panel_scroll(s.settings()), "歌词样式"); s.tabs.addTab(panel_scroll(s.title_settings()), "歌名设置"); s.tabs.addTab(s.animation_settings(), "动画")

        # ---- 中: 预览 + 播放控制 ----
        mid = QWidget(); M = QVBoxLayout(mid); M.setContentsMargins(4,4,4,4); M.setSpacing(4); M.addWidget(s.preview, 1)
        progress = QHBoxLayout()
        s.sl = QSlider(Qt.Horizontal); s.sl.sliderMoved.connect(lambda v: s.seek(v / 1000))
        s.lab = QLabel("00:00.00"); progress.addWidget(s.sl, 1); progress.addWidget(s.lab)
        M.addLayout(progress)
        actions = QHBoxLayout()
        s.play = QPushButton("播放/暂停"); s.play.clicked.connect(s.toggle)
        s.demo_btn=QPushButton("实时演示"); s.demo_btn.setToolTip("不需要音频，循环模拟卡拉OK逐字变色和动画效果"); s.demo_btn.clicked.connect(s.toggle_demo)
        s.batch_in=QLineEdit(); s.batch_in.setPlaceholderText("输入文件夹"); s.batch_in.setToolTip("批量生成的音频输入文件夹"); s.batch_in.setMaximumWidth(145)
        bi=QPushButton("…"); bi.setToolTip("选择批量输入文件夹"); bi.setFixedWidth(26); bi.clicked.connect(s.pick_batch_input)
        s.batch_out=QLineEdit(); s.batch_out.setPlaceholderText("输出文件夹"); s.batch_out.setToolTip("批量生成的输出文件夹"); s.batch_out.setMaximumWidth(145)
        bo=QPushButton("…"); bo.setToolTip("选择批量输出文件夹"); bo.setFixedWidth(26); bo.clicked.connect(s.pick_batch_output)
        bat=QPushButton("批量生成"); bat.clicked.connect(s.batch)
        srt=QPushButton("导出SRT"); srt.clicked.connect(s.export_srt)
        exp=QPushButton("导出MP4"); exp.clicked.connect(s.export_mp4)
        actions.addWidget(s.play); actions.addWidget(s.demo_btn); actions.addStretch(1); actions.addWidget(QLabel("输入")); actions.addWidget(s.batch_in); actions.addWidget(bi); actions.addWidget(QLabel("输出")); actions.addWidget(s.batch_out); actions.addWidget(bo); actions.addWidget(bat); actions.addWidget(srt); actions.addWidget(exp)
        M.addLayout(actions)

        left_scroll=QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.NoFrame); left_scroll.setMinimumSize(0,0); left_scroll.setWidget(left)
        top = QSplitter(Qt.Horizontal); top.addWidget(left_scroll); top.addWidget(mid); top.addWidget(s.tabs)
        top.setStretchFactor(0, 2); top.setStretchFactor(1, 7); top.setStretchFactor(2, 2); top.setSizes([280, 960, 360])

        # ---- 下: 轨道 ----
        bot = QWidget(); B = QVBoxLayout(bot); B.setContentsMargins(0, 0, 0, 0)
        tb = QHBoxLayout()
        for txt, kind in (("+字幕轨", "sub"), ("+视频轨", "video"), ("+音频轨", "audio")):
            b = QPushButton(txt); b.clicked.connect(lambda _, k=kind, t=txt: s.add_track(k)); tb.addWidget(b)
        tb.addWidget(QLabel("Ctrl+滚轮缩放时间轴")); tb.addStretch(); B.addLayout(tb)
        s.sc = QScrollArea(); s.sc.setWidget(s.tl); B.addWidget(s.sc)
        main = QSplitter(Qt.Vertical); main.addWidget(top); main.addWidget(bot); main.setStretchFactor(0, 1); main.setStretchFactor(1, 0); main.setSizes([560, 390])
        s.setCentralWidget(main)
        s.apply_theme(s.st.get("theme","专业深色"))

    def resizeEvent(s, e):
        super().resizeEvent(e)
        # 上方横向三栏由 splitter 等比重新分配；下方轨道同步到整个窗口宽度。
        if hasattr(s,"tl"):
            QTimer.singleShot(0, s.tl.fit)

    # ---- 右: 通用设置、字幕设置 + 动画 ----
    def general_settings(s):
        w=QWidget(); v=QVBoxLayout(w); v.setContentsMargins(8,8,8,8); v.setSpacing(8)
        skin=QGroupBox("界面皮肤"); sf=QFormLayout(skin); s.theme_combo=QComboBox(); s.theme_combo.addItems(list(THEMES)); s.theme_combo.setCurrentText(s.st.get("theme","专业深色")); s.theme_combo.currentTextChanged.connect(s.apply_theme); sf.addRow("皮肤",s.theme_combo)
        adv=QGroupBox("高级识别设置（一般保持默认即可）"); af=QFormLayout(adv)
        s.rap=QCheckBox("说唱也识别（实验功能，默认关闭）"); af.addRow("识别范围",s.rap)
        s.pinyin_fix=QCheckBox("按歌词拼音纠错（默认开启）"); s.pinyin_fix.setChecked(True); af.addRow("文字纠错",s.pinyin_fix)
        s.order=QLineEdit("只检测唱歌；此项已不再强制四段"); s.order.setToolTip("已知的段落顺序，用逗号分隔"); af.addRow("段落顺序",s.order)
        s.bpm_sp=QDoubleSpinBox(); s.bpm_sp.setRange(0,300); s.bpm_sp.setDecimals(2); s.bpm_sp.setSpecialValueText("自动")
        s.off_sp=QDoubleSpinBox(); s.off_sp.setRange(-30,600); s.off_sp.setDecimals(3); s.off_sp.setSingleStep(0.01)
        s.grid_cb=QComboBox(); s.grid_cb.addItems(["8","16","32","64"]); s.grid_cb.setCurrentText("64")
        s.an_grid_cb=QComboBox(); s.an_grid_cb.addItems(["4","8","16","32","64"]); s.an_grid_cb.setCurrentText("8")
        bb=QPushButton("检测BPM"); bb.clicked.connect(s.detect_bpm)
        af.addRow("BPM（0=自动）",s.bpm_sp); af.addRow("首拍秒",s.off_sp); af.addRow("标注每几拍",s.grid_cb); af.addRow("分析窗口拍数",s.an_grid_cb); af.addRow("",bb)
        for wd in (s.bpm_sp,s.off_sp): wd.valueChanged.connect(lambda _: s.tl.update())
        s.grid_cb.currentTextChanged.connect(lambda _: s.tl.update())
        v.addWidget(skin); v.addWidget(adv)
        note=QLabel("高级设置说明：\n• 说唱识别：默认关闭，只保留唱歌段，稳定性更高。\n• 拼音纠错：用下载歌词修正 ASR 同音错字。\n• 段落顺序：仅在启用说唱识别时使用。\n• BPM、首拍和标注网格：控制时间轴拍点标记；BPM 为 0 时自动检测。\n• 分析窗口拍数：控制唱歌检测的分析窗口大小。")
        note.setWordWrap(True); note.setStyleSheet("color:#aeb8cc;padding:6px;"); v.addWidget(note); v.addStretch(1)
        return w

    def settings(s):
        w = QWidget(); f = QFormLayout(w); st = s.st
        fc = QFontComboBox(); fc.setCurrentFont(QFont(st["font"])); fc.currentFontChanged.connect(lambda ft: s.setst("font", ft.family()))
        fontlinks = QWidget(); fl = QHBoxLayout(fontlinks); fl.setContentsMargins(0, 0, 0, 0)
        gf = QPushButton("下载中文字体")
        gf.setToolTip("打开 Google Fonts 的 Noto Sans SC 页面，下载后安装即可使用")
        gf.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://fonts.google.com/noto/specimen/Noto+Sans+SC")))
        nc = QPushButton("Noto CJK 下载")
        nc.setToolTip("打开 Noto CJK 官方下载页")
        nc.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/notofonts/noto-cjk/releases")))
        fl.addWidget(gf); fl.addWidget(nc)
        sz = QSpinBox(); sz.setRange(10, 300); sz.setValue(st["size"]); sz.valueChanged.connect(lambda v: s.setst("size", v))
        sw = QSpinBox(); sw.setRange(0, 20); sw.setValue(st["sw"]); sw.valueChanged.connect(lambda v: s.setst("sw", v))
        bd = QCheckBox("加粗"); bd.setChecked(st.get("bold", False)); bd.toggled.connect(lambda v: s.setst("bold", v))
        it = QCheckBox("倾斜"); it.setChecked(st.get("italic", False)); it.toggled.connect(lambda v: s.setst("italic", v))
        fm = QComboBox(); fm.addItems(["纯色", "渐变色", "循环跑色"]); fm.setCurrentText(st.get("fill_mode", "纯色")); fm.currentTextChanged.connect(lambda v: s.setst("fill_mode", v))
        cs = QDoubleSpinBox(); cs.setRange(0.1, 10.0); cs.setSingleStep(0.1); cs.setValue(float(st.get("cycle_speed", 1.0))); cs.setSuffix(" 倍"); cs.valueChanged.connect(lambda v: s.setst("cycle_speed", v))
        ga=QDoubleSpinBox(); ga.setRange(-180,180); ga.setValue(float(st.get("gradient_angle",45))); ga.setSuffix("°"); ga.valueChanged.connect(lambda v:s.setst("gradient_angle",v))
        cp=QComboBox(); cp.addItem("颜色预设"); cp.addItems(COLOR_PRESETS); cp.currentTextChanged.connect(lambda v:s.apply_color_preset(st,v))
        lm=QComboBox(); lm.addItems(["横排","竖排"]); lm.setCurrentText(st.get("layout_mode","横排")); lm.currentTextChanged.connect(lambda v:s.setst("layout_mode",v))
        lsq=QDoubleSpinBox(); lsq.setRange(-50,200); lsq.setSingleStep(1); lsq.setValue(float(st.get("letter_spacing",0))); lsq.setSuffix(" px"); lsq.valueChanged.connect(lambda v:s.setst("letter_spacing",v))
        for lab, wd in (("字体", fc), ("", fontlinks), ("字号", sz), ("文字颜色", s.colorbtn("color")), ("描边颜色", s.colorbtn("stroke")),
                        ("描边粗细", sw), ("", bd), ("", it), ("填充模式", fm), ("渐变第二颜色", s.colorbtn("color2")),
                        ("跑色速度", cs), ("渐变角度", ga), ("颜色预设", cp), ("显示模式", lm), ("字间距", lsq)):
            f.addRow(lab, wd)
        kc = QCheckBox("逐字高亮(卡拉OK)"); kc.setChecked(st["karaoke"]); kc.toggled.connect(lambda v: s.setst("karaoke", v))
        f.addRow("", kc); f.addRow("高亮颜色", s.colorbtn("hl"))
        bi = QPushButton("选择背景图片"); bi.clicked.connect(s.pick_bg)
        f.addRow("背景颜色", s.colorbtn("bg")); f.addRow("背景图片", bi)
        f.addRow(QLabel("提示: 预览里拖动文字可移位置, 双击改文字"))
        return w

    def animation_settings(s):
        w=QWidget(); f=QFormLayout(w); f.setContentsMargins(6,4,6,4); f.setVerticalSpacing(4); st=s.st
        cur_scale=QSpinBox(); cur_scale.setRange(0,100); cur_scale.setValue(int(st.get("current_scale",30))); cur_scale.setSuffix(" %"); cur_scale.valueChanged.connect(lambda v:s.setst("current_scale",v))
        op=QSpinBox(); op.setRange(0,100); op.setValue(int(st.get("other_opacity",65))); op.setSuffix(" %"); op.valueChanged.connect(lambda v:s.setst("other_opacity",v))
        ia=QComboBox(); ia.addItems(ANIM_IN); ia.setCurrentText(st.get("in_anim","淡入")); ia.currentTextChanged.connect(lambda v:s.setst("in_anim",v))
        ispd=QDoubleSpinBox(); ispd.setRange(.1,10); ispd.setValue(float(st.get("in_speed",1))); ispd.setSuffix(" 倍"); ispd.valueChanged.connect(lambda v:s.setst("in_speed",v))
        oa=QComboBox(); oa.addItems(ANIM_OUT); oa.setCurrentText(st.get("out_anim","淡出")); oa.currentTextChanged.connect(lambda v:s.setst("out_anim",v))
        ospd=QDoubleSpinBox(); ospd.setRange(.1,10); ospd.setValue(float(st.get("out_speed",1))); ospd.setSuffix(" 倍"); ospd.valueChanged.connect(lambda v:s.setst("out_speed",v))
        la=QComboBox(); la.addItems(ANIM_LOOP); la.setCurrentText(st.get("loop_anim","无")); la.currentTextChanged.connect(lambda v:s.setst("loop_anim",v))
        lspd=QDoubleSpinBox(); lspd.setRange(.1,10); lspd.setValue(float(st.get("loop_speed",1))); lspd.setSuffix(" 倍"); lspd.valueChanged.connect(lambda v:s.setst("loop_speed",v))
        cr=QComboBox(); cr.addItems(["循环","1","2","3","4","5"]); cr.setCurrentText(str(st.get("current_row","循环"))); cr.currentTextChanged.connect(lambda v:s.setst("current_row",v))
        slotbox=QGroupBox("歌词槽位（勾选几个就显示几行）"); slotbox.setStyleSheet("QGroupBox{margin-top:10px;} QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;padding:0 4px;}"); sg=QGridLayout(slotbox); sg.setContentsMargins(6,14,6,6); sg.setVerticalSpacing(3); sg.addWidget(QLabel("槽位"),0,0); sg.addWidget(QLabel("启用"),0,1); sg.addWidget(QLabel("X"),0,2); sg.addWidget(QLabel("Y"),0,3); sg.addWidget(QLabel("角度"),0,4)
        for i,sl in enumerate(st["lyric_slots"][:5]):
            num=QLabel(str(i+1)); en=QCheckBox(); en.setChecked(sl.get("enabled",False)); en.toggled.connect(lambda v,i=i:s.slotset(i,"enabled",v))
            xq=QDoubleSpinBox(); xq.setRange(0,1); xq.setSingleStep(.01); xq.setValue(float(sl.get("x",.5))); xq.valueChanged.connect(lambda v,i=i:s.slotset(i,"x",v))
            yq=QDoubleSpinBox(); yq.setRange(0,1); yq.setSingleStep(.01); yq.setValue(float(sl.get("y",.5))); yq.valueChanged.connect(lambda v,i=i:s.slotset(i,"y",v))
            aq=QDoubleSpinBox(); aq.setRange(-180,180); aq.setValue(float(sl.get("angle",0))); aq.setSuffix("°"); aq.valueChanged.connect(lambda v,i=i:s.slotset(i,"angle",v))
            for col,wd in enumerate((num,en,xq,yq,aq)): sg.addWidget(wd,i+1,col)
        for lab,wd in (("当前行放大",cur_scale),("非当前行透明度",op),("入场动画",ia),("入场速度",ispd),("出场动画",oa),("出场速度",ospd),("循环动画",la),("循环速度",lspd)):
            f.addRow(lab,wd)
        f.addRow("当前行位置",cr); f.addRow(slotbox); f.addRow(QLabel("动画设置已集中到此页；歌词样式只保留文字外观。"));
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame); scroll.setMinimumSize(0,0); scroll.setWidget(w); return scroll

    def title_settings(s):
        w=QWidget(); f=QFormLayout(w); ts=s.st["title_style"]
        s.title_edit=QLineEdit(s.st.get("title_text","")); s.title_edit.setPlaceholderText("输入歌名")
        s.title_edit.textChanged.connect(lambda v: (s.st.__setitem__("title_text",v),s.preview.update()))
        fc=QFontComboBox(); fc.setCurrentFont(QFont(ts["font"])); fc.currentFontChanged.connect(lambda ft:s.setts("font",ft.family()))
        sz=QSpinBox(); sz.setRange(10,300); sz.setValue(ts["size"]); sz.valueChanged.connect(lambda v:s.setts("size",v))
        bd=QCheckBox("加粗"); bd.setChecked(ts.get("bold",True)); bd.toggled.connect(lambda v:s.setts("bold",v))
        it=QCheckBox("倾斜"); it.setChecked(ts.get("italic",False)); it.toggled.connect(lambda v:s.setts("italic",v))
        fm=QComboBox(); fm.addItems(["纯色","渐变色","循环跑色"]); fm.setCurrentText(ts.get("fill_mode","纯色")); fm.currentTextChanged.connect(lambda v:s.setts("fill_mode",v))
        sw=QSpinBox(); sw.setRange(0,20); sw.setValue(ts.get("sw",2)); sw.valueChanged.connect(lambda v:s.setts("sw",v))
        cs=QDoubleSpinBox(); cs.setRange(0.1,10); cs.setSingleStep(0.1); cs.setValue(float(ts.get("cycle_speed",1.0))); cs.setSuffix(" 倍"); cs.valueChanged.connect(lambda v:s.setts("cycle_speed",v))
        lm=QComboBox(); lm.addItems(["横排","竖排"]); lm.setCurrentText(ts.get("layout_mode","横排")); lm.currentTextChanged.connect(lambda v:s.setts("layout_mode",v))
        lsq=QDoubleSpinBox(); lsq.setRange(-50,200); lsq.setSingleStep(1); lsq.setValue(float(ts.get("letter_spacing",0))); lsq.setSuffix(" px"); lsq.valueChanged.connect(lambda v:s.setts("letter_spacing",v))
        ga=QDoubleSpinBox(); ga.setRange(-180,180); ga.setValue(float(ts.get("gradient_angle",45))); ga.setSuffix("°"); ga.valueChanged.connect(lambda v:s.setts("gradient_angle",v))
        cp=QComboBox(); cp.addItem("颜色预设"); cp.addItems(COLOR_PRESETS); cp.currentTextChanged.connect(lambda v:s.apply_color_preset(ts,v))
        la=QComboBox(); la.addItems(ANIM_LOOP); la.setCurrentText(ts.get("loop_anim","无")); la.currentTextChanged.connect(lambda v:s.setts("loop_anim",v))
        ls=QDoubleSpinBox(); ls.setRange(.1,10); ls.setValue(float(ts.get("loop_speed",1))); ls.setSuffix(" 倍"); ls.valueChanged.connect(lambda v:s.setts("loop_speed",v))
        stay=QDoubleSpinBox(); stay.setRange(0,3600); stay.setValue(float(ts.get("stay",0))); stay.setSuffix(" 秒"); stay.valueChanged.connect(lambda v:s.setts("stay",v))
        gap=QDoubleSpinBox(); gap.setRange(0,3600); gap.setValue(float(ts.get("interval",0))); gap.setSuffix(" 秒"); gap.valueChanged.connect(lambda v:s.setts("interval",v))
        xp=QDoubleSpinBox(); xp.setRange(0,1); xp.setSingleStep(0.01); xp.setValue(ts.get("x",0.5)); xp.valueChanged.connect(lambda v:s.setts("x",v))
        yp=QDoubleSpinBox(); yp.setRange(0,1); yp.setSingleStep(0.01); yp.setValue(ts.get("y",0.12)); yp.valueChanged.connect(lambda v:s.setts("y",v))
        for lab,wd in (("歌名",s.title_edit),("字体",fc),("字号",sz),("",bd),("",it),("显示模式",lm),("字间距",lsq),("填充模式",fm),("文字颜色",s.colorbtn("color",ts)),("渐变第二颜色",s.colorbtn("color2",ts)),("渐变角度",ga),("颜色预设",cp),("描边颜色",s.colorbtn("stroke",ts)),("描边粗细",sw),("水平位置",xp),("垂直位置",yp),("循环动画",la),("循环速度",ls),("停留时长",stay),("出现间隔",gap)):
            f.addRow(lab,wd)
        f.addRow(QLabel("歌名样式和歌词样式相互独立")); return w

    def colorbtn(s, key, target=None):
        target=target or s.st
        b = QPushButton()
        paint = lambda: b.setStyleSheet(f"background:{target[key]};min-height:22px")
        def pick():
            c = QColorDialog.getColor(QColor(target[key]), s)
            if c.isValid(): target[key]=c.name(); s.preview.update(); paint()
        b.clicked.connect(pick); paint(); return b

    def setts(s, k, v): s.st["title_style"][k]=v; s.preview.update()

    def apply_color_preset(s, target, name):
        if name in COLOR_PRESETS:
            target["color"],target["color2"]=COLOR_PRESETS[name]; s.preview.update()

    def setst(s, k, v): s.st[k] = v; s.preview.update()

    def slotset(s, i, k, v): s.st["lyric_slots"][i][k]=v; s.preview.update()

    def apply_theme(s, name):
        if name not in THEMES: return
        s.st["theme"]=name; QApplication.instance().setStyleSheet(THEMES[name])

    # ---- 播放 / 时间 ----
    def _make_demo_cues(s):
        lines=["风起的日子笑看落花","雪舞的时节举杯向月","这样的心情这样的路","我们一起走过","希望你能爱我到地老天荒","希望你能陪我到海角天涯","就算一切重来我也不会改变","我选择你你选择我"]
        out=[]; t=0.6
        for text in lines:
            dur=max(2.3,len(text)*.22); step=dur/max(1,len(text)); words=[(ch,t+i*step,t+(i+1)*step) for i,ch in enumerate(text)]
            out.append(Cue(t,t+dur,text,0,words)); t+=dur+.35
        return out

    def toggle_demo(s):
        if s.demo_mode:
            s.demo_timer.stop(); s.demo_mode=False; s.demo_cues=None; s.demo_btn.setText("实时演示"); s.dur=max([c.end for c in s.cues],default=60.0); s.sl.setRange(0,int(s.dur*1000)); s.tl.fit(); s.set_t(0); s.preview.update(); s.tl.update(); return
        s.demo_mode=True; s.demo_cues=list(s.cues) if s.cues else s._make_demo_cues(); s.t=0; s.dur=max(c.end for c in s.demo_cues)+.2; s.sl.setRange(0,int(s.dur*1000)); s.tl.fit(); s.demo_btn.setText("暂停演示"); s.demo_timer.start(); s.preview.update(); s.tl.update()

    def demo_tick(s):
        if not s.demo_mode: return
        s.t+=.033
        if s.t>=s.dur: s.t=0
        s.set_t(s.t); s.preview.update()

    def on_dur(s, ms): s.dur = ms / 1000; s.sl.setRange(0, ms); s.tl.fit()

    def set_t(s, t, from_player=False):
        s.t = t
        if not s.sl.isSliderDown(): s.sl.setValue(int(t * 1000))
        s.lab.setText(fmt(t)); s.preview.update(); s.tl.update()
        x = int(s.tl.X(t)); bar = s.sc.horizontalScrollBar(); vw=s.sc.viewport().width()
        # 预览进度条、时间轴红线和时间轴横向滚动位置共用同一个 t。
        if bar.maximum()>0 and not (bar.value()+30 <= x <= bar.value()+vw-30):
            bar.setValue(max(0,min(bar.maximum(),int(x-vw*.35))))

    def seek(s, t):
        t = max(0, t); s.player.setPosition(int(t * 1000)); s.set_t(t)

    def toggle(s):
        s.player.pause() if s.player.playbackState() == QMediaPlayer.PlayingState else s.player.play()

    def open_audio(s):
        p, _ = QFileDialog.getOpenFileName(s, "选择音频", "", "音频 (*.mp3 *.wav *.flac *.m4a *.ogg)")
        if not p: return
        s.bpm_sp.setValue(0); s.off_sp.setValue(0)
        s.path = p; s.song.setText(os.path.basename(p)); s.player.setSource(QUrl.fromLocalFile(p))
        title=re.sub(r"[（(].*?[）)]", "", os.path.splitext(os.path.basename(p))[0]).strip()
        s.st["title_text"]=title
        if hasattr(s,"title_edit"): s.title_edit.setText(title)
        s.q.setText(re.sub(r"[_\-\.]+", " ", os.path.splitext(os.path.basename(p))[0]))
        try:
            import soundfile as sf
            y, sr = sf.read(p, always_2d=True); y = np.abs(y.mean(1)); n = int(sr * 0.01)
            s.env = y[:len(y) // n * n].reshape(-1, n).max(1)
        except Exception:
            s.env = None
        s.sync_audio_duration()
        s.tl.update()

    def sync_audio_duration(s):
        if not s.path: return
        duration=0.0
        try:
            import soundfile as sf
            inf=sf.info(s.path); duration=float(inf.frames/inf.samplerate)
        except Exception:
            try:
                duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",s.path],text=True).strip())
            except Exception: pass
        if duration>0:
            s.dur=duration; s.sl.setRange(0,int(duration*1000)); s.tl.fit()

    def pick_batch_input(s):
        p=QFileDialog.getExistingDirectory(s,"选择批量输入文件夹",s.batch_in.text())
        if p: s.batch_in.setText(p)

    def pick_batch_output(s):
        p=QFileDialog.getExistingDirectory(s,"选择批量输出文件夹",s.batch_out.text())
        if p: s.batch_out.setText(p)

    def import_video(s):
        p, _ = QFileDialog.getOpenFileName(s, "导入视频", "", "视频 (*.mp4 *.mov *.mkv *.avi *.webm)")
        if not p: return
        s.video_path=p
        s.statusBar().showMessage("已导入视频: " + os.path.basename(p), 8000)
        s.tl.update()

    def import_subtitle(s):
        p, _ = QFileDialog.getOpenFileName(s, "导入字幕", "", "字幕 (*.srt *.vtt *.txt)")
        if not p: return
        try:
            raw=open(p,encoding="utf-8-sig").read()
        except UnicodeDecodeError:
            raw=open(p,encoding="gb18030",errors="replace").read()
        pat=re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})")
        imported=[]
        blocks=re.split(r"\n\s*\n",raw.replace("\r\n","\n"))
        for block in blocks:
            m=pat.search(block)
            if not m: continue
            vals=list(m.groups()); a=int(vals[0])*3600+int(vals[1])*60+int(vals[2])+int(vals[3].ljust(3,"0")[:3])/1000
            b=int(vals[4])*3600+int(vals[5])*60+int(vals[6])+int(vals[7].ljust(3,"0")[:3])/1000
            text="\n".join(x.strip() for x in block[m.end():].splitlines() if x.strip() and not re.fullmatch(r"\d+",x.strip()))
            text=re.sub(r"<[^>]+>","",text).strip()
            if text and b>a: imported.append(Cue(a,b,text,0))
        if not imported:
            return QMessageBox.information(s,"导入字幕","没有识别到有效的 SRT/VTT 时间轴字幕")
        s.cues=[c for c in s.cues if c.track!=0]+imported
        s.dur=max(s.dur,max(c.end for c in imported)); s.sl.setRange(0,int(s.dur*1000)); s.tl.fit(); s.refill()
        s.statusBar().showMessage(f"已导入字幕: {os.path.basename(p)}（{len(imported)}条）",8000)

    # ---- 字幕数据 ----
    def refill(s):
        s.cues.sort(key=lambda c: c.start); s.blk = True
        s.tb.setRowCount(len(s.cues))
        for i, c in enumerate(s.cues):
            for j, v in enumerate((c.text, fmt(c.start), fmt(c.end))): s.tb.setItem(i, j, QTableWidgetItem(v))
        s.blk = False; s.tl.update(); s.preview.update()

    def cell(s, r, col):
        if s.blk or r >= len(s.cues): return
        c, v = s.cues[r], s.tb.item(r, col).text()
        try:
            if col == 0: c.text = v
            elif col == 1: c.start = parse(v)
            else: c.end = parse(v)
        except ValueError:
            pass
        s.tl.update(); s.preview.update()

    def sel(s, c):
        s.cur = c
        if c in s.cues: s.tb.selectRow(s.cues.index(c))

    def add_cue(s):
        tr = s.tl.active if s.tracks[s.tl.active][1] == "sub" else 0
        s.cues.append(Cue(s.t, s.t + 2, "新字幕", tr)); s.refill()

    def add_track(s, kind):
        n = sum(1 for t in s.tracks if t[1] == kind) + 1
        s.tracks.append([{"sub": "字幕", "video": "视频", "audio": "音频"}[kind] + str(n), kind]); s.tl.fit()

    def export_srt(s):
        p, _ = QFileDialog.getSaveFileName(s, "导出", "", "SRT (*.srt)")
        if not p: return
        def ts(t): return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(t % 1 * 1000):03d}"
        with open(p, "w", encoding="utf-8") as f:
            for i, c in enumerate(sorted(s.cues, key=lambda c: c.start), 1):
                f.write(f"{i}\n{ts(c.start)} --> {ts(c.end)}\n{c.text}\n\n")

    # ---- 歌词 / 识别 ----
    def dl_lyrics(s):
        src, title, txt = fetch_lyrics(s.q.text())
        if txt:
            s.lyrics.setPlainText(txt); s.statusBar().showMessage(f"歌词来源: {src} | {title}", 8000)
        else:
            QMessageBox.information(s, "歌词", "各来源的歌词都太少或没找到, 改一下搜索词再试")

    def run_asr(s):
        if not s.path: return QMessageBox.information(s, "提示", "先导入音频")
        s.btn_asr.setEnabled(False); s.bar.setValue(0)
        order = s.get_order()
        s.worker = Worker(s.path, s.lyrics.toPlainText(), s.rap.isChecked(), order,
                          s.bpm_sp.value(), s.off_sp.value(), int(s.an_grid_cb.currentText()), s.pinyin_fix.isChecked())
        s.worker.lab.connect(s.set_cells); s.worker.found.connect(s.set_bpm)
        s.worker.prog.connect(s.bar.setValue); s.worker.lyr.connect(s.lyrics.setPlainText)
        s.worker.done.connect(s.asr_done); s.worker.err.connect(s.asr_err); s.worker.start()

    def get_order(s):
        mp = {"说唱": "rap", "唱歌": "sing"}
        return [mp.get(x.strip(), "sing") for x in s.order.text().replace("，", ",").split(",") if x.strip()] or ["sing"]

    def set_bpm(s, b, o): s.bpm_sp.setValue(b); s.off_sp.setValue(o)

    def pick_bg(s):
        p, _ = QFileDialog.getOpenFileName(s, "背景图片", "", "图片 (*.png *.jpg *.jpeg *.bmp)")
        if p: s.setst("bg_img", p)

    def run_bg(s, w):
        s.job = w; s.bar.setValue(0)
        w.prog.connect(s.bar.setValue); w.done.connect(s.fin)
        if hasattr(w, "err"): w.err.connect(s.fail)
        w.start()

    def fin(s, r):
        s.bar.setValue(100)
        QMessageBox.information(s, "完成", r if isinstance(r, str) else "批量完成" + ("\n失败:\n" + "\n".join(r) if r else ""))

    def fail(s, m): QMessageBox.critical(s, "失败", m)

    def export_mp4(s):
        if not s.path or not s.cues: return QMessageBox.information(s, "提示", "先导入音频并识别/添加字幕")
        out, _ = QFileDialog.getSaveFileName(s, "导出MP4", os.path.splitext(s.path)[0] + ".mp4", "MP4 (*.mp4)")
        if out: s.run_bg(RenderWorker(s.path, list(s.cues), dict(s.st), out))

    def batch(s):
        in_dir=s.batch_in.text().strip(); out_dir=s.batch_out.text().strip()
        if in_dir:
            fs=[os.path.join(in_dir,n) for n in sorted(os.listdir(in_dir)) if n.lower().endswith((".mp3",".wav",".flac",".m4a",".ogg"))]
        else:
            fs, _ = QFileDialog.getOpenFileNames(s, "选择多首音频", "", "音频 (*.mp3 *.wav *.flac *.m4a *.ogg)")
        if not fs: return QMessageBox.information(s,"批量生成","输入文件夹中没有找到音频文件")
        d=out_dir or QFileDialog.getExistingDirectory(s,"选择输出文件夹")
        if d:
            s.batch_in.setText(os.path.dirname(fs[0]) if in_dir else in_dir); s.batch_out.setText(d); s.run_bg(BatchWorker(fs,d,dict(s.st),s.rap.isChecked(),s.get_order()))

    def closeEvent(s, e):
        try: json.dump(s.st, open(STYLE_FILE, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception: pass
        super().closeEvent(e)

    def set_cells(s, c): s.cells = c; s.tl.update()

    def detect_bpm(s):
        if not s.path: return QMessageBox.information(s, "提示", "先导入音频")
        import librosa
        y, sr = librosa.load(s.path, sr=22050, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        s.bpm_sp.setValue(float(np.atleast_1d(tempo)[0]))
        if len(beats): s.off_sp.setValue(float(librosa.frames_to_time(beats[0], sr=sr)))

    def asr_done(s, cues):
        s.sync_audio_duration(); s.cues = [c for c in s.cues if c.track != 0] + cues; s.refill()
        s.bar.setValue(100); s.btn_asr.setEnabled(True)
        s.info.setText(f"识别完成: {len(cues)} 句字幕 | 拼音纠正 {getattr(s.worker,'fixed_count',0)} 句 | BPM {s.bpm_sp.value():.1f}")

    def asr_err(s, m):
        s.btn_asr.setEnabled(True); QMessageBox.critical(s, "识别失败", m)


if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    w = Main(); w.show(); sys.exit(app.exec())
