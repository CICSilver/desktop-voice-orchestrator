"""Generate anti-distortion TTS synthesis prompt manifests for M3 personal ASR vocabulary expansion.

Features:
  * Optionally reads the user's local NetEase Cloud Music SQLite database (`webdb.dat`, copied
    read-only to a temp file so a running client is never locked) or a custom JSON music library,
    merged with a curated Mandarin pop artist/track seed bank and common desktop app aliases
    (including mixed Chinese-English desktop apps: `Steam`, `Clash`, `Discord`, `QQ`,
    `PowerShell`, `VS Code`, plus `浏览器`, `任务管理器`, `终端`, `网易云音乐`, etc.);
  * Applies strict TTS front-end text normalization:
      - Converts ALL Arabic numerals and `%` into natural spoken Chinese (`5%` -> `百分之五`,
        `15%` -> `百分之十五`, `100%` -> `百分之百`); zero `[0-9%]` characters allowed;
      - Enforces 4 to 20 spoken units per utterance (`4 <= speech_unit_count <= 20`, counting
        each Chinese character as 1 unit and each whitelisted English app word by its syllable
        equivalent so `"小克，打开 Steam"` or `"打开 QQ"` is never wrongly discarded);
      - Strips all exotic punctuation, brackets, quotes, kana, or wild Latin garbage: only clean
        CJK ideographs, natural intra-sentence Chinese commas `，`, and whitelisted English app
        tokens (`Steam`, `Clash`, `Discord`, `QQ`, `PowerShell`, `VS`, `Code`) are emitted;
  * Balances four complementary categories:
      1. `new_cmd_boost`: reinforces previously under-trained commands (`换歌`, `切歌`, `放每日推荐`,
         `顺序播放`, `列表循环`, `单曲循环`, `随机播放`, `打开桌面歌词`, `关闭桌面歌词`, `收藏这首歌`);
      2. `music_search_nl`: natural-language song/artist/playlist requests (prefix `小克`, suffix
         `小克`, and bare follow-up turns);
      3. `colloquial_control_multi`: conversational volume/playback/desktop-app controls (both
         single-step app launches like `"小克，打开 Steam"` and cross-domain compound chains like
         `"小克，打开 Steam，再放点背景音乐"`, `"先打开 Clash，然后把声音关小一点，小克"`);
      4. `negative_colloquial`: everyday non-command statements mentioning music/audio/game words
         without wake words, preventing command-hallucination bias.

Usage:
    python tools/build_nl_tts_prompts.py \\
        --out-dir "H:/我的云端硬盘/desktop-voice-orchestrator-data/m3_tts_prep"

    python tools/build_nl_tts_prompts.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PURE_CJK_RE = re.compile(r"^[\u4e00-\u9fa5]+$")
WAKE_HOMOPHONES_RE = re.compile(r"小[克客课可科柯刻壳]")

# Strictly whitelisted English desktop application words allowed in mixed ZH-EN TTS prompts
ALLOWED_EN_APP_WORDS: dict[str, str] = {
    "steam": "Steam",
    "clash": "Clash",
    "discord": "Discord",
    "qq": "QQ",
    "powershell": "PowerShell",
    "vs": "VS",
    "code": "Code",
}

# Spoken-syllable unit weights for length / speaking-rate gating
EN_WORD_UNIT_WEIGHTS: dict[str, int] = {
    "Steam": 2,
    "Clash": 2,
    "Discord": 2,
    "QQ": 2,
    "PowerShell": 3,
    "VS": 2,
    "Code": 2,
}

# Every clause between `，` must consist only of CJK chars and space-separated whitelisted Latin tokens
VALID_TTS_CHARS_RE = re.compile(r"^[\u4e00-\u9fa5A-Za-z， ]+$")

CN_DIGITS = "零一二三四五六七八九"

# Curated high-frequency Mandarin artists & representative pure-Chinese song titles
SEED_ARTIST_SONGS: list[tuple[str, list[str]]] = [
    ("周杰伦", ["晴天", "七里香", "夜曲", "稻香", "青花瓷", "告白气球", "兰亭序", "一路向北", "简单爱", "花海", "搁浅", "反方向的钟"]),
    ("陈奕迅", ["十年", "孤勇者", "浮夸", "红玫瑰", "好久不见", "富士山下", "爱情转移", "不要说话", "陪你度过漫长岁月", "K歌之王"]),
    ("林俊杰", ["江南", "修炼爱情", "不为谁而作的歌", "曹操", "小酒窝", "可惜没如果", "背对背拥抱", "美人鱼", "一千年以后"]),
    ("邓紫棋", ["泡沫", "光年之外", "来自天堂的魔鬼", "喜欢你", "句号", "多远都要在一起", "倒数", "画"]),
    ("王菲", ["红豆", "匆匆那年", "传奇", "如愿", "人间", "我愿意", "流年", "容易受伤的女人"]),
    ("孙燕姿", ["遇见", "天黑黑", "我怀念的", "绿光", "开始懂了", "逆光", "第一天"]),
    ("毛不易", ["消愁", "像我这样的人", "平凡的一天", "借", "无名的人", "盛夏", "一荤一素"]),
    ("薛之谦", ["演员", "丑八怪", "认真的雪", "刚刚好", "天外来物", "绅士", "意外"]),
    ("李荣浩", ["模特", "李白", "年少有为", "麻雀", "乌梅子酱", "不将就", "戒烟"]),
    ("周深", ["大鱼", "达拉崩吧", "灯火里的中国", "小美满", "光亮", "起风了", "花开忘忧"]),
    ("莫文蔚", ["忽然之间", "这世界那么多人", "盛夏的果实", "阴天", "慢慢喜欢你", "他不爱我"]),
    ("张学友", ["吻别", "一路上有你", "遥远的她", "她来听我的演唱会", "一千个伤心的理由"]),
    ("五月天", ["倔强", "突然好想你", "知足", "温柔", "恋爱ing", "干杯", "私奔到月球", "我不愿让你一个人"]),
    ("陶喆", ["普通朋友", "小镇姑娘", "就是爱你", "爱很简单", "黑色柳丁", "找自己"]),
    ("王力宏", ["大城小爱", "唯一", "花田错", "需要人陪", "改变自己", "心中的日月"]),
    ("蔡依林", ["日不落", "倒带", "说爱你", "爱情三十六计", "舞娘", "柠檬草的味道"]),
    ("张杰", ["逆战", "天下", "这就是爱", "他不懂", "只要平凡", "最美的太阳"]),
    ("朴树", ["平凡之路", "那些花儿", "生如夏花", "白桦林", "清白之年"]),
    ("赵雷", ["成都", "南方姑娘", "鼓楼", "我记得", "画", "少年锦时"]),
    ("李健", ["贝加尔湖畔", "风吹麦浪", "传奇", "异乡人", "假如爱有天意"]),
    ("梁静茹", ["勇气", "可惜不是你", "暖暖", "会呼吸的痛", "宁夏", "分手快乐"]),
    ("许嵩", ["有何不可", "素颜", "断桥残雪", "清明雨上", "雅俗共赏", "惊鸿一面"]),
    ("汪苏泷", ["一笑倾城", "有点甜", "万有引力", "小星星", "大娱乐家"]),
    ("郭静", ["往未来飞的客机", "下一个天亮", "心墙", "在树上唱歌"]),
    ("告五人", ["爱人错过", "给你一瓶魔法药水", "带我去找夜生活", "唯一", "披星戴月的想你"]),
    ("房东的猫", ["云烟成雨", "秋酿", "美好事物", "下一站茶山刘"]),
    ("张惠妹", ["听海", "哭不出来", "如果你也听说", "连名带姓", "三天三夜"]),
    ("伍佰", ["突然的自我", "挪威的森林", "泪桥", "LastDance", "浪人情歌"]),
    ("蔡琴", ["渡口", "被遗忘的时光", "恰似你的温柔", "你的眼神"]),
    ("草东没有派对", ["山海", "大风吹", "烂泥"]),
]

PLAYLIST_AND_GENRES: list[str] = [
    "我喜欢的音乐", "每日推荐", "私人漫游", "私人雷达",
    "纯音乐", "轻音乐", "钢琴曲", "白噪音", "助眠音乐",
    "华语经典", "粤语老歌", "民谣歌单", "摇滚歌单", "爵士乐",
    "古典音乐", "学习背景音乐", "写代码歌单", "运动歌单", "放松音乐",
]

DESKTOP_APPS_ZH: list[str] = [
    "网易云音乐", "音乐播放器", "浏览器", "终端", "记事本",
    "计算器", "任务管理器", "文件资源管理器", "系统设置", "微信",
]

DESKTOP_APPS_EN: list[str] = [
    "Steam", "Clash", "Discord", "QQ", "PowerShell", "VS Code",
]

DESKTOP_APPS: list[str] = DESKTOP_APPS_EN + DESKTOP_APPS_ZH


def int_to_chinese(n: int) -> str:
    """Converts an integer 0..9999 into natural spoken Mandarin."""
    if n < 0:
        return "负" + int_to_chinese(-n)
    if n < 10:
        return CN_DIGITS[n]
    if n == 10:
        return "十"
    if n < 20:
        return "十" + CN_DIGITS[n - 10]
    if n < 100:
        tens, ones = divmod(n, 10)
        return CN_DIGITS[tens] + "十" + (CN_DIGITS[ones] if ones else "")
    if n == 100:
        return "一百"
    if n < 1000:
        hundreds, rem = divmod(n, 100)
        if rem == 0:
            return CN_DIGITS[hundreds] + "百"
        if rem < 10:
            return CN_DIGITS[hundreds] + "百零" + CN_DIGITS[rem]
        tens, ones = divmod(rem, 10)
        return CN_DIGITS[hundreds] + "百" + CN_DIGITS[tens] + "十" + (CN_DIGITS[ones] if ones else "")
    thousands, rem = divmod(n, 1000)
    if rem == 0:
        return CN_DIGITS[thousands] + "千"
    if rem < 100:
        return CN_DIGITS[thousands] + "千零" + int_to_chinese(rem)
    return CN_DIGITS[thousands] + "千" + int_to_chinese(rem)


def normalize_for_tts(text: str) -> str:
    """Converts `%` and Arabic digits into spoken Chinese, canonicalizes whitelisted English app words, and sanitizes spacing."""
    # 100% -> 百分之百, 15% -> 百分之十五
    def repl_pct(m: re.Match) -> str:
        val = int(m.group(1))
        return "百分之百" if val == 100 else f"百分之{int_to_chinese(val)}"

    out = re.sub(r"(\d+)\s*%", repl_pct, text)
    out = re.sub(r"\d+", lambda m: int_to_chinese(int(m.group(0))), out)
    # Expand compact "vscode" -> "VS Code" before word tokenization
    out = re.sub(r"(?<![A-Za-z])vscode(?![A-Za-z])", "VS Code", out, flags=re.IGNORECASE)
    out = re.sub(r"[;；!！?？。、,]+", "，", out)

    # Canonicalize whitelisted English app words (e.g., steam -> Steam, powershell -> PowerShell)
    def repl_en_word(m: re.Match) -> str:
        w = m.group(0)
        return ALLOWED_EN_APP_WORDS.get(w.lower(), w)

    out = re.sub(r"[A-Za-z]+", repl_en_word, out)

    # Collapse whitespace: remove spaces between CJK chars and around `，`,
    # keep exactly one space between CJK <-> Latin and Latin <-> Latin tokens.
    out = re.sub(r"\s*，\s*", "，", out)
    out = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", out)
    out = re.sub(r"([\u4e00-\u9fa5])\s*([A-Za-z])", r"\1 \2", out)
    out = re.sub(r"([A-Za-z])\s*([\u4e00-\u9fa5])", r"\1 \2", out)
    out = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1 \2", out)
    out = re.sub(r" +", " ", out).strip(" ，")
    out = re.sub(r"，+", "，", out).strip("， ")
    return out


def hanzi_only(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fa5]", text))


def normalize_target_text(text: str) -> str:
    """Normalized ASR target / deduplication key: lowercase ASCII letters + CJK ideographs, no spaces/punctuation."""
    return "".join(re.findall(r"[a-z\u4e00-\u9fa5]", text.lower()))


def extract_en_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text)


def speech_unit_count(tts_text: str) -> int:
    """Counts effective spoken length: 1 per Chinese character + syllable weight per whitelisted English token."""
    hz_count = len(hanzi_only(tts_text))
    en_units = sum(EN_WORD_UNIT_WEIGHTS.get(tok, max(2, len(tok) // 3)) for tok in extract_en_tokens(tts_text))
    return hz_count + en_units


def is_valid_tts_prompt(tts_text: str, min_chars: int = 4, max_chars: int = 20) -> bool:
    if not tts_text or not VALID_TTS_CHARS_RE.fullmatch(tts_text):
        return False
    if re.search(r"[0-9%]", tts_text):
        return False
    if "，，" in tts_text or "  " in tts_text or tts_text.startswith(("，", " ")) or tts_text.endswith(("，", " ")):
        return False
    if " ，" in tts_text or "， " in tts_text:
        return False

    en_tokens = extract_en_tokens(tts_text)
    for tok in en_tokens:
        if ALLOWED_EN_APP_WORDS.get(tok.lower()) != tok:
            return False

    # Require at least 2 Chinese characters in every prompt
    if len(hanzi_only(tts_text)) < 2:
        return False

    units = speech_unit_count(tts_text)
    return min_chars <= units <= max_chars


def clean_cjk_entity(name: str, min_len: int = 2, max_len: int = 7) -> str | None:
    """Extracts a clean pure-Chinese song or artist name, rejecting kana/Latin/symbols."""
    name = name.strip()
    # Strip trailing "Official" if artist name is pure Chinese before it (e.g. 洛天依Official -> 洛天依)
    name = re.sub(r"(?:Official|工作室|乐队)$", "", name).strip()
    # Strip parenthetical suffixes like (Live), （翻自 xxx）
    name = re.split(r"[\(（\[【\-—·:：/]", name)[0].strip()
    name = normalize_for_tts(name).replace("，", "").replace(" ", "")
    if PURE_CJK_RE.fullmatch(name) and min_len <= len(name) <= max_len:
        # Exclude names containing wake-word homophones so search prompts don't double-wake strangely
        if not WAKE_HOMOPHONES_RE.search(name):
            return name
    return None


def load_music_from_webdb(webdb_path: Path | None) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Reads (artist, song) pairs, solo artists, and solo songs from NetEase webdb.dat safely."""
    pairs: list[tuple[str, str]] = []
    artists_set: set[str] = set()
    songs_set: set[str] = set()

    if webdb_path is None or not webdb_path.is_file():
        return pairs, [], []

    tmp_file = Path(tempfile.gettempdir()) / f"dvo_m3_webdb_{os.getpid()}.sqlite"
    try:
        shutil.copy2(webdb_path, tmp_file)
        conn = sqlite3.connect(str(tmp_file))
        cur = conn.cursor()
        for table in ("historyTracks", "dbTrack"):
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not cur.fetchone():
                continue
            cur.execute(f"SELECT jsonStr FROM [{table}]")
            for (jstr,) in cur.fetchall():
                if not jstr:
                    continue
                try:
                    obj = json.loads(jstr)
                except Exception:
                    continue
                song = clean_cjk_entity(str(obj.get("name", "")), min_len=2, max_len=7)
                raw_artists = obj.get("artists") or obj.get("ar") or []
                clean_artists = []
                for a in raw_artists:
                    if isinstance(a, dict):
                        ca = clean_cjk_entity(str(a.get("name", "")), min_len=2, max_len=5)
                        if ca:
                            clean_artists.append(ca)
                if song:
                    songs_set.add(song)
                for ca in clean_artists[:2]:
                    artists_set.add(ca)
                    if song:
                        pairs.append((ca, song))
        conn.close()
    except Exception as exc:
        print(f"Warning: could not read webdb.dat ({exc}); falling back to seed bank.", file=sys.stderr)
    finally:
        tmp_file.unlink(missing_ok=True)

    return pairs, sorted(artists_set), sorted(songs_set)


def build_music_catalog(webdb_path: Path | None, custom_json: Path | None) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    pair_list: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    artist_set: set[str] = set()
    song_set: set[str] = set()

    def add_pair(artist: str, song: str) -> None:
        ca = clean_cjk_entity(artist, 2, 5)
        cs = clean_cjk_entity(song, 2, 7)
        if ca:
            artist_set.add(ca)
        if cs:
            song_set.add(cs)
        if ca and cs and (ca, cs) not in seen_pairs:
            seen_pairs.add((ca, cs))
            pair_list.append((ca, cs))

    # 1. User's personal NetEase history & library first!
    db_pairs, db_artists, db_songs = load_music_from_webdb(webdb_path)
    for a, s in db_pairs:
        add_pair(a, s)
    for a in db_artists:
        artist_set.add(a)
    for s in db_songs:
        song_set.add(s)

    # 2. Optional custom JSON ({ "artists": [...], "songs": [...], "pairs": [["周杰伦", "晴天"], ...] })
    if custom_json and custom_json.is_file():
        cdata = json.loads(custom_json.read_text("utf-8"))
        for a, s in cdata.get("pairs", []):
            add_pair(a, s)
        for a in cdata.get("artists", []):
            ca = clean_cjk_entity(a, 2, 5)
            if ca:
                artist_set.add(ca)
        for s in cdata.get("songs", []):
            cs = clean_cjk_entity(s, 2, 7)
            if cs:
                song_set.add(cs)

    # 3. Curated Mandarin seed library
    for artist, tracks in SEED_ARTIST_SONGS:
        for t in tracks:
            add_pair(artist, t)

    return pair_list, sorted(artist_set), sorted(song_set)


def wrap_wake(body: str, mode: str, wake: str = "小克") -> str:
    if mode == "prefix":
        return f"{wake}，{body}"
    if mode == "suffix":
        return f"{body}，{wake}"
    return body


def generate_candidate_pools(
    pairs: list[tuple[str, str]],
    artists: list[str],
    songs: list[str],
    rng: random.Random,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "new_cmd_boost": [],
        "music_search_nl": [],
        "colloquial_control_multi": [],
        "negative_colloquial": [],
    }
    seen_norm: set[str] = set()

    def register(category: str, raw_text: str, subtag: str, entities: list[str] | None = None) -> None:
        tts_text = normalize_for_tts(raw_text)
        if not is_valid_tts_prompt(tts_text, min_chars=4, max_chars=20):
            return
        norm = normalize_target_text(tts_text)
        if norm in seen_norm:
            return
        seen_norm.add(norm)
        buckets[category].append({
            "category": category,
            "subtag": subtag,
            "tts_text": tts_text,
            "text": norm,
            "char_count": speech_unit_count(tts_text),
            "entities": entities or [],
            "strict_exact": not bool(entities),
        })

    # ------------------------------------------------------------------
    # 1. Category: new_cmd_boost (missed / new commands reinforcement)
    # ------------------------------------------------------------------
    core_new_cmds = [
        ("next", ["换歌", "切歌", "换一首", "换一首歌", "下一首", "下一首歌", "切到下一首", "帮我换首歌",
                  "换下一首歌", "切一下歌", "换别的歌听", "这首听腻了换一首", "跳到下一首歌"]),
        ("daily", ["放每日推荐", "播放每日推荐", "打开每日推荐", "每日推荐", "来点每日推荐",
                   "听今天的每日推荐", "帮我放每日推荐", "放一下每日推荐", "开始播放每日推荐"]),
        ("play_mode", ["顺序播放", "列表循环", "单曲循环", "随机播放",
                       "改成顺序播放", "设为顺序播放", "切换到顺序播放", "开启顺序播放",
                       "改成列表循环", "设为列表循环", "切换到列表循环", "开启列表循环",
                       "改成单曲循环", "设为单曲循环", "切换到单曲循环", "把这首歌单曲循环",
                       "改成随机播放", "设为随机播放", "切换到随机播放", "开启随机播放模式"]),
        ("lyrics", ["打开桌面歌词", "关闭桌面歌词", "显示桌面歌词", "隐藏桌面歌词",
                    "把桌面歌词打开", "把桌面歌词关掉", "把桌面歌词关了", "开启桌面歌词",
                    "关掉桌面歌词", "帮我打开桌面歌词", "帮我关闭桌面歌词", "桌面歌词打开一下",
                    "桌面歌词关一下", "把歌词显示出来", "把桌面上的歌词关掉"]),
        ("like", ["收藏这首歌", "喜欢这首歌", "我喜欢这首歌", "收藏一下这首歌", "把这首歌收藏一下",
                  "把这首歌加到喜欢", "给这首歌点个红心", "收藏当前这首歌", "帮我收藏这首歌",
                  "这首歌不错收藏一下", "加到我喜欢的音乐", "把当前歌曲收藏"]),
    ]
    particles = ["", "吧", "呀", "呗"]
    polite_prefixes = ["", "帮我", "麻烦", "请", "先"]

    for subtag, phrases in core_new_cmds:
        for ph in phrases:
            for wmode in ("prefix", "suffix", "bare"):
                register("new_cmd_boost", wrap_wake(ph, wmode), subtag)
                for p in particles[1:]:
                    if not ph.endswith(("吧", "呀", "呗", "一下")):
                        register("new_cmd_boost", wrap_wake(ph + p, wmode), subtag)
            for pre in polite_prefixes[1:]:
                if not ph.startswith(("帮我", "把", "这首", "我")):
                    for wmode in ("prefix", "suffix", "bare"):
                        register("new_cmd_boost", wrap_wake(pre + ph, wmode), subtag)

    # Compound combinations specifically pairing two new commands so short words like 换歌/切歌/顺序播放
    # appear in diverse acoustic contexts
    new_cmd_atoms = [
        ("换歌", "next"), ("切歌", "next"), ("换一首歌", "next"), ("下一首", "next"),
        ("放每日推荐", "daily"), ("打开每日推荐", "daily"), ("播放每日推荐", "daily"),
        ("顺序播放", "mode"), ("列表循环", "mode"), ("单曲循环", "mode"), ("随机播放", "mode"),
        ("打开桌面歌词", "lyrics"), ("关闭桌面歌词", "lyrics"), ("显示桌面歌词", "lyrics"), ("隐藏桌面歌词", "lyrics"),
        ("收藏这首歌", "like"), ("喜欢这首歌", "like"), ("把这首歌收藏", "like"),
    ]
    connectors = ["，再", "，然后", "，接着", "，顺便"]
    for a1, g1 in new_cmd_atoms:
        for a2, g2 in new_cmd_atoms:
            if g1 == g2:
                continue
            for conn in connectors:
                chain = f"{a1}{conn}{a2}"
                register("new_cmd_boost", wrap_wake(chain, "prefix"), f"{g1}+{g2}")
                register("new_cmd_boost", wrap_wake(chain, "suffix"), f"{g1}+{g2}")
                register("new_cmd_boost", chain, f"{g1}+{g2}")

    # ------------------------------------------------------------------
    # 2. Category: music_search_nl (artist/song/playlist natural requests)
    # ------------------------------------------------------------------
    pair_templates_prefix = [
        "小克，我想听{a}的{s}",
        "小克，播放{a}的{s}",
        "小克，帮我放{a}的{s}",
        "小克，来一首{a}的{s}",
        "小克，放首{a}唱的{s}",
        "小克，搜一下{a}的{s}",
        "小克，给我放{a}的{s}",
        "小克，点一首{a}的{s}",
    ]
    pair_templates_suffix = [
        "我想听{a}的{s}，小克",
        "播放{a}的{s}，小克",
        "帮我放首{a}的{s}，小克",
        "来一首{a}的{s}，小克",
        "放一下{a}的{s}，小克",
    ]
    pair_templates_bare = [
        "我想听{a}的{s}",
        "帮我播放{a}的{s}",
        "来一首{a}的{s}",
        "放一首{a}的{s}吧",
        "切换成{a}的{s}",
    ]

    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)
    for artist, song in shuffled_pairs:
        t1 = rng.choice(pair_templates_prefix).format(a=artist, s=song)
        t2 = rng.choice(pair_templates_suffix).format(a=artist, s=song)
        t3 = rng.choice(pair_templates_bare).format(a=artist, s=song)
        register("music_search_nl", t1, "artist_song", [artist, song])
        register("music_search_nl", t2, "artist_song", [artist, song])
        register("music_search_nl", t3, "artist_song", [artist, song])

    # Add extra rich templates for top iconic pairs so common requests get thorough phrasing diversity
    for artist, tracks in SEED_ARTIST_SONGS[:18]:
        for song in tracks[:5]:
            ca = clean_cjk_entity(artist, 2, 5)
            cs = clean_cjk_entity(song, 2, 7)
            if not (ca and cs):
                continue
            for tpl in pair_templates_prefix + pair_templates_suffix[:3] + pair_templates_bare[:2]:
                register("music_search_nl", tpl.format(a=ca, s=cs), "artist_song_core", [ca, cs])

    artist_templates = [
        "小克，我想听{a}的歌",
        "小克，随便放几首{a}的歌",
        "小克，来点{a}的音乐",
        "小克，播放{a}的热门歌曲",
        "帮我放{a}的歌，小克",
        "我想听{a}唱的歌，小克",
        "来几首{a}的歌吧",
        "播放{a}的歌单",
    ]
    shuffled_artists = list(artists)
    rng.shuffle(shuffled_artists)
    for artist in shuffled_artists[:160]:
        for tpl in rng.sample(artist_templates, k=min(4, len(artist_templates))):
            register("music_search_nl", tpl.format(a=artist), "artist_only", [artist])

    song_templates = [
        "小克，我想听{s}这首歌",
        "小克，帮我播放{s}",
        "小克，放一首{s}",
        "播放{s}这首歌，小克",
        "帮我搜一下{s}，小克",
        "我想听一首{s}",
    ]
    shuffled_songs = list(songs)
    rng.shuffle(shuffled_songs)
    for song in shuffled_songs[:220]:
        for tpl in rng.sample(song_templates, k=min(3, len(song_templates))):
            register("music_search_nl", tpl.format(s=song), "song_only", [song])

    playlist_templates = [
        "小克，播放{g}",
        "小克，放一点{g}",
        "小克，我想听{g}",
        "小克，来点{g}吧",
        "帮我放{g}，小克",
        "打开{g}，小克",
        "播放{g}吧",
        "给我来点安静的{g}",
    ]
    for genre in PLAYLIST_AND_GENRES:
        for tpl in playlist_templates:
            register("music_search_nl", tpl.format(g=genre), "playlist_genre", [])

    # ------------------------------------------------------------------
    # 3. Category: colloquial_control_multi (volume %, natural control, 2-step chains, EN/ZH apps)
    # ------------------------------------------------------------------
    pct_values = [1, 2, 3, 5, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 70, 80, 100]
    rel_vol_up_verbs = ["调大音量", "增加音量", "把声音调大", "把音量开大", "声音加大", "音量升高"]
    rel_vol_down_verbs = ["降低音量", "调小音量", "把声音关小", "把音量调低", "声音减小", "把声音调小"]
    abs_vol_verbs = ["把音量调到", "音量设为", "声音调到", "把声音开到"]

    for pct in pct_values:
        if pct <= 25:
            for v in rel_vol_up_verbs:
                register("colloquial_control_multi", f"小克，{v}{pct}%", "vol_pct_up")
                register("colloquial_control_multi", f"{v}{pct}%，小克", "vol_pct_up")
                register("colloquial_control_multi", f"{v}{pct}%", "vol_pct_up")
            for v in rel_vol_down_verbs:
                register("colloquial_control_multi", f"小克，{v}{pct}%", "vol_pct_down")
                register("colloquial_control_multi", f"{v}{pct}%，小克", "vol_pct_down")
                register("colloquial_control_multi", f"{v}{pct}%", "vol_pct_down")
        if pct in (10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100):
            for v in abs_vol_verbs:
                register("colloquial_control_multi", f"小克，{v}{pct}%", "vol_pct_abs")
                register("colloquial_control_multi", f"{v}{pct}%，小克", "vol_pct_abs")

    colloquial_playback = [
        "太吵了把声音关小一点", "声音太小了调大一点", "音乐有点吵小声一点",
        "把音乐先停一下", "先别放歌了", "继续放刚才那首歌", "把音乐声音开大一点",
        "先把电脑静音一下", "恢复刚才的音量", "把背景音乐关掉吧", "继续播放音乐吧",
        "帮我把音乐暂停一下", "开始放点歌听听", "声音稍微轻一点点", "声音再响一点",
    ]
    for cp in colloquial_playback:
        register("colloquial_control_multi", f"小克，{cp}", "colloquial_single")
        register("colloquial_control_multi", f"{cp}，小克", "colloquial_single")
        register("colloquial_control_multi", cp, "colloquial_single")

    # Rich two-step compound commands (volume + media)
    step1_vol = [
        "先把声音关小5%", "先把声音关小10%", "先把声音关小15%", "先把音量降低10%",
        "先把声音调大5%", "先把音量增加10%", "先把音量调大15%", "先把声音调小一点",
        "先把声音开大一点", "把音量降低20%", "把声音调大8%", "先把音量调小12%",
    ]
    step2_media = [
        "再换首歌", "再切歌", "然后下一首", "再放每日推荐", "再打开桌面歌词",
        "再关闭桌面歌词", "再收藏这首歌", "再改成单曲循环", "再设为随机播放",
        "然后顺序播放", "再切换列表循环", "然后继续播放音乐", "再暂停音乐",
    ]
    for s1 in step1_vol:
        for s2 in step2_media:
            register("colloquial_control_multi", f"小克，{s1}，{s2}", "two_step_vol_media")
            register("colloquial_control_multi", f"{s1}，{s2}，小克", "two_step_vol_media")

    # Two-step: control + play specific song
    top_short_pairs = [(a, s) for a, s in shuffled_pairs if len(a) + len(s) <= 6][:60]
    for a, s in top_short_pairs:
        register("colloquial_control_multi", f"小克，声音小一点，再放{a}的{s}", "two_step_song", [a, s])
        register("colloquial_control_multi", f"小克，打开歌词，再放{a}的{s}", "two_step_song", [a, s])

    # Desktop app launch & window management phrases (covering EN + ZH desktop apps!)
    app_single_templates = [
        "小克，打开 {app}",
        "小克，帮我打开 {app}",
        "小克，启动 {app}",
        "小克，切到 {app}",
        "小克，切换到 {app}",
        "小克，把 {app} 打开",
        "打开 {app}，小克",
        "帮我启动 {app}，小克",
        "帮我把 {app} 打开，小克",
        "切换到 {app}，小克",
        "启动 {app}，小克",
        "打开 {app}",
        "帮我打开 {app}",
        "启动 {app}",
        "切到 {app}",
        "小克，帮我启动 {app}",
    ]
    for app in DESKTOP_APPS_EN:
        for tpl in app_single_templates:
            register("colloquial_control_multi", tpl.format(app=app), "desktop_app_en")

    for app in DESKTOP_APPS_ZH:
        for tpl in app_single_templates[:12]:
            register("colloquial_control_multi", tpl.format(app=app), "desktop_app_zh")

    # Window management standalone
    for win_cmd in ("小克，最小化当前窗口", "小克，把当前窗口最小化", "把当前窗口最小化，小克"):
        register("colloquial_control_multi", win_cmd, "desktop_window")

    # Cross-domain two-step compound phrases: Mixed EN/ZH desktop app + media / volume control
    cross_domain_curated = [
        "小克，打开 Steam，再放点背景音乐",
        "先打开 Clash，然后把声音关小一点，小克",
        "小克，打开 Discord，再把音乐暂停",
        "小克，打开 PowerShell，再放点纯音乐",
        "小克，打开 VS Code，顺便放每日推荐",
        "打开 QQ，再把声音关小10%，小克",
        "小克，启动 Steam，然后播放每日推荐",
        "小克，打开 Clash，再放首歌听听",
        "小克，先把音乐暂停，再打开 Discord",
        "先把声音调小一点，再打开 Steam，小克",
        "小克，先把音量降低10%，再打开 Discord",
        "打开 VS Code，然后播放纯音乐，小克",
        "小克，启动 PowerShell，再打开桌面歌词",
        "小克，打开 QQ，顺便把音乐关小一点",
    ]
    for cd_txt in cross_domain_curated:
        sub = "two_step_app_en" if extract_en_tokens(cd_txt) else "two_step_app_zh"
        register("colloquial_control_multi", cd_txt, sub)

    app_open_clauses = [
        "打开 {app}",
        "先打开 {app}",
        "启动 {app}",
        "帮我打开 {app}",
    ]
    media_follow_clauses = [
        "再放点背景音乐",
        "再把音乐暂停",
        "然后把声音关小一点",
        "再放点纯音乐",
        "再换首歌",
        "再放每日推荐",
        "再把声音调小10%",
        "再把音量调小5%",
        "然后继续播放音乐",
        "顺便打开桌面歌词",
    ]
    for app in DESKTOP_APPS_EN + ["浏览器", "终端", "任务管理器"]:
        sub = "two_step_app_en" if app in DESKTOP_APPS_EN else "two_step_app_zh"
        for c1 in app_open_clauses:
            for c2 in media_follow_clauses:
                a_str = c1.format(app=app)
                register("colloquial_control_multi", f"小克，{a_str}，{c2}", sub)
                register("colloquial_control_multi", f"{a_str}，{c2}，小克", sub)

    # Reverse order: media/volume first -> open EN desktop app second
    media_first_clauses = [
        "先把音乐暂停",
        "先把声音关小一点",
        "先把音量调小10%",
        "先换一首歌",
        "先放点轻音乐",
    ]
    for app in DESKTOP_APPS_EN:
        for m1 in media_first_clauses:
            register("colloquial_control_multi", f"小克，{m1}，再打开 {app}", "two_step_app_en")
            register("colloquial_control_multi", f"{m1}，再启动 {app}，小克", "two_step_app_en")

    # ------------------------------------------------------------------
    # 4. Category: negative_colloquial (non-wake everyday statements with music/control/app words)
    # ------------------------------------------------------------------
    neg_templates = [
        "这首{s}的前奏听起来真舒服",
        "{a}唱的这首{s}还挺好听的",
        "我昨天把{a}的歌循环听了一下午",
        "最近每天上下班都在听{s}",
        "感觉耳机里的声音比音箱清楚多了",
        "这个游戏的背景音乐做得很有氛围",
        "刚才那段视频的音量是不是太大了",
        "等打完这一局游戏再去倒杯水喝",
        "他把桌面歌词的颜色调成了浅蓝色",
        "网易云今天的每日推荐还挺合胃口的",
        "这首歌我已经单曲循环好几天了",
        "我平时写代码的时候喜欢随机播放纯音乐",
        "外面下大雨了先把窗户关小一点",
        "电脑风扇转起来的声音有点吵人",
        "刚才那一波团战配合得真不错",
        "等会儿开会记得先把麦克风静音",
        "这个播放列表里的老歌都挺经典的",
        "我先把这几首喜欢的歌收藏到歌单里",
        "邻居放音乐的声音稍微有点大",
        "浏览器开了太多标签页有点卡顿",
        "今天 Steam 夏季特卖有好几款游戏打折",
        "刚才在 Discord 频道里听大家聊得挺热闹",
        "用 PowerShell 跑这个脚本速度还挺快的",
        "我手机上的 QQ 刚才响了好几声",
    ]
    for tpl in neg_templates:
        if "{a}" in tpl or "{s}" in tpl:
            for a, s in shuffled_pairs[:80]:
                txt = tpl.format(a=a, s=s)
                if not WAKE_HOMOPHONES_RE.search(txt):
                    register("negative_colloquial", txt, "neg_music_chat", [a, s] if "{a}" in tpl else [s])
        else:
            if not WAKE_HOMOPHONES_RE.search(tpl):
                register("negative_colloquial", tpl, "neg_daily_chat")

    return buckets


def build_pilot_25(full_items: list[dict]) -> list[dict]:
    """Curates 25 representative, high-value pilot prompts covering every M3 target capability
    (including 4 mixed Chinese-English desktop app utterances for immediate CosyVoice2 listening check).
    """
    desired_anchors = [
        # 1-8: Previously missed / new commands (换歌, 切歌, 每日推荐, modes, 桌面歌词, 收藏)
        "小克，换歌",
        "小克，切歌",
        "放每日推荐，小克",
        "小克，顺序播放",
        "小克，切换成单曲循环",
        "设为随机播放，小克",
        "小克，打开桌面歌词",
        "小克，收藏这首歌",
        # 9-14: Natural-language song/artist/playlist search
        "小克，我想听周杰伦的晴天",
        "帮我放陈奕迅的孤勇者，小克",
        "小克，来一首邓紫棋的泡沫",
        "小克，我想听周深的歌",
        "播放莫文蔚的忽然之间",
        "小克，放点适合写代码的纯音乐",
        # 15-19: Spoken percentages + two-step media/volume commands
        "小克，先把声音关小百分之十，再换首歌",
        "小克，把音量调大百分之十五",
        "收藏这首歌，然后改成列表循环，小克",
        "小克，太吵了把声音关小一点",
        "先暂停音乐，再关闭桌面歌词，小克",
        # 20-23: Mixed Chinese-English desktop app launches (single-step + compound!)
        "小克，打开 Steam，再放点背景音乐",
        "帮我启动 Clash，小克",
        "小克，打开 Discord，再把音乐暂停",
        "小克，打开 PowerShell",
        # 24-25: Non-command colloquial contrast
        "这首歌我已经单曲循环好几天了",
        "今天 Steam 夏季特卖有好几款游戏打折",
    ]

    by_norm = {it["text"]: it for it in full_items}
    pilot: list[dict] = []
    used_norm: set[str] = set()

    for anchor in desired_anchors:
        norm_tts = normalize_for_tts(anchor)
        norm_key = normalize_target_text(norm_tts)
        if norm_key in by_norm:
            item = dict(by_norm[norm_key])
        else:
            has_ent = [
                name
                for name in ("周杰伦", "晴天", "陈奕迅", "孤勇者", "邓紫棋", "泡沫", "周深", "莫文蔚", "忽然之间")
                if name in norm_tts
            ]
            en_toks = extract_en_tokens(norm_tts)
            sub = "desktop_app_en" if en_toks else "curated"
            item = {
                "category": "colloquial_control_multi" if en_toks else "pilot_anchor",
                "subtag": sub,
                "tts_text": norm_tts,
                "text": norm_key,
                "char_count": speech_unit_count(norm_tts),
                "entities": has_ent,
                "strict_exact": not bool(has_ent),
            }
        if is_valid_tts_prompt(item["tts_text"], 4, 20) and item["text"] not in used_norm:
            used_norm.add(item["text"])
            pilot.append(item)

    # Top up to exactly 25 if needed
    for it in full_items:
        if len(pilot) >= 25:
            break
        if it["text"] not in used_norm:
            used_norm.add(it["text"])
            pilot.append(dict(it))

    for idx, it in enumerate(pilot[:25], 1):
        it["id"] = f"pilot_{idx:02d}"
    return pilot[:25]


def sample_balanced_full(buckets: dict[str, list[dict]], target_total: int, rng: random.Random) -> list[dict]:
    quotas = {
        "new_cmd_boost": int(round(target_total * 0.25)),          # ~900
        "music_search_nl": int(round(target_total * 0.42)),        # ~1512
        "colloquial_control_multi": int(round(target_total * 0.25)),  # ~900
        "negative_colloquial": target_total - int(round(target_total * 0.25)) * 2 - int(round(target_total * 0.42)),  # ~288
    }

    selected: list[dict] = []
    leftovers: list[dict] = []

    for cat, quota in quotas.items():
        pool = list(buckets.get(cat, []))
        rng.shuffle(pool)
        if cat == "colloquial_control_multi" and quota >= 200:
            # Guarantee robust representation of mixed ZH-EN single & compound desktop-app prompts
            en_app_items = [x for x in pool if x.get("subtag") in ("desktop_app_en", "two_step_app_en")]
            other_items = [x for x in pool if x.get("subtag") not in ("desktop_app_en", "two_step_app_en")]
            en_quota = min(len(en_app_items), max(160, int(round(quota * 0.24))))
            cat_picked = en_app_items[:en_quota] + other_items[: max(0, quota - en_quota)]
            selected.extend(cat_picked)
            picked_norms = {x["text"] for x in cat_picked}
            leftovers.extend([x for x in pool if x["text"] not in picked_norms])
        else:
            selected.extend(pool[:quota])
            leftovers.extend(pool[quota:])

    if len(selected) < target_total and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_total - len(selected)])

    rng.shuffle(selected)
    selected = selected[:target_total]
    for idx, it in enumerate(selected, 1):
        it["id"] = f"tts_{idx:04d}"
    return selected


def verify_manifest_strict(items: list[dict]) -> None:
    seen = set()
    for it in items:
        tts_text = it["tts_text"]
        norm = it["text"]
        assert not re.search(r"[0-9%]", tts_text), f"Forbidden digit/% in: {tts_text}"
        assert is_valid_tts_prompt(tts_text, 4, 20), f"Invalid prompt chars/length/whitelist in: {tts_text}"
        for tok in extract_en_tokens(tts_text):
            assert ALLOWED_EN_APP_WORDS.get(tok.lower()) == tok, f"Non-whitelisted English token {tok} in: {tts_text}"
        assert re.fullmatch(r"[a-z\u4e00-\u9fa5]+", norm), f"Invalid norm text: {norm}"
        assert norm == normalize_target_text(tts_text), f"Norm mismatch {norm} vs {tts_text}"
        units = speech_unit_count(tts_text)
        assert 4 <= units <= 20, f"Spoken unit count {units} out of [4, 20] bounds: {tts_text}"
        assert norm not in seen, f"Duplicate normalized text: {norm}"
        seen.add(norm)


def run_self_test() -> int:
    """Runs offline unit tests verifying ZH-EN app normalization, whitelist gating, and pilot/full generation."""
    # 1. Normalization & length unit tests
    assert normalize_for_tts("小克，打开 Steam") == "小克，打开 Steam"
    assert normalize_target_text("小克，打开 Steam") == "小克打开steam"
    assert is_valid_tts_prompt("小克，打开 Steam")
    assert is_valid_tts_prompt("打开 QQ")
    assert is_valid_tts_prompt(normalize_for_tts("先打开 Clash，然后把声音关小10%，小克"))
    assert normalize_for_tts("先打开 Clash，然后把声音关小10%，小克") == "先打开 Clash，然后把声音关小百分之十，小克"
    assert is_valid_tts_prompt("小克，打开 VS Code")

    # Wild non-whitelisted Latin words or unexpanded digits must be rejected
    assert not is_valid_tts_prompt("小克，打开 FoobarApp")
    assert not is_valid_tts_prompt("小克，调大音量15%")

    # 2. Full end-to-end seed generation verification without touching user disk
    rng = random.Random(20261111)
    pairs, artists, songs = build_music_catalog(None, None)
    buckets = generate_candidate_pools(pairs, artists, songs, rng)
    full_items = sample_balanced_full(buckets, 3600, rng)
    pilot_items = build_pilot_25(full_items)

    verify_manifest_strict(pilot_items)
    verify_manifest_strict(full_items)
    assert len(pilot_items) == 25
    assert len(full_items) == 3600

    pilot_en = [it for it in pilot_items if extract_en_tokens(it["tts_text"])]
    assert len(pilot_en) >= 3, f"Expected >=3 mixed ZH-EN pilot prompts, got {len(pilot_en)}"

    full_en = [it for it in full_items if extract_en_tokens(it["tts_text"])]
    assert len(full_en) >= 150, f"Expected >=150 mixed ZH-EN prompts in full_3600, got {len(full_en)}"
    for must_word in ("Steam", "Clash", "Discord", "QQ", "PowerShell"):
        assert any(must_word in it["tts_text"] for it in full_items), f"Missing {must_word} in full_items"

    print(
        f"SELF-TEST PASSED! pilot_en={len(pilot_en)}/25, full_en={len(full_en)}/3600; "
        f"whitelist & numeral normalization verified."
    )
    return 0


def main() -> int:
    default_webdb = Path(os.path.expandvars(r"%LOCALAPPDATA%\NetEase\CloudMusic\Library\webdb.dat"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory for pilot_prompts_25.jsonl and full_prompts_3600.jsonl")
    parser.add_argument("--webdb", type=Path, default=default_webdb,
                        help="path to NetEase CloudMusic webdb.dat (optional, auto-detected)")
    parser.add_argument("--no-webdb", action="store_true", help="ignore local NetEase webdb.dat")
    parser.add_argument("--custom-music", type=Path, help="optional custom JSON of artists/songs")
    parser.add_argument("--full-count", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--self-test", action="store_true", help="run offline unit tests and exit")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if args.out_dir is None:
        parser.error("--out-dir is required unless --self-test is set")

    rng = random.Random(args.seed)
    webdb_path = None if args.no_webdb else args.webdb
    pairs, artists, songs = build_music_catalog(webdb_path, args.custom_music)

    buckets = generate_candidate_pools(pairs, artists, songs, rng)
    total_raw_pool = sum(len(v) for v in buckets.values())

    full_items = sample_balanced_full(buckets, args.full_count, rng)
    pilot_items = build_pilot_25(full_items)

    # Ensure every pilot and full prompt passes strict assertions
    verify_manifest_strict(pilot_items)
    verify_manifest_strict(full_items)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = args.out_dir / "pilot_prompts_25.jsonl"
    full_path = args.out_dir / f"full_prompts_{args.full_count}.jsonl"

    with open(pilot_path, "w", encoding="utf-8", newline="\n") as f:
        for it in pilot_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(full_path, "w", encoding="utf-8", newline="\n") as f:
        for it in full_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    cat_counts = Counter(it["category"] for it in full_items)
    en_mixed_count = sum(1 for it in full_items if extract_en_tokens(it["tts_text"]))
    pilot_en_count = sum(1 for it in pilot_items if extract_en_tokens(it["tts_text"]))
    len_vals = [it["char_count"] for it in full_items]
    summary = {
        "webdb_detected": bool(webdb_path and webdb_path.is_file()),
        "catalog_pairs": len(pairs),
        "catalog_artists": len(artists),
        "catalog_songs": len(songs),
        "candidate_pool_size": total_raw_pool,
        "pilot_count": len(pilot_items),
        "pilot_mixed_zh_en_count": pilot_en_count,
        "full_count": len(full_items),
        "full_mixed_zh_en_count": en_mixed_count,
        "whitelisted_en_app_words": sorted(ALLOWED_EN_APP_WORDS.values()),
        "category_distribution": dict(cat_counts),
        "char_length_min": min(len_vals),
        "char_length_max": max(len_vals),
        "char_length_mean": round(sum(len_vals) / len(len_vals), 2),
    }
    (args.out_dir / "prompts_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )

    print("=== NL TTS Prompts Generation Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPilot 25 prompts -> {pilot_path}")
    for it in pilot_items:
        print(f"  [{it['id']}] ({it['char_count']:2d}单元 | {it['category']:24s}) {it['tts_text']}")
    print(f"\nFull {len(full_items)} prompts -> {full_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
