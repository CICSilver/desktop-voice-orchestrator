"""Desktop Voice Orchestrator — Router SFT Dataset Generator & Teacher Distillation (M4 Slice C).

Generates high-diversity Chinese spoken command and rejection datasets
with Mandarin ASR phonetic corruption, dynamic app catalogs, and strict schema validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Support importing local schema whether run directly or as package
try:
    from .schema import (
        ACTION_TYPES,
        build_system_prompt,
        create_action,
        validate_router_plan,
    )
except ImportError:
    from schema import (
        ACTION_TYPES,
        build_system_prompt,
        create_action,
        validate_router_plan,
    )

APP_CATALOG_MASTER: list[dict[str, Any]] = [
    {
        "id": "powershell",
        "display_name": "PowerShell",
        "aliases": ["powershell", "终端", "命令行", "pwsh"],
    },
    {
        "id": "steam",
        "display_name": "Steam",
        "aliases": ["steam", "蒸汽平台", "死盯", "斯蒂姆"],
    },
    {
        "id": "clash",
        "display_name": "Clash",
        "aliases": ["clash", "clash verge", "代理", "克拉什", "可拉什", "梯子", "挂梯子"],
    },
    {
        "id": "discord",
        "display_name": "Discord",
        "aliases": ["discord", "迪斯科", "迪斯扣", "迪斯科德"],
    },
    {
        "id": "qq",
        "display_name": "QQ",
        "aliases": ["qq", "腾讯qq", "扣扣", "企鹅"],
    },
    {
        "id": "chrome",
        "display_name": "Google Chrome",
        "aliases": ["chrome", "谷歌浏览器", "浏览器"],
    },
    {
        "id": "vscode",
        "display_name": "Visual Studio Code",
        "aliases": ["vscode", "代码编辑器", "写代码的软件"],
    },
    {
        "id": "netease",
        "display_name": "网易云音乐",
        "aliases": ["网易云", "云音乐", "听歌软件"],
    },
    {
        "id": "wechat",
        "display_name": "微信",
        "aliases": ["wechat", "微讯"],
    },
    {
        "id": "potplayer",
        "display_name": "PotPlayer",
        "aliases": ["potplayer", "播放器", "视频播放器"],
    },
]

ASR_HOMOPHONE_MAP: dict[str, list[str]] = {
    "steam": ["死盯", "斯蒂姆", "斯丁", "四丁"],
    "clash": ["克拉什", "可拉什", "卡拉什"],
    "discord": ["迪斯科", "迪斯扣", "迪斯科德"],
    "qq": ["扣扣"],
    "晴天": ["情天", "擎天"],
    "歌单": ["割单", "格单"],
    "切歌": ["车歌", "惬歌"],
    "暂停": ["战停", "站停"],
    "音量": ["阴凉", "应亮"],
    "周杰伦": ["纠结伦", "周接龙"],
    "梯子": ["提子", "体子"],
}


def apply_asr_corruption(text: str, corruption_prob: float = 0.3, rng: random.Random | None = None) -> str:
    """Randomly inject realistic SenseVoice/Paraformer Mandarin ASR homophones into utterance."""
    if rng is None:
        rng = random.Random()
    if rng.random() > corruption_prob:
        return text

    candidates = [k for k in ASR_HOMOPHONE_MAP if k in text.lower()]
    if not candidates:
        return text

    target_word = rng.choice(candidates)
    replacement = rng.choice(ASR_HOMOPHONE_MAP[target_word])
    # Replace the first occurrence case-insensitively
    idx = text.lower().find(target_word)
    if idx >= 0:
        return text[:idx] + replacement + text[idx + len(target_word):]
    return text


def sample_app_catalog(rng: random.Random) -> list[dict[str, Any]]:
    """Sample a dynamic subset of apps (3 to 8 apps) to prevent memorizing static list."""
    k = rng.randint(3, min(8, len(APP_CATALOG_MASTER)))
    return rng.sample(APP_CATALOG_MASTER, k)


def build_self_test_seed_dataset() -> list[dict[str, Any]]:
    """Build deterministic offline seed dataset covering all 14 actions, edge cases, and hard negatives."""
    fixed_rng = random.Random(42)
    standard_catalog = APP_CATALOG_MASTER[:5]  # powershell, steam, clash, discord, qq

    seeds: list[tuple[str, list[dict[str, Any]], str, list[dict[str, Any]]]] = []

    # 1. Bucket 1: 11 basic media actions
    basic_media_cases = [
        ("播放音乐", [create_action("media.play")], "media.play", standard_catalog),
        ("把音乐打开", [create_action("media.play")], "media.play", standard_catalog),
        ("继续播放音乐", [create_action("media.play")], "media.play", standard_catalog),
        ("暂停播放", [create_action("media.pause")], "media.pause", standard_catalog),
        ("先别放了停一下", [create_action("media.pause")], "media.pause", standard_catalog),
        ("把音乐关了", [create_action("media.pause")], "media.pause", standard_catalog),
        ("切歌", [create_action("media.next")], "media.next", standard_catalog),
        ("换下一首歌", [create_action("media.next")], "media.next", standard_catalog),
        ("这首听腻了换一首", [create_action("media.next")], "media.next", standard_catalog),
        ("我喜欢这首歌", [create_action("media.like")], "media.like", standard_catalog),
        ("收藏一下", [create_action("media.like")], "media.like", standard_catalog),
        ("播放每日推荐", [create_action("media.play_daily")], "media.play_daily", standard_catalog),
        ("来点日推歌曲", [create_action("media.play_daily")], "media.play_daily", standard_catalog),
        ("切换到顺序播放", [create_action("media.mode.order")], "media.mode.order", standard_catalog),
        ("按顺序放", [create_action("media.mode.order")], "media.mode.order", standard_catalog),
        ("开启列表循环", [create_action("media.mode.list_loop")], "media.mode.list_loop", standard_catalog),
        ("循环整个歌单", [create_action("media.mode.list_loop")], "media.mode.list_loop", standard_catalog),
        ("单曲循环这首歌", [create_action("media.mode.single_loop")], "media.mode.single_loop", standard_catalog),
        ("一直放这一首", [create_action("media.mode.single_loop")], "media.mode.single_loop", standard_catalog),
        ("随机播放", [create_action("media.mode.shuffle")], "media.mode.shuffle", standard_catalog),
        ("随便乱序播放", [create_action("media.mode.shuffle")], "media.mode.shuffle", standard_catalog),
        ("打开桌面歌词", [create_action("media.lyrics.show")], "media.lyrics.show", standard_catalog),
        ("显示歌词", [create_action("media.lyrics.show")], "media.lyrics.show", standard_catalog),
        ("关闭桌面歌词", [create_action("media.lyrics.hide")], "media.lyrics.hide", standard_catalog),
        ("把歌词隐藏起来", [create_action("media.lyrics.hide")], "media.lyrics.hide", standard_catalog),
    ]
    seeds.extend(basic_media_cases)

    # 2. Bucket 2: Audio Volume Adjust (defaults +-5, explicit +-1..+-20)
    volume_cases = [
        ("大声点", [create_action("audio.volume.adjust", volume_delta_percent=5)], "volume.default_up", standard_catalog),
        ("声音大一点", [create_action("audio.volume.adjust", volume_delta_percent=5)], "volume.default_up", standard_catalog),
        ("小声点", [create_action("audio.volume.adjust", volume_delta_percent=-5)], "volume.default_down", standard_catalog),
        ("声音稍微调小一点", [create_action("audio.volume.adjust", volume_delta_percent=-5)], "volume.default_down", standard_catalog),
        ("音量调大10%", [create_action("audio.volume.adjust", volume_delta_percent=10)], "volume.explicit", standard_catalog),
        ("声音调小15%", [create_action("audio.volume.adjust", volume_delta_percent=-15)], "volume.explicit", standard_catalog),
        ("音量加8点", [create_action("audio.volume.adjust", volume_delta_percent=8)], "volume.explicit", standard_catalog),
        ("音量降低20%", [create_action("audio.volume.adjust", volume_delta_percent=-20)], "volume.explicit", standard_catalog),
        ("声音增大15%", [create_action("audio.volume.adjust", volume_delta_percent=15)], "volume.explicit", standard_catalog),
    ]
    seeds.extend(volume_cases)

    # 3. Bucket 3: Media Search & Play (song, playlist, artist)
    search_cases = [
        (
            "放一首周杰伦的晴天",
            [create_action("media.search_play", search_query="晴天", search_type="song", search_artist="周杰伦")],
            "search.song_with_artist",
            standard_catalog,
        ),
        (
            "放晴天",
            [create_action("media.search_play", search_query="晴天", search_type="song", search_artist=None)],
            "search.song_query_only",
            standard_catalog,
        ),
        (
            "想听陈奕迅的十年",
            [create_action("media.search_play", search_query="十年", search_type="song", search_artist="陈奕迅")],
            "search.song_with_artist",
            standard_catalog,
        ),
        (
            "播放告白气球",
            [create_action("media.search_play", search_query="告白气球", search_type="song", search_artist=None)],
            "search.song_query_only",
            standard_catalog,
        ),
        (
            "看书有点闷来点轻音乐歌单",
            [create_action("media.search_play", search_query="轻音乐", search_type="playlist", search_artist=None)],
            "search.playlist",
            standard_catalog,
        ),
        (
            "放一下适合写代码的歌单",
            [create_action("media.search_play", search_query="适合写代码", search_type="playlist", search_artist=None)],
            "search.playlist",
            standard_catalog,
        ),
        (
            "搜个经典粤语歌单放一下",
            [create_action("media.search_play", search_query="经典粤语", search_type="playlist", search_artist=None)],
            "search.playlist",
            standard_catalog,
        ),
        (
            "来点林俊杰的歌",
            [create_action("media.search_play", search_query="林俊杰", search_type="song", search_artist="林俊杰")],
            "search.song_artist_query",
            standard_catalog,
        ),
    ]
    seeds.extend(search_cases)

    # 4. Bucket 4: Dynamic App Catalog & App Open
    app_open_cases = [
        ("打开steam", [create_action("app.open", app_id="steam")], "app.open_direct", standard_catalog),
        ("帮我开一下死盯", [create_action("app.open", app_id="steam")], "app.open_alias_asr", standard_catalog),
        ("启动蒸汽平台", [create_action("app.open", app_id="steam")], "app.open_alias", standard_catalog),
        ("帮我挂上代理", [create_action("app.open", app_id="clash")], "app.open_infer", standard_catalog),
        ("打开克拉什", [create_action("app.open", app_id="clash")], "app.open_alias_asr", standard_catalog),
        ("启动梯子软件", [create_action("app.open", app_id="clash")], "app.open_infer", standard_catalog),
        ("上一下迪斯科", [create_action("app.open", app_id="discord")], "app.open_alias", standard_catalog),
        ("把discord打开", [create_action("app.open", app_id="discord")], "app.open_direct", standard_catalog),
        ("帮我打开QQ", [create_action("app.open", app_id="qq")], "app.open_direct", standard_catalog),
        ("打开扣扣", [create_action("app.open", app_id="qq")], "app.open_alias", standard_catalog),
        ("把终端打开", [create_action("app.open", app_id="powershell")], "app.open_alias", standard_catalog),
        ("启动命令行", [create_action("app.open", app_id="powershell")], "app.open_alias", standard_catalog),
    ]
    seeds.extend(app_open_cases)

    # 5. Bucket 5: Compositional multi-intent (2~3 actions)
    multi_intent_cases = [
        (
            "太吵了把声音关小点再切首歌",
            [create_action("audio.volume.adjust", volume_delta_percent=-5), create_action("media.next")],
            "multi.volume_next",
            standard_catalog,
        ),
        (
            "打开steam然后随机播放",
            [create_action("app.open", app_id="steam"), create_action("media.mode.shuffle")],
            "multi.app_mode",
            standard_catalog,
        ),
        (
            "先暂停然后把音量调大10%接着换一首",
            [
                create_action("media.pause"),
                create_action("audio.volume.adjust", volume_delta_percent=10),
                create_action("media.next"),
            ],
            "multi.pause_volume_next",
            standard_catalog,
        ),
        (
            "收藏一下并单曲循环",
            [create_action("media.like"), create_action("media.mode.single_loop")],
            "multi.like_loop",
            standard_catalog,
        ),
        (
            "声音稍微调大点然后再放每日推荐",
            [create_action("audio.volume.adjust", volume_delta_percent=5), create_action("media.play_daily")],
            "multi.volume_daily",
            standard_catalog,
        ),
        (
            "帮我打开终端接着把音乐暂停一下",
            [create_action("app.open", app_id="powershell"), create_action("media.pause")],
            "multi.app_pause",
            standard_catalog,
        ),
        (
            "切歌然后打开桌面歌词",
            [create_action("media.next"), create_action("media.lyrics.show")],
            "multi.next_lyrics",
            standard_catalog,
        ),
        (
            "挂上代理然后打开迪斯科语音",
            [create_action("app.open", app_id="clash"), create_action("app.open", app_id="discord")],
            "multi.apps_open",
            standard_catalog,
        ),
    ]
    seeds.extend(multi_intent_cases)

    # 6. Bucket 6: Hard Negatives & Safe Rejection (actions -> [])
    negatives_catalog_without_ps = [
        a for a in APP_CATALOG_MASTER if a["id"] in ("steam", "clash", "discord", "qq")
    ]
    negative_cases = [
        ("算了还是别切了", [], "negative.self_cancellation", standard_catalog),
        ("不用打开了，刚才看错了", [], "negative.self_cancellation", standard_catalog),
        ("算了别放了等会儿再听吧", [], "negative.self_cancellation", standard_catalog),
        ("取消取消，先不用调音量了", [], "negative.self_cancellation", standard_catalog),
        ("今晚不玩了先别开死盯了", [], "negative.self_cancellation", standard_catalog),
        ("在很久很久以前有一座古老高耸的城堡", [], "negative.reading_aloud", standard_catalog),
        ("生命是一场奇妙的冒险旅程不能轻易放弃", [], "negative.reading_aloud", standard_catalog),
        ("听妈妈的话别让她受伤想快快长大才能保护她", [], "negative.reading_lyrics", standard_catalog),
        ("为你翘课的那一天花落的那一天教室的那一间", [], "negative.reading_lyrics", standard_catalog),
        ("哈哈哈笑死我了这波怎么说兄弟们", [], "negative.chitchat_game", standard_catalog),
        ("你刚才那个大招放得也太歪了吧笑死", [], "negative.chitchat_game", standard_catalog),
        ("走走走中路集合开团了兄弟们速度", [], "negative.chitchat_game", standard_catalog),
        ("这首歌真好听旋律真的太赞了", [], "negative.chitchat_sentiment", standard_catalog),
        ("周杰伦是哪一年出生的你知道吗", [], "negative.question_qa", standard_catalog),
        ("今天外面多少度下不下雨啊", [], "negative.question_weather", standard_catalog),
        ("怎么写一个快速排序算法帮我查下", [], "negative.question_code", standard_catalog),
        ("帮我打开Photoshop作图软件", [], "negative.unregistered_app", standard_catalog),
        ("启动一下AutoCAD画图", [], "negative.unregistered_app", standard_catalog),
        ("把网易云音乐打开一下", [], "negative.unregistered_app", negatives_catalog_without_ps),
        ("音量调大80%", [], "negative.out_of_range_volume", standard_catalog),
        ("声音直接加50个点太小了", [], "negative.out_of_range_volume", standard_catalog),
        ("把音量调小70%", [], "negative.out_of_range_volume", standard_catalog),
    ]
    seeds.extend(negative_cases)

    # Convert to standard SFT samples
    samples: list[dict[str, Any]] = []
    for text, actions, category, catalog in seeds:
        system_prompt = build_system_prompt(catalog)
        allowed_ids = [a["id"] for a in catalog]
        plan = {"actions": actions}

        valid, err = validate_router_plan(plan, allowed_app_ids=allowed_ids)
        if not valid:
            raise ValueError(f"Self-test seed invalid for text '{text}': {err}")

        assistant_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": assistant_json},
                ],
                "category": category,
                "app_ids": allowed_ids,
            }
        )

    return samples


def expand_synthetic_dataset(
    target_count: int,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Synthesize high-diversity dataset using offline procedural templates & noise injection."""
    if rng is None:
        rng = random.Random(42)

    base_samples = build_self_test_seed_dataset()
    samples: list[dict[str, Any]] = list(base_samples)

    verbs_play = ["播放", "放一下", "开一下", "听听", "来一首", "整点", "放首", "开启"]
    verbs_pause = ["暂停", "暂停一下", "停一下", "别放了", "停止播放", "先把音乐停了", "先安静一下"]
    verbs_next = ["切歌", "换歌", "下一首", "换下一首", "换一首歌", "跳过这首", "切下一曲"]
    verbs_open = ["打开", "启动", "帮我开一下", "运行", "上一下", "开一下"]

    popular_songs = [
        ("周杰伦", ["晴天", "七里香", "稻香", "青花瓷", "夜曲", "告白气球", "安静", "简单爱"]),
        ("林俊杰", ["江南", "可惜没如果", "修炼爱情", "背对背拥抱", "一千年以后"]),
        ("陈奕迅", ["十年", "红玫瑰", "孤勇者", "富士山下", "好久不见"]),
        ("王菲", ["红豆", "传奇", "匆匆那年", "流年"]),
        ("孙燕姿", ["遇见", "开始懂了", "绿光", "我怀念的"]),
        ("邓紫棋", ["泡沫", "光年之外", "句号", "来自天堂的魔鬼"]),
    ]

    playlists = [
        "轻音乐",
        "适合写代码",
        "深度工作纯音乐",
        "华语经典流行",
        "催眠助眠",
        "ACG动漫热歌",
        "爵士咖啡馆",
        "下雨天抒情歌",
        "健身燃脂BGM",
        "游戏原声大碟",
    ]

    chit_chats = [
        "哈哈哈笑死我了，这波操作太秀了",
        "你刚才看到那个敌人了吗，差点被阴了",
        "走走走下把继续，必须找回场子",
        "今晚吃什么啊，有点想吃火锅",
        "今天上班真累，先打把游戏放松一下",
        "这个视频做得很用心，点赞收藏了",
        "哇这首歌的前奏一响，鸡皮疙瘩就起来了",
        "刚才麦克风有点喷麦，现在好了吗",
        "明天早上九点有个例会别忘了参加",
        "这个键盘打字手感真的很清脆",
        "听妈妈的话晚点再恋爱吧",
        "海阔天空在勇敢以后要拿执着将命运的锁打破",
        "白日依山尽黄河入海流欲穷千里目更上一层楼",
    ]

    unregistered_app_queries = [
        "帮我打开Photoshop修改海报",
        "启动一下CAD看施工图纸",
        "打开剪映剪辑一下刚才的视频",
        "把Excel表格打开我看下数据",
        "启动网易云游戏助手",
        "打开虚幻引擎编辑器",
        "帮我启动原神客户端",
        "打开百度网盘下载文件",
    ]

    while len(samples) < target_count:
        catalog = sample_app_catalog(rng)
        allowed_ids = [a["id"] for a in catalog]
        catalog_ids_set = set(allowed_ids)
        system_prompt = build_system_prompt(catalog)

        bucket = rng.randint(1, 6)

        if bucket == 1:
            # Basic media action
            act_type = rng.choice(
                [
                    "media.play",
                    "media.pause",
                    "media.next",
                    "media.like",
                    "media.play_daily",
                    "media.mode.order",
                    "media.mode.list_loop",
                    "media.mode.single_loop",
                    "media.mode.shuffle",
                    "media.lyrics.show",
                    "media.lyrics.hide",
                ]
            )
            if act_type == "media.play":
                txt = f"{rng.choice(verbs_play)}音乐"
            elif act_type == "media.pause":
                txt = rng.choice(verbs_pause)
            elif act_type == "media.next":
                txt = rng.choice(verbs_next)
            elif act_type == "media.like":
                txt = rng.choice(["收藏这首歌", "我喜欢这首歌", "喜欢这个", "点亮红心", "收藏一下这曲子"])
            elif act_type == "media.play_daily":
                txt = rng.choice(["播放每日推荐", "来点每日推荐", "听一下日推", "放每日推荐歌单"])
            elif act_type == "media.mode.order":
                txt = rng.choice(["切换到顺序播放", "顺序播放", "按顺序播放歌单"])
            elif act_type == "media.mode.list_loop":
                txt = rng.choice(["开启列表循环", "列表循环播放", "循环整个歌单"])
            elif act_type == "media.mode.single_loop":
                txt = rng.choice(["单曲循环这首歌", "单曲循环", "一直放这一首吧"])
            elif act_type == "media.mode.shuffle":
                txt = rng.choice(["随机播放", "随便乱序放一首", "开启随机模式"])
            elif act_type == "media.lyrics.show":
                txt = rng.choice(["打开桌面歌词", "显示歌词", "把歌词弹出来"])
            else:
                txt = rng.choice(["关闭桌面歌词", "把歌词隐藏起来", "关掉歌词"])

            plan = {"actions": [create_action(act_type)]}
            cat = f"bucket1.{act_type}"

        elif bucket == 2:
            # Volume adjust
            is_relative = rng.random() < 0.5
            if is_relative:
                is_up = rng.random() < 0.5
                vol = 5 if is_up else -5
                txt = rng.choice(["声音大一点", "大声点", "调大音量"]) if is_up else rng.choice(["声音小一点", "小声点", "调小音量", "太吵了声音调低点"])
                cat = "bucket2.volume.default"
            else:
                vol_step = rng.choice([2, 5, 8, 10, 12, 15, 20])
                sign = 1 if rng.random() < 0.5 else -1
                vol = sign * vol_step
                txt = f"音量调大{vol_step}%" if sign > 0 else f"声音调小{vol_step}%"
                cat = "bucket2.volume.explicit"
            plan = {"actions": [create_action("audio.volume.adjust", volume_delta_percent=vol)]}

        elif bucket == 3:
            # Search play
            is_playlist = rng.random() < 0.35
            if is_playlist:
                pname = rng.choice(playlists)
                txt = rng.choice([f"放一下{pname}歌单", f"搜个{pname}歌单放放", f"来点{pname}歌单听听", f"看书有点闷来点{pname}歌单"])
                plan = {"actions": [create_action("media.search_play", search_query=pname, search_type="playlist")]}
                cat = "bucket3.search.playlist"
            else:
                artist, songs = rng.choice(popular_songs)
                song = rng.choice(songs)
                with_artist = rng.random() < 0.6
                if with_artist:
                    txt = rng.choice([f"放一首{artist}的{song}", f"想听{artist}的{song}", f"播放{artist}唱的{song}"])
                    plan = {"actions": [create_action("media.search_play", search_query=song, search_type="song", search_artist=artist)]}
                else:
                    txt = rng.choice([f"放一首{song}", f"播放{song}", f"来首{song}听听"])
                    plan = {"actions": [create_action("media.search_play", search_query=song, search_type="song")]}
                cat = "bucket3.search.song"

        elif bucket == 4:
            # App open from current catalog
            target_app = rng.choice(catalog)
            app_id = target_app["id"]
            name_variant = target_app["display_name"]
            if target_app.get("aliases") and rng.random() < 0.7:
                name_variant = rng.choice(target_app["aliases"])
            txt = f"{rng.choice(verbs_open)}{name_variant}"
            plan = {"actions": [create_action("app.open", app_id=app_id)]}
            cat = "bucket4.app.open"

        elif bucket == 5:
            # Multi-intent composition (2 actions)
            combo_type = rng.choice(["volume_next", "app_shuffle", "pause_open", "like_loop"])
            if combo_type == "volume_next":
                txt = "太吵了把声音关小点然后再切首歌"
                actions = [
                    create_action("audio.volume.adjust", volume_delta_percent=-5),
                    create_action("media.next"),
                ]
            elif combo_type == "app_shuffle":
                chosen_app = rng.choice(catalog)
                name_variant = chosen_app.get("aliases", [chosen_app["display_name"]])[0]
                txt = f"打开{name_variant}然后开启随机播放"
                actions = [
                    create_action("app.open", app_id=chosen_app["id"]),
                    create_action("media.mode.shuffle"),
                ]
            elif combo_type == "pause_open":
                chosen_app = rng.choice(catalog)
                txt = f"先暂停一下音乐接着把{chosen_app['display_name']}启动"
                actions = [
                    create_action("media.pause"),
                    create_action("app.open", app_id=chosen_app["id"]),
                ]
            else:
                txt = "我喜欢这首歌给它加个红心然后单曲循环"
                actions = [
                    create_action("media.like"),
                    create_action("media.mode.single_loop"),
                ]
            plan = {"actions": actions}
            cat = "bucket5.multi"

        else:
            # Hard negatives -> {"actions": []}
            neg_type = rng.choice(["cancel", "chitchat", "unregistered_app", "out_of_range_vol"])
            if neg_type == "cancel":
                txt = rng.choice(
                    [
                        "算了还是别切歌了",
                        "不用打开了刚才看错人了",
                        "等等先别放音乐了",
                        "算了算了我自己去弄不用调了",
                        "取消刚才那条指令",
                    ]
                )
                cat = "bucket6.negative.cancel"
            elif neg_type == "chitchat":
                txt = rng.choice(chit_chats)
                cat = "bucket6.negative.chitchat"
            elif neg_type == "unregistered_app":
                txt = rng.choice(unregistered_app_queries)
                cat = "bucket6.negative.unregistered_app"
            else:
                out_vol = rng.choice([30, 40, 50, 80, 90])
                txt = f"音量调大{out_vol}%" if rng.random() < 0.5 else f"把声音调小{out_vol}%"
                cat = "bucket6.negative.out_of_range_vol"
            plan = {"actions": []}

        # Apply ASR phonetic corruption to user query with some probability
        txt = apply_asr_corruption(txt, corruption_prob=0.25, rng=rng)

        valid, err = validate_router_plan(plan, allowed_app_ids=allowed_ids)
        if not valid:
            continue

        assistant_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": txt},
                    {"role": "assistant", "content": assistant_json},
                ],
                "category": cat,
                "app_ids": allowed_ids,
            }
        )

    return samples[:target_count]


def run_self_test() -> int:
    """Execute offline deterministic self-test and verify 100% schema alignment."""
    print("=== [Self-Test] 1. Generating deterministic offline seed dataset ===")
    seeds = build_self_test_seed_dataset()
    print(f"Generated {len(seeds)} seed samples across all 14 actions and edge cases.")

    print("\n=== [Self-Test] 2. Verifying strict schema and catalog validation ===")
    action_type_counts: dict[str, int] = {k: 0 for k in ACTION_TYPES}
    negative_count = 0
    multi_count = 0

    for i, s in enumerate(seeds):
        plan = json.loads(s["messages"][2]["content"])
        valid, msg = validate_router_plan(plan, allowed_app_ids=s["app_ids"])
        if not valid:
            print(f"❌ Sample #{i} failed validation: {msg}")
            return 1

        actions = plan["actions"]
        if not actions:
            negative_count += 1
        else:
            if len(actions) > 1:
                multi_count += 1
            for act in actions:
                action_type_counts[act["type"]] += 1

    print("Coverage report across 14 action types:")
    for atype, cnt in action_type_counts.items():
        print(f"  - {atype:<24}: {cnt} instances")
        if cnt == 0:
            print(f"❌ Action type {atype} has zero occurrences in seeds!")
            return 1

    print(f"  - (negatives actions==[]): {negative_count} samples")
    print(f"  - (multi-intent actions>1): {multi_count} samples")

    if len(seeds) < 60:
        print(f"❌ Seed dataset size ({len(seeds)}) is below required minimum of 60 samples!")
        return 1

    print("\n=== [Self-Test] 3. Testing procedural expansion up to 100 samples ===")
    expanded = expand_synthetic_dataset(100, rng=random.Random(1234))
    if len(expanded) != 100:
        print(f"❌ Expanded dataset count mismatch: expected 100, got {len(expanded)}")
        return 1

    for i, item in enumerate(expanded):
        plan = json.loads(item["messages"][2]["content"])
        valid, msg = validate_router_plan(plan, allowed_app_ids=item["app_ids"])
        if not valid:
            print(f"❌ Expanded item #{i} invalid: {msg}")
            return 1

    print("✅ All 100 expanded samples passed strict validate_router_plan checks!")
    print("\n=== [Self-Test] Summary: ALL CHECKS PASSED (exit code 0) ===")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Router SFT Dataset Generator & Teacher Distillation")
    parser.add_argument("--self-test", action="store_true", help="Run offline deterministic self-test and exit")
    parser.add_argument("--output-dir", type=str, default="dataset/router", help="Output directory for train/val files")
    parser.add_argument("--num-samples", type=int, default=3000, help="Total target number of samples")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Ratio of train split vs val split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files")
    parser.add_argument("--model-teacher", type=str, default="gemini-2.5-flash", help="Teacher model identifier")

    args = parser.parse_args()

    if args.self_test:
        sys.exit(run_self_test())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "val.jsonl"

    rng = random.Random(args.seed)
    print(f"Generating {args.num_samples} samples (train_ratio={args.train_ratio}, seed={args.seed})...")

    # Generate synthetic samples
    dataset = expand_synthetic_dataset(args.num_samples, rng=rng)
    rng.shuffle(dataset)

    split_idx = int(len(dataset) * args.train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"Writing {len(train_data)} train samples to {train_file} ...")
    with open(train_file, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Writing {len(val_data)} val samples to {val_file} ...")
    with open(val_file, "w", encoding="utf-8") as f:
        for row in val_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Dataset generation completed successfully.")


if __name__ == "__main__":
    main()
