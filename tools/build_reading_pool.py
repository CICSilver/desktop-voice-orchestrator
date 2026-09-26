"""Build the sentence pool for the collection page's reading mode.

Writes web/reading-sentences.json from two sources:

  * AISHELL-1 transcripts (Apache-2.0, https://www.openslr.org/33/): ordinary
    Mandarin across news, technology, sports and entertainment;
  * the colloquial sentences below, several of which use command words
    (音乐, 音量, 播放, 暂停...) without being commands, so a recognizer
    fine-tuned on this speaker does not learn that such words always mean one.

Sentences resembling the wake word (小克 and its homophones) are excluded, so
reading them can serve as wake-word negatives.

    python tools/build_reading_pool.py --transcript <aishell_transcript_v0.8.txt>
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

WAKE_LIKE = re.compile("小[克客课可科柯刻]")

COLLOQUIAL = [
    "这首歌的音乐还挺好听的",
    "刚才音量是不是有点大",
    "我在找一个好用的音乐播放器",
    "暂停一下我去倒杯水",
    "视频播放到一半就卡住了",
    "今天晚上吃点什么好呢",
    "这个游戏的背景音乐做得不错",
    "你觉得这个音量合适吗",
    "等会儿再听歌吧",
    "耳机的声音比音箱小一点",
    "这个音乐节的票很难抢",
    "电脑风扇的声音越来越大了",
    "这首歌我已经循环播放一整天了",
    "音量键好像失灵了",
    "你先别暂停让我听完这一段",
    "降低一点期待也不是坏事",
    "增加一点运动量对身体好",
    "明天早上八点有个会",
    "周末去哪里玩比较好",
    "最近天气变化挺大的",
    "这段代码跑起来有点慢",
    "他说下午会把东西送过来",
    "打开窗户以后凉快多了",
    "帮我看看这个文件有没有问题",
    "晚上记得把衣服收进来",
    "这个视频的配乐有点吵",
    "我觉得这里还可以再改改",
    "先把这局打完再说",
    "你那边信号好像不太好",
    "这把武器的伤害也太高了",
    "刚才那一波打得还行",
    "快递说今天下午能到",
    "这周的工作有点多",
    "我去楼下买瓶水",
    "屏幕亮度调低一点眼睛舒服些",
    "昨天那部电影的音乐很有感觉",
    "会议暂停十分钟大家休息一下",
    "这个播放量涨得挺快的",
    "声音开大一点我听不清",
    "你们那边能听到我说话吗",
    "今天的音乐课改到下午了",
    "这台音箱的低音很足",
    "等我把这个视频看完",
    "外面好像下雨了",
    "咖啡喝多了晚上睡不着",
    "我把歌单分享给你了",
    "这个版本的更新内容不少",
    "麦克风是不是没开",
    "今天不想做饭点个外卖吧",
    "这首曲子的节奏有点快",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("web/reading-sentences.json"))
    parser.add_argument("--count", type=int, default=3000, help="AISHELL sentences to keep")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=22)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    seen, candidates = set(), []
    for line in args.transcript.read_text("utf-8").splitlines():
        _, _, text = line.partition(" ")
        text = text.replace(" ", "")
        if not args.min_chars <= len(text) <= args.max_chars or text in seen:
            continue
        if WAKE_LIKE.search(text) or not re.fullmatch(r"[一-鿿]+", text):
            continue
        seen.add(text)
        candidates.append(text)
    random.Random(args.seed).shuffle(candidates)
    colloquial = [text for text in COLLOQUIAL if not WAKE_LIKE.search(text)]
    pool = {
        "sources": {
            "aishell": "AISHELL-1 transcripts, Apache-2.0, https://www.openslr.org/33/",
            "colloquial": "tools/build_reading_pool.py",
        },
        "colloquial": colloquial,
        "aishell": candidates[:args.count],
    }
    args.out.write_text(json.dumps(pool, ensure_ascii=False, indent=0) + "\n", "utf-8")
    print(f"{len(pool['aishell'])} AISHELL + {len(colloquial)} colloquial sentences -> {args.out}")


if __name__ == "__main__":
    main()
