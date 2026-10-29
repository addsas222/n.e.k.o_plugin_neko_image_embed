"""查询词的中英别名。

存在理由：CLIP ViT-B/32 的文本塔是英文训练的。实测「猫」这类纯中文查询几乎靠猜，
而角色与用户日常说的是中文。判定链给出中文 caption 时，`store` 的词法加权能兜住；
但在没有模型、没有 caption 的情况下（本机默认状态），只能靠这张表把常见词映射成英文。

表很小、可覆盖：``[image_embed].query_aliases`` 里的键值会合并在默认表之上。
"""

from __future__ import annotations

DEFAULT_ALIASES: dict[str, str] = {
    "猫": "cat",
    "猫咪": "cat",
    "小猫": "kitten",
    "狗": "dog",
    "狗狗": "dog",
    "小狗": "puppy",
    "鸟": "bird",
    "鱼": "fish",
    "兔子": "rabbit",
    "仓鼠": "hamster",
    "动物": "animal",
    "人": "person",
    "女孩": "girl",
    "男孩": "boy",
    "自拍": "selfie",
    "合照": "group photo",
    "风景": "landscape",
    "天空": "sky",
    "海": "sea",
    "山": "mountain",
    "花": "flower",
    "树": "tree",
    "夜景": "night view",
    "城市": "city",
    "室内": "indoor",
    "食物": "food",
    "美食": "food",
    "蛋糕": "cake",
    "咖啡": "coffee",
    "奶茶": "milk tea",
    "表情包": "meme sticker",
    "梗图": "meme",
    "贴纸": "sticker",
    "截图": "screenshot",
    "聊天记录": "chat screenshot",
    "游戏": "game",
    "动漫": "anime",
    "插画": "illustration",
    "像素": "pixel art",
    "像素风": "pixel art",
    "可爱": "cute",
    "搞笑": "funny",
    "漂亮": "beautiful",
    "暗": "dark",
    "亮": "bright",
}


def translate_query(query: str, extra: dict[str, str] | None = None) -> str:
    """把查询里的中文别名替换成英文；没有命中就原样返回。"""
    text = (query or "").strip()
    if not text:
        return ""
    table = dict(DEFAULT_ALIASES)
    if extra:
        for key, value in extra.items():
            if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip():
                table[key.strip()] = value.strip()

    if text in table:
        return table[text]

    translated = text
    # 长词优先，避免「猫咪」被「猫」先吃掉一半
    for key in sorted(table, key=len, reverse=True):
        if key in translated:
            translated = translated.replace(key, f" {table[key]} ")
    return " ".join(translated.split())
