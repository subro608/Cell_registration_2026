"""Shared text rendering: Helvetica regular + oblique for 'in vivo'/'ex vivo' italic."""

import cv2, numpy as np, re
from PIL import Image, ImageDraw, ImageFont

HELV_PATH = '/System/Library/Fonts/Helvetica.ttc'

# Cache fonts by size
_font_cache = {}

def _get_fonts(size):
    if size not in _font_cache:
        _font_cache[size] = (
            ImageFont.truetype(HELV_PATH, size, index=0),  # Regular
            ImageFont.truetype(HELV_PATH, size, index=2),  # Oblique
        )
    return _font_cache[size]

_LATIN_RE = re.compile(r'((?:in|ex)\s+vivo)', re.IGNORECASE)

def put_text_mixed(frame, text, org, font_unused, scale, color, thickness,
                   line_type=cv2.LINE_AA):
    """Draw text with 'in vivo'/'ex vivo' in Helvetica Oblique, rest in Regular.

    Args match cv2.putText signature for easy drop-in replacement.
    scale: cv2 font scale — mapped to PIL pixel size (scale * 32).
    color: BGR tuple.
    """
    pil_size = max(12, int(round(scale * 32)))
    font_reg, font_it = _get_fonts(pil_size)

    # Convert BGR frame to RGB PIL image
    h, w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # PIL color is RGB, cv2 color is BGR
    rgb_color = (color[2], color[1], color[0])

    x, y_base = org
    # cv2 putText y is baseline; PIL y is top. Adjust: subtract ascent.
    ascent = font_reg.getbbox('A')[3]
    y_pil = y_base - ascent

    parts = _LATIN_RE.split(text)
    for part in parts:
        if not part:
            continue
        is_italic = bool(re.match(r'(?:in|ex)\s+vivo$', part.strip(), re.IGNORECASE))
        f = font_it if is_italic else font_reg
        draw.text((x, y_pil), part, font=f, fill=rgb_color)
        bbox = f.getbbox(part)
        x += bbox[2] - bbox[0]

    # Write back to frame
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def text_width_mixed(text, scale):
    """Compute total pixel width of mixed regular/italic text."""
    pil_size = max(12, int(round(scale * 32)))
    font_reg, font_it = _get_fonts(pil_size)

    parts = _LATIN_RE.split(text)
    total = 0
    for part in parts:
        if not part:
            continue
        is_italic = bool(re.match(r'(?:in|ex)\s+vivo$', part.strip(), re.IGNORECASE))
        f = font_it if is_italic else font_reg
        bbox = f.getbbox(part)
        total += bbox[2] - bbox[0]
    return total
