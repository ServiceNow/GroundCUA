import re
import os
import backoff
import math
import base64
from io import BytesIO
from loguru import logger
from PIL import Image, ImageDraw

def clean_invalid_json_escapes(s: str) -> str:
    if s is None:
        return None
    s = re.sub(r"^```json\s*|\s*```$", "", s.strip())
    s = s.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', s)
    return s

def draw_coords(image, x, y):
    width, height = image.size
    absolute_x = int(x * width)
    absolute_y = int(y * height)

    draw = ImageDraw.Draw(image)
    circle_size = 40

    draw.ellipse(
        [
            absolute_x - circle_size // 2,
            absolute_y - circle_size // 2,
            absolute_x + circle_size // 2,
            absolute_y + circle_size // 2
        ],
        outline="red",
        width=3
    )

    point_radius = 2
    draw.ellipse(
        [
            absolute_x - point_radius,
            absolute_y - point_radius,
            absolute_x + point_radius,
            absolute_y + point_radius
        ],
        fill="red"
    )
    return image

def parse_coordinates_from_line(line, max_num = 2):
    if not line:
        return None

    if line.startswith((
        "pyautogui.click",
        "pyautogui.moveTo",
        "pyautogui.dragTo",
        "pyautogui.doubleClick",
        "pyautogui.rightClick",
        "pyautogui.middleClick",
        "pyautogui.tripleClick",
        "computer.tripleClick",
    )):
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        floats = [float(n) for n in numbers][:max_num]
        return tuple(floats)

    return None

def parse_coordinates_from_code(code, max_num = 2):
    if not code:
        return None

    all_coords = []
    code_lines = code.split("\n")
    for line in code_lines:
        line = line.strip()
        if not line:
            continue

        coords = parse_coordinates_from_line(line)
        if coords:
            x, y = coords
            all_coords.append((x, y))

    return all_coords


def draw_coords_from_code(image, code):
    coords = parse_coordinates_from_code(code)
    print(coords)
    if coords:
        for x, y in coords:
            image = draw_coords(image, x, y)
    return image


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_time=600,
    max_tries=10,
    jitter=backoff.full_jitter,
)
def call_llm(
    client,
    messages,
    model,
    temperature = 0,
    ):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.exception(f"Retrying... Error calling LLM: {str(e)}")
        raise

def load_image(image_name, image_folder=None):
    if image_folder is None:
        image_path = image_name
    else:
        image_path = os.path.join(image_folder, image_name)
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image = Image.open(BytesIO(image_data))
    return image

def image_to_base64(pil_image):
    image_bytes = BytesIO()
    pil_image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64

def crop_image_patch(image: Image.Image, x: float, y: float, patch_size: int = 300, target_size: int = 300) -> Image.Image:
    """
    Crop a square patch centered at coordinates (x,y) and resize based on patch_size and target_size ratio.

    Args:
        image: PIL Image
        x: Pixel x coordinate
        y: Pixel y coordinate
        patch_size: Size of square patch to crop from original image (default 300)
        target_size: Maximum dimension for resize (default 300)

    Returns:
        PIL Image of cropped and resized patch with original aspect ratio
    """
    width, height = image.size

    center_x = int(x * width)
    center_y = int(y * height)

    half_size = patch_size // 2
    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    right = min(image.width, center_x + half_size)
    bottom = min(image.height, center_y + half_size)

    patch = image.crop((left, top, right, bottom))

    scaling_factor = target_size / patch_size

    w, h = patch.size
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    patch = patch.resize((new_w, new_h))

    return patch

def draw_bounding_box(image, center_x, center_y):
    width, height = image.size
    absolute_x = int(center_x * width)
    absolute_y = int(center_y * height)

    draw = ImageDraw.Draw(image)
    circle_size = 40

    draw.ellipse(
        [
            absolute_x - circle_size // 2,
            absolute_y - circle_size // 2,
            absolute_x + circle_size // 2,
            absolute_y + circle_size // 2
        ],
        outline="red",
        width=3
    )

    point_radius = 2
    draw.ellipse(
        [
            absolute_x - point_radius,
            absolute_y - point_radius,
            absolute_x + point_radius,
            absolute_y + point_radius
        ],
        fill="red"
    )

    return image

def draw_bounding_box_and_crop_patch(image, code):
    image_patch = None
    code_lines = code.split("\n")
    for line in code_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith((
            "pyautogui.click",
            "pyautogui.moveTo",
            "pyautogui.dragTo",
            "pyautogui.doubleClick",
            "pyautogui.rightClick",
            "pyautogui.middleClick",
            "pyautogui.tripleClick",
            "computer.tripleClick",
            )):

            match = re.search(r"x\s*=\s*([0-9.]+).*?y\s*=\s*([0-9.]+)", line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                if 0 <= x <= 1 and 0 <= y <= 1:
                    image = draw_bounding_box(image, x, y)
                    if not image_patch:
                        image_patch = crop_image_patch(image, x, y)

    return image, image_patch

def smart_resize(
    height: int,
    width: int,
    factor=28,
    min_pixels=3136,
    max_pixels=12845056,
    max_aspect_ratio_allowed: float | None = None,
    size_can_be_smaller_than_factor: bool = False,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if not size_can_be_smaller_than_factor and (height < factor or width < factor):
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor} "
            f"(when size_can_be_smaller_than_factor is False)"
        )
    elif max_aspect_ratio_allowed is not None and max(height, width) / min(height, width) > max_aspect_ratio_allowed:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_aspect_ratio_allowed}, "
            f"got {max(height, width) / min(height, width)}"
            f"(when max_aspect_ratio_allowed is not None)"
        )
    h_bar = max(1, round(height / factor)) * factor
    w_bar = max(1, round(width / factor)) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(1, math.floor(height / beta / factor)) * factor
        w_bar = max(1, math.floor(width / beta / factor)) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_code_relative_to_absolute(code: str, image: Image.Image) -> tuple[str, tuple[int, int], list[tuple[float, float]]]:
    """
        Convert relative coordinates to absolute coordinates.
    """
    original_w, original_h = image.size
    resized_h, resized_w = smart_resize(original_h, original_w)

    lines = code.split("\n")
    new_lines = []
    all_coords = []

    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append(line)
            continue

        coords = parse_coordinates_from_line(line)
        if coords and len(coords) == 2:
            x_rel, y_rel = coords
            all_coords.append((x_rel, y_rel))
            x_abs = int(x_rel * resized_w)
            y_abs = int(y_rel * resized_h)

            if "x=" in line and "y=" in line:
                line = re.sub(r"x\s*=\s*([-+]?\d*\.\d+|[-+]?\d+)", f"x={x_abs}", line)
                line = re.sub(r"y\s*=\s*([-+]?\d*\.\d+|[-+]?\d+)", f"y={y_abs}", line)
            else:
                line = re.sub(
                    r"([-+]?\d*\.\d+|[-+]?\d+)",
                    lambda m, c=[x_abs, y_abs]: str(c.pop(0)) if c else m.group(0),
                    line,
                    count=2
                )

        new_lines.append(line)

    return "\n".join(new_lines), (resized_w, resized_h), all_coords


def convert_code_absolute_to_relative(
    code: str,
    image: Image.Image,
    coord_type: str = "qwen25"  # or "absolute"
) -> tuple[str, tuple[int, int], list[tuple[float, float]]]:
    """
        Convert different types of coordinate to relative coordinates.
    """
    orig_w, orig_h = image.size
    if coord_type == "qwen25":
        resized_h, resized_w = smart_resize(orig_h, orig_w)
    elif coord_type == "absolute":
        resized_h, resized_w = orig_h, orig_w
    else:
        raise ValueError(f"Unsupported coord_type: {coord_type}. Use 'qwen25' or 'absolute'.")

    def to_rel(val: int, denom: int) -> str:
        return f"{val / denom:.4f}".rstrip("0").rstrip(".")

    lines, new_lines, all_coords = code.split("\n"), [], []

    for line in lines:
        l_strip = line.strip()
        if not l_strip:
            new_lines.append(line)
            continue

        coords = parse_coordinates_from_line(l_strip)
        if coords and len(coords) == 2:
            x_abs, y_abs = coords
            x_rel, y_rel = x_abs / resized_w, y_abs / resized_h
            all_coords.append((x_rel, y_rel))

            if "x=" in l_strip and "y=" in l_strip:
                line = re.sub(r"x\s*=\s*[-+]?\d+(\.\d+)?", f"x={to_rel(x_abs, resized_w)}", line)
                line = re.sub(r"y\s*=\s*[-+]?\d+(\.\d+)?", f"y={to_rel(y_abs, resized_h)}", line)
            else:
                repl_vals = [to_rel(x_abs, resized_w), to_rel(y_abs, resized_h)]
                line = re.sub(
                    r"[-+]?\d+(\.\d+)?",
                    lambda m, c=repl_vals: (c.pop(0) if c else m.group(0)),
                    line,
                    count=2,
                )
        new_lines.append(line)

    return "\n".join(new_lines), (resized_w, resized_h), all_coords
