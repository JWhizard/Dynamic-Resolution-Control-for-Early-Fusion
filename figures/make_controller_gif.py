"""Animated replay of the amortised sequence controller on real M3FD clips.

Nothing here is staged: frames come from the M3FD val split, detections come from
the cached ms_fusion_seed42 run *at the resolution the controller has in force*,
and the resolution itself is produced by replaying
``results/processed/sequence/controller.json`` over the clip.

The camera panel shows the frame resampled to the inference size and blown back
up with nearest-neighbour, so the size the detector actually sees is visible as
blockiness rather than only as a number.

    python figures/make_controller_gif.py

Writes figures/controller_stream.gif, and copies it to ../milestone2_figures/
when that folder exists.
"""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
RAW = Path("/mnt/12TB_Drive/whisenaj/Datasets/M3FD_Detection")
HERE = Path(__file__).resolve().parent
OUT = HERE / "controller_stream.gif"
ALSO = REPO.parent / "milestone2_figures"      # the slide deck's figure folder, if present
CTRL = REPO / "results" / "processed" / "sequence" / "controller.json"

import sys
sys.path.insert(0, str(REPO))
from controller_sequence import SequenceController  # noqa: E402
from probe_features import PEOPLE, SMALL  # noqa: E402

ACTIONS = (480, 544, 576, 640)
CONF = 0.10

# ------------------------------------------------------------------ canvas
W, H = 940, 672
IMG = (10, 42, 690, 552)          # 680 x 510, 4:3
PANEL = (700, 42, 930, 552)       # 230 wide
TRACK = (10, 560, 930, 648)       # 88 tall, so four levels stay apart

INK = (18, 18, 18)
PAPER = (252, 252, 250)
GREY = (150, 150, 150)          # rules and lead lines only; all type is INK
FAINT = (218, 218, 214)
HILITE = (208, 62, 40)            # the small-People trigger
PERSON = (255, 165, 0)            # any People candidate

F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
fnt = {k: ImageFont.truetype(p, s) for k, (p, s) in {
    "title": (FB, 17), "sub": (F, 12), "big": (FB, 34), "lab": (FB, 11),
    "body": (F, 12), "small": (F, 10), "tick": (F, 10), "banner": (FB, 13),
}.items()}


def clips():
    man = json.loads((REPO / "splits" / "m3fd_grouped_seed42.json").read_text())
    groups, val = man["groups"], sorted(man["splits"]["val"])
    by = {}
    for f in val:
        by.setdefault(groups[f], []).append(f)
    return [
        (by[496], "the window runs out", "sixteen frames go by, then the tree lowers the size"),
        (by[83][:16], "the override", "a small pedestrian cuts the window short"),
    ]


def replay(frames, caches):
    """Per-frame record of what the controller did and why."""
    c = SequenceController.from_json(CTRL)
    out = []
    for i, f in enumerate(frames):
        before = c.current_res
        res = c.choose(f)
        scheduled = res != before or (before == res and c.frames_in_window == 1 and i)
        det = caches[res][f]
        held = c.current_res
        c.update(f, det)
        out.append({
            "id": f, "res": res, "det": det,
            "scheduled": res != before,
            "override": c.current_res != held,
            "left": c.interval - c.frames_in_window,
            "feat": c.previous_features,
        })
    return out


def camera(frame_id, res, det, flag):
    """The frame as the detector saw it, with the boxes it returned."""
    w, h = IMG[2] - IMG[0], IMG[3] - IMG[1]
    im = Image.open(RAW / "Vis" / f"{frame_id}.png").convert("RGB")
    sw = res                                   # long side of the inference input
    sh = max(1, round(sw * im.height / im.width))
    im = im.resize((sw, sh), Image.LANCZOS).resize((w, h), Image.NEAREST)
    g = ImageDraw.Draw(im)
    sx, sy = w / 1024, h / 768
    for x1, y1, x2, y2, conf, cls in det:
        if conf < CONF:
            continue
        area = (x2 - x1) * (y2 - y1)
        small_person = cls == PEOPLE and area < SMALL
        col = HILITE if small_person else (PERSON if cls == PEOPLE else (255, 255, 255))
        box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        g.rectangle(box, outline=(0, 0, 0), width=4)
        g.rectangle(box, outline=col, width=2)
        if small_person:
            g.rectangle([box[0] - 9, box[1] - 9, box[2] + 9, box[3] + 9], outline=col, width=2)
    if flag:
        text, col = flag
        tw = g.textlength(text, font=fnt["banner"])
        g.rectangle([8, 8, 8 + tw + 18, 34], fill=(255, 255, 255), outline=col, width=3)
        g.text((17, 13), text, font=fnt["banner"], fill=INK)
    return im


def panel(d, i, n):
    """Resolution, the window countdown, and what the summary says."""
    w, h = PANEL[2] - PANEL[0], PANEL[3] - PANEL[1]
    im = Image.new("RGB", (w, h), PAPER)
    g = ImageDraw.Draw(im)
    g.rectangle([0, 0, w - 1, h - 1], outline=FAINT, width=1)

    g.text((14, 14), "SIZE IN FORCE", font=fnt["lab"], fill=INK)
    g.text((14, 30), f"{d['res']} px", font=fnt["big"], fill=INK)

    g.line([14, 82, w - 14, 82], fill=FAINT, width=1)
    g.text((14, 92), "WINDOW", font=fnt["lab"], fill=INK)
    g.text((14, 108), "restarted — 16 frames to go" if d["override"]
           else f"{d['left']} of 16 frames left",
           font=fnt["body"], fill=INK)
    for k in range(16):                        # 16 pips, filled as the window runs
        x = 14 + k * ((w - 28) / 16)
        done = k < 16 - d["left"]
        g.rectangle([x, 130, x + (w - 28) / 16 - 2, 138],
                    fill=INK if done else FAINT)

    g.line([14, 156, w - 14, 156], fill=FAINT, width=1)
    f = d["feat"]
    g.text((14, 166), "THIS FRAME'S SUMMARY", font=fnt["lab"], fill=INK)
    rows = [
        ("candidates ≥ 0.10", f"{f['n_conf10']:.0f}", INK),
        ("of them People", f"{f['n_people_conf10']:.0f}", PERSON if f["n_people_conf10"] else INK),
        ("People under 32\u00b2 px", f"{f['n_small_people_conf10']:.0f}",
         HILITE if f["n_small_people_conf10"] else INK),
        ("mean conf. top 10", f"{f['mean_conf_top10']:.2f}", INK),
        ("median box px²", f"{f['median_area_conf25']:,.0f}", INK),
    ]
    for k, (lab, val, _col) in enumerate(rows):
        y = 186 + k * 19
        g.text((14, y), lab, font=fnt["small"], fill=INK)
        g.text((w - 14 - g.textlength(val, font=fnt["body"]), y - 1), val,
               font=fnt["body"], fill=INK)

    g.line([14, 288, w - 14, 288], fill=FAINT, width=1)
    g.text((14, 298), "THE RULE IN PLAY", font=fnt["lab"], fill=INK)
    if d["override"]:
        msg, col = ("a People candidate under\n32\u00b2 px: 640 px on the next\nframe, window restarts", HILITE)
    elif d["scheduled"]:
        msg, col = (f"the window ended: the tree\nread the last summary and\nset {d['res']} px", INK)
    else:
        msg, col = ("hold — the summary is kept,\nnothing is decided until the\nwindow ends", INK)
    g.multiline_text((14, 316), msg, font=fnt["body"], fill=INK, spacing=5)

    g.text((14, h - 26), f"frame {i + 1} of {n}  ·  {d['id']}", font=fnt["small"], fill=INK)
    return im


def track(rec, i):
    """The whole clip's resolution path, drawn up to the current frame."""
    w, h = TRACK[2] - TRACK[0], TRACK[3] - TRACK[1]
    im = Image.new("RGB", (w, h), PAPER)
    g = ImageDraw.Draw(im)
    left, right, top, bot = 54, w - 12, 12, h - 30
    ys = {r: bot - (ACTIONS.index(r) / 3) * (bot - top) for r in ACTIONS}
    for r, y in ys.items():
        for x in range(left, right, 6):
            g.line([x, y, x + 3, y], fill=FAINT, width=1)
        g.text((left - 10 - g.textlength(str(r), font=fnt["tick"]), y - 6),
               str(r), font=fnt["tick"], fill=INK)
    n = len(rec)
    xs = [left + (k + 0.5) * (right - left) / n for k in range(n)]

    def path(upto, col, width):
        pts = []
        for k in range(upto + 1):
            y = ys[rec[k]["res"]]
            if pts and pts[-1][1] != y:
                pts.append((xs[k] - (xs[1] - xs[0]) / 2, pts[-1][1]))
                pts.append((xs[k] - (xs[1] - xs[0]) / 2, y))
            pts.append((xs[k], y))
        if len(pts) > 1:
            g.line(pts, fill=col, width=width, joint="curve")

    path(n - 1, FAINT, 3)
    path(i, INK, 3)
    g.ellipse([xs[i] - 5, ys[rec[i]["res"]] - 5, xs[i] + 5, ys[rec[i]["res"]] + 5], fill=INK)
    g.text((left, h - 20), "frame 1", font=fnt["tick"], fill=INK)
    g.text((right - g.textlength(f"frame {n}", font=fnt["tick"]), h - 20),
           f"frame {n}", font=fnt["tick"], fill=INK)
    return im


def compose(rec, i, head, sub):
    d = rec[i]
    flag = None
    if d["override"]:
        flag = ("SMALL PEDESTRIAN — 640 px NEXT FRAME", HILITE)
    elif d["scheduled"]:
        flag = (f"WINDOW ENDED — {d['res']} px", (30, 30, 30))
    canvas = Image.new("RGB", (W, H), PAPER)
    g = ImageDraw.Draw(canvas)
    g.text((10, 10), head, font=fnt["title"], fill=INK)
    g.text((10 + g.textlength(head, font=fnt["title"]) + 12, 14), sub, font=fnt["sub"], fill=INK)
    canvas.paste(camera(d["id"], d["res"], d["det"], flag), IMG[:2])
    canvas.paste(panel(d, i, len(rec)), PANEL[:2])
    canvas.paste(track(rec, i), TRACK[:2])
    g.text((10, H - 18),
           "M3FD val split · boxes are the detector's own output at the size in force · "
           "the picture is resampled to that size, so you see what the detector sees",
           font=fnt["small"], fill=INK)
    return canvas


def main():
    caches = {r: pickle.load(open(REPO / "results" / "cache" / f"ms_fusion_seed42_{r}_val.pkl", "rb"))
              for r in ACTIONS}
    pages, durations = [], []
    for frames, head, sub in clips():
        rec = replay(frames, caches)
        for i, d in enumerate(rec):
            pages.append(compose(rec, i, head, sub))
            # linger on the frames where the controller actually does something
            durations.append(1500 if (d["scheduled"] or d["override"]) else
                             (1400 if i == len(rec) - 1 else 400))
    write_gif(pages, durations)
    if ALSO.is_dir():
        shutil.copy2(OUT, ALSO / OUT.name)
    print(OUT, len(pages), "frames", OUT.stat().st_size // 1024, "KB")


def write_gif(pages, durations):
    """ffmpeg with one palette for the whole clip if it is available, PIL otherwise.

    A single global palette is worth several megabytes here, because every frame
    is a photograph.  The concat demuxer carries the per-frame durations, so the
    file holds one image per step rather than a padded frame rate.
    """
    if not shutil.which("ffmpeg"):
        q = [p.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE) for p in pages]
        q[0].save(OUT, save_all=True, append_images=q[1:], duration=durations,
                  loop=0, optimize=True, disposal=2)
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lines = []
        for i, (page, ms) in enumerate(zip(pages, durations)):
            page.save(tmp / f"f{i:04d}.png")
            lines += [f"file 'f{i:04d}.png'", f"duration {ms / 1000:.2f}"]
        lines.append(f"file 'f{len(pages) - 1:04d}.png'")   # concat needs the tail twice
        (tmp / "list.txt").write_text("\n".join(lines))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(tmp / "list.txt"), "-filter_complex",
             "[0:v]split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];"
             "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
             "-loop", "0", str(OUT)], check=True)


if __name__ == "__main__":
    main()
