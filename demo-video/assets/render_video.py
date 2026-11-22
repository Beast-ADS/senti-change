from __future__ import annotations

import json
import math
import re
import subprocess
import textwrap
import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as geom_transform

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
CASE = Path(__file__).resolve().parents[2] / "outputs" / "ENC-0012"
RECORD = json.loads((CASE / "evidence.json").read_text(encoding="utf-8"))
EVIDENCE = RECORD["evidence"]
W, H, FPS = 1920, 1080, 24
NAVY = (7, 17, 30)
NAVY2 = (13, 29, 48)
WHITE = (241, 246, 251)
MUTED = (160, 178, 196)
CYAN = (72, 196, 211)
ORANGE = (255, 131, 78)

SCENES = [
    ("THE CHALLENGE", "Waterbody change is\nhard to see at scale.",
     "A changing shoreline is an observation.\nIt is not proof of encroachment.", "baseline"),
    ("CHANGE SENTINEL", "From satellite observations\nto traceable cases.",
     "Geospatial screening for waterbody change\nand human investigation.", "overlay"),
    ("LIVE CATALOG ACQUISITION", "A real catalog query.\nA recorded, dated result.",
     "Verified study area · Hussain Sagar\nQuery recorded · 05 Nov 2026", "overlay"),
    ("SEASON-MATCHED COMPARISON", "Compare like months\nwith like months.",
     "Same-calendar-month baseline and recent observations", "pair"),
    ("OBSERVED CHANGE CANDIDATE", "A measurable signal.\nNot a legal conclusion.",
     "Potential persistent water-extent loss\ninside the historical footprint", "overlay"),
    ("EVIDENCE FUSION", "Independent sources\ncan disagree.",
     "This disagreement stays visible in the case record.", "overlay"),
    ("UNCERTAINTY & PRIORITY", "A score for review.\nNever a probability.",
     "The sources disagree. The case stays\nat potential change for human investigation.", "overlay"),
    ("TRACEABLE CASE", "Case ENC-0012 keeps\nevidence together.",
     "Dates · AOI provenance · source findings\nmeasurements · geospatial exports", "overlay"),
    ("CLOSING", "From satellite observations\nto traceable investigations.",
     "Change Sentinel · waterbody change screening\nfor human review", "overlay"),
]


def get_font(size, bold=False):
    paths = ([r"C:\Windows\Fonts\seguisb.ttf", r"C:\Windows\Fonts\segoeuib.ttf"] if bold
             else [r"C:\Windows\Fonts\segoeui.ttf"])
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        proposal = f"{line} {word}".strip()
        if line and draw.textbbox((0, 0), proposal, font=fnt)[2] > width:
            lines.append(line)
            line = word
        else:
            line = proposal
    if line:
        lines.append(line)
    return lines


def map_assets():
    before_path = CASE / "baseline_matched_frequency.tif"
    recent_path = CASE / "recent_matched_frequency.tif"
    mask_path = CASE / "change_mask.tif"
    aoi_path = CASE / "aoi.geojson"
    if not all(p.is_file() for p in (before_path, recent_path, mask_path, aoi_path)):
        raise FileNotFoundError("ENC-0012 is missing required georeferenced map evidence")
    with rasterio.open(before_path) as b, rasterio.open(recent_path) as r, rasterio.open(mask_path) as m:
        if (b.crs != r.crs or b.transform != r.transform or b.shape != r.shape or
                m.crs != b.crs or m.transform != b.transform or m.shape != b.shape):
            raise ValueError("ENC-0012 before, after and loss rasters are not spatially aligned")
        baseline, recent, loss = b.read(1), r.read(1), m.read(1) == 1
        valid = (baseline != b.nodata) & (recent != r.nodata)
        transform, crs, bounds = b.transform, b.crs, b.bounds
    gj = json.loads(aoi_path.read_text(encoding="utf-8"))
    geom = shape(gj.get("geometry", gj))
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geom = geom_transform(tx.transform, geom)
    ASSETS.mkdir(parents=True, exist_ok=True)

    def render_map(arr, title, overlay=False):
        t = np.clip(np.nan_to_num(arr, nan=0.0), 0, 1)[..., None]
        lo, mid, hi = (np.array(c, dtype=np.float32) for c in ([18, 38, 58], [30, 102, 153], [71, 197, 207]))
        a = np.clip(t/0.55, 0, 1)
        rgb = lo*(1-a)+mid*a
        b = np.clip((t-0.55)/0.45, 0, 1)
        rgb = rgb*(1-b)+hi*b
        rgb[~valid] = [11, 25, 40]
        if overlay:
            rgb[loss] = 0.35*rgb[loss]+0.65*np.array(ORANGE)
        im = Image.fromarray(rgb.astype(np.uint8), "RGB")
        # Preserve the exported raster aspect ratio; never stretch the pixels to a map box.
        scale = 610/im.height
        outw, outh = round(im.width*scale), round(im.height*scale)
        im = im.resize((outw, outh), Image.Resampling.NEAREST)
        panel = Image.new("RGB", (760, 790), (10, 23, 39))
        d = ImageDraw.Draw(panel)
        d.text((26, 18), title, font=get_font(22, True), fill=WHITE)
        d.text((26, 51), "Observed-water frequency · exported 10 m grid", font=get_font(15), fill=MUTED)
        x0, y0 = (760-outw)//2, 88
        panel.paste(im, (x0, y0))
        d = ImageDraw.Draw(panel)

        def pix(point):
            col, row = (~transform) * (point[0], point[1])
            return x0+col*scale, y0+row*scale

        polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polygons:
            pts = [pix(p) for p in poly.exterior.coords]
            if len(pts) > 1:
                d.line(pts, fill=(231, 240, 247), width=2, joint="curve")
        d.text((26, 722), f"EPSG:{crs.to_epsg()} · 10 m · {bounds.left/1000:.1f}–{bounds.right/1000:.1f} km E",
               font=get_font(14), fill=MUTED)
        if overlay:
            d.rounded_rectangle((26, 754, 44, 772), radius=4, fill=ORANGE)
            d.text((54, 751), "Observed-loss candidate", font=get_font(14), fill=WHITE)
        d.text((455, 754), "Case ENC-0012 exports", font=get_font(13), fill=(125, 150, 172))
        return panel

    bmap = render_map(baseline, "BEFORE · MATCHED BASELINE")
    rmap = render_map(recent, "AFTER · RECENT MATCHED MONTHS")
    omap = render_map(recent, "HUSSAIN SAGAR · RECENT OBSERVATIONS", True)
    bmap.save(ASSETS / "baseline_frequency.png")
    rmap.save(ASSETS / "recent_frequency.png")
    pair = Image.new("RGB", (1560, 790), NAVY)
    pair.paste(bmap, (0, 0))
    pair.paste(rmap, (800, 0))
    pair.save(ASSETS / "before_after_comparison.png")
    omap.save(ASSETS / "georeferenced_candidate_map.png")


def background():
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top = np.array(NAVY, dtype=np.float32)[None, None, :]
    bot = np.array(NAVY2, dtype=np.float32)[None, None, :]
    pix = np.broadcast_to(top*(1-y)+bot*y, (H, W, 3)).astype(np.uint8).copy()
    im = Image.fromarray(pix)
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(80, W, 80):
        d.line((x, 110, x, 1025), fill=(78, 125, 158, 14), width=1)
    for yy in range(150, 1020, 80):
        d.line((75, yy, 1845, yy), fill=(78, 125, 158, 14), width=1)
    return im


def slide(scene, i):
    im = background()
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle((82, 40, 110, 68), radius=8, fill=CYAN)
    d.line((88, 56, 96, 48, 103, 61), fill=NAVY, width=3)
    d.text((124, 39), "CHANGE SENTINEL", font=get_font(23, True), fill=WHITE)
    d.text((1600, 44), f"{i+1:02d} / 09", font=get_font(18, True), fill=MUTED)
    d.line((82, 91, 1838, 91), fill=(89, 126, 153, 95), width=1)
    label, title, body, kind = scene
    d.text((96, 128), label, font=get_font(18, True), fill=CYAN)
    if kind == "pair":
        pair = Image.open(ASSETS/"before_after_comparison.png").convert("RGB")
        pair.thumbnail((1740, 650), Image.Resampling.LANCZOS)
        im.paste(pair, ((W-pair.width)//2, 186))
        for j, line in enumerate(title.split("\n")):
            d.text((100, 170+j*48), line, font=get_font(39, True), fill=WHITE)
        d.text((100, 847), "100% of historical footprint comparable · maps use the same projected grid",
               font=get_font(21), fill=MUTED)
    else:
        y = 216
        for line in title.split("\n"):
            d.text((100, y), line, font=get_font(55 if i != 8 else 49, True), fill=WHITE)
            y += 68
        y += 12
        for line in body.split("\n"):
            d.text((105, y), line, font=get_font(23), fill=MUTED)
            y += 34
        mapfile = "baseline_frequency.png" if kind == "baseline" else "georeferenced_candidate_map.png"
        mapim = Image.open(ASSETS/mapfile).convert("RGB")
        mapim.thumbnail((720, 740), Image.Resampling.LANCZOS)
        d.rounded_rectangle((1070, 165, 1818, 918), radius=18,
                            fill=(16, 34, 54, 235), outline=(77, 123, 153, 160), width=2)
        im.paste(mapim, (1070+(748-mapim.width)//2, 178))
        cards = {
            0: [("MONITOR", "Broad areas"), ("COMPARE", "Across dates"), ("VERIFY", "Before action")],
            1: [("SCREEN", "Water extent"), ("TRACE", "Evidence sources"), ("REVIEW", "Human decision")],
            2: [("640", "catalog items"), ("287", "read successfully"), ("0", "read failures")],
            4: [("51.9 ha", "loss candidate"), ("209.29 ha", "historical footprint"), ("±14.63 ha", "boundary uncertainty")],
            8: [("OBSERVE", "Satellite evidence"), ("TRACE", "Geographic record"), ("VERIFY", "Human review")],
        }.get(i, [])
        cy = 624
        for j, (head, sub) in enumerate(cards):
            x = 100+j*300
            d.rounded_rectangle((x, cy, x+282, cy+116), radius=13,
                                fill=(13, 29, 46, 240), outline=(67, 112, 145, 160), width=1)
            d.text((x+16, cy+14), head, font=get_font(22 if len(head)>12 else 26, True),
                   fill=ORANGE if j == len(cards)-1 else CYAN)
            d.text((x+16, cy+59), sub, font=get_font(17), fill=WHITE)
        if i == 2:
            d.text((102, 786), "ACCEPTED OBSERVATIONS", font=get_font(14, True), fill=(126, 150, 173))
            d.text((102, 811), "09 Nov 2022  →  14 Oct 2026 · latest usable scene 22 days old",
                   font=get_font(18, True), fill=WHITE)
        elif i == 4:
            c = EVIDENCE["coordinates"]["centroid_wgs84"]
            d.text((102, 786), f"FLAGGED CENTROID  ·  {c[0]:.5f}°N, {c[1]:.5f}°E", font=get_font(17, True), fill=WHITE)
            d.text((102, 811), "E1 · observed extent loss · not a construction finding", font=get_font(17), fill=ORANGE)
        elif i == 5:
            d.text((101, 790), "SENTINEL-1  ·  CONTRADICTS  ·  recent radar still reads water", font=get_font(16, True), fill=ORANGE)
            d.text((101, 817), "DYNAMIC WORLD  ·  CONTRADICTS  ·  mean water probability 0.70", font=get_font(16, True), fill=ORANGE)
        elif i == 6:
            d.text((101, 790), "50 / 100 · HEURISTIC, NOT A PROBABILITY", font=get_font(17, True), fill=CYAN)
            d.text((101, 817), "POTENTIAL CHANGE · REVIEW AND FIELD CONFIRMATION REQUIRED", font=get_font(15, True), fill=ORANGE)
        elif i == 7:
            d.text((101, 792), "AOI STATUS · VERIFIED  |  CRS · EPSG:32644", font=get_font(17, True), fill=CYAN)
            d.text((101, 819), "JRC historical coverage ends 2020", font=get_font(16), fill=MUTED)
    d.line((82, 1037, 1838, 1037), fill=(89, 126, 153, 95), width=1)
    d.text((96, 1047), "SATELLITE-DERIVED SCREENING · NOT A LEGAL FINDING", font=get_font(14, True), fill=(145, 164, 181))
    return im.convert("RGB")


def subtitle(base, text):
    im = base.copy().convert("RGBA")
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((145, 880, 1775, 1018), radius=18, fill=(3, 10, 18, 222), outline=(84, 121, 149, 140), width=1)
    im.alpha_composite(layer)
    d = ImageDraw.Draw(im)
    lines = wrap(d, text, get_font(26), 1490)[:3]
    y = 880+(138-len(lines)*35)//2
    d.multiline_text((197, y), "\n".join(lines), font=get_font(26), fill=WHITE, spacing=5)
    return im.convert("RGB")


def parse_script():
    raw = (ROOT/"narration_script.txt").read_text(encoding="utf-8-sig")
    chunks, current = [], None
    for line in raw.splitlines():
        if line.startswith("SCENE "):
            if current:
                chunks.append(current)
            current = {"heading": line, "text": []}
        elif current and line.strip():
            current["text"].append(line.strip())
    if current:
        chunks.append(current)
    sentences = []
    for i, chunk in enumerate(chunks):
        text = " ".join(chunk["text"])
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        for sentence in parts:
            if sentence.strip():
                sentences.append({"scene": i, "text": sentence.strip()})
    return sentences


def make_voice_manifest():
    if RECORD.get("synthetic_inputs") is not False or (EVIDENCE.get("aoi") or {}).get("status") != "verified":
        raise ValueError("Refusing to render with a synthetic or unverified case")
    acq = EVIDENCE.get("acquisition") or {}
    if acq.get("status") != "complete" or acq.get("read_ok") != 287 or acq.get("failed") != 0:
        raise ValueError("ENC-0012 acquisition record did not match the verified live case")
    map_assets()
    (ASSETS/"audio").mkdir(exist_ok=True)
    segments = parse_script()
    for i, seg in enumerate(segments, 1):
        seg["path"] = str((ASSETS/"audio"/f"line_{i:02d}.wav").resolve())
    manifest = ASSETS/"voice_manifest.json"
    manifest.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    ps = r'''param([string]$Manifest)
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice("Microsoft Zira Desktop")
$voice.Rate = -1
$voice.Volume = 100
foreach ($item in $items) {
  $voice.SetOutputToWaveFile($item.path)
  $voice.Speak($item.text)
  $voice.SetOutputToNull()
}
$voice.Dispose()
'''
    ps_path = ASSETS/"synthesize_voice.ps1"
    ps_path.write_text(ps, encoding="utf-8")
    speech_ok = False
    try:
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_path), "-Manifest", str(manifest)], check=True, capture_output=True, text=True)
        speech_ok = all(Path(seg["path"]).is_file() for seg in segments)
    except (subprocess.SubprocessError, OSError):
        pass
    if not speech_ok:
        # Keep a caption-led film renderable when local Windows speech is unavailable.
        # A quiet AAC track preserves broad player compatibility without implying voice narration.
        for seg in segments:
            samples = max(1, round(max(2.4, len(seg["text"].split()) / 2.4) * 16000))
            with wave.open(seg["path"], "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(np.zeros(samples, dtype="<i2").tobytes())
    return segments


def wav_data(path):
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(f"Unexpected voice WAV format: {path}")
        return wf.getframerate(), np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").copy()


def timestamp(seconds):
    ms = round(seconds*1000)
    hh, ms = divmod(ms, 3_600_000)
    mm, ms = divmod(ms, 60_000)
    ss, ms = divmod(ms, 1000)
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"


def render(segments):
    by_scene = {i: [] for i in range(len(SCENES))}
    sample_rate = None
    for seg in segments:
        sr, samples = wav_data(seg["path"])
        if sample_rate is None:
            sample_rate = sr
        if sr != sample_rate:
            raise ValueError("Narration clips have inconsistent sample rates")
        seg["samples"] = samples
        seg["frames"] = max(1, math.ceil(len(samples)/sr*FPS))
        seg["pad_to"] = round(seg["frames"]*sr/FPS)
        seg["audio"] = np.pad(samples, (0, max(0, seg["pad_to"]-len(samples))))
        by_scene[seg["scene"]].append(seg)

    frames, audio_parts, cues = [], [], []
    absolute_frame = 0
    for scene_i, scene in enumerate(SCENES):
        base = slide(scene, scene_i)
        for part_i, seg in enumerate(by_scene[scene_i]):
            start = absolute_frame
            frames.extend([subtitle(base, seg["text"])]*seg["frames"])
            audio_parts.append(seg["audio"])
            absolute_frame += seg["frames"]
            cues.append((start/FPS, absolute_frame/FPS, seg["text"]))
            if part_i < len(by_scene[scene_i])-1:
                pause = round(0.22*FPS)
                frames.extend([base]*pause)
                audio_parts.append(np.zeros(round(pause*sample_rate/FPS), np.int16))
                absolute_frame += pause
        if scene_i < len(SCENES)-1:
            next_base = slide(SCENES[scene_i+1], scene_i+1)
            pad_frames, fade_frames = round(0.9*FPS), round(0.6*FPS)
            audio_parts.append(np.zeros(round(pad_frames*sample_rate/FPS), np.int16))
            for j in range(pad_frames):
                if j < pad_frames-fade_frames:
                    frames.append(base)
                else:
                    alpha = (j-(pad_frames-fade_frames)+1)/fade_frames
                    frames.append(Image.blend(base, next_base, alpha))
            absolute_frame += pad_frames
        else:
            hold = round(1.2*FPS)
            frames.extend([base]*hold)
            audio_parts.append(np.zeros(round(hold*sample_rate/FPS), np.int16))
            absolute_frame += hold

    full_audio = np.concatenate(audio_parts).astype(np.int16)
    srt = []
    for n, (a, b, text) in enumerate(cues, 1):
        srt.extend([str(n), f"{timestamp(a)} --> {timestamp(b)}", "\n".join(textwrap.wrap(text, width=72)), ""])
    (ROOT/"Change_Sentinel_Demo.srt").write_text("\n".join(srt), encoding="utf-8")

    target = ROOT/"Change_Sentinel_Demo.mp4"
    container = av.open(str(target), "w", format="mp4")
    vs = container.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = {"crf": "19", "preset": "medium", "movflags": "+faststart"}
    aus = container.add_stream("aac", rate=sample_rate)
    aus.layout = "mono"
    aus.bit_rate = 128000
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    audio_at, chunk_size = 0, 1024
    for i, image in enumerate(frames):
        target_audio = min(len(full_audio), round((i+1)*sample_rate/FPS))
        while audio_at < target_audio:
            end = min(audio_at+chunk_size, len(full_audio))
            af = av.AudioFrame.from_ndarray(full_audio[audio_at:end].reshape(1, -1), format="s16", layout="mono")
            af.sample_rate = sample_rate
            af.pts = audio_at
            af.time_base = Fraction(1, sample_rate)
            for converted in resampler.resample(af):
                for packet in aus.encode(converted):
                    container.mux(packet)
            audio_at = end
        vf = av.VideoFrame.from_image(image)
        vf.pts = i
        vf.time_base = Fraction(1, FPS)
        for packet in vs.encode(vf):
            container.mux(packet)
    while audio_at < len(full_audio):
        end = min(audio_at+chunk_size, len(full_audio))
        af = av.AudioFrame.from_ndarray(full_audio[audio_at:end].reshape(1, -1), format="s16", layout="mono")
        af.sample_rate, af.pts = sample_rate, audio_at
        af.time_base = Fraction(1, sample_rate)
        for converted in resampler.resample(af):
            for packet in aus.encode(converted):
                container.mux(packet)
        audio_at = end
    for converted in resampler.resample(None):
        for packet in aus.encode(converted):
            container.mux(packet)
    for packet in aus.encode(None):
        container.mux(packet)
    for packet in vs.encode(None):
        container.mux(packet)
    container.close()
    return target, absolute_frame/FPS, len(cues)


def verify_video(path, duration):
    container = av.open(str(path))
    streams = {s.type: s for s in container.streams}
    if "video" not in streams or "audio" not in streams:
        raise RuntimeError("Rendered MP4 must contain video and narration audio")
    v = streams["video"]
    if (v.codec_context.width, v.codec_context.height) != (W, H):
        raise RuntimeError("Rendered video has the wrong resolution")
    if abs(float(container.duration/av.time_base)-duration) > 1.0:
        raise RuntimeError("Rendered duration does not match the frame timeline")
    samples = []
    decoded = 0
    for frame in container.decode(video=0):
        decoded += 1
        if decoded in (1, max(1, round(duration*FPS/2)), max(1, round(duration*FPS))):
            img = frame.to_ndarray(format="rgb24")
            samples.append((float(img.mean()), float(img.std())))
    if decoded < FPS*120 or any(std < 3 for _, std in samples):
        raise RuntimeError("Video decode or frame-content check failed")
    if len(list(container.decode(audio=0))) == 0:
        raise RuntimeError("Narration audio could not be decoded")
    container.close()
    return decoded, streams["audio"].codec_context.name


def main():
    if RECORD.get("synthetic_inputs") is not False or (EVIDENCE.get("aoi") or {}).get("status") != "verified":
        raise ValueError("Refusing to create demo from synthetic or unverified data")
    if EVIDENCE.get("evidence_fusion", {}).get("sources", {}).get("sentinel1", {}).get("verdict") != "contradicts":
        raise ValueError("ENC-0012 source findings changed; review script before rendering")
    segments = make_voice_manifest()
    target, duration, cues = render(segments)
    frame_count, audio_codec = verify_video(target, duration)
    notes = [
        "# Production notes", "", "## Genuine source material", "",
        "- Case ENC-0012: `synthetic_inputs=false`; verified Hussain Sagar AOI.",
        "- Recorded Sentinel-2 query on 2026-11-05: 640 items found, 353 cloud-filtered, 287 read, 0 failed.",
        "- Accepted scene period: 2022-11-09 through 2026-10-14; latest usable scene was 22 days old at analysis time.",
        "- Maps are rendered from the case baseline/recent frequency GeoTIFFs and exported candidate mask; before, after and mask were verified aligned at EPSG:32644, 10 m.",
        "- Recorded source results: Sentinel-1 contradicts the candidate; Dynamic World contradicts it; JRC supports historical context only and ends in 2020.",
        "- Case state is potential change, score 50/100 heuristic, not a probability. This case has no ground-truth label; no scientific accuracy is claimed.",
        "", "## Capture and limitations", "",
        "- No live screen recording was captured. The browser capture helper could not identify the open browser URL, so this video uses authentic georeferenced case exports plus custom explanatory slides; it does not present a fake UI recording.",
        "- The water-frequency maps are derived analysis rasters, not raw-color satellite imagery.",
        "- Captions are burned in and supplied as SRT. Local Windows text-to-speech was unavailable in this session, so the MP4 contains a silent AAC compatibility track rather than narration.",
        "- No background music is included because no licensed audio asset was available.",
        "- The project database and `.env` were not read or modified for video production. The existing case output files were read only.",
        "", f"Output: 1920x1080, 24 fps, {duration:.2f} s ({duration/60:.2f} min), H.264 video + {audio_codec} audio; {frame_count} frames; {cues} subtitle cues."
    ]
    (ROOT/"production_notes.md").write_text("\n".join(notes)+"\n", encoding="utf-8")
    print(json.dumps({"mp4": str(target), "duration_s": duration, "resolution": f"{W}x{H}",
                      "frames": frame_count, "audio_codec": audio_codec, "subtitle_cues": cues,
                      "bytes": target.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
