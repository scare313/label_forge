# app.py
# LabelForge — Streamlit SKU Label Maker (FBA + MRP)
# OOP + SOLID, unique previews, catalog moved to a separate tab

from __future__ import annotations
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =========================
# Domain Models & Enums
# =========================

class LabelType(Enum):
    FBA = auto()
    MRP = auto()

@dataclass
class Settings:
    # General
    dpi: int = 300
    # FBA mm
    fba_label_width_mm: float = 50.0
    fba_label_height_mm: float = 25.0
    fba_title_scale: float = 0.30
    fba_small_scale: float = 0.50
    fba_footer_pad_px: int = 30
    # MRP mm
    mrp_label_width_mm: float = 40.0
    mrp_label_height_mm: float = 30.0
    mrp_brand_scale: float = 0.50
    mrp_title_scale: float = 0.50
    mrp_mrp_scale: float = 0.60
    mrp_small_scale: float = 0.50
    mrp_bottom_pad_px: int = 30
    mrp_show_inclusive_line: bool = True
    mrp_max_title_lines: int = 2
    # Fonts
    font_path_regular: str = "assets/DejaVuSans.ttf"
    font_path_bold: str = "assets/DejaVuSans-Bold.ttf"
    # What to print
    print_fba_labels: bool = True
    print_mrp_labels: bool = True
    # UI
    limit_preview: int = 12

@dataclass
class SkuRecord:
    SKU: str
    ASIN: str = ""
    FNSKU: str = ""
    Brand: str = ""
    Title: str = ""
    MRP: str = ""
    HSN: str = ""
    GST: str = ""  # store field for UI; mapped from "GST%"
    MfgMonthYear: str = ""
    Condition: str = "New"
    Category: str = ""

    @staticmethod
    def from_dict(d: Dict) -> "SkuRecord":
        return SkuRecord(
            SKU=str(d.get("SKU", "")).strip(),
            ASIN=str(d.get("ASIN", "")).strip(),
            FNSKU=str(d.get("FNSKU", "")).strip(),
            Brand=str(d.get("Brand", "")).strip(),
            Title=str(d.get("Title", "")).strip(),
            MRP=str(d.get("MRP", "")).strip(),
            HSN=str(d.get("HSN", "")).strip(),
            GST=str(d.get("GST%", d.get("GST", ""))).strip(),
            MfgMonthYear=str(d.get("MfgMonthYear", "")).strip(),
            Condition=str(d.get("Condition", "New")).strip() or "New",
            Category=str(d.get("Category", "")).strip(),
        )

    def to_json_obj(self) -> Dict:
        # Persist as original schema with "GST%"
        return {
            "ASIN": self.ASIN,
            "FNSKU": self.FNSKU,
            "Brand": self.Brand,
            "Title": self.Title,
            "MRP": self.MRP,
            "HSN": self.HSN,
            "GST%": self.GST,
            "MfgMonthYear": self.MfgMonthYear,
            "Condition": self.Condition,
            "Category": self.Category,
        }

# =========================
# Utilities
# =========================

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))

def wrap_text(draw, text: str, font, max_width: int, max_lines: int = 2):
    words = (text or "").split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
                cur = w
            else:
                while draw.textlength(w, font=font) > max_width and len(w) > 1:
                    w = w[:-1]
                lines.append(w)
                cur = ""
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines and " ".join(words) != " ".join(lines):
        t, ell = lines[-1], "…"
        while t and draw.textlength(t + ell, font=font) > max_width:
            t = t[:-1]
        lines[-1] = (t + ell) if t else ell
    return lines

# =========================
# Services & Repositories
# =========================

class Paths:
    DATA_DIR = Path("data")
    ASSETS_DIR = Path("assets")
    SETTINGS_PATH = DATA_DIR / "settings.json"
    CATALOG_PATH = DATA_DIR / "catalog.json"

    @staticmethod
    def ensure():
        Paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Paths.ASSETS_DIR.mkdir(parents=True, exist_ok=True)

class SettingsService:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Settings:
        if not self.path.exists():
            s = Settings()
            self.save(s)
            return s
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            return Settings(**{
                **asdict(Settings()),
                **obj
            })
        except Exception:
            # fallback defaults
            return Settings()

    def save(self, s: Settings) -> None:
        self.path.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")

class CatalogRepository:
    def __init__(self, path: Path):
        self.path = path

    def load_all(self) -> Dict[str, SkuRecord]:
        if not self.path.exists():
            self.save_all({})
            return {}
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return {}
            return {sku: SkuRecord.from_dict({"SKU": sku, **data}) for sku, data in obj.items()}
        except Exception:
            return {}

    def save_all(self, catalog: Dict[str, SkuRecord | Dict]) -> None:
        # Accept dict of SkuRecord or plain dict -> convert to JSON schema
        out = {}
        for sku, val in catalog.items():
            if isinstance(val, SkuRecord):
                out[sku] = val.to_json_obj()
            else:
                out[sku] = SkuRecord.from_dict({"SKU": sku, **val}).to_json_obj()
        self.path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    def upsert(self, rec: SkuRecord) -> None:
        all_items = self.load_all()
        all_items[rec.SKU] = rec
        self.save_all(all_items)

    def to_dataframe(self) -> pd.DataFrame:
        d = self.load_all()
        rows = []
        for sku, rec in d.items():
            rows.append({
                "SKU": rec.SKU,
                "ASIN": rec.ASIN, "FNSKU": rec.FNSKU, "Brand": rec.Brand,
                "Title": rec.Title, "MRP": rec.MRP, "HSN": rec.HSN, "GST%": rec.GST,
                "MfgMonthYear": rec.MfgMonthYear, "Condition": rec.Condition,
                "Category": rec.Category
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["SKU","ASIN","FNSKU","Brand","Title","MRP","HSN","GST%","MfgMonthYear","Condition","Category"]
        )

class FontPack:
    def __init__(self, regular_path: str, bold_path: str):
        self.regular_path = regular_path
        self.bold_path = bold_path
        self.fallback_used = False

    def load(self, path: str, size: int):
        try:
            return ImageFont.truetype(path, int(size))
        except Exception:
            self.fallback_used = True
            return ImageFont.load_default()

# -------- Barcode Service (Code128B) --------

CODE128_PATTERNS = [
    '212222','222122','222221','121223','121322','131222','122213','122312','132212','221213',
    '221312','231212','112232','122132','122231','113222','123122','123221','223211','221132',
    '221231','213212','223112','312131','311222','321122','321221','312212','322112','322211',
    '212123','212321','232121','111323','131123','131321','112313','132113','132311','211313',
    '231113','231311','112133','112331','132131','113123','113321','133121','313121','211331',
    '231131','213113','213311','213131','311123','311321','331121','312113','312311','332111',
    '314111','221411','431111','111224','111422','121124','121421','141122','141221','112214',
    '112412','122114','122411','142112','142211','241211','221114','413111','241112','134111',
    '111242','121142','121241','114212','124112','124211','411212','421112','421211','212141',
    '214121','412121','111143','111341','131141','114113','114311','411113','411311','113141',
    '114131','311141','411131','211412','211214','211232','2331112'
]
START_B = 104
STOP = 106

class BarcodeService:
    def code128b_values(self, data: str) -> List[int]:
        values = [START_B]
        for ch in data:
            o = ord(ch)
            if not (32 <= o <= 126):
                raise ValueError("Code128B supports ASCII 32..126")
            values.append(o - 32)
        checksum = START_B
        for i, v in enumerate(values[1:], start=1):
            checksum += v * i
        checksum %= 103
        values.extend([checksum, STOP])
        return values

    def draw_code128(self, draw: ImageDraw.ImageDraw, xy: Tuple[int,int],
                     width_px: int, height_px: int, data: str, ink=0):
        patterns = [CODE128_PATTERNS[v] for v in self.code128b_values(data)]
        modules = sum(sum(int(c) for c in p) for p in patterns)
        px_per_module = max(1, width_px // modules)
        total_width = px_per_module * modules
        x, y = xy
        curx = x + (width_px - total_width)//2
        bar = True
        for p in patterns:
            for n in p:
                w = px_per_module * int(n)
                if bar:
                    draw.rectangle([curx, y, curx + w - 1, y + height_px - 1], fill=ink)
                curx += w
                bar = not bar

# -------- PDF Service --------

class PdfService:
    def images_to_pdf_bytes(self, images: List[Image.Image]) -> Optional[io.BytesIO]:
        if not images:
            return None
        buf = io.BytesIO()
        rgb0 = images[0].convert("RGB")
        rest = [im.convert("RGB") for im in images[1:]]
        rgb0.save(buf, format="PDF", save_all=True, append_images=rest)
        buf.seek(0)
        return buf

# =========================
# Label Renderers (OCP via interface)
# =========================

class ILabelRenderer(ABC):
    @abstractmethod
    def render(self, item: SkuRecord, settings: Settings, fonts: FontPack) -> Image.Image:
        ...

class FbaLabelRenderer(ILabelRenderer):
    def __init__(self, barcode: BarcodeService):
        self.barcode = barcode

    def render(self, item: SkuRecord, settings: Settings, fonts: FontPack) -> Image.Image:
        dpi = int(settings.dpi)
        W = mm_to_px(settings.fba_label_width_mm, dpi)
        H = mm_to_px(settings.fba_label_height_mm, dpi)
        margin = int(0.06 * W)
        base_title = int(H * 0.30)
        base_small = int(H * 0.18)
        title_px = int(base_title * settings.fba_title_scale)
        small_px = int(base_small * settings.fba_small_scale)
        bottom_pad = int(settings.fba_footer_pad_px)

        img = Image.new("L", (W, H), color=255)
        d = ImageDraw.Draw(img)
        f_title = fonts.load(settings.font_path_bold, title_px)
        f_small = fonts.load(settings.font_path_regular, small_px)

        # Title
        y = margin // 2
        line = wrap_text(d, item.Title, f_title, W - 2 * margin, max_lines=1)
        if line:
            d.text((margin, y), line[0], font=f_title, fill=0)
            y += f_title.size + 2

        # Barcode (FNSKU)
        fnsku = item.FNSKU
        bar_top = max(y, int(H * 0.30))
        bar_h = int(H * 0.40)
        bar_w = W - 2 * margin
        if fnsku:
            self.barcode.draw_code128(d, (margin, bar_top), bar_w, bar_h, fnsku, ink=0)
            hr_top = min(H - f_small.size - bottom_pad, bar_top + bar_h + 1)
            txt_w = d.textlength(fnsku, font=f_small)
            d.text(((W - txt_w) // 2, hr_top), fnsku, font=f_small, fill=0)
        else:
            d.rectangle([margin, bar_top, W - margin, bar_top + bar_h], outline=0, width=2)
            d.text((margin + 4, bar_top + bar_h // 2 - f_small.size // 2), "FNSKU MISSING", font=f_small, fill=0)

        # Footer
        cond = item.Condition or "New"
        sku = item.SKU
        footer_top = H - f_small.size - bottom_pad
        d.text((margin, footer_top), cond, font=f_small, fill=0)
        sku_w = d.textlength(sku, font=f_small)
        d.text((W - margin - sku_w, footer_top), sku, font=f_small, fill=0)
        return img

class MrpLabelRenderer(ILabelRenderer):
    """Enhanced MRP label design with clean layout and strong visual hierarchy."""
    def render(self, item: SkuRecord, settings: Settings, fonts: FontPack) -> Image.Image:
        dpi = int(settings.dpi)
        W = mm_to_px(settings.mrp_label_width_mm, dpi)
        H = mm_to_px(settings.mrp_label_height_mm, dpi)
        margin = int(W * 0.07)  # proportional margin for balanced look

        # Base font sizes scaled by settings
        base_brand = int(H * 0.22)
        base_title = int(H * 0.20)
        base_mrp = int(H * 0.42)
        base_small = int(H * 0.16)

        brand_px = int(base_brand * settings.mrp_brand_scale)
        title_px = int(base_title * settings.mrp_title_scale)
        mrp_px = int(base_mrp * settings.mrp_mrp_scale)
        small_px = int(base_small * settings.mrp_small_scale)
        bottom_pad = int(settings.mrp_bottom_pad_px)

        img = Image.new("L", (W, H), color=255)
        d = ImageDraw.Draw(img)

        # Fonts
        f_brand = fonts.load(settings.font_path_bold, brand_px)
        f_title = fonts.load(settings.font_path_regular, title_px)
        f_mrp = fonts.load(settings.font_path_bold, mrp_px)
        f_small = fonts.load(settings.font_path_regular, small_px)

        # Extract fields
        brand = (item.Brand or "").strip()
        title = (item.Title or "").strip()
        mrp = (item.MRP or "").strip()
        mfg = (item.MfgMonthYear or "").strip()
        hsn = (item.HSN or "").strip()
        gst = (item.GST or "").strip()

        # Start layout
        y = margin

        # Brand (uppercase for emphasis)
        if brand:
            d.text((margin, y), brand.upper(), font=f_brand, fill=0)
            y += f_brand.size + 4

        # Title (wrap to max lines)
        lines = wrap_text(d, title, f_title, W - 2 * margin, max_lines=settings.mrp_max_title_lines)
        for ln in lines:
            d.text((margin, y), ln, font=f_title, fill=0)
            y += f_title.size + 2

        # Price (centered, bold)
        y += 6
        mrp_text = f"MRP ₹{mrp}" if mrp else "MRP —"
        tw = d.textlength(mrp_text, font=f_mrp)
        d.text(((W - tw) // 2, y), mrp_text, font=f_mrp, fill=0)
        y += f_mrp.size + 2

        # Inclusive line
        if settings.mrp_show_inclusive_line:
            incl = "(Inclusive of all taxes)"
            ti = d.textlength(incl, font=f_small)
            d.text(((W - ti) // 2, y), incl, font=f_small, fill=0)
            y += f_small.size + 4

        # Hairline separator for neatness
        band_y = H - (f_small.size + bottom_pad + 4)
        if band_y > y + 4:
            d.line((margin, band_y, W - margin, band_y), fill=0, width=1)

        # Bottom info band
        left_text = f"Mfg: {mfg}" if mfg else ""
        right_parts = []
        if hsn:
            right_parts.append(f"HSN {hsn}")
        if gst:
            right_parts.append(f"GST {gst}%")
        right_text = "   ".join(right_parts)

        d.text((margin, band_y + 4), left_text, font=f_small, fill=0)
        rt_w = d.textlength(right_text, font=f_small)
        d.text((W - margin - rt_w, band_y + 4), right_text, font=f_small, fill=0)

        return img

# =========================
# Business Logic
# =========================

class ValidationService:
    REQUIRED_FBA = ["SKU", "Title", "FNSKU"]
    REQUIRED_MRP = ["SKU", "Title", "Brand", "MRP"]

    def missing_for_fba(self, rec: SkuRecord) -> List[str]:
        return [k for k in self.REQUIRED_FBA if not getattr(rec, k, "").strip()]

    def missing_for_mrp(self, rec: SkuRecord) -> List[str]:
        # MRP can be "0" for some use cases; treat empty only when truly missing
        miss = []
        if not (rec.SKU or "").strip():
            miss.append("SKU")
        if not (rec.Title or "").strip():
            miss.append("Title")
        if not (rec.Brand or "").strip():
            miss.append("Brand")
        if rec.MRP is None or str(rec.MRP).strip() == "":
            miss.append("MRP")
        return miss

class QueueManager:
    STATE_KEY = "queue_items"

    def __init__(self):
        if self.STATE_KEY not in st.session_state:
            st.session_state[self.STATE_KEY] = []  # [{SKU: str, Qty: int}]
    def get(self) -> List[Dict]:
        return st.session_state[self.STATE_KEY]
    def add(self, sku: str, qty: int):
        st.session_state[self.STATE_KEY].append({"SKU": sku, "Qty": int(qty)})
    def clear(self):
        st.session_state[self.STATE_KEY] = []
    
    def remove(self, sku: str):
        st.session_state[self.STATE_KEY] = [item for item in st.session_state[self.STATE_KEY] if item["SKU"] != sku]
    def update_qty(self, sku: str, qty: int):
        for item in st.session_state[self.STATE_KEY]:
            if item["SKU"] == sku:
                item["Qty"] = qty


# =========================
# Pages (UI)
# =========================

class BasePage(ABC):
    @abstractmethod
    def render(self): ...

class LabelGeneratorPage(BasePage):
    def __init__(
        self,
        settings_service: SettingsService,
        catalog_repo: CatalogRepository,
        validation: ValidationService,
        pdf_service: PdfService,
        fba_renderer: ILabelRenderer,
        mrp_renderer: ILabelRenderer,
    ):
        self.settings_service = settings_service
        self.catalog_repo = catalog_repo
        self.validation = validation
        self.pdf_service = pdf_service
        self.fba_renderer = fba_renderer
        self.mrp_renderer = mrp_renderer
        self.queue = QueueManager()

    def render(self):
        st.header("🧾 Label Generator")

        # ---------- Sidebar: Settings ----------
        with st.sidebar:
            st.header("⚙️ Settings")
            s = self.settings_service.load()
            s.dpi = st.number_input("DPI", 150, 1200, value=int(s.dpi), step=25)

            st.subheader("FBA Label (mm)")
            c1, c2 = st.columns(2)
            with c1:
                s.fba_label_width_mm = st.number_input("Width", 10.0, 200.0, value=float(s.fba_label_width_mm), step=1.0)
            with c2:
                s.fba_label_height_mm = st.number_input("Height", 10.0, 200.0, value=float(s.fba_label_height_mm), step=1.0)
            s.fba_title_scale = st.slider("Title scale", 0.2, 1.2, float(s.fba_title_scale), 0.05)
            s.fba_small_scale = st.slider("Small text scale", 0.2, 1.2, float(s.fba_small_scale), 0.05)
            s.fba_footer_pad_px = st.number_input("Footer bottom padding (px)", 0, 200, value=int(s.fba_footer_pad_px), step=2)

            st.subheader("MRP Label (mm)")
            c3, c4 = st.columns(2)
            with c3:
                s.mrp_label_width_mm = st.number_input("Width ", 10.0, 200.0, value=float(s.mrp_label_width_mm), step=1.0)
            with c4:
                s.mrp_label_height_mm = st.number_input("Height ", 10.0, 200.0, value=float(s.mrp_label_height_mm), step=1.0)
            s.mrp_brand_scale = st.slider("Brand scale", 0.2, 1.2, float(s.mrp_brand_scale), 0.05)
            s.mrp_title_scale = st.slider("Title scale ", 0.2, 1.2, float(s.mrp_title_scale), 0.05)
            s.mrp_mrp_scale   = st.slider("MRP scale   ", 0.2, 1.2, float(s.mrp_mrp_scale), 0.05)
            s.mrp_small_scale = st.slider("Small scale ", 0.2, 1.2, float(s.mrp_small_scale), 0.05)
            s.mrp_bottom_pad_px = st.number_input("Bottom padding (px)", 0, 200, value=int(s.mrp_bottom_pad_px), step=2)
            s.mrp_show_inclusive_line = st.checkbox("Show '(Inclusive of all taxes)'", value=bool(s.mrp_show_inclusive_line))
            s.mrp_max_title_lines = st.slider("MRP title max lines", 1, 3, int(s.mrp_max_title_lines), 1)

            st.subheader("Fonts")
            s.font_path_regular = st.text_input("Regular font path", s.font_path_regular)
            s.font_path_bold = st.text_input("Bold font path", s.font_path_bold)

            st.subheader("What to Print")
            colp, colq = st.columns(2)
            with colp:
                s.print_fba_labels = st.toggle("Print FBA", value=bool(s.print_fba_labels))
            with colq:
                s.print_mrp_labels = st.toggle("Print MRP", value=bool(s.print_mrp_labels))

            s.limit_preview = st.number_input("Preview up to N labels", 0, 100, value=int(s.limit_preview), step=1)

            if st.button("💾 Save Settings", use_container_width=True):  # ★
                self.settings_service.save(s)
                st.success("Settings saved.", icon="✅")
                st.cache_data.clear()

        # ---------- Build Queue ----------
        st.subheader("🧺 Print Queue")

        df = self.catalog_repo.to_dataframe()
        sku_options = df["SKU"].dropna().astype(str).unique().tolist()

        # Row: Select SKU | Qty | Add (aligned)
        add_c1, add_c2, add_c3 = st.columns([2, 1, 1], vertical_alignment="center")  # ★
        with add_c1:
            sku_pick = st.selectbox(
                "Select SKU",
                options=[""] + sku_options,
                index=0,
                label_visibility="collapsed",        # ★ remove label to align
                placeholder="Select SKU…"            # ★ hint instead of label
            )
        with add_c2:
            qty_pick = st.number_input(
                "Qty",
                min_value=1,
                value=1,
                step=1,
                label_visibility="collapsed"         # ★ remove label to align
            )
        with add_c3:
            if st.button("➕ Add", use_container_width=True):  # ★ full width
                if not sku_pick:
                    st.warning("Pick an SKU to add.", icon="ℹ️")
                else:
                    q_now = self.queue.get()
                    # If SKU already present, increase instead of duplicating
                    for item in q_now:
                        if item["SKU"] == sku_pick:
                            item["Qty"] = int(item["Qty"]) + int(qty_pick)
                            st.rerun()
                    self.queue.add(sku_pick, int(qty_pick))
                    st.rerun()

        # Queue table: SKU | Qty (editable) | Remove (aligned)
        q = self.queue.get()
        if q:
            st.caption("Tip: Edit quantities inline, or remove an SKU.")

            # Header row for visual clarity (no actual labels on inputs)
            hdr1, hdr2, hdr3 = st.columns([3, 2, 1], vertical_alignment="center")  # ★
            with hdr1: st.markdown("**SKU**")
            with hdr2: st.markdown("**Quantity**")
            with hdr3: st.markdown("**Action**")

            # Draw each row with centered vertical alignment
            # Using index snapshot; we rerun immediately after removal to keep indices consistent
            for idx, item in enumerate(list(q)):  # iterate snapshot
                c1, c2, c3 = st.columns([3, 2, 1], vertical_alignment="center")  # ★
                with c1:
                    # Plain text keeps vertical rhythm consistent vs st.write()
                    st.text(item["SKU"])
                with c2:
                    new_qty = st.number_input(
                        f"qty_edit_{idx}",
                        min_value=1,
                        value=int(item["Qty"]),
                        step=1,
                        label_visibility="collapsed"          # ★ no label, aligns vertically
                    )
                    if new_qty != item["Qty"]:
                        item["Qty"] = int(new_qty)
                with c3:
                    if st.button("Remove", key=f"remove_{idx}", use_container_width=True):  # ★ full width
                        q.pop(idx)
                        st.rerun()

            # Clear all (aligned and full-width)
            if st.button("🧹 Clear Queue", type="secondary", use_container_width=True):  # ★
                self.queue.clear()
                st.rerun()

        # ---------- Generate & Preview (unique per SKU) ----------
        st.subheader("🖨️ Preview & Download")
        if st.button("Generate Labels", type="primary", use_container_width=True):  # ★
            if not q:
                st.error("Queue is empty. Add at least one SKU.", icon="⚠️")
                st.stop()

            s = self.settings_service.load()
            all_items = self.catalog_repo.load_all()

            # Attach catalog info
            selection: List[Tuple[SkuRecord, int]] = []
            for row in q:
                sku = row["SKU"]
                qty = int(row["Qty"])
                rec = all_items.get(sku)
                if not rec:
                    rec = SkuRecord(SKU=sku)  # triggers validation form
                selection.append((rec, qty))

            # Validate and prompt if missing
            missing_any = False
            need_fba = s.print_fba_labels
            need_mrp = s.print_mrp_labels

            for rec, _qty in selection:
                miss = []
                if need_fba: miss += self.validation.missing_for_fba(rec)
                if need_mrp: miss += self.validation.missing_for_mrp(rec)
                miss = sorted(set(miss))
                if miss:
                    missing_any = True
                    with st.form(f"fix_{rec.SKU or 'UNKNOWN'}"):
                        st.warning(f"**{rec.SKU or '(no SKU)'}** is missing: {', '.join(miss)}", icon="⚠️")
                        cols = st.columns(3)
                        updates = {}
                        fields = ["SKU","Title","FNSKU","Brand","MRP","HSN","GST%","MfgMonthYear","Condition","ASIN","Category"]
                        for i, field in enumerate([f for f in fields if f.split('%')[0] in miss or f in miss]):
                            with cols[i % 3]:
                                key = f"{rec.SKU}_{field}"
                                placeholder = getattr(rec, field.replace("GST%", "GST"), "")
                                updates[field] = st.text_input(field, value=str(placeholder), key=key)
                        if st.form_submit_button("Save"):
                            new = SkuRecord.from_dict({
                                "SKU": updates.get("SKU", rec.SKU) or rec.SKU,
                                "ASIN": updates.get("ASIN", rec.ASIN) or rec.ASIN,
                                "FNSKU": updates.get("FNSKU", rec.FNSKU) or rec.FNSKU,
                                "Brand": updates.get("Brand", rec.Brand) or rec.Brand,
                                "Title": updates.get("Title", rec.Title) or rec.Title,
                                "MRP": updates.get("MRP", rec.MRP) or rec.MRP,
                                "HSN": updates.get("HSN", rec.HSN) or rec.HSN,
                                "GST%": updates.get("GST%", rec.GST) or rec.GST,
                                "MfgMonthYear": updates.get("MfgMonthYear", rec.MfgMonthYear) or rec.MfgMonthYear,
                                "Condition": updates.get("Condition", rec.Condition) or rec.Condition,
                                "Category": updates.get("Category", rec.Category) or rec.Category
                            })
                            self.catalog_repo.upsert(new)
                            st.success(f"Updated {new.SKU}. Click **Generate Labels** again.", icon="✅")
                            st.stop()
            if missing_any:
                st.info("Fill the missing fields above, then click **Generate Labels** again.", icon="ℹ️")
                st.stop()

            # Unique previews per SKU
            unique_by_sku: Dict[str, SkuRecord] = {rec.SKU: rec for rec, _q in selection}
            unique_items = list(unique_by_sku.values())

            fonts = FontPack(s.font_path_regular, s.font_path_bold)
            fba_preview_imgs, mrp_preview_imgs = [], []

            if s.print_fba_labels:
                for rec in unique_items:
                    fba_preview_imgs.append(self.fba_renderer.render(rec, s, fonts))
            if s.print_mrp_labels:
                for rec in unique_items:
                    mrp_preview_imgs.append(self.mrp_renderer.render(rec, s, fonts))

            # Show previews (unique only)
            show_n = s.limit_preview
            if show_n and (fba_preview_imgs or mrp_preview_imgs):
                st.markdown("**Preview (unique labels per SKU)**")
                tabs_prev = [t for t in (["FBA Preview"] if fba_preview_imgs else []) + (["MRP Preview"] if mrp_preview_imgs else [])] or ["Preview"]
                prev_tabs = st.tabs(tabs_prev)
                tab_idx = 0
                if fba_preview_imgs:
                    with prev_tabs[tab_idx]:
                        cols = st.columns(6)
                        for i, im in enumerate(fba_preview_imgs[:show_n]):
                            with cols[i % 6]:
                                st.image(im, caption="FBA", use_container_width=True, clamp=True)
                    tab_idx += 1
                if mrp_preview_imgs:
                    with prev_tabs[tab_idx]:
                        cols = st.columns(6)
                        for i, im in enumerate(mrp_preview_imgs[:show_n]):
                            with cols[i % 6]:
                                st.image(im, caption="MRP", use_container_width=True, clamp=True)

            # Build full print set by quantity (for PDF)
            expanded_fba, expanded_mrp = [], []
            if s.print_fba_labels:
                for rec, qty in selection:
                    for _ in range(qty):
                        expanded_fba.append(self.fba_renderer.render(rec, s, fonts))
            if s.print_mrp_labels:
                for rec, qty in selection:
                    for _ in range(qty):
                        expanded_mrp.append(self.mrp_renderer.render(rec, s, fonts))

            pdf = PdfService()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if s.print_fba_labels and expanded_fba:
                buf = pdf.images_to_pdf_bytes(expanded_fba)
                st.download_button("⬇️ Download FBA PDF", data=buf, file_name=f"fba_labels_{ts}.pdf", mime="application/pdf", use_container_width=True)
            if s.print_mrp_labels and expanded_mrp:
                buf = pdf.images_to_pdf_bytes(expanded_mrp)
                st.download_button("⬇️ Download MRP PDF", data=buf, file_name=f"mrp_labels_{ts}.pdf", mime="application/pdf", use_container_width=True)

            if fonts.fallback_used:
                st.warning("Fallback bitmap font used. Set valid .ttf files in **Settings → Fonts** for crisp print.", icon="🅰️")

class CatalogPage(BasePage):
    def __init__(self, catalog_repo: CatalogRepository):
        self.catalog_repo = catalog_repo

    @st.cache_data(show_spinner=False)
    def _load_df_cached(_self) -> pd.DataFrame:
        return _self.catalog_repo.to_dataframe()

    def _clear_cache(self):
        try:
            st.cache_data.clear()
        except Exception:
            pass

    def _search_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["SKU","Title","Brand","Category","ASIN"]:
            if col in df.columns:
                df[col] = df[col].fillna("")
        q = st.text_input("🔎 Search SKU / Title / ASIN", placeholder="Type to search…")
        c1, c2 = st.columns(2)
        with c1:
            brands = sorted([b for b in df["Brand"].dropna().unique().tolist() if str(b).strip()])
            brand_sel = st.multiselect("Brand", brands, placeholder="All")
        with c2:
            cats = sorted([c for c in df["Category"].dropna().unique().tolist() if str(c).strip()])
            cat_sel = st.multiselect("Category", cats, placeholder="All")

        mask = pd.Series(True, index=df.index)
        if q:
            ql = q.lower()
            mask &= (
                df["SKU"].str.lower().str.contains(ql) |
                df["Title"].str.lower().str.contains(ql) |
                df["ASIN"].str.lower().str.contains(ql)
            )
        if brand_sel:
            mask &= df["Brand"].isin(brand_sel)
        if cat_sel:
            mask &= df["Category"].isin(cat_sel)
        return df[mask]

    def render(self):
        st.header("📚 Catalog Management")
        df = self._load_df_cached()

        filtered = self._search_filter(df)
        st.dataframe(filtered.reset_index(drop=True), width='stretch', height=320)

        with st.expander("➕ Add or edit SKU"):
            with st.form("add_edit_form", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    sku = st.text_input("SKU*", placeholder="SKU123")
                    asin = st.text_input("ASIN")
                    fnsku = st.text_input("FNSKU")
                with c2:
                    brand = st.text_input("Brand")
                    title = st.text_input("Title*")
                    mrp = st.text_input("MRP", placeholder="e.g., 499")
                with c3:
                    hsn = st.text_input("HSN")
                    gst = st.text_input("GST%", placeholder="e.g., 18")
                    cond = st.text_input("Condition", value="New")

                d1, d2 = st.columns(2)
                with d1:
                    mfg = st.text_input("MfgMonthYear", placeholder="Sep-2025")
                with d2:
                    category = st.text_input("Category", placeholder="e.g., Cables")

                if st.form_submit_button("Save to Catalog"):
                    if not sku.strip() or not title.strip():
                        st.error("SKU and Title are required.", icon="⚠️")
                    else:
                        rec = SkuRecord.from_dict({
                            "SKU": sku, "ASIN": asin, "FNSKU": fnsku, "Brand": brand,
                            "Title": title, "MRP": mrp, "HSN": hsn, "GST%": gst,
                            "MfgMonthYear": mfg, "Condition": cond, "Category": category
                        })
                        self.catalog_repo.upsert(rec)
                        st.success(f"Saved SKU '{rec.SKU}'.", icon="✅")
                        self._clear_cache()

        st.subheader("🔁 Import / Export")
        c1, c2 = st.columns(2)
        with c1:
            upl = st.file_uploader("Import catalog.json", type=["json"])
            if upl:
                try:
                    data = json.loads(upl.read().decode("utf-8"))
                    if isinstance(data, dict):
                        # validate minimally
                        cleaned = {}
                        for sku, v in data.items():
                            cleaned[str(sku)] = SkuRecord.from_dict({"SKU": sku, **(v or {})}).to_json_obj()
                        self.catalog_repo.save_all({sku: SkuRecord.from_dict({"SKU": sku, **v}) for sku, v in cleaned.items()})
                        st.success("Catalog imported.", icon="✅")
                        self._clear_cache()
                    else:
                        st.error("Invalid JSON format. Expected an object keyed by SKU.", icon="⚠️")
                except Exception as e:
                    st.error(f"Failed to import: {e}", icon="⚠️")
        with c2:
            full = self.catalog_repo.load_all()
            wire = {sku: rec.to_json_obj() for sku, rec in full.items()}
            st.download_button("Export catalog.json", data=json.dumps(wire, indent=2), file_name="catalog.json")

# =========================
# Application Root
# =========================

class App:
    def __init__(self):
        st.set_page_config(page_title="LabelForge — SKU Label Maker", page_icon="🧾", layout="wide")
        self._apply_responsive_css()

        Paths.ensure()
        self.settings_service = SettingsService(Paths.SETTINGS_PATH)
        self.catalog_repo = CatalogRepository(Paths.CATALOG_PATH)
        self.validation = ValidationService()
        self.pdf_service = PdfService()
        self.barcode = BarcodeService()

        # Renderers (DIP: depend on abstractions)
        self.fba_renderer: ILabelRenderer = FbaLabelRenderer(self.barcode)
        self.mrp_renderer: ILabelRenderer = MrpLabelRenderer()

        # Pages
        self.page_generator = LabelGeneratorPage(
            self.settings_service, self.catalog_repo, self.validation, self.pdf_service,
            self.fba_renderer, self.mrp_renderer
        )
        self.page_catalog = CatalogPage(self.catalog_repo)

    def _apply_responsive_css(self):
        st.markdown("""
        <style>
        button, .stButton>button { padding: 0.6rem 1rem; font-size: 1rem; }
        .stNumberInput input, .stSelectbox select, .stTextInput input { height: 2.6rem; font-size: 1rem; }
        .stDataFrame, .stDataEditor { font-size: 0.95rem; }
        @media (max-width: 640px) { .stColumns { display: block; } }
        </style>
        """, unsafe_allow_html=True)


    def run(self):
        st.title("🧾 LabelForge — Streamlit SKU Label Maker (FBA + MRP)")
        tabs = st.tabs(["🏠 Label Generator", "📚 Catalog"])
        with tabs[0]:
            self.page_generator.render()
        with tabs[1]:
            self.page_catalog.render()

# =========================
# Entry
# =========================

if __name__ == "__main__":
    App().run()