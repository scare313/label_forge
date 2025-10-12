import json, io, os, re
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ------------------ Constants & Paths ------------------
APP_TITLE = "SKU Label Maker (FBA + MRP)"
DATA_DIR = Path("data")
ASSETS_DIR = Path("assets")
CATALOG_PATH = DATA_DIR / "catalog.json"
SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "dpi": 300,
    "print_fba_labels": True,
    "print_mrp_labels": True,
    # FBA label (mm)
    "fba_label_width_mm": 50,
    "fba_label_height_mm": 25,
    "fba_title_scale": 0.30,
    "fba_small_scale": 0.50,
    "fba_footer_pad_px": 30,
    # MRP label (mm)
    "mrp_label_width_mm": 40,
    "mrp_label_height_mm": 30,
    "mrp_brand_scale": 0.50,
    "mrp_title_scale": 0.50,
    "mrp_mrp_scale": 0.60,
    "mrp_small_scale": 0.50,
    "mrp_bottom_pad_px": 30,
    "mrp_show_inclusive_line": True,
    "mrp_max_title_lines": 2,
    # Fonts (override in UI)
    "font_path_regular": str((ASSETS_DIR / "DejaVuSans.ttf").as_posix()),
    "font_path_bold": str((ASSETS_DIR / "DejaVuSans-Bold.ttf").as_posix()),
    # UX
    "limit_preview": 12,           # show first N labels as preview to keep UI fast
    "zpl_enable": False
}

REQUIRED_FOR_FBA = ["SKU", "Title", "FNSKU"]
REQUIRED_FOR_MRP = ["SKU", "Title", "Brand", "MRP"]

# ------------------ Utilities ------------------
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path: Path, default):
    if not path.exists():
        path.write_text(json.dumps(default, indent=2), encoding="utf-8")
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def save_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

@st.cache_data(show_spinner=False)
def load_catalog_df() -> pd.DataFrame:
    catalog = load_json(CATALOG_PATH, default={})
    # Normalize to DataFrame
    rows = []
    for sku, info in catalog.items():
        row = {"SKU": sku}
        row.update(info or {})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["SKU","ASIN","FNSKU","Brand","Title","MRP","HSN","GST%","MfgMonthYear","Condition","Category"])
    return pd.DataFrame(rows)

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))

class FontPack:
    def __init__(self, regular_path, bold_path):
        self.regular_path = regular_path
        self.bold_path = bold_path
        self.fallback_used = False

    def load(self, path, size):
        try:
            return ImageFont.truetype(path, int(size))
        except Exception:
            self.fallback_used = True
            return ImageFont.load_default()

# ---- Code128B (same logic as your script) ----
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

def code128b_values(data: str):
    values = [START_B]
    for ch in data:
        o = ord(ch)
        if not (32 <= o <= 126):
            raise ValueError('Code128B supports ASCII 32..126')
        values.append(o - 32)
    checksum = START_B
    for i, v in enumerate(values[1:], start=1):
        checksum += v * i
    checksum %= 103
    values.append(checksum)
    values.append(STOP)
    return values

def code128_draw(draw: ImageDraw.ImageDraw, xy, width_px: int, height_px: int, data: str, ink=0):
    patterns = [CODE128_PATTERNS[v] for v in code128b_values(data)]
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

def wrap_text(draw, text: str, font, max_width: int, max_lines: int = 2):
    words = (text or '').split()
    lines, cur = [], ''
    for w in words:
        test = (cur + ' ' + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur); cur = w
            else:
                # hyphenate long chunk
                while draw.textlength(w, font=font) > max_width and len(w) > 1:
                    w = w[:-1]
                lines.append(w); cur = ''
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines and ' '.join(words) != ' '.join(lines):
        t = lines[-1]; ell = '…'
        while t and draw.textlength(t + ell, font=font) > max_width:
            t = t[:-1]
        lines[-1] = (t + ell) if t else ell
    return lines

# ------------- Renderers (adapted from your v2.2) -------------
def render_fba_label(item, cfg, fonts: FontPack):
    dpi = int(cfg["dpi"])
    W = mm_to_px(float(cfg['fba_label_width_mm']), dpi)
    H = mm_to_px(float(cfg['fba_label_height_mm']), dpi)
    margin = int(0.06 * W)

    base_title = int(H*0.30)
    base_small = int(H*0.18)
    title_px = int(base_title * float(cfg.get('fba_title_scale', 0.30)))
    small_px = int(base_small * float(cfg.get('fba_small_scale', 0.50)))
    bottom_pad = int(float(cfg.get('fba_footer_pad_px', 30)))

    img = Image.new('L', (W, H), color=255)
    d = ImageDraw.Draw(img)
    f_title = fonts.load(cfg["font_path_bold"], title_px)
    f_small = fonts.load(cfg["font_path_regular"], small_px)

    # Title (1 line)
    y = margin//2
    title = str(item.get('Title') or '')
    line = wrap_text(d, title, f_title, W - 2*margin, max_lines=1)
    if line:
        d.text((margin, y), line[0], font=f_title, fill=0)
        y += f_title.size + 2

    # Barcode
    fnsku = str(item.get('FNSKU') or '')
    bar_top = max(y, int(H*0.30))
    bar_height = int(H*0.40)
    bar_width = W - 2*margin
    if fnsku:
        code128_draw(d, (margin, bar_top), bar_width, bar_height, fnsku, ink=0)
        hr_top = min(H - f_small.size - bottom_pad, bar_top + bar_height + 1)
        txt_w = d.textlength(fnsku, font=f_small)
        d.text(((W - txt_w)//2, hr_top), fnsku, font=f_small, fill=0)
    else:
        d.rectangle([margin, bar_top, W-margin, bar_top+bar_height], outline=0, width=2)
        d.text((margin+4, bar_top + bar_height//2 - f_small.size//2), 'FNSKU MISSING', font=f_small, fill=0)

    # Footer
    cond = str(item.get('Condition') or 'New')
    sku = str(item.get('SKU') or '')
    footer_top = H - f_small.size - bottom_pad
    d.text((margin, footer_top), cond, font=f_small, fill=0)
    sku_w = d.textlength(sku, font=f_small)
    d.text((W - margin - sku_w, footer_top), sku, font=f_small, fill=0)
    return img

def render_mrp_label(item, cfg, fonts: FontPack):
    dpi = int(cfg["dpi"])
    W = mm_to_px(float(cfg['mrp_label_width_mm']), dpi)
    H = mm_to_px(float(cfg['mrp_label_height_mm']), dpi)
    margin = int(0.07 * W)

    base_brand = int(H*0.22)
    base_title = int(H*0.20)
    base_mrp   = int(H*0.40)
    base_small = int(H*0.16)
    brand_px = int(base_brand * float(cfg.get('mrp_brand_scale', 0.50)))
    title_px = int(base_title * float(cfg.get('mrp_title_scale', 0.50)))
    mrp_px   = int(base_mrp   * float(cfg.get('mrp_mrp_scale', 0.60)))
    small_px = int(base_small * float(cfg.get('mrp_small_scale', 0.50)))
    bottom_pad = int(float(cfg.get('mrp_bottom_pad_px', 30)))

    img = Image.new('L', (W, H), color=255)
    d = ImageDraw.Draw(img)

    f_brand = fonts.load(cfg["font_path_bold"], brand_px)
    f_title = fonts.load(cfg["font_path_regular"], title_px)
    f_mrp   = fonts.load(cfg["font_path_bold"], mrp_px)
    f_small = fonts.load(cfg["font_path_regular"], small_px)

    brand = str(item.get('Brand') or '').strip()
    title = str(item.get('Title') or '').strip()
    mrp = str(item.get('MRP') or '').strip()
    mfg = str(item.get('MfgMonthYear') or '').strip()
    hsn = str(item.get('HSN') or '').strip()
    gst = str(item.get('GST%') or '').strip()

    y = margin//2
    # Header: Brand + Title
    if brand:
        d.text((margin, y), brand, font=f_brand, fill=0)
        y += f_brand.size + 1
    lines = wrap_text(d, title, f_title, W - 2*margin, max_lines=int(cfg.get('mrp_max_title_lines', 2)))
    for ln in lines:
        d.text((margin, y), ln, font=f_title, fill=0)
        y += f_title.size
    y += 1

    # MRP
    mrp_text = f"MRP ₹{mrp}" if mrp else "MRP —"
    tw = d.textlength(mrp_text, font=f_mrp)
    d.text(((W - tw)//2, y), mrp_text, font=f_mrp, fill=0)
    y += f_mrp.size

    # Inclusive line
    if bool(cfg.get('mrp_show_inclusive_line', True)):
        incl = "(Inclusive of all taxes)"
        tw2 = d.textlength(incl, font=f_small)
        d.text(((W - tw2)//2, y - 2), incl, font=f_small, fill=0)
        y += f_small.size + 2

    # Bottom band
    band_y = H - (f_small.size + bottom_pad + 2)
    if band_y > y + 2:
        d.line((margin, band_y, W - margin, band_y), fill=0, width=1)

    left = []
    if mfg: left.append(f"Mfg: {mfg}")
    right = []
    if hsn: right.append(f"HSN {hsn}")
    if gst: right.append(f"GST {gst}%")

    d.text((margin, band_y + 2), '  '.join(left), font=f_small, fill=0)
    rt_text = '   '.join(right)
    rt_w = d.textlength(rt_text, font=f_small)
    d.text((W - margin - rt_w, band_y + 2), rt_text, font=f_small, fill=0)
    return img

# ------------------ Validation & Expansion ------------------
def validate_item_for_fba(item):
    missing = [k for k in REQUIRED_FOR_FBA if not str(item.get(k,"")).strip()]
    return missing

def validate_item_for_mrp(item):
    missing = [k for k in REQUIRED_FOR_MRP if not str(item.get(k,"")).strip()]
    return missing

def expand_rows(selection: list[dict]):
    """Expand items by quantity to build a print queue."""
    expanded = []
    for it in selection:
        qty = int(it.get("Qty", 0) or 0)
        if qty <= 0: continue
        for _ in range(qty):
            expanded.append({k:v for k,v in it.items() if k != "Qty"})
    return expanded

def images_to_pdf_bytes(images):
    if not images:
        return None
    buf = io.BytesIO()
    pil0 = images[0].convert("RGB")
    rest = [im.convert("RGB") for im in images[1:]]
    pil0.save(buf, format="PDF", save_all=True, append_images=rest)
    buf.seek(0)
    return buf

# ------------------ UI Helpers ------------------
def apply_responsive_css():
    st.markdown("""
        <style>
        /* Mobile-friendly spacings and tap targets */
        button, .stButton>button {
            padding: 0.6rem 1rem;
            font-size: 1rem;
        }
        .stNumberInput input, .stSelectbox select, .stTextInput input {
            height: 2.6rem;
            font-size: 1rem;
        }
        .stDataFrame, .stDataEditor { font-size: 0.95rem; }
        .tight { margin-top: -0.5rem; }
        .pill { border-radius: 999px; padding: 0.25rem 0.75rem; background: #eef2ff; }
        @media (max-width: 640px) {
            .stColumns { display: block; }
        }
        </style>
    """, unsafe_allow_html=True)

def sku_search_filter(df: pd.DataFrame):
    df = df.copy()
    # Fill NaNs
    for col in ["SKU","Title","Brand","Category"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    q = st.text_input("🔎 Search by SKU / Title / ASIN", placeholder="Type to search…", key="q")
    col1, col2 = st.columns(2)
    brands = sorted([b for b in df["Brand"].dropna().unique().tolist() if str(b).strip()])
    cats   = sorted([c for c in df["Category"].dropna().unique().tolist() if str(c).strip()])
    with col1:
        brand_sel = st.multiselect("Brand", brands, placeholder="All")
    with col2:
        cat_sel = st.multiselect("Category", cats, placeholder="All")

    mask = pd.Series(True, index=df.index)
    if q:
        ql = q.lower()
        mask &= (
            df["SKU"].str.lower().str.contains(ql) |
            df.get("Title","").str.lower().str.contains(ql) |
            df.get("ASIN","").str.lower().str.contains(ql)
        )
    if brand_sel:
        mask &= df["Brand"].isin(brand_sel)
    if cat_sel:
        mask &= df["Category"].isin(cat_sel)
    return df[mask]

def catalog_dict_from_df(df: pd.DataFrame) -> dict:
    out = {}
    for _, r in df.iterrows():
        sku = str(r.get("SKU","")).strip()
        if not sku: continue
        info = {
            "ASIN": r.get("ASIN",""),
            "FNSKU": r.get("FNSKU",""),
            "Brand": r.get("Brand",""),
            "Title": r.get("Title",""),
            "MRP": str(r.get("MRP","") or ""),
            "HSN": str(r.get("HSN","") or ""),
            "GST%": str(r.get("GST%","") or ""),
            "MfgMonthYear": str(r.get("MfgMonthYear","") or ""),
            "Condition": r.get("Condition","New") or "New",
            "Category": r.get("Category",""),
        }
        out[sku] = info
    return out

def upsert_catalog_row(df: pd.DataFrame, row: dict):
    df = df.copy()
    sku = row.get("SKU","").strip()
    if not sku: return df
    # Update or append
    exists = (df["SKU"] == sku) if "SKU" in df.columns else pd.Series([False]*len(df))
    if exists.any():
        idx = df[exists].index[0]
        for k,v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

# ------------------ Main App ------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")
    apply_responsive_css()
    ensure_dirs()

    st.title("🧾 SKU Label Maker")
    st.caption("Generate **FBA (ASIN)** and **MRP** labels with search, settings, and bulk printing.")

    # ---------- Sidebar: Settings ----------
    with st.sidebar:
        st.header("⚙️ Settings")
        settings = load_json(SETTINGS_PATH, DEFAULT_SETTINGS)
        # General
        settings["dpi"] = st.number_input("DPI", min_value=150, max_value=1200, value=int(settings["dpi"]), step=25)
        st.subheader("FBA Label (mm)")
        colA, colB = st.columns(2)
        with colA:
            settings["fba_label_width_mm"] = st.number_input("Width", min_value=10.0, value=float(settings["fba_label_width_mm"]), step=1.0)
        with colB:
            settings["fba_label_height_mm"] = st.number_input("Height", min_value=10.0, value=float(settings["fba_label_height_mm"]), step=1.0)
        settings["fba_title_scale"] = st.slider("Title scale", 0.2, 1.2, float(settings["fba_title_scale"]), 0.05)
        settings["fba_small_scale"] = st.slider("Small text scale", 0.2, 1.2, float(settings["fba_small_scale"]), 0.05)
        settings["fba_footer_pad_px"] = st.number_input("Footer bottom padding (px)", min_value=0, value=int(settings["fba_footer_pad_px"]), step=2)

        st.subheader("MRP Label (mm)")
        colC, colD = st.columns(2)
        with colC:
            settings["mrp_label_width_mm"] = st.number_input("Width ", min_value=10.0, value=float(settings["mrp_label_width_mm"]), step=1.0)
        with colD:
            settings["mrp_label_height_mm"] = st.number_input("Height ", min_value=10.0, value=float(settings["mrp_label_height_mm"]), step=1.0)
        settings["mrp_brand_scale"] = st.slider("Brand scale", 0.2, 1.2, float(settings["mrp_brand_scale"]), 0.05)
        settings["mrp_title_scale"] = st.slider("Title scale ", 0.2, 1.2, float(settings["mrp_title_scale"]), 0.05)
        settings["mrp_mrp_scale"]   = st.slider("MRP scale   ", 0.2, 1.2, float(settings["mrp_mrp_scale"]), 0.05)
        settings["mrp_small_scale"] = st.slider("Small scale ", 0.2, 1.2, float(settings["mrp_small_scale"]), 0.05)
        settings["mrp_bottom_pad_px"] = st.number_input("Bottom padding (px)", min_value=0, value=int(settings["mrp_bottom_pad_px"]), step=2)
        settings["mrp_show_inclusive_line"] = st.checkbox("Show '(Inclusive of all taxes)' line", value=bool(settings["mrp_show_inclusive_line"]))
        settings["mrp_max_title_lines"] = st.slider("MRP title max lines", 1, 3, int(settings["mrp_max_title_lines"]), 1)

        st.subheader("Fonts")
        settings["font_path_regular"] = st.text_input("Regular font path", value=settings["font_path_regular"])
        settings["font_path_bold"]    = st.text_input("Bold font path", value=settings["font_path_bold"])

        st.subheader("What to Print")
        colP, colQ = st.columns(2)
        with colP:
            settings["print_fba_labels"] = st.toggle("Print FBA", value=bool(settings["print_fba_labels"]))
        with colQ:
            settings["print_mrp_labels"] = st.toggle("Print MRP", value=bool(settings["print_mrp_labels"]))

        settings["limit_preview"] = st.number_input("Preview up to N labels", min_value=0, max_value=100, value=int(settings["limit_preview"]), step=1)

        if st.button("💾 Save Settings", use_container_width=True):
            save_json(SETTINGS_PATH, settings)
            st.success("Settings saved.", icon="✅")
            st.cache_data.clear()

    # ---------- Catalog: Load and Search ----------
    df = load_catalog_df()
    st.subheader("📚 Catalog")
    st.caption("Search & filter by SKU, title, brand, or category. Edit inline and **Save**.")
    filtered = sku_search_filter(df)
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=320)

    # ---------- Add / Edit SKU ----------
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
            c4, c5 = st.columns(2)
            with c4:
                mfg = st.text_input("MfgMonthYear", placeholder="Sep-2025")
            with c5:
                category = st.text_input("Category", placeholder="e.g., Cables")

            submitted = st.form_submit_button("Save to Catalog")
            if submitted:
                row = {
                    "SKU": sku.strip(),
                    "ASIN": asin.strip(),
                    "FNSKU": fnsku.strip(),
                    "Brand": brand.strip(),
                    "Title": title.strip(),
                    "MRP": re.sub(r"[^\d.]", "", mrp.strip()) if mrp else "",
                    "HSN": hsn.strip(),
                    "GST%": gst.strip(),
                    "MfgMonthYear": mfg.strip(),
                    "Condition": cond.strip() or "New",
                    "Category": category.strip(),
                }
                if not row["SKU"] or not row["Title"]:
                    st.error("SKU and Title are required.", icon="⚠️")
                else:
                    df2 = upsert_catalog_row(df, row)
                    # persist
                    save_json(CATALOG_PATH, catalog_dict_from_df(df2))
                    st.success(f"Saved SKU '{row['SKU']}'.", icon="✅")
                    st.cache_data.clear()

    # ---------- Build Print List ----------
    st.subheader("🧺 Print Queue")
    st.caption("Add SKUs and quantities. Missing data will be requested before printing.")
    # assemble options list
    sku_options = df["SKU"].dropna().astype(str).unique().tolist()
    if "queue" not in st.session_state:
        st.session_state.queue = []

    add_col1, add_col2, add_col3 = st.columns([2,2,1])
    with add_col1:
        sku_pick = st.selectbox("Select SKU", options=[""] + sku_options, index=0)
    with add_col2:
        qty_pick = st.number_input("Qty", min_value=1, value=1, step=1, key="qty_pick")
    with add_col3:
        if st.button("➕ Add"):
            if not sku_pick:
                st.warning("Pick an SKU to add.", icon="ℹ️")
            else:
                st.session_state.queue.append({"SKU": sku_pick, "Qty": int(qty_pick)})

    # Show queue with editable quantities and remove buttons
    if st.session_state.queue:
        qdf = pd.DataFrame(st.session_state.queue)
        st.dataframe(qdf, use_container_width=True)
        if st.button("🧹 Clear Queue", type="secondary"):
            st.session_state.queue = []

    # ---------- Generate Labels ----------
    st.subheader("🖨️ Preview & Download")
    generate = st.button("Generate Labels", type="primary", use_container_width=True)
    if generate:
        if not st.session_state.queue:
            st.error("Queue is empty. Add at least one SKU.", icon="⚠️")
        else:
            # Build selection by merging with catalog
            cat = load_json(CATALOG_PATH, default={})
            selection = []
            missing_sku_records = []
            for row in st.session_state.queue:
                sku = row["SKU"]
                qty = row["Qty"]
                info = cat.get(sku, {})
                rec = {"SKU": sku, "Qty": qty}
                rec.update(info)
                # inject defaults to avoid KeyErrors
                rec.setdefault("Title","")
                rec.setdefault("Brand","")
                rec.setdefault("FNSKU","")
                rec.setdefault("MRP","")
                rec.setdefault("Condition","New")
                selection.append(rec)

                # Validate required fields upfront
                need_fba = settings["print_fba_labels"]
                need_mrp = settings["print_mrp_labels"]
                miss = []
                if need_fba:
                    miss += validate_item_for_fba(rec)
                if need_mrp:
                    miss += validate_item_for_mrp(rec)
                miss = sorted(set(miss))
                if miss:
                    missing_sku_records.append((sku, miss))

            if missing_sku_records:
                st.error("Some SKUs are missing required fields. Please fill them below.", icon="⚠️")
                # Inline editors for missing fields
                for sku, miss in missing_sku_records:
                    st.markdown(f"**{sku}** missing: " + ", ".join(miss))
                    with st.form(f"fix_{sku}"):
                        cols = st.columns(3)
                        updates = {}
                        for i, field in enumerate(miss):
                            with cols[i % 3]:
                                updates[field] = st.text_input(field, key=f"{sku}_{field}")
                        if st.form_submit_button("Save"):
                            # Persist update
                            df_now = load_catalog_df()
                            # seed existing row values so we don't wipe other fields
                            base = df_now[df_now["SKU"] == sku].to_dict("records")
                            base0 = base[0] if base else {"SKU": sku}
                            base0.update({k: v for k,v in updates.items() if v is not None})
                            df_new = upsert_catalog_row(df_now, base0)
                            save_json(CATALOG_PATH, catalog_dict_from_df(df_new))
                            st.success(f"Updated {sku}. Re-run 'Generate Labels'.", icon="✅")
                            st.stop()
            else:
                # No missing -> render
                fonts = FontPack(settings["font_path_regular"], settings["font_path_bold"])
                expanded = expand_rows(selection)

                fba_imgs, mrp_imgs = [], []
                if settings["print_fba_labels"]:
                    for item in expanded:
                        fba_imgs.append(render_fba_label(item, settings, fonts))
                if settings["print_mrp_labels"]:
                    for item in expanded:
                        mrp_imgs.append(render_mrp_label(item, settings, fonts))

                # Previews
                show_n = int(settings.get("limit_preview", 12))
                if show_n and (fba_imgs or mrp_imgs):
                    st.write("**Preview** (first {} labels)".format(show_n))
                    cols = st.columns(6)
                    i = 0
                    for im in (fba_imgs[:show_n] if fba_imgs else []):
                        with cols[i % 6]:
                            st.image(im, caption="FBA", use_column_width=True, clamp=True)
                        i += 1
                    for im in (mrp_imgs[:show_n] if mrp_imgs else []):
                        with cols[i % 6]:
                            st.image(im, caption="MRP", use_column_width=True, clamp=True)
                        i += 1

                # Downloads
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if settings["print_fba_labels"] and fba_imgs:
                    fba_pdf = images_to_pdf_bytes(fba_imgs)
                    st.download_button(
                        "⬇️ Download FBA PDF",
                        data=fba_pdf,
                        file_name=f"fba_labels_{ts}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                if settings["print_mrp_labels"] and mrp_imgs:
                    mrp_pdf = images_to_pdf_bytes(mrp_imgs)
                    st.download_button(
                        "⬇️ Download MRP PDF",
                        data=mrp_pdf,
                        file_name=f"mrp_labels_{ts}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                # Font fallback warning
                if fonts.fallback_used:
                    st.warning("Fallback bitmap font used. Set valid .ttf files in **Settings → Fonts** for sharper print.", icon="🅰️")

    # ---------- Import/Export ----------
    st.subheader("🔁 Import / Export")
    c1, c2 = st.columns(2)
    with c1:
        upl = st.file_uploader("Import catalog.json", type=["json"])
        if upl:
            try:
                data = json.loads(upl.read().decode("utf-8"))
                if isinstance(data, dict):
                    save_json(CATALOG_PATH, data)
                    st.success("Catalog imported.", icon="✅"); st.cache_data.clear()
                else:
                    st.error("Invalid JSON format. Expected an object keyed by SKU.", icon="⚠️")
            except Exception as e:
                st.error(f"Failed to import: {e}", icon="⚠️")
    with c2:
        cat_now = load_json(CATALOG_PATH, default={})
        st.download_button("Export catalog.json", data=json.dumps(cat_now, indent=2), file_name="catalog.json")

    st.caption("Tip: Catalog fields supported — SKU, ASIN, FNSKU, Brand, Title, MRP, HSN, GST%, MfgMonthYear, Condition, Category.")

if __name__ == "__main__":
    main()