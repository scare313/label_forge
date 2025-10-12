# Tag and Pack — Streamlit SKU Label Maker (FBA + MRP)

Generate beautiful, print-ready **FBA (ASIN)** and **MRP** labels from a clean, mobile-friendly **Streamlit** UI.  
Built for fast e‑commerce workflows (Amazon FBA, Flipkart, Meesho) with **search & filter**, **JSON-backed catalog**, **inline validation**, and **PDF exports**.

> ⚙️ **No Excel required**: Manage your SKU master in `data/catalog.json`.  
> 🎯 **Warehouse-friendly**: Big tap targets, responsive layout, quick add-to-queue.  
> 🖨️ **Print-ready**: High-DPI raster labels (PDF). Optional **ZPL** can be added.

---

## ✨ Features

- **Search & Filter** by **SKU / Title / ASIN** with **Brand** and **Category** facets
- **Inline catalog editor** with validation prompts for missing fields
- **User settings** persisted in `data/settings.json` (DPI, label size, fonts, scales)
- **Responsive UI**: mobile/warehouse friendly
- **Preview first N labels** and **download PDFs** (FBA + MRP)
- **Code128** barcode for **FNSKU** (FBA label)
- **Import/Export** catalog via JSON
- **Graceful error handling**: font fallbacks, missing fields, invalid JSON

---

## 📁 Project Structure

```
labeler/
├─ app.py                    # Streamlit app (UI + logic)
├─ requirements.txt          # Python deps
├─ run_labeler.bat           # Windows: activate venv & run
├─ data/
│  ├─ catalog.json           # SKU master (auto-created if missing)
│  └─ settings.json          # UI/print defaults (auto-created if missing)
└─ assets/
   ├─ DejaVuSans.ttf
   └─ DejaVuSans-Bold.ttf
```

---

## 🚀 Quick Start

### 1) Create & activate a virtual environment

```bash
cd labeler
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

Or on Windows, just double‑click **`run_labeler.bat`**.

---

## ⚙️ Configuration

### `data/settings.json` (sample)

```json
{
  "dpi": 300,
  "fba_label_width_mm": 50,
  "fba_label_height_mm": 30,
  "mrp_label_width_mm": 50,
  "mrp_label_height_mm": 30,
  "font_path_regular": ".\\fonts\\NotoSans-Regular.ttf",
  "font_path_bold": ".\\fonts\\NotoSans-Bold.ttf",
  "print_fba_labels": true,
  "print_mrp_labels": true,
  "fba_title_scale": 0.30,
  "fba_small_scale": 0.50,
  "fba_footer_pad_px": 30,
  "mrp_brand_scale": 0.50,
  "mrp_title_scale": 0.50,
  "mrp_mrp_scale": 0.60,
  "mrp_small_scale": 0.50,
  "mrp_bottom_pad_px": 30,
  "mrp_show_inclusive_line": true,
  "mrp_max_title_lines": 2,
  "limit_preview": 12
}
```

### `data/catalog.json` (sample)

```json
{
  "SKU123": {
    "ASIN": "B0XYZ12345",
    "FNSKU": "X001ABCDEF",
    "Brand": "Acme",
    "Title": "Super Fast Charger 20W",
    "MRP": "699",
    "HSN": "8504",
    "GST%": "18",
    "MfgMonthYear": "Oct-2025",
    "Condition": "New",
    "Category": "Chargers"
  }
}
```

---

## 🧭 Using the App

1. **Open the app** → `streamlit run app.py`
2. **Load/Edit Catalog**
   - Use **Search** to find SKUs (by SKU/Title/ASIN) and filter by **Brand/Category**
   - Expand **Add or edit SKU** to add new SKUs or update existing
3. **Build Print Queue**
   - Select SKU from dropdown, set **Qty**, click **➕ Add**
   - Repeat for multiple SKUs
4. **Generate**
   - Click **Generate Labels**
   - If any required fields are missing, fill them in the inline form
   - View **previews**, then **download PDFs** for FBA/MRP

---

## 🖨️ Printing Tips

- Ensure the **DPI** and **label dimensions (mm)** match your printer/label stock
- Use **TTF fonts** for sharp print (avoid PIL bitmap fallback)
- Print PDFs at **100% scale** (no fit-to-page) to preserve dimensions
- For thermal printers (Zebra), consider adding **ZPL** output (see Roadmap)

---

## 🧪 Requirements

- Python 3.9+
- Packages (from `requirements.txt`):
  ```txt
  streamlit==1.38.0
  pillow==10.4.0
  pandas==2.2.2
  reportlab==4.2.2
  ```

---

## 🧩 Customization

- **Fonts**: Set `font_path_regular` and `font_path_bold` (e.g., Noto Sans, Roboto)
- **Label Design**:
  - FBA: Title, **Code128** barcode (FNSKU), Condition, SKU
  - MRP: Brand, Title, **MRP ₹**, inclusive taxes line, bottom band (Mfg/HSN/GST)
- **Performance**:
  - Use `limit_preview` to cap thumbnails
  - Batch many SKUs—PDF export avoids browser memory bloat

---

## ❗ Troubleshooting

- **Blurry/Blocky text**
  - The app fell back to a bitmap font. Provide valid `.ttf` in **Settings → Fonts**.
- **PDF prints at wrong size**
  - Disable “Fit to page”; print at **Actual size (100%)**.
- **Port already in use**
  - Run: `streamlit run app.py --server.port 8502`
- **Non-Latin characters missing**
  - Use a Unicode font (e.g., **Noto Sans** family) and set both regular/bold paths.
- **Nothing happens on Generate**
  - Ensure **queue** has at least one SKU and missing fields are filled.

---

## 🔮 Roadmap

- ✅ QR code on MRP (to product page or internal SKU URL)
- ✅ ZPL export for Zebra printers
- 🖼️ Brand logo support per SKU
- 🧩 Excel import (map → JSON) for migration
- 🧾 Bulk operations: CSV import for quantities
- 👤 Roles & presets (Warehouse vs. Admin settings)

---

## 📜 License

MIT License