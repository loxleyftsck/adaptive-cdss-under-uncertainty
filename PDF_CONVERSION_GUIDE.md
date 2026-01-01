# PDF Conversion Guide

## Converting research_paper.md to PDF

---

## 📄 FILE CREATED: research_paper.md

✅ Academic paper telah dibuat di:  
`C:\Users\LENOVO\Documents\adaptive-cdss-under-uncertainty\research_paper.md`

**Content:**

- Full academic paper format
- Your name: Herald Michain Samuel Theo Ginting
- Email: <heraldmsamueltheo@gmail.com>
- GitHub: loxleyftsck
- Repository link included
- Complete with abstract, methodology, bibliography (21 references)

---

## 🔧 METHOD 1: Pandoc (Recommended - Best Quality)

### Install Pandoc (if not installed)

**Windows:**

```powershell
# Download from https://pandoc.org/installing.html
# OR via Chocolatey:
choco install pandoc

# Install LaTeX (required for PDF):
choco install miktex
```

**Or download installers:**

- Pandoc: <https://github.com/jgm/pandoc/releases/latest>
- MiKTeX: <https://miktex.org/download>

### Convert to PDF

```powershell
cd C:\Users\LENOVO\Documents\adaptive-cdss-under-uncertainty

pandoc research_paper.md -o research_paper.pdf `
  --pdf-engine=xelatex `
  -V geometry:margin=1in `
  -V fontsize=11pt `
  -V documentclass=article `
  --number-sections `
  --toc
```

**Result:** Professional academic PDF dengan table of contents dan numbering.

---

## 🔧 METHOD 2: VSCode Extension (Easy)

### Install Extension

1. Open VSCode
2. Install extension: **"Markdown PDF"** by yzane
3. Open `research_paper.md`
4. Press `Ctrl+Shift+P`
5. Type: "Markdown PDF: Export (pdf)"
6. Done! PDF will be in same folder

**Pros:** No command line, one-click conversion  
**Cons:** Less control over formatting

---

## 🔧 METHOD 3: Online Converter (No Installation)

### Option A: Markdown to PDF

1. Go to: <https://www.markdowntopdf.com/>
2. Upload `research_paper.md`
3. Click "Convert"
4. Download PDF

### Option B: Dillinger

1. Go to: <https://dillinger.io/>
2. Paste markdown content
3. Click "Export as" → "PDF"
4. Download

**Pros:** Works immediately, no installation  
**Cons:** Less formatting control, needs internet

---

## 🔧 METHOD 4: Microsoft Word (If you have Office)

1. Open VSCode or any text editor
2. Open `research_paper.md`
3. Copy all content
4. Open Microsoft Word
5. Paste (Word will auto-format markdown headers)
6. Manual formatting adjustments:
   - Set title to bigger font
   - Format equations if needed
   - Add page breaks
7. Save As → PDF

**Pros:** Full manual control  
**Cons:** More manual work

---

## 🔧 METHOD 5: Google Docs (Free)

1. Open Google Docs: <https://docs.google.com/>
2. Create new document
3. Copy content from `research_paper.md`
4. Paste
5. Format manually (headers, titles, equations)
6. File → Download → PDF Document

**Pros:** Free, accessible anywhere  
**Cons:** Manual formatting

---

## ⚡ RECOMMENDED APPROACH

**If you have 10 minutes:**
→ Install Pandoc + MiKTeX → Use Method 1 (BEST quality)

**If you want instant result:**
→ Use Method 3 (Online converter)

**If you want full control:**
→ Use Method 4 or 5 (Manual formatting in Word/Docs)

---

## 📋 CHECKLIST BEFORE CONVERTING

✅ File location: `C:\Users\LENOVO\Documents\adaptive-cdss-under-uncertainty\research_paper.md`  
✅ Your name included: Herald Michain Samuel Theo Ginting  
✅ Email included: <heraldmsamueltheo@gmail.com>  
✅ GitHub: loxleyftsck  
✅ Repository link: <https://github.com/loxleyftsck/adaptive-cdss-under-uncertainty>  
✅ 21 scholarly references included  
✅ Mathematical equations formatted  
✅ Professional structure (Abstract, Methods, Results, Conclusion)

---

## 🎨 POST-CONVERSION TIPS

After creating PDF, you may want to:

1. **Add header/footer** with your name and date
2. **Check equation rendering** - LaTeX equations might need adjustment
3. **Verify links** - GitHub repository link should be clickable
4. **Page numbers** - Add if needed via PDF editor
5. **Cover page** - Optional: Add a cover page with university/org logo

---

## 🆘 TROUBLESHOOTING

**"Pandoc not found"**  
→ Install Pandoc first (see Method 1 install instructions)

**"pdflatex not found"**  
→ Install MiKTeX or use `--pdf-engine=wkhtmltopdf` instead

**"Math equations not rendering"**  
→ Use `--mathjax` flag or online converter

**"Prefer simpler PDF"**  
→ Remove `--toc` and `--number-sections` flags

---

**Current Status:** ⏳ Pandoc command is running (if you installed it)

Check your folder for `research_paper.pdf` in a few seconds!
