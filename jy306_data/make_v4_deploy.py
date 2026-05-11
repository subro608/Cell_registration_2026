#!/usr/bin/env python3
"""
Create deploy version of viewer_warped_invivo_3d_v4.html that loads
patch strip from external PNG and calcium videos from patches/ folder.
"""
import re

SRC = '/Users/neurolab/neuroinformatics/margaret/3d_viewer/viewer_warped_invivo_3d_v4.html'
DST = '/Users/neurolab/neuroinformatics/invivo-exvivo-cell-registration/viewer_v4.html'

print("Reading source HTML...")
with open(SRC) as f:
    html = f.read()
print(f"  Source size: {len(html)/1e6:.1f} MB")

# 1. Replace patchStripB64 with empty string — we'll load from external PNG
print("Replacing patchStripB64...")
html = re.sub(
    r'const patchStripB64="[^"]*"',
    'const patchStripB64=""',
    html
)

# 2. Replace calciumB64 array with empty — we'll load from patches/ folder
print("Replacing calciumB64...")
html = re.sub(
    r'const calciumB64=\[.*?\];',
    'const calciumB64=[];',
    html,
    flags=re.DOTALL
)

# 3. Replace the patchStripImg loading code to use external PNG
# Find: patchStripImg=new Image(); patchStripImg.onload=...
# The existing code loads from base64. We need to change the src.
# Find the line that sets patchStripImg.src
old_load = "patchStripImg.src='data:image/png;base64,'+patchStripB64;"
new_load = "patchStripImg.src='patch_strip_v4.png';"
if old_load in html:
    html = html.replace(old_load, new_load)
    print("  Replaced patch strip src to external PNG")
else:
    # Try alternate pattern
    print("  WARNING: Could not find patch strip src pattern, trying regex...")
    html = re.sub(
        r"patchStripImg\.src='data:image/png;base64,'\+patchStripB64;",
        "patchStripImg.src='patch_strip_v4.png';",
        html
    )

# 4. Replace calcium video loading to use external mp4 files
# Find the code that sets video src from calciumB64
old_calcium = "vid.src='data:video/mp4;base64,'+calciumB64[idx];"
new_calcium = "vid.src='patches/patch_'+idx+'.mp4';"
old_calcium_check = "if(HAS_CALCIUM && typeof calciumB64!=='undefined' && calciumB64[idx])"
new_calcium_check = "if(HAS_CALCIUM)"
if old_calcium in html:
    html = html.replace(old_calcium, new_calcium)
    html = html.replace(old_calcium_check, new_calcium_check)
    print("  Replaced calcium video src to external mp4")
else:
    print("  WARNING: Could not find calcium video src pattern")

print(f"  Deploy size: {len(html)/1e6:.1f} MB")

with open(DST, 'w') as f:
    f.write(html)
print(f"Written to {DST}")
