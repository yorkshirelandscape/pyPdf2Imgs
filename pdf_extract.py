"""PDF image extraction backend (mutool-based). No UI dependencies."""
import os
import re
import glob
import shutil
import hashlib
import subprocess
from PIL import Image, ImageChops

MUTOOL = 'mutool'

# GUI apps launched from Finder/Explorer (as opposed to a terminal) don't
# inherit the shell's PATH, so a plain subprocess.run(['mutool', ...]) can
# fail even when mutool is properly installed. Fall back to the locations
# common package managers actually use.
MUTOOL_FALLBACK_PATHS = [
    '/opt/homebrew/bin/mutool',        # Homebrew, Apple Silicon
    '/usr/local/bin/mutool',           # Homebrew, Intel Mac / common Linux
    '/opt/local/bin/mutool',           # MacPorts
    '/usr/bin/mutool',                 # Linux distro packages
    '/snap/bin/mutool',                # Linux snap packages
    r'C:\Program Files\mupdf\mutool.exe',
    r'C:\Program Files (x86)\mupdf\mutool.exe',
]


class MutoolNotFoundError(RuntimeError):
    pass


def find_mutool():
    found = shutil.which(MUTOOL)
    if found:
        return found
    for path in MUTOOL_FALLBACK_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _resolve_mutool():
    path = find_mutool()
    if not path:
        raise MutoolNotFoundError(
            "Could not find 'mutool'. Install MuPDF tools (e.g. 'brew install "
            "mupdf-tools' on macOS, 'apt install mupdf-tools' on Linux, or "
            "download from https://mupdf.com/downloads/ on Windows)."
        )
    return path


def run_mutool_info(pdf_path):
    mutool = _resolve_mutool()
    result = subprocess.run([mutool, 'info', '-I', pdf_path], capture_output=True, text=True)
    return result.stdout


def run_mutool_extract(pdf_path, outdir):
    mutool = _resolve_mutool()
    pdf_path = os.path.abspath(pdf_path)
    cmd = [mutool, 'extract', pdf_path]
    return subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)


def parse_image_info(info_text):
    images = []
    lines = info_text.splitlines()
    in_images_section = False
    for line in lines:
        if line.strip().startswith('Images ('):
            in_images_section = True
            continue
        if in_images_section:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            page = int(parts[0])
            dims = [p for p in parts[2].split(' ') if 'x' in p and p[0].isdigit()]
            obj_num = int(parts[2].split('(')[1].split(' ')[0])
            if dims:
                width, height = [int(d) for d in dims[0].split('x')]
            else:
                width = height = '?'
            images.append({'page': page, 'obj_num': obj_num, 'width': width, 'height': height})
    return images


def find_extracted_file(img_dir, obj_num):
    """mutool extract picks the file extension per-object based on its encoding
    (.jpg for DCT/JPEG streams, .png for everything else it has to decode). Find
    whatever it actually produced for this object number rather than assuming png."""
    matches = glob.glob(os.path.join(img_dir, f"image-{obj_num:04d}.*"))
    return matches[0] if matches else None


_PAM_MODE_BY_TUPLTYPE = {
    ('CMYK', 4): 'CMYK',
    ('RGB_ALPHA', 4): 'RGBA',
    ('RGB', 3): 'RGB',
    ('GRAYSCALE_ALPHA', 2): 'LA',
    ('GRAYSCALE', 1): 'L',
    ('BLACKANDWHITE', 1): '1',
}


def _read_pam(path):
    """Parse a PAM (Portable Arbitrary Map) file. mutool falls back to this
    format when decoded pixel data doesn't fit cleanly into PNG or JPEG (e.g.
    raw CMYK from an indexed-color image), and Pillow cannot open PAM files
    natively."""
    with open(path, 'rb') as f:
        data = f.read()
    if not data.startswith(b'P7\n'):
        raise ValueError(f'Not a PAM file: {path}')
    header_end = data.index(b'ENDHDR\n') + len(b'ENDHDR\n')
    fields = {}
    for line in data[:header_end].decode('ascii', errors='replace').splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ('WIDTH', 'HEIGHT', 'DEPTH', 'MAXVAL'):
            fields[parts[0]] = int(parts[1])
        elif len(parts) == 2 and parts[0] == 'TUPLTYPE':
            fields['TUPLTYPE'] = parts[1]
    width, height, depth = fields['WIDTH'], fields['HEIGHT'], fields['DEPTH']
    tupltype = fields.get('TUPLTYPE')
    pixel_data = data[header_end:header_end + width * height * depth]

    if (tupltype, depth) == ('CMYK_ALPHA', 5):
        # Pillow has no native 5-channel mode; split the interleaved bands
        # (each Nth byte is one channel) and reassemble as RGBA.
        size = (width, height)
        c, m, y, k, a = (
            Image.frombytes('L', size, pixel_data[i::5]) for i in range(5)
        )
        rgb = Image.merge('CMYK', (c, m, y, k)).convert('RGB')
        return Image.merge('RGBA', (*rgb.split(), a))

    mode = _PAM_MODE_BY_TUPLTYPE.get((tupltype, depth))
    if mode is None:
        raise ValueError(f"Unsupported PAM tuple type/depth: {tupltype}/{depth}")
    return Image.frombytes(mode, (width, height), pixel_data)


def open_extracted_image(path):
    """Open an image file mutool produced, handling formats Pillow can't
    read natively (currently just PAM)."""
    if path.lower().endswith('.pam'):
        return _read_pam(path)
    return Image.open(path)


def canonical_png_path(img_dir, obj_num):
    return os.path.join(img_dir, f"image-{obj_num:04d}.png")


def normalize_to_png(img_dir, obj_num):
    """Ensure image-{obj_num}.png exists as an RGBA PNG, regardless of the
    extension/mode mutool originally produced. Removes the stale non-png
    original once converted. Returns the canonical png path, or None if the
    object wasn't extracted at all."""
    png_path = canonical_png_path(img_dir, obj_num)
    src_path = find_extracted_file(img_dir, obj_num)
    if src_path is None:
        return None
    if src_path == png_path:
        # Already the canonical file; still make sure it has alpha.
        with Image.open(png_path) as im:
            if im.mode != 'RGBA':
                im.convert('RGBA').save(png_path)
        return png_path
    with open_extracted_image(src_path) as im:
        im.convert('RGBA').save(png_path)
    os.remove(src_path)
    return png_path


def merge_smask_alpha(img_dir):
    """Pair RGB(A) images with grayscale/indexed images of matching size as
    alpha masks, then composite and trim the transparent border. Works across
    mixed jpg/png sources found in img_dir."""
    files = sorted(os.listdir(img_dir))
    image_info = []
    for fname in files:
        if fname.startswith('image-') and fname.lower().endswith(('.png', '.jpg', '.jpeg', '.pam')):
            fpath = os.path.join(img_dir, fname)
            try:
                with open_extracted_image(fpath) as im:
                    image_info.append({'fname': fname, 'size': im.size, 'mode': im.mode, 'fpath': fpath})
            except Exception as e:
                print(f"[WARN] Could not open {fname}: {e}")

    def extract_obj_num(fname):
        m = re.search(r'image-(\d+)', fname)
        return int(m.group(1)) if m else -1

    image_info_sorted = sorted(image_info, key=lambda x: extract_obj_num(x['fname']))
    used_masks = set()
    merged_count = 0
    for i, rgb in enumerate(image_info_sorted):
        if rgb['mode'] not in ('RGB', 'RGBA'):
            continue
        best_mask = None
        for j in range(i + 1, len(image_info_sorted)):
            mask = image_info_sorted[j]
            if mask['fname'] == rgb['fname'] or mask['fname'] in used_masks:
                continue
            if mask['mode'] in ('RGB', 'RGBA'):
                continue
            if abs(mask['size'][0] - rgb['size'][0]) <= 2 and abs(mask['size'][1] - rgb['size'][1]) <= 2:
                best_mask = mask
                break
        if not best_mask:
            continue
        rgb_path = rgb['fpath']
        mask_path = best_mask['fpath']
        try:
            with open_extracted_image(rgb_path) as im:
                im = im.convert('RGBA')
                with open_extracted_image(mask_path) as smask:
                    smask = smask.convert('L')
                    if smask.size != im.size:
                        resample = getattr(Image, 'Resampling', Image).LANCZOS
                        smask = smask.resize(im.size, resample)
                    r, g, b, _ = im.split()
                    im_rgba = Image.merge('RGBA', (r, g, b, smask))
                    alpha = im_rgba.split()[-1]
                    bbox = alpha.getbbox()
                    if bbox:
                        im_rgba = im_rgba.crop(bbox)
                    # Alpha compositing means the merged result must live in a
                    # png, even if the source image was a jpg.
                    obj_num = extract_obj_num(rgb['fname'])
                    out_path = canonical_png_path(img_dir, obj_num)
                    im_rgba.save(out_path)
                    if out_path != rgb_path and os.path.exists(rgb_path):
                        os.remove(rgb_path)
                    merged_count += 1
                    used_masks.add(best_mask['fname'])
        except Exception as e:
            print(f"[WARN] Failed to merge {rgb['fname']} + {best_mask['fname']}: {e}")
    print(f"[INFO] Merged {merged_count} image+mask pairs.")


def trim_whitespace(im):
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bbox = alpha.getbbox()
        return im.crop(bbox) if bbox else im
    bg = im.getpixel((0, 0))
    if isinstance(bg, int):
        bg = (bg,)
    bg_img = Image.new(im.mode, im.size, bg)
    diff = ImageChops.difference(im, bg_img)
    bbox = diff.getbbox()
    return im.crop(bbox) if bbox else im


def hash_image_file(path):
    hasher = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(8192)
            if not buf:
                break
            hasher.update(buf)
    return hasher.hexdigest()


def ahash_image(img, hash_size=8):
    img = img.convert('L')
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    img = img.resize((hash_size, hash_size), resample)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    return ''.join('1' if p > avg else '0' for p in pixels)


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def extract_images(pdf_path, outdir):
    """Full pipeline: mutool info -> extract -> smask merge -> normalize ->
    thumbnails -> exact + perceptual dedup. Returns a list of image dicts:
    {filename, meta, orig_path, thumb_path, save_name}."""
    info = run_mutool_info(pdf_path)
    images = parse_image_info(info)
    if not images:
        return []

    run_mutool_extract(pdf_path, outdir)
    merge_smask_alpha(outdir)

    img_files = []
    for img in images:
        obj_num = img['obj_num']
        try:
            orig_path = normalize_to_png(outdir, obj_num)
        except Exception as e:
            # However complete our format handling is, one image mutool
            # produces in a shape we don't recognize (or can't decode)
            # shouldn't take down every other image in the file with it.
            print(f"[WARN] Skipping image object {obj_num}, could not normalize: {e}")
            continue
        if orig_path is None:
            continue
        img_files.append({
            'filename': os.path.basename(orig_path),
            'meta': img,
            'orig_path': orig_path,
            'thumb_path': os.path.join(outdir, f"thumb-{obj_num:04d}.png"),
            'save_name': None,
        })

    # Exact-duplicate removal (identical bytes).
    seen_hashes = set()
    unique = []
    for img in img_files:
        try:
            h = hash_image_file(img['orig_path'])
        except Exception:
            continue
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(img)
    img_files = unique
    if not img_files:
        return []

    # Thumbnails (trimmed + capped to 160x160).
    for img in img_files:
        orig_path = img['orig_path']
        thumb_path = img['thumb_path']
        if os.path.exists(thumb_path):
            continue
        try:
            with Image.open(orig_path) as im:
                im_trimmed = trim_whitespace(im)
                im_trimmed.thumbnail((160, 160))
                im_trimmed.save(thumb_path, 'PNG')
        except Exception as e:
            print(f"[WARN] Failed to generate thumbnail for {orig_path}: {e}")

    # Perceptual dedup: group visually-similar images, keep the largest of each group.
    hash_area_img = []
    for img in img_files:
        thumb_path = img['thumb_path']
        if not os.path.exists(thumb_path):
            continue
        try:
            with Image.open(thumb_path) as thumb_img:
                h = ahash_image(thumb_img)
        except Exception as e:
            print(f"[WARN] Could not perceptual-hash {thumb_path}: {e}")
            continue
        try:
            with Image.open(img['orig_path']) as orig_img:
                area = orig_img.width * orig_img.height
        except Exception:
            area = 0
        hash_area_img.append((h, area, img))

    threshold = 5
    kept = []
    used = set()
    for i, (h1, area1, img1) in enumerate(hash_area_img):
        if i in used:
            continue
        group = [(area1, img1, i)]
        for j in range(i + 1, len(hash_area_img)):
            if j in used:
                continue
            h2, area2, img2 = hash_area_img[j]
            if hamming_distance(h1, h2) <= threshold:
                group.append((area2, img2, j))
                used.add(j)
        group.sort(key=lambda x: x[0], reverse=True)
        kept.append(group[0][1])
        used.add(i)

    return kept


def make_extract_dir(base_tmpdir=None):
    """Create/clear a fresh extraction directory."""
    import tempfile
    if base_tmpdir and os.path.exists(base_tmpdir):
        shutil.rmtree(base_tmpdir)
    return tempfile.mkdtemp(prefix='pypdf2imgs_')
