import os
import subprocess
import shutil
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageChops
import re
import threading
import hashlib

# --- CONFIG ---
MUTOOL = 'mutool'
DEBUG = True  # Set to True to use a persistent directory for debugging


# --- Map object numbers to metadata and collect image file info ---
def get_image_files_with_metadata(img_dir, images):
    obj_map = {}
    for img in images:
        num = str(img['obj_num']).zfill(4)
        obj_map[num] = img
    img_files = []
    for img in images:
        obj_num = str(img['obj_num']).zfill(4)
        meta = obj_map.get(obj_num, {})
        thumb_path = os.path.join(img_dir, f"thumb-{obj_num}.png")
        orig_path = os.path.join(img_dir, f"image-{obj_num}.png")  # Only extract when saving

        # --- Ensure alpha channel for extracted images ---
        if os.path.exists(orig_path):
            try:
                im = Image.open(orig_path)
                if im.mode != 'RGBA':
                    im = im.convert('RGBA')
                    im.save(orig_path)
            except Exception as e:
                print(f"[WARN] Could not ensure alpha for {orig_path}: {e}")
        img_files.append({
            'filename': f"image-{obj_num}.png",
            'meta': meta,
            'orig_path': orig_path,
            'thumb_path': thumb_path,
        })
    return img_files

def run_mutool_info(pdf_path):
    result = subprocess.run([MUTOOL, 'info', '-I', pdf_path], capture_output=True, text=True)
    return result.stdout


def run_mutool_extract(pdf_path, outdir, image_obj_nums):
    pdf_path = os.path.abspath(pdf_path)
    if image_obj_nums is None:
        cmd = [MUTOOL, 'extract', pdf_path]
    else:
        cmd = [MUTOOL, 'extract', pdf_path]
        for num in image_obj_nums:
            cmd.append(str(num))
    print(f"[DEBUG] Extracting images to: {os.path.abspath(outdir)}")
    result = subprocess.run(cmd, cwd=outdir)
    print(f"[DEBUG] mutool extract stdout: {result.stdout}")
    print(f"[DEBUG] mutool extract stderr: {result.stderr}")
    print(f"[DEBUG] Files in extraction dir after extraction: {os.listdir(outdir)}")
    return result

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
            page_obj = int(parts[1].strip('():').split(' ')[0])
            dims = [p for p in parts[2].split(' ') if 'x' in p and p[0].isdigit()]
            obj_num = int(parts[2].split('(')[1].split(' ')[0])
            # Try to extract color space/mode info if present
            mode = None
            if '[ DCT' in parts[2] or '[ JPX' in parts[2]:
                # Look for ICC, DeviceRGB, DeviceGray, etc.
                if 'DeviceGray' in parts[2] or 'Gray' in parts[2]:
                    mode = 'L'
                elif 'DeviceRGB' in parts[2] or 'RGB' in parts[2]:
                    mode = 'RGB'
                elif 'DeviceCMYK' in parts[2] or 'CMYK' in parts[2]:
                    mode = 'CMYK'
            if dims:
                width, height = [int(d) for d in dims[0].split('x')]
            else:
                width = height = '?'
            images.append({'page': page, 'obj_num': obj_num, 'width': width, 'height': height, 'mode': mode})
    return images



def merge_smask_alpha(img_dir):
    """
    For each image file, try to pair RGB(A) images with grayscale or non-RGB(A) images of similar size as masks.
    Print debug info for all candidates and pairings.
    """
    files = sorted(os.listdir(img_dir))
    # Gather all images with their mode and size
    image_info = []
    for fname in files:
        if fname.startswith('image-') and (fname.endswith('.png') or fname.endswith('.jpg') or fname.endswith('.jpeg')):
            fpath = os.path.join(img_dir, fname)
            try:
                with Image.open(fpath) as im:
                    image_info.append({
                        'fname': fname,
                        'size': im.size,
                        'mode': im.mode,
                        'fpath': fpath
                    })
            except Exception as e:
                print(f"[WARN] Could not open {fname}: {e}")
    # Print all image info for debugging
    print("[DEBUG] All image files:")
    for info in image_info:
        print(f"  {info['fname']}: mode={info['mode']}, size={info['size']}")
    merged_count = 0
    used_masks = set()
    # Group by page and position if possible (using filename number as proxy)
    def extract_obj_num(fname):
        m = re.search(r'image-(\d+)', fname)
        return int(m.group(1)) if m else -1
    image_info_sorted = sorted(image_info, key=lambda x: extract_obj_num(x['fname']))
    for i, rgb in enumerate(image_info_sorted):
        if rgb['mode'] not in ('RGB', 'RGBA'):
            continue
        # Try to find a mask: next image on same page, similar size, not already used, and not RGB(A)
        best_mask = None
        for j in range(i+1, len(image_info_sorted)):
            mask = image_info_sorted[j]
            if mask['fname'] == rgb['fname'] or mask['fname'] in used_masks:
                continue
            if mask['mode'] in ('RGB', 'RGBA'):
                continue
            # Must be very close in size
            if abs(mask['size'][0] - rgb['size'][0]) <= 2 and abs(mask['size'][1] - rgb['size'][1]) <= 2:
                best_mask = mask
                break
        if best_mask:
            rgb_path = rgb['fpath']
            mask_path = best_mask['fpath']
            print(f"[DEBUG] Pairing {rgb['fname']} (mode={rgb['mode']}, size={rgb['size']}) with {best_mask['fname']} (mode={best_mask['mode']}, size={best_mask['size']})")
            try:
                with Image.open(rgb_path) as im:
                    im = im.convert('RGBA')
                    with Image.open(mask_path) as smask:
                        smask = smask.convert('L')
                        if smask.size != im.size:
                            try:
                                resample = Image.Resampling.LANCZOS
                            except AttributeError:
                                # Pillow < 9.1.0: fallback to Image.LANCZOS or Image.NEAREST
                                resample = getattr(Image, 'LANCZOS', getattr(Image, 'NEAREST', 0))
                            smask = smask.resize(im.size, resample)
                        r, g, b, _ = im.split()
                        im_rgba = Image.merge('RGBA', (r, g, b, smask))
                        # --- Trim after mask application ---
                        from PIL import ImageChops
                        def trim_transparency_local(im):
                            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                                alpha = im.convert('RGBA').split()[-1]
                                bbox = alpha.getbbox()
                                if bbox:
                                    return im.crop(bbox)
                            return im
                        im_rgba = trim_transparency_local(im_rgba)
                        outpath = rgb_path
                        im_rgba.save(outpath)
                        print(f"[INFO] Merged {rgb['fname']} + {best_mask['fname']} -> {rgb['fname']} (overwritten, trimmed)")
                        merged_count += 1
                        used_masks.add(best_mask['fname'])
            except Exception as e:
                print(f"[WARN] Failed to merge {rgb['fname']} + {best_mask['fname']}: {e}")
    print(f"[INFO] Merged {merged_count} image+mask pairs.")

def main():
    spinner = None
    spinner_frames = []
    spinner_frame_idx = 0
    spinner_job = None

    def load_spinner_gif(path='spinner.gif'):
        # Load all frames of the GIF
        # PIL already imported at top
        frames = []
        try:
            im = Image.open(path)
            while True:
                frames.append(ImageTk.PhotoImage(im.copy(), master=root))
                try:
                    im.seek(im.tell() + 1)
                except EOFError:
                    break
        except Exception as e:
            print(f"[WARN] Could not load spinner.gif: {e}")
        return frames

    def show_spinner():
        nonlocal spinner, spinner_frames, spinner_frame_idx, spinner_job
        canvas.itemconfig(frame_id, state='hidden')
        spinner_frames = load_spinner_gif()
        spinner_frame_idx = 0
        if not spinner_frames:
            raise RuntimeError("spinner.gif not found or failed to load. Spinner is required.")
        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 200
        spinner = tk.Label(canvas, image=spinner_frames[0], borderwidth=0, highlightthickness=0, bg='white')
        spinner.place(x=w//2, y=h//2, anchor='center')
        def animate():
            nonlocal spinner_frame_idx, spinner_job
            if not spinner_frames or spinner is None:
                return
            spinner.config(image=spinner_frames[spinner_frame_idx])
            spinner_frame_idx = (spinner_frame_idx + 1) % len(spinner_frames)
            spinner_job = root.after(80, animate)  # 80ms per frame ~12.5fps
        animate()

    def clear_spinner():
        nonlocal spinner, spinner_job
        if spinner_job is not None:
            root.after_cancel(spinner_job)
            spinner_job = None
        if spinner is not None:
            spinner.destroy()
            spinner = None
        # Restore the frame to the canvas after spinner is cleared
        canvas.itemconfig(frame_id, state='normal')
        print('[DEBUG] clear_spinner: frame_id state set to normal')
        print('[DEBUG] clear_spinner: canvas items:', canvas.find_all())
        print('[DEBUG] clear_spinner: frame.winfo_manager():', frame.winfo_manager())
        print('[DEBUG] clear_spinner: frame.winfo_ismapped():', frame.winfo_ismapped())
        
    root = tk.Tk()
    root.title('PDF Image Selector')
    win_w, win_h = 965, 800  # Increased width for more toolbar padding and radio spacing (was 950)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (win_w // 2)
    y = (screen_height // 2) - (win_h // 2)
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    # --- Open PDF logic ---
    def open_pdf():
        nonlocal img_files_with_meta, tmpdir
        # Step 1: Show file dialog first (no spinner yet)
        pdf_path = filedialog.askopenfilename(title='Select PDF file', filetypes=[('PDF files', '*.pdf')], parent=root)
        if not pdf_path:
            return
        # Step 2: Now show spinner and start processing
        show_spinner()
        def worker():
            nonlocal img_files_with_meta, tmpdir
            # Step 0: Use the selected pdf_path
            # Step 1: Read PDF Info
            info = run_mutool_info(pdf_path)
            images = parse_image_info(info)
            if not images:
                def show_no_images():
                    clear_spinner()
                    messagebox.showinfo('No Images', 'No images found in PDF.', parent=root)
                root.after(0, show_no_images)
                return
            # Step 2: Prepare extraction dir
            if DEBUG:
                debug_extract_dir = os.path.abspath(os.path.join(os.getcwd(), 'debug_extracted_images'))
                if os.path.exists(debug_extract_dir):
                    for f in os.listdir(debug_extract_dir):
                        fp = os.path.join(debug_extract_dir, f)
                        try:
                            if os.path.isfile(fp) or os.path.islink(fp):
                                os.unlink(fp)
                            elif os.path.isdir(fp):
                                shutil.rmtree(fp)
                        except Exception as e:
                            print(f"[DEBUG] Failed to delete {fp}: {e}")
                else:
                    os.makedirs(debug_extract_dir)
                tmpdir = debug_extract_dir
            else:
                if tmpdir and os.path.exists(tmpdir):
                    shutil.rmtree(tmpdir)
                tmpdir = tempfile.mkdtemp()
            # Step 3: Extract images
            run_mutool_extract(pdf_path, tmpdir, None)
            # Step 4: Apply SMask
            merge_smask_alpha(tmpdir)
            # Step 5: Deduplicate
            def hash_image_file(path):
                hasher = hashlib.sha1()
                with open(path, 'rb') as f:
                    while True:
                        buf = f.read(8192)
                        if not buf:
                            break
                        hasher.update(buf)
                return hasher.hexdigest()
            def deduplicate_images(imgs):
                seen = set()
                unique_imgs = []
                for img in imgs:
                    img_path = img['orig_path']
                    try:
                        h = hash_image_file(img_path)
                    except Exception:
                        continue
                    if h not in seen:
                        seen.add(h)
                        unique_imgs.append(img)
                return unique_imgs
            imgs_with_meta = get_image_files_with_metadata(tmpdir, images)
            imgs_with_meta = deduplicate_images(imgs_with_meta)
            if not imgs_with_meta:
                def show_no_imgs_extracted():
                    clear_spinner()
                    messagebox.showinfo('No Images', 'No images extracted.', parent=root)
                root.after(0, show_no_imgs_extracted)
                return
            # Step 6: Generate thumbnails
            total_thumbs = len(images)
            for i, img in enumerate(images):
                obj_num = img.get('obj_num')
                if not obj_num:
                    continue
                orig_path = os.path.join(tmpdir, f"image-{obj_num:04d}.png")
                thumb_path = os.path.join(tmpdir, f"thumb-{obj_num:04d}.png")
                if not os.path.exists(orig_path):
                    continue
                if os.path.exists(thumb_path):
                    continue
                try:
                    with Image.open(orig_path) as im:
                        def trim_whitespace(im):
                            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                                alpha = im.convert("RGBA").split()[-1]
                                bbox = alpha.getbbox()
                                if bbox:
                                    return im.crop(bbox)
                                else:
                                    return im
                            else:
                                bg = im.getpixel((0, 0))
                                if isinstance(bg, int):
                                    bg = (bg,)
                                bg_img = Image.new(im.mode, im.size, bg)
                                diff = ImageChops.difference(im, bg_img)
                                bbox = diff.getbbox()
                                if bbox:
                                    return im.crop(bbox)
                                else:
                                    return im
                        im_trimmed = trim_whitespace(im)
                        im_trimmed.thumbnail((160, 160))
                        im_trimmed.save(thumb_path, 'PNG')
                except Exception as e:
                    print(f"[WARN] Failed to generate trimmed thumbnail for image {orig_path}: {e}")
            # Step 7: Perceptual Deduplication (after all thumbs are written)
            def ahash_image(img, hash_size=8):
                img = img.convert('L')
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = getattr(Image, 'LANCZOS', getattr(Image, 'BICUBIC', 0))
                img = img.resize((hash_size, hash_size), resample)
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                bits = ''.join(['1' if p > avg else '0' for p in pixels])
                return bits
            def hamming_distance(a, b):
                return sum(x != y for x, y in zip(a, b))
            hash_area_img = []
            for img in imgs_with_meta:
                thumb_path = img.get('thumb_path')
                orig_path = img.get('orig_path')
                if not thumb_path or not os.path.exists(thumb_path):
                    continue
                try:
                    with Image.open(thumb_path) as thumb_img:
                        h = ahash_image(thumb_img)
                except Exception as e:
                    print(f"[WARN] Could not perceptual-hash thumbnail {thumb_path}: {e}")
                    continue
                try:
                    with Image.open(orig_path) as orig_img:
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
                for j in range(i+1, len(hash_area_img)):
                    if j in used:
                        continue
                    h2, area2, img2 = hash_area_img[j]
                    if hamming_distance(h1, h2) <= threshold:
                        group.append((area2, img2, j))
                        used.add(j)
                group.sort(reverse=True)
                kept.append(group[0][1])
                used.add(i)
            imgs_with_meta = kept
            img_files_with_meta = imgs_with_meta
            def finish_render():
                clear_spinner()
                print('[DEBUG] finish_render: after clear_spinner, before render_images')
                print('[DEBUG] finish_render: canvas items:', canvas.find_all())
                print('[DEBUG] finish_render: frame.winfo_manager():', frame.winfo_manager())
                print('[DEBUG] finish_render: frame.winfo_ismapped():', frame.winfo_ismapped())
                render_images()
                print('[DEBUG] finish_render: after render_images')
                print('[DEBUG] finish_render: canvas items:', canvas.find_all())
                print('[DEBUG] finish_render: frame.winfo_manager():', frame.winfo_manager())
                print('[DEBUG] finish_render: frame.winfo_ismapped():', frame.winfo_ismapped())
            root.after(0, finish_render)
        threading.Thread(target=worker, daemon=True).start()

    def select_all():
        suppress_traces['value'] = True
        for img in img_files_with_meta:
            if 'select_var' not in img:
                img['select_var'] = tk.BooleanVar(value=True)
            img['select_var'].set(True)
        suppress_traces['value'] = False
        for img in img_files_with_meta:
            if 'check_canvas' in img:
                draw_check_checkbox = img['draw_check_checkbox']
                draw_check_checkbox(img['check_canvas'], True)

    def select_none():
        suppress_traces['value'] = True
        for img in img_files_with_meta:
            if 'select_var' not in img:
                img['select_var'] = tk.BooleanVar(value=True)
            img['select_var'].set(False)
        suppress_traces['value'] = False
        for img in img_files_with_meta:
            if 'check_canvas' in img:
                draw_check_checkbox = img['draw_check_checkbox']
                draw_check_checkbox(img['check_canvas'], False)

    def save_selected():
        if not tmpdir or not os.path.exists(tmpdir):
            messagebox.showerror('No Images', 'No images to save. Please open a PDF first.', parent=root)
            return
        kept = 0
        export_fmt = export_format_var.get().upper()
        page_img_counter = {}
        outdir = filedialog.askdirectory(title='Select output folder', parent=root)
        if not outdir:
            return
        for img in img_files_with_meta:
            var = img.get('select_var')
            if var and var.get():
                img_file = img['filename']
                meta = img['meta']
                pg = meta.get('page', '?')
                pg_str = f"{int(pg):03}" if isinstance(pg, int) or (isinstance(pg, str) and str(pg).isdigit()) else str(pg)
                if pg_str not in page_img_counter:
                    page_img_counter[pg_str] = 1
                else:
                    page_img_counter[pg_str] += 1
                per_page_idx = page_img_counter[pg_str]
                idx_str = f"{per_page_idx:02}"
                out_name = f"pg{pg_str}-{idx_str}.{export_fmt.lower()}"
                full_img_path = os.path.join(tmpdir, img_file) if tmpdir else None
                if not full_img_path or not os.path.exists(full_img_path):
                    continue
                pil_img = Image.open(full_img_path)
                out_path = os.path.join(outdir, out_name)
                pil_img.save(out_path, export_fmt)
                kept += 1
        messagebox.showinfo('Done', f'Saved {kept} images to {outdir}', parent=root)
        render_images()


    # --- Toolbar and controls ---
    toolbar = ttk.Frame(root)
    toolbar.pack(side='top', fill='x')

    # --- Toolbar layout: [Open PDF] {flex} [Select All] [Select None] {flex} O PNG O WEBP [Save Selected] ---
    checks = []
    export_format_var = tk.StringVar(value='PNG')
    toolbar.grid_columnconfigure(0, weight=0)
    toolbar.grid_columnconfigure(1, weight=1)
    toolbar.grid_columnconfigure(2, weight=0)
    toolbar.grid_columnconfigure(3, weight=0)
    toolbar.grid_columnconfigure(4, weight=1)
    toolbar.grid_columnconfigure(5, weight=0)
    toolbar.grid_columnconfigure(6, weight=0)
    btn_open = tk.Button(toolbar, text='📂 Open PDF', bg='#bbdefb', fg='black', activebackground='#1976d2', activeforeground='white', command=open_pdf)
    btn_open.grid(row=0, column=0, padx=(8, 0), pady=5, sticky='w')
    btn_all = ttk.Button(toolbar, text='Select All', command=select_all)
    btn_all.grid(row=0, column=2, padx=(0, 2), pady=5)
    btn_none = ttk.Button(toolbar, text='Select None', command=select_none)
    btn_none.grid(row=0, column=3, padx=(2, 0), pady=5)
    format_controls = ttk.Frame(toolbar)
    rb_png = ttk.Radiobutton(format_controls, text='PNG', variable=export_format_var, value='PNG', width=5)
    rb_webp = ttk.Radiobutton(format_controls, text='WEBP', variable=export_format_var, value='WEBP', width=5)
    rb_png.pack(side='left', padx=(0,1), pady=5)
    rb_webp.pack(side='left', padx=(1,0), pady=5)
    format_controls.grid(row=0, column=5, padx=(0, 0), pady=5)
    btn_save = tk.Button(toolbar, text='💾 Save Selected', bg='#43a047', fg='black', activebackground='#a5d6a7', activeforeground='black', command=save_selected)
    btn_save.grid(row=0, column=6, padx=(8, 12), pady=5, sticky='e')

    # --- Placeholder for image state ---
    img_files_with_meta = []
    tmpdir = None

    # --- Canvas and image grid ---
    container = ttk.Frame(root)
    container.pack(side='top', fill='both', expand=True)

    class ScrollFrame():
        def __init__(self, container):
            self.canvas = tk.Canvas(container, highlightthickness=0)
            self.vsb = ttk.Scrollbar(container, orient='vertical', command=self.canvas.yview)
            self.canvas.configure(yscrollcommand=self.vsb.set)
            self.canvas.pack(side='left', fill='both', expand=True)
            self.vsb.pack(side='right', fill='y')
            self.frame = ttk.Frame(self.canvas)
            self.frame_id = self.canvas.create_window((0, 0), window=self.frame, anchor='nw')

            def _update_scrollregion(self):
                bbox = self.canvas.bbox(self.frame_id)
                if bbox:
                    self.canvas.configure(scrollregion=bbox)

            self.frame.bind('<Configure>', _update_scrollregion)
            self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(self.frame_id))

            def _on_mousewheel(self, event):
                self.canvas.yview_scroll(event.delta/(-1*event.delta), "units")

            def _bind_mousewheel(self, event):
                self.canvas.bind_all("<MouseWheel>", lambda *args, passed=self.canvas: self._on_mousewheel(*args, passed))

            def _unbind_mousewheel(self, event):
                self.canvas.unbind_all("<MouseWheel>")

            def setup_mousewheel(self):
                self.frame.bind('<Enter>', lambda *args, passed=self.canvas: self._bind_mousewheel(*args, passed))
                self.frame.bind('<Leave>', lambda *args, passed=self.canvas: self._unbind_mousewheel(*args, passed))

            self.setup_mousewheel = setup_mousewheel


    sf = ScrollFrame(container)

    thumbs = []
    thumb_size = 160

    # --- Batch selection optimization ---
    suppress_traces = {'value': False}  # Mutable container for closure


    # Prompt for PDF on startup, but after main window is visible and centered
    def prompt_pdf_after_window_shown():
        # Wait for the window to be fully drawn and centered
        root.update_idletasks()
        root.deiconify()  # Ensure window is shown
        root.lift()
        root.after(150, open_pdf)  # Give a short delay for window manager to show

    # Hide window until ready to show (prevents off-screen dialog)
    root.withdraw()
    root.after(50, lambda: root.deiconify())
    root.after(100, prompt_pdf_after_window_shown)


    # --- Render images grid ---
    def render_images():
        print('[DEBUG] render_images: start, img_files_with_meta:', len(img_files_with_meta))
        print('[DEBUG] img_files_with_meta:', img_files_with_meta)
        for widget in frame.winfo_children():
            widget.destroy()
        thumbs.clear()
        checks.clear()
        print('[DEBUG] render_images: cleared widgets and thumbs')
        frame.update_idletasks()
        n_imgs = len(img_files_with_meta)
        cols = 4
        pad_y = 18
        pad_x = 24
        thumb_w = thumb_size

        # --- Click to show full-res image ---
        def show_full_res(event, orig_path=None, meta=None, img_file=None):
            if not orig_path or not os.path.exists(orig_path):
                messagebox.showerror('Error', f'Image file not found.', parent=root)
                return
            if not meta:
                meta = {}
            try:
                top = tk.Toplevel(root)
                top.title(f"Full Image: {img_file if img_file else ''}")
                root.update_idletasks()
                x = root.winfo_rootx() + (root.winfo_width() // 2) - 400
                y = root.winfo_rooty() + (root.winfo_height() // 2) - 300
                top.geometry(f"800x600+{x}+{y}")
                top.transient(root)
                top.grab_set()
                # Scrollable canvas for large images
                canvas = tk.Canvas(top, highlightthickness=0)
                hsb = ttk.Scrollbar(top, orient='horizontal', command=canvas.xview)
                vsb = ttk.Scrollbar(top, orient='vertical', command=canvas.yview)
                canvas.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
                canvas.pack(side='left', fill='both', expand=True)
                vsb.pack(side='right', fill='y')
                hsb.pack(side='bottom', fill='x')
                pil_full = Image.open(orig_path)
                img_w, img_h = pil_full.size
                # Resize if too large for screen
                screen_w = root.winfo_screenwidth()
                screen_h = root.winfo_screenheight()
                max_w = min(1600, screen_w - 100)
                max_h = min(1200, screen_h - 100)
                scale = min(1.0, max_w / img_w, max_h / img_h)
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = getattr(Image, 'LANCZOS', getattr(Image, 'BICUBIC', 0))
                if scale < 1.0:
                    pil_disp = pil_full.resize((int(img_w*scale), int(img_h*scale)), resample)
                else:
                    pil_disp = pil_full

                # Center image in canvas
                canvas.update_idletasks()
                c_w = canvas.winfo_width() or 800
                c_h = canvas.winfo_height() or 600
                x0 = max((c_w - pil_disp.width) // 2, 0)
                y0 = max((c_h - pil_disp.height) // 2, 0)
                full_img = ImageTk.PhotoImage(pil_disp, master=top)
                img_id = canvas.create_image(x0, y0, anchor='nw', image=full_img)
                canvas.config(scrollregion=(0, 0, pil_disp.width, pil_disp.height))

                # Store image reference
                if not hasattr(show_full_res, '_img_refs'):
                    show_full_res._img_refs = []
                show_full_res._img_refs.append(full_img)

                # Show meta info at top
                meta_str = f"{img_file if img_file else ''} | {meta.get('width', '?')} x {meta.get('height', '?')}"
                label = ttk.Label(top, text=meta_str, background='#222', foreground='#fff', anchor='w')
                label.place(x=0, y=0, relwidth=1)
                
                # Center image on resize
                def center_image(event=None):
                    c_w = canvas.winfo_width()
                    c_h = canvas.winfo_height()
                    x0 = max((c_w - pil_disp.width) // 2, 0)
                    y0 = max((c_h - pil_disp.height) // 2, 0)
                    canvas.coords(img_id, x0, y0)
                canvas.bind('<Configure>', center_image)

                # Close on Escape
                top.bind('<Escape>', lambda e: top.destroy())
            except Exception as e:
                messagebox.showerror('Error', f'Could not open image: {e}', parent=root)
        # --- End click ---

        page_img_counter = {}
        for idx, img in enumerate(img_files_with_meta):
            print(f'[DEBUG] Drawing image {idx}:', img)
            row = idx // cols
            col = idx % cols
            img_file = img['filename']
            meta = img['meta']
            orig_path = img['orig_path']
            thumb_path = img.get('thumb_path')
            if 'select_var' not in img:
                img['select_var'] = tk.BooleanVar(value=True)
            checks.append(img['select_var'])
            display_path = thumb_path if thumb_path and os.path.exists(thumb_path) else orig_path
            pil_img = Image.open(display_path)
            pil_img.thumbnail((thumb_size, thumb_size))

            # No compositing: let transparency be handled by default (may show as white or system default)
            thumb = ImageTk.PhotoImage(pil_img, master=root)
            thumbs.append(thumb)

            # Use tk.Label for thumbnail, no border/highlight, default bg
            panel = tk.Label(frame, image=thumb, borderwidth=0, relief='flat')
            panel.grid(row=row*2, column=col, padx=pad_x, pady=pad_y, sticky='ns')
            
            # --- Make thumbnail look clickable: change cursor on hover ---
            def on_enter(e, panel=panel):
                panel.config(cursor='hand2')
            def on_leave(e, panel=panel):
                panel.config(cursor='')
            panel.bind('<Enter>', on_enter)
            panel.bind('<Leave>', on_leave)
            # --- End clickable style ---

            panel.bind('<Button-1>', lambda event, orig_path=orig_path, meta=meta, img_file=img_file: show_full_res(event, orig_path, meta, img_file))
            # --- End click ---
            pg = meta.get('page', '?') if meta else '?'
            width = meta.get('width', '?') if meta else '?'
            height = meta.get('height', '?') if meta else '?'
            pg_str = f"{int(pg):03}" if (isinstance(pg, int) or (isinstance(pg, str) and str(pg).isdigit())) else str(pg)
            if pg_str not in page_img_counter:
                page_img_counter[pg_str] = 1
            else:
                page_img_counter[pg_str] += 1
            per_page_idx = page_img_counter[pg_str]
            idx_str = f"{per_page_idx:02}"
            meta_frame = ttk.Frame(frame)
            meta_frame.grid(row=row*2+1, column=col, sticky='n')

            # --- Top group: file name entry and checkbox, centered ---
            top_group = ttk.Frame(meta_frame)
            top_group.pack(side='top', anchor='n', pady=(0,0), fill='x')
            entry_width = round(18 * 2 / 3)
            default_name = img.get('save_name') or f"pg{pg_str}-{idx_str}"
            if 'save_name' not in img:
                img['save_name'] = default_name
            entry = ttk.Entry(top_group, width=entry_width)
            entry.insert(0, img['save_name'])

            # Set background to match checkbox background
            entry.configure(background='#f5f5f5')
            entry.pack(side='left', anchor='n', padx=(0, 0), pady=(0,0))

            def update_save_name(event=None, img=img, entry=entry):
                val = entry.get().strip()
                if val:
                    img['save_name'] = val
            entry.bind('<FocusOut>', update_save_name)
            entry.bind('<Return>', update_save_name)

            def draw_check_checkbox(canvas, checked):
                try:
                    canvas.delete('all')
                    if checked:
                        canvas.create_text(10, 10, text='✔️', font=('Arial', 10), anchor='center')
                except tk.TclError:
                    pass

            check_checkbox = tk.Canvas(top_group, width=20, height=20, highlightthickness=0, bd=0, relief='flat', bg='#f5f5f5')

            # Use pady=(2,0) to nudge the checkbox down for top alignment with entry
            check_checkbox.pack(side='left', anchor='n', padx=(12,0), pady=(2,0))

            def update_check_checkbox(*args, canvas=check_checkbox, v=img['select_var']):
                if not suppress_traces['value']:
                    draw_check_checkbox(canvas, v.get())
            img['select_var'].trace_add('write', update_check_checkbox)
            check_checkbox.bind('<Button-1>', lambda e, v=img['select_var']: v.set(not v.get()))
            draw_check_checkbox(check_checkbox, img['select_var'].get())
            img['check_canvas'] = check_checkbox
            img['draw_check_checkbox'] = draw_check_checkbox

            # --- Dimensions label below, centered ---
            dim_str = f"{width} x {height}"
            label2 = ttk.Label(meta_frame, text=dim_str, anchor='center', justify='center')
            label2.pack(side='top', anchor='center', pady=(2,0))

        frame.update_idletasks()
        canvas.update_idletasks()
        root.update_idletasks()
        # Update scrollregion to the bounding box of the frame window (not 'all')
        bbox = canvas.bbox(frame_id)
        print('[DEBUG] render_images: done')
        print('[DEBUG] frame children after render:', frame.winfo_children())
        print('[DEBUG] canvas bbox (frame_id):', bbox)
        if bbox:
            canvas.config(scrollregion=bbox)

    root.mainloop() 

if __name__ == '__main__':
    main()
