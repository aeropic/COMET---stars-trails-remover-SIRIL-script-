###########################################################
#                                                         #
#             AEROPIC COMET cleaner utility               #
#                                                         #
#                        V2.0                             #
#                                                         #
###########################################################

import sys, os
import numpy as np
import cv2
from astropy.io import fits
from scipy.ndimage import map_coordinates, maximum_filter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QSlider, QPushButton, QHBoxLayout, 
                             QFileDialog, QCheckBox, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
import sirilpy as s

class AEROPIC_Master_Comet_EN(QMainWindow):
    def __init__(self):
        super().__init__()
        # Establishing link with Siril instance. Essential for direct memory pixel data exchange.
        # This allows the script to work on the current loaded image without disk I/O overhead.
        self.siril = s.SirilInterface()
        try: self.siril.connect()
        except: sys.exit(1)
            
        self.header_text = (
            "\n###########################################################\n"
            "#                                                         #\n"
            "#             AEROPIC COMET cleaner utility               #\n"
            "#                                                         #\n"
            "#                         V2.0                            #\n"
            "#                                                         #\n"
            "###########################################################"
        )
        try: self.siril.log(self.header_text)
        except: print(self.header_text)        
        
        # Load active image data. We work in float32 to prevent rounding errors during 
        # interpolation and background averaging.
        self.current_file = self.siril.get_image_filename()
        raw = self.siril.get_image_pixeldata()
        self.data = raw.astype(np.float32)
        self.original_data = self.data.copy()
        self.h, self.w = self.data.shape[-2:]
        self.c = self.data.shape[0] if self.data.ndim == 3 else 1
        
        # FITS Header preservation is critical to keep Astrometry (WCS) and metadata intact.
        # This ensures the output can still be Plate-Solved or used for photometry.
        self.header_fits = fits.Header()
        if self.current_file.lower().endswith(('.fit', '.fits', '.fz')):
            try:
                with fits.open(self.current_file) as hdul: self.header_fits = hdul[0].header
            except: pass
        
        # State variables. 
        # 'special_radii' stores per-star manual radius overrides.
        # 'restore_masks' stores coordinates/radii of pixels to be reverted to original state.
        self.data_stars, self.detected_coords = None, [] 
        self.special_radii, self.history, self.redo_stack = {}, [], []
        self.restore_masks = [] 
        self.offset, self.pan_start, self.p1, self.p2 = [0, 0], None, None, None
        self.mouse_pos, self.disp_min, self.disp_max = (0, 0), 0, 1.0
        self.ghost_radius = 40
        self.vh, self.vw = 900, 1400 
        self.is_alt_pressed = False 
        self.is_ghost_mod_pressed = False # CTRL+SHIFT state
        
        self.init_ui()
        self.update_stats()
        self.setup_cv()

    def init_ui(self):
        """Build the control panel using PyQt6. Focus on accessibility and real-time feedback."""
        self.setWindowTitle("AEROPIC - COMET cleaner utility")
        # ADAPTED: Increased width to 450 to accommodate the longer User Manual text.
        self.setFixedWidth(450); self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        
        # Star reference loader: Needs a non-comet-aligned image (e.g. one star-aligned frame)
        # to find the real position of the stars before they were trailed.
        self.btn_load = QPushButton("⚠️ LOAD STAR REFERENCE (REQUIRED)")
        self.btn_load.setStyleSheet("background-color: #FF9800; color: black; font-weight: bold; height: 35px;")
        self.btn_load.clicked.connect(self.load_stars_ref); layout.addWidget(self.btn_load)
        
        # Trail Geometry: Define the width of the cleaning beam globally.
        self.sld_r, _ = self.add_sld("GLOBAL TRAIL RADIUS (px)", 1, 150, 20, layout)
        
        # Softness: Defines the transition zone between cleaned trail and original background.
        # Helps to avoid sharp 'cookie-cutter' edges in the final image.
        self.sld_soft, _ = self.add_sld("SOFTNESS / BLEND (%)", 0, 100, 50, layout)
        
        # Detection Sensitivity: Standard Sigma clipping approach on the reference image.
        self.sld_sens, self.lbl_sens = self.add_sld("STAR THRESHOLD (Sigma)", 5, 200, 30, layout)
        self.sld_sens.valueChanged.connect(self.update_star_count)
        
        # Overlays toggle: visual debugging of which stars will be processed.
        self.chk_show_stars = QCheckBox("Show stars centers"); self.chk_show_stars.setChecked(True); layout.addWidget(self.chk_show_stars)
        
        # Tool sizes & Zoom: 
        # Restore brush allows painting back the comet or important nebulosity.
        self.sld_mask_r, _ = self.add_sld("PRESERVE AREA BRUSH SIZE", 10, 800, 100, layout)
        self.sld_stretch, _ = self.add_sld("DISPLAY STRETCH", 1, 100, 80, layout)
        self.sld_z, self.lbl_z = self.add_sld("ZOOM (%)", 1, 150, 30, layout)

        # History Management: Allowing safe experimentation with radii and vectors.
        h_nav = QHBoxLayout()
        btn_u = QPushButton("⬅️ UNDO"); btn_u.clicked.connect(self.undo); h_nav.addWidget(btn_u)
        btn_r = QPushButton("REDO ➡️"); btn_r.clicked.connect(self.redo); h_nav.addWidget(btn_r)
        layout.addLayout(h_nav)

        # Action buttons: Main processing and Output.
        self.btn_run = QPushButton("🚀 RUN - CLEAN STAR TRAILS")
        self.btn_run.setEnabled(False); self.btn_run.setStyleSheet("background: #424242; color: #888; height: 50px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_catalog_clean); layout.addWidget(self.btn_run)
        
        self.btn_save = QPushButton("💾 SAVE TrailLess IMAGE"); self.btn_save.clicked.connect(self.save_fits); layout.addWidget(self.btn_save)

        # Hotkeys help: ADAPTED Manual tools description.
        help_frame = QFrame(); help_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        help_lay = QVBoxLayout(help_frame)
        help_lay.addWidget(QLabel("<b>MANUAL TOOLS:</b><br>"
                                  "CTRL + Click: Set P1 | SHIFT + Click: Set P2<br>"
                                  "ALT : draw preserve area ghost | ALT+Click: Apply<br>"
                                  "ALT + Scroll : change size of preserve area<br>"
                                  "ALT + left click : draw the final preserve area<br>"
                                  "Right-Click: Pan in zoomed image<br>"
                                  "Ctrl+Shift+Scroll: bright stars Ghost size | Ctrl+Shift+Click: Apply"))
        layout.addWidget(help_frame)

    def add_sld(self, txt, mi, ma, v, lay):
        """Helper to create labeled sliders."""
        h = QHBoxLayout(); lbl = QLabel(str(v/10 if "Sigma" in txt else v)); h.addWidget(QLabel(f"<b>{txt}</b>")); h.addStretch(); h.addWidget(lbl); lay.addLayout(h)
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(mi, ma); s.setValue(v); s.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        s.valueChanged.connect(lambda val: lbl.setText(str(val/10 if "Sigma" in txt else val))); lay.addWidget(s)
        return s, lbl

    def update_stats(self):
        """Calculate image statistics for the visual stretch (display only, does not affect data)."""
        sample = self.data[0][::4, ::4]; self.disp_min = np.percentile(sample, 2); self.disp_max = np.percentile(sample, 99.9)

    def load_stars_ref(self):
        """Load the star reference FITS to find center coordinates."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Ref FITS", "", "*.fit*")
        if path:
            with fits.open(path) as hdul: self.data_stars = hdul[0].data.astype(np.float32)
            if self.data_stars.ndim == 3: self.data_stars = self.data_stars[0]
            self.update_star_count()
            self.btn_run.setEnabled(True); self.btn_run.setStyleSheet("background: #1A237E; color: white; height: 50px; font-weight: bold;")

    def update_star_count(self):
        """Real-time peak detection on reference image based on Sigma slider."""
        if self.data_stars is None: return
        thresh = np.nanmean(self.data_stars) + (self.sld_sens.value()/10.0) * np.nanstd(self.data_stars)
        peaks = (self.data_stars > thresh) & (maximum_filter(self.data_stars, size=20) == self.data_stars)
        self.detected_coords = [tuple(c) for c in np.argwhere(peaks)]
        self.btn_load.setText(f"✅ {len(self.detected_coords)} STARS"); self.btn_load.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

    def run_catalog_clean(self):
        """Main Algorithm: Smart Background Interpolation along the trail vector."""
        if not self.p1 or not self.p2: 
            QMessageBox.warning(self, "Missing Vector", "Please define the trail vector first.")
            return
            
        # MATH THEORY: Build the coordinate system relative to the trail.
        # v_u = unit vector along the trail.
        # n_u = unit vector normal (perpendicular) to the trail.
        v_vec = np.array([self.p2[0]-self.p1[0], self.p2[1]-self.p1[1]])
        length = np.linalg.norm(v_vec); v_u, n_u = v_vec/length, np.array([-v_vec[1], v_vec[0]])/length
        glob_r, softness = self.sld_r.value(), self.sld_soft.value()/100.0
        total_stars = len(self.detected_coords)
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.history.append(self.data.copy()); self.redo_stack.clear()
            
            # ITERATIVE MASKING LOGIC:
            # 1. We create a global mask of all areas currently considered 'contaminated' (value 1).
            # 2. For each star, we check if the background sampling points (Left and Right) 
            #    fall in a 'clean' area (value 0).
            # 3. If a star is stuck between other trails, it's deferred to the next iteration.
            mask_global = np.zeros((self.h, self.w), dtype=np.uint8)
            for ry, rx in self.detected_coords:
                r_eff = self.special_radii.get((ry, rx), glob_r)
                cv2.line(mask_global, (int(rx), int(ry)), (int(rx+v_vec[1]), int(ry+v_vec[0])), 1, int(r_eff * 2.2))

            stars_to_process, iteration, force_mode = self.detected_coords.copy(), 0, False
            while stars_to_process and iteration < 100:
                iteration += 1; remaining_stars = []; something_cleaned = False
                for ry, rx in stars_to_process:
                    r_eff = self.special_radii.get((ry, rx), glob_r)
                    # Temporarily clear own trail from mask to allow sampling
                    cv2.line(mask_global, (int(rx), int(ry)), (int(rx+v_vec[1]), int(ry+v_vec[0])), 0, int(r_eff * 2.2))
                    
                    # Generate cleaning grid (Meshgrid for the trail rectangular area)
                    t_range, o_range = np.arange(-r_eff, length + r_eff), np.arange(-r_eff, r_eff + 1)
                    T, O = np.meshgrid(t_range, o_range); CY, CX = ry + T * v_u[0] + O * n_u[0], rx + T * v_u[1] + O * n_u[1]
                    m = (CY >= 0) & (CY < self.h - 1) & (CX >= 0) & (CX < self.w - 1)
                    iy, ix = CY[m].astype(int), CX[m].astype(int)
                    
                    # Compute sampling coordinates (Left 'gx' and Right 'dx' relative to trail center)
                    gy, gx = CY[m] + (r_eff + 4) * n_u[0], CX[m] + (r_eff + 4) * n_u[1]
                    dy, dx = CY[m] - (r_eff + 4) * n_u[0], CX[m] - (r_eff + 4) * n_u[1]
                    
                    # Check background health
                    in_g = (gy >= 0) & (gy < self.h); in_d = (dy >= 0) & (dy < self.h)
                    clean_g = np.zeros_like(in_g, dtype=bool); clean_d = np.zeros_like(in_d, dtype=bool)
                    if np.any(in_g): clean_g[in_g] = (map_coordinates(mask_global, [gy[in_g], gx[in_g]], order=0) == 0)
                    if np.any(in_d): clean_d[in_d] = (map_coordinates(mask_global, [dy[in_d], dx[in_d]], order=0) == 0)

                    # STRATEGY: If both sides are contaminated by other trails, we wait.
                    # Unless 'force_mode' is True (last resort).
                    if not np.all(clean_g | clean_d) and not force_mode:
                        cv2.line(mask_global, (int(rx), int(ry)), (int(rx+v_vec[1]), int(ry+v_vec[0])), 1, int(r_eff * 2.2))
                        remaining_stars.append((ry, rx)); continue
                    
                    # Calculate Weighting Mask (Soft edges)
                    w_m = np.clip(np.where(np.abs(O[m])/r_eff < (1-softness), 1.0, (1.0 - np.abs(O[m])/r_eff) / (softness + 1e-6)), 0, 1)
                    for i in range(self.c):
                        # Interpolate background values from clean sides
                        vg = map_coordinates(self.data[i], [np.clip(gy, 0, self.h-1), np.clip(gx, 0, self.w-1)], order=1)
                        vd = map_coordinates(self.data[i], [np.clip(dy, 0, self.h-1), np.clip(gx, 0, self.w-1)], order=1)
                        val = np.zeros_like(vg)
                        both = clean_g & clean_d; only_g = clean_g & ~clean_d; only_d = clean_d & ~clean_g
                        val[both] = (vg[both] + vd[both]) / 2.0
                        val[only_g] = vg[only_g]; val[only_d] = vd[only_d]
                        neither = ~(clean_g | clean_d)
                        if np.any(neither): val[neither] = vg[neither] if np.sum(in_g) >= np.sum(in_d) else vd[neither]
                        # Final pixel replacement with alpha-blending
                        self.data[i, iy, ix] = self.data[i, iy, ix] * (1 - w_m) + val * w_m
                    something_cleaned = True
                
                self.siril.log(f"Iteration {iteration}: {len(remaining_stars)} stars left")
                if not something_cleaned:
                    if force_mode: break # Give up if no progress in force mode
                    force_mode = True # No clean neighbors? Try with whatever we have.
                else: force_mode = False
                stars_to_process = remaining_stars
            
            # Post-Processing: RE-APPLY RESTORE MASKS
            # This is crucial! Global cleaning might destroy the comet head. 
            # We re-paste the original pixels stored in 'restore_masks'.
            for ry_m, rx_m, r_m in self.restore_masks:
                Y, X = np.ogrid[:self.h, :self.w]; mask = (X - rx_m)**2 + (Y - ry_m)**2 <= r_m**2
                for i in range(self.c): self.data[i][mask] = self.original_data[i][mask]
            
            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Processing Complete", f"{total_stars - len(remaining_stars)} stars cleaned in {iteration} iterations.")
        finally:
            while QApplication.overrideCursor(): QApplication.restoreOverrideCursor()

    def setup_cv(self):
        """Configure OpenCV window and mouse callback for low-latency visual feedback."""
        cv2.namedWindow("AEROPIC View", cv2.WINDOW_NORMAL); cv2.setMouseCallback("AEROPIC View", self.on_mouse)
        self.timer = QTimer(); self.timer.timeout.connect(self.loop); self.timer.start(40)

    def on_mouse(self, event, x, y, flags, param):
        """User interaction handler."""
        self.mouse_pos = (x, y)
        self.is_alt_pressed = bool(flags & cv2.EVENT_FLAG_ALTKEY)
        self.is_ghost_mod_pressed = (flags & cv2.EVENT_FLAG_SHIFTKEY) and (flags & cv2.EVENT_FLAG_CTRLKEY)
        
        z = self.sld_z.value()/100.0
        # Convert screen mouse coords to image space coords
        rx, ry = int((x + self.offset[1])/z), self.h - int((y + self.offset[0])/z)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_ghost_mod_pressed and self.detected_coords:
                # Assign manual radius to the closest detected star
                dists = [np.sqrt((ry-c[0])**2 + (rx-c[1])**2) for c in self.detected_coords]
                idx = np.argmin(dists); self.special_radii[self.detected_coords[idx]] = self.ghost_radius
            elif self.is_alt_pressed:
                # Manual pixel restoration (Red circle tool)
                self.history.append(self.data.copy()); br = self.sld_mask_r.value()
                self.restore_masks.append((ry, rx, br))
                Y, X = np.ogrid[:self.h, :self.w]; mask = (X - rx)**2 + (Y - ry)**2 <= br**2
                for i in range(self.c): self.data[i][mask] = self.original_data[i][mask]
            elif flags & cv2.EVENT_FLAG_CTRLKEY: self.p1 = (ry, rx)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY: self.p2 = (ry, rx)
        elif event == cv2.EVENT_RBUTTONDOWN: self.pan_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
            if self.pan_start: 
                self.offset[1]-=(x-self.pan_start[0]); self.offset[0]-=(y-self.pan_start[1]); self.pan_start=(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.is_ghost_mod_pressed: 
                self.ghost_radius = np.clip(self.ghost_radius + (5 if flags > 0 else -5), 5, 300)
            elif self.is_alt_pressed: 
                self.sld_mask_r.setValue(np.clip(self.sld_mask_r.value() + (25 if flags > 0 else -25), 10, 800))

    def loop(self):
        """Visual loop: Render image with stretch, zoom and overlays."""
        img = np.transpose(self.data, (1, 2, 0))[:, :, ::-1] if self.c > 1 else self.data
        s_max = self.disp_min + (self.disp_max - self.disp_min) * ((101 - self.sld_stretch.value())/100.0)
        disp = (np.clip((np.flipud(img)-self.disp_min)/(s_max-self.disp_min), 0, 1)*255).astype(np.uint8)
        if disp.ndim == 2: disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        
        z = self.sld_z.value()/100.0; disp_z = cv2.resize(disp, None, fx=z, fy=z)
        view = np.zeros((self.vh, self.vw, 3), dtype=np.uint8)
        self.offset[0] = np.clip(self.offset[0], 0, max(0, disp_z.shape[0]-self.vh))
        self.offset[1] = np.clip(self.offset[1], 0, max(0, disp_z.shape[1]-self.vw))
        h_part, w_part = min(self.vh, disp_z.shape[0]), min(self.vw, disp_z.shape[1])
        view[:h_part, :w_part] = disp_z[self.offset[0]:self.offset[0]+h_part, self.offset[1]:self.offset[1]+w_part]
        
        # 1. DRAW Star centers & radii (Yellow/Orange)
        if self.chk_show_stars.isChecked() and self.data_stars is not None:
            glob_r = self.sld_r.value()
            for ry, rx in self.detected_coords:
                r_eff = self.special_radii.get((ry, rx), glob_r)
                color = (0, 165, 255) if (ry, rx) in self.special_radii else (0, 255, 255)
                cx, cy = int(rx*z-self.offset[1]), int((self.h-ry)*z-self.offset[0])
                if 0<=cx<self.vw and 0<=cy<self.vh: cv2.circle(view, (cx, cy), int(r_eff*z), color, 1)
        
        # 2. DRAW Static Restore Masks (Red)
        for ry_m, rx_m, r_m in self.restore_masks:
            cx, cy = int(rx_m*z-self.offset[1]), int((self.h-ry_m)*z-self.offset[0])
            if 0<=cx<self.vw and 0<=cy<self.vh: cv2.circle(view, (cx, cy), int(r_m*z), (0, 0, 255), 1)

        # 3. DRAW Dynamic Ghost Circles (Brushes)
        if self.is_alt_pressed: # Red Ghost
            cv2.circle(view, self.mouse_pos, int(self.sld_mask_r.value() * z), (0, 0, 255), 2)
        if self.is_ghost_mod_pressed: # Orange Ghost
            cv2.circle(view, self.mouse_pos, int(self.ghost_radius * z), (0, 165, 255), 2)

        # 4. DRAW Trail Vector (p1 -> p2)
        if self.p1:
            p1v = (int(self.p1[1]*z-self.offset[1]), int((self.h-self.p1[0])*z-self.offset[0]))
            if 0<=p1v[0]<self.vw and 0<=p1v[1]<self.vh: cv2.circle(view, p1v, 4, (255,0,0), -1)
            if self.p2:
                p2v = (int(self.p2[1]*z-self.offset[1]), int((self.h-self.p2[0])*z-self.offset[0]))
                if 0<=p2v[0]<self.vw and 0<=p2v[1]<self.vh: cv2.line(view, p1v, p2v, (0,255,0), 2)
        
        cv2.imshow("AEROPIC View", view); cv2.waitKey(1)

    def undo(self): 
        if self.history: self.redo_stack.append(self.data.copy()); self.data = self.history.pop()
    def redo(self):
        if self.redo_stack: self.history.append(self.data.copy()); self.data = self.redo_stack.pop()
        
    def save_fits(self):
        """Export result as FITS with original header preservation."""
        base, ext = os.path.splitext(self.current_file); out = f"{base}_TrailLess{ext}"
        fits.PrimaryHDU(data=self.data, header=self.header_fits).writeto(out, overwrite=True)
        self.siril.log(f"Image saved: {out}")
        QMessageBox.information(self, "Image Saved", f"File saved as:\n{out}")

if __name__ == "__main__":
    app = QApplication(sys.argv); win = AEROPIC_Master_Comet_EN(); win.show(); sys.exit(app.exec())
