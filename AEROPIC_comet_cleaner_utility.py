import sys, os, time
import numpy as np
import cv2
from astropy.io import fits
from scipy.ndimage import map_coordinates, maximum_filter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QSlider, QPushButton, QMessageBox, QHBoxLayout, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
import sirilpy as s
from sirilpy import SirilConnectionError

class AEROPIC_Master_Comet_EN(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
            print("Connected successfully!")
        except SirilConnectionError as e:
            print(f"Connection failed: {e}")
            QMessageBox.critical(self, "Siril Error", f"Connection failed: {e}")
            sys.exit(1)

        self.current_file = self.siril.get_image_filename()
        raw = self.siril.get_image_pixeldata()
        self.data = raw.astype(np.float32)
        self.original_data = self.data.copy()
        self.h, self.w = self.data.shape[-2:]
        self.c = self.data.shape[0] if self.data.ndim == 3 else 1
        
        self.header = fits.Header()
        if self.current_file.lower().endswith(('.fit', '.fits', '.fz')):
            try:
                with fits.open(self.current_file) as hdul: self.header = hdul[0].header
            except: pass
        
        self.data_stars = None
        self.history, self.redo_stack = [], []
        self.offset, self.pan_start = [0, 0], None
        self.p1, self.p2 = None, None 
        self.masks = [] 
        self.mouse_pos = (0, 0)
        self.disp_min, self.disp_max = 0, 1.0
        
        self.init_ui()
        self.update_stats()
        self.setup_cv()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def update_stats(self):
        sample = self.data[0][::4, ::4]
        self.disp_min = np.percentile(sample, 2); self.disp_max = np.percentile(sample, 99.9)
        self.sld_stretch.setValue(80) 

    def init_ui(self):
        self.setWindowTitle("AEROPIC - COMET MASTER PRO")
        self.setFixedWidth(420); self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        
        self.btn_load = QPushButton("⚠️ LOAD STAR REFERENCE (REQUIRED)")
        self.btn_load.setStyleSheet("background-color: #FF9800; color: black; font-weight: bold; height: 35px;")
        self.btn_load.clicked.connect(self.load_stars_ref); layout.addWidget(self.btn_load)
        
        self.sld_r, _ = self.add_sld("CLEAN RADIUS (px)", 1, 100, 20, layout)
        self.sld_soft, _ = self.add_sld("SOFTNESS / BLEND (%)", 0, 100, 50, layout)
        
        # Le curseur Sigma déclenche maintenant update_star_count
        self.sld_sens, self.lbl_sens = self.add_sld("STAR THRESHOLD (Sigma)", 5, 200, 30, layout)
        self.sld_sens.valueChanged.connect(self.update_star_count)
        
        self.sld_mask_r, self.lbl_mask_r = self.add_sld("RESTORE BRUSH SIZE", 10, 800, 100, layout)
        self.sld_stretch, _ = self.add_sld("DISPLAY STRETCH", 1, 100, 80, layout)
        self.sld_z, self.lbl_z = self.add_sld("ZOOM (%)", 1, 150, 30, layout)
        self.sld_z.valueChanged.connect(lambda v: self.lbl_z.setText(f"{v}%"))

        h_nav = QHBoxLayout()
        btn_undo = QPushButton("⬅️ UNDO DATA"); btn_undo.clicked.connect(self.undo); h_nav.addWidget(btn_undo)
        btn_redo = QPushButton("REDO DATA ➡️"); btn_redo.clicked.connect(self.redo); h_nav.addWidget(btn_redo)
        layout.addLayout(h_nav)

        self.btn_run = QPushButton("🚀 RUN - CLEAN STAR TRAILS")
        self.btn_run.setEnabled(False); self.btn_run.setStyleSheet("background: #424242; color: #888; height: 50px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_catalog_clean); layout.addWidget(self.btn_run)
        
        self.btn_save = QPushButton("💾 SAVE TrailLess IMAGE")
        self.btn_save.clicked.connect(self.save_fits); layout.addWidget(self.btn_save)

    def add_sld(self, txt, mi, ma, v, lay):
        h_lay = QHBoxLayout(); lbl_val = QLabel(str(v))
        h_lay.addWidget(QLabel(f"<b>{txt}</b>")); h_lay.addStretch(); h_lay.addWidget(lbl_val); lay.addLayout(h_lay)
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(mi, ma); s.setValue(v); s.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        s.valueChanged.connect(lambda val: lbl_val.setText(str(val/10 if "Sigma" in txt else val))); lay.addWidget(s)
        return s, lbl_val

    def load_stars_ref(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Star Reference", "", "*.fit *.fits *.tif *.tiff")
        if path:
            if path.lower().endswith(('.tif', '.tiff')):
                data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                data = data.astype(np.float32) / (65535.0 if data.dtype == np.uint16 else 255.0)
                self.data_stars = data.T if data.ndim == 2 else data.mean(axis=2).T
            else:
                with fits.open(path) as hdul: self.data_stars = hdul[0].data.astype(np.float32)
                if self.data_stars.ndim == 3: self.data_stars = self.data_stars[0]
            
            self.update_star_count()
            self.btn_run.setEnabled(True); self.btn_run.setStyleSheet("background: #1A237E; color: white; height: 50px; font-weight: bold;")

    def update_star_count(self):
        if self.data_stars is None: return
        # Calcul du seuil et détection rapide des pics
        thresh = np.nanmean(self.data_stars) + (self.sld_sens.value() / 10.0) * np.nanstd(self.data_stars)
        peaks = (self.data_stars > thresh) & (maximum_filter(self.data_stars, size=20) == self.data_stars)
        count = np.sum(peaks)
        self.btn_load.setText(f"✅ {count} STARS DETECTED"); self.btn_load.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

    def setup_cv(self):
        cv2.namedWindow("AEROPIC View", cv2.WINDOW_NORMAL); cv2.setMouseCallback("AEROPIC View", self.on_mouse)
        self.timer = QTimer(); self.timer.timeout.connect(self.loop); self.timer.start(40)

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y); z = self.get_zoom_factor()
        rx, ry = int((x + self.offset[1]) / z), self.h - int((y + self.offset[0]) / z) 
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_ALTKEY: self.masks.append([ry, rx, self.sld_mask_r.value()])
            elif flags & cv2.EVENT_FLAG_CTRLKEY: self.p1 = (ry, rx)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY: self.p2 = (ry, rx)
        elif event == cv2.EVENT_RBUTTONDOWN: self.pan_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
            if self.pan_start:
                self.offset[1] -= (x - self.pan_start[0]); self.offset[0] -= (y - self.pan_start[1])
                self.pan_start = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 25 if flags > 0 else -25
            self.sld_mask_r.setValue(np.clip(self.sld_mask_r.value() + delta, 10, 800))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Z:
            if e.modifiers() & Qt.KeyboardModifier.ControlModifier: self.undo()
            elif self.masks: self.masks.pop()
        elif e.key() == Qt.Key.Key_C: self.masks = []

    def get_zoom_factor(self): return max(0.01, self.sld_z.value() / 100.0)

    def clean_trail(self, yd, xd, ref_src):
        v = np.array([self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]])
        length = np.linalg.norm(v); v_u, n_u = v/length, np.array([-v[1], v[0]])/length
        radius, softness = self.sld_r.value(), self.sld_soft.value() / 100.0
        t_range, o_range = np.arange(-length/2, length/2), np.arange(-radius, radius + 1)
        T, O = np.meshgrid(t_range, o_range)
        CY, CX = yd + T * v_u[0] + O * n_u[0], xd + T * v_u[1] + O * n_u[1]
        dist_norm = np.abs(O) / radius
        weight = np.clip(np.where(dist_norm < (1-softness), 1.0, (1.0 - dist_norm) / (softness + 1e-6)), 0, 1)
        mask = (CY >= 0) & (CY < self.h - 1) & (CX >= 0) & (CX < self.w - 1)
        iy, ix, w_m = CY[mask].astype(int), CX[mask].astype(int), weight[mask]
        for i in range(self.c):
            v1 = map_coordinates(ref_src[i], [CY[mask] + (radius+3) * n_u[0], CX[mask] + (radius+3) * n_u[1]], order=1)
            v2 = map_coordinates(ref_src[i], [CY[mask] - (radius+3) * n_u[0], CX[mask] - (radius+3) * n_u[1]], order=1)
            self.data[i, iy, ix] = self.data[i, iy, ix] * (1 - w_m) + ((v1 + v2) / 2.0) * w_m

    def run_catalog_clean(self):
        if not self.p1 or not self.p2: return
        self.history.append(self.data.copy()); self.redo_stack.clear()
        noise_ref = self.data.copy()
        thresh = np.nanmean(self.data_stars) + (self.sld_sens.value() / 10.0) * np.nanstd(self.data_stars)
        peaks = (self.data_stars > thresh) & (maximum_filter(self.data_stars, size=20) == self.data_stars)
        coords = np.argwhere(peaks)
        for ry, rx in coords: self.clean_trail(ry, rx, noise_ref)
        if self.masks:
            yy, xx = np.indices((self.h, self.w)); full_m = np.zeros((self.h, self.w), dtype=np.float32)
            for my, mx, mr in self.masks:
                dist = np.sqrt((yy - my)**2 + (xx - mx)**2)
                full_m = np.maximum(full_m, np.clip((mr - dist) / 5.0, 0, 1))
            for i in range(self.c): 
                self.data[i] = self.data[i] * (1 - full_m) + self.original_data[i] * full_m
        QMessageBox.information(self, "Done", f"{len(coords)} stars cleaned.")

    def loop(self):
        img = np.transpose(self.data, (1, 2, 0))[:, :, ::-1] if self.c > 1 else self.data
        s_max = self.disp_min + (self.disp_max - self.disp_min) * ((101 - self.sld_stretch.value()) / 100.0)
        disp = (np.clip((np.flipud(img)-self.disp_min)/(s_max-self.disp_min), 0, 1)*255).astype(np.uint8)
        if disp.ndim == 2: disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        z = self.get_zoom_factor(); disp_z = cv2.resize(disp, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
        vh, vw = 900, 1400; self.offset = [np.clip(self.offset[0], 0, max(0, disp_z.shape[0]-vh)), np.clip(self.offset[1], 0, max(0, disp_z.shape[1]-vw))]
        view = np.ascontiguousarray(disp_z[self.offset[0]:self.offset[0]+vh, self.offset[1]:self.offset[1]+vw])
        for my, mx, mr in self.masks:
            cv2.circle(view, (int(mx*z-self.offset[1]), int((self.h-my)*z-self.offset[0])), int(mr*z), (0, 0, 255), 2)
        if QApplication.queryKeyboardModifiers() & Qt.KeyboardModifier.AltModifier:
            cv2.circle(view, self.mouse_pos, int(self.sld_mask_r.value()*z), (255, 255, 255), 1, cv2.LINE_AA)
        if self.p1:
            p1v = (int(self.p1[1]*z-self.offset[1]), int((self.h-self.p1[0])*z-self.offset[0]))
            cv2.circle(view, p1v, 4, (255, 0, 0), -1)
            if self.p2: cv2.line(view, p1v, (int(self.p2[1]*z-self.offset[1]), int((self.h-self.p2[0])*z-self.offset[0])), (0, 255, 0), 2)
        cv2.imshow("AEROPIC View", view); cv2.waitKey(1)

    def undo(self): 
        if self.history: self.redo_stack.append(self.data.copy()); self.data = self.history.pop()
    def redo(self):
        if self.redo_stack: self.history.append(self.data.copy()); self.data = self.redo_stack.pop()
    def save_fits(self):
        base, ext = os.path.splitext(self.current_file); out = f"{base}_TrailLess{ext}"
        if ext.lower() in ['.tif', '.tiff']:
            save_data = (np.transpose(self.data, (1, 2, 0)) * 65535.0).astype(np.uint16) if self.c > 1 else (self.data * 65535.0).astype(np.uint16)
            cv2.imwrite(out, save_data)
        else: fits.PrimaryHDU(data=self.data, header=self.header).writeto(out, overwrite=True)
        QMessageBox.information(self, "Saved", f"File saved:\n{out}")

if __name__ == "__main__":
    app = QApplication(sys.argv); win = AEROPIC_Master_Comet_EN(); win.show(); sys.exit(app.exec())