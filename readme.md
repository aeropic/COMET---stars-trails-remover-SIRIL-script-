# AEROPIC - COMET stars trails remover

This Python-based utility is designed to work seamlessly with **Siril** via `sirilpy`. It allows for the precise removal of star trails in comet-aligned stacked picture while preserving the natural noise grain of the background, preventing "flat" or "plastic" artifacts.

---

## 🛠 Installation in Siril

To have **AEROPIC - COMET stars trails remover** appear directly in your Siril top menu, follow these steps:

1.  **Requirements**: 
    * Ensure **Siril 1.2.0+** is installed on your system.
    * Your Python environment must have `sirilpy` installed: this is done by default install of SIRIL.
2.  **Create a Folder**: Create a folder anywhere on your computer (e.g., named `Aeropic`).
3.  **Add the Script**: Place the `AEROPIC_comet_cleaner_utility.py` file inside this folder.
4.  **Configure Siril**:
    * Open **Siril** and go to the hamburger menu **Preferences**.
    * Navigate to the **Scripts** tab.
    * Add or paste the path to your `Aeropic` folder (link to the **folder**, not the script file itself). (eg : C:\Users\ALAIN\AppData\Local\siril-scripts\Aeropic)
5.  **Restart/Refresh**: After clicking **Apply**, a new entry will appear in your **Scripts menu** containing the tool.

---

## 🚀 User Manual

### 0. Loading the comet picture
the comet stacked image shall be opened in SIRIL
The script will use it when opening the script from the **Scripts menu**

### 1. Loading the Stars Reference
You **must** provide a star reference image.
* **Optimal Reference:** Use a "Star-only" mask or a slightly stretched version of your stack where stars are clearly defined.
* The **Load Button** dynamically updates the star count based on your **Sigma** setting (e.g., `✅ 452 STARS DETECTED`).

### 2. Defining the Trail (The Vector)
* **Ctrl + Left Click**: Set the start of a star trail (Blue point).
* **Shift + Left Click**: Set the end of the same star trail (Green line).
* This defines the **length and angle** applied to all detected stars.

### 3. Local Protection (Comet Mask)
* **Alt + Left Click**: Draw a "Restore" mask over the comet's head or tail.
* **Mouse Wheel**: Change the brush size.
* Red circles indicate areas that will be restored from original data, keeping the comet intact.

---

## ⌨️ Shortcuts & Navigation

| Key | Action |
| :--- | :--- |
| **Right Click + Drag** | Pan / Move inside the zoomed image |
| **Zoom Slider** | Zoom level |
| **Ctrl + Z** | Undo last data modification |
| **Z (Key)** | Remove the last restoration mask |
| **C (Key)** | Clear all restoration masks |
| **Alt (Hold)** | Shows the brush size preview |


---

