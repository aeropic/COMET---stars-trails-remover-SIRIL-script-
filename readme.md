# AEROPIC - COMET stars trails remover

This Python-based utility is designed to work with **Siril**. It allows for the precise removal of star trails in comet-aligned stacked picture while preserving the natural noise grain of the background, preventing "flat" or "plastic" artifacts.

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
<img width="758" height="257" alt="install" src="https://github.com/user-attachments/assets/2e134417-d8c0-4fe8-bddf-52e59bb20aa6" />

---

## 🚀 User Manual

### 0. Loading the comet picture
the comet stacked image shall be opened in SIRIL
The script will use it when opening the script from the **Scripts menu**

### 1. Loading the Stars Reference
You **must** provide a star reference image. (Orange warning)
<img width="422" height="672" alt="PPqO4F5PR0 pngload" src="https://github.com/user-attachments/assets/f92f4a47-3112-4b35-8966-706973b2cb37" />



* **Optimal Reference:** Use a "Star-only" mask obtained from the stacked on stars picture (starmask.Fit). A normal starmask should work, but, if needed you can  slightly stretch a starmask to make  the stars bearely visible. You may also use a raw image saved in fit or tiff (not tested yet)
* The **Load Button** becomes green and shows the total amount of detected stars (e.g., `✅ 26 STARS DETECTED`).
* <img width="422" height="517" alt="stars" src="https://github.com/user-attachments/assets/77264139-4c0e-4411-93e5-886e1126caf4" />

* if the **show stars centers** is ticked, yellow circles are drawn in the picture. THey reflect the actual position of the **star trail radius** slider

* play with the **sigma slider** to dynamically increase (slider to the left)/decrease (to the right) the star count to get a reasonable amount (e.g., `✅ 117 STARS DETECTED`). This is the number of trails that will be erased in the picture, don't target too high numbers !
When you start seeing clusters of stars around the comet nucleus, this means you are at the limit of stars detection noise...
<img width="1081" height="792" alt="cluster" src="https://github.com/user-attachments/assets/eae85c10-91d3-462e-941b-ea86fb548798" />


### 2. Defining the Trail (The Vector)
* **DISPLAY STRETCH**: if you do not see the trails enough, play with the display stretch slider. It is just used for the display!
* **Ctrl + Left Click**: Set the start of a star trail (Blue point).
* **Shift + Left Click**: Set the end of the same star trail (Green line).
* This defines the **length and angle** applied to all detected stars.
* start and end points can be modified if needed (just repeat the Ctrl + click or shift+click)
<img width="1148" height="695" alt="trail" src="https://github.com/user-attachments/assets/6f551c97-7759-4d7d-92d4-825035eb6853" />

### 3. Local Protection (Comet Mask)
* **Alt + Left Click**: Draw a "Restore" mask over the aeras you want to keep unchanged (eg: comet's head or tail).
* **Mouse Wheel**: Change the brush size.
* Red circles indicate areas that will be restored from original data, keeping the area intact.
<img width="1701" height="895" alt="before" src="https://github.com/user-attachments/assets/05463103-13dd-412d-a867-8b09e9631a85" />

### 4. run the script
* **RUN**:this will run the sript and erase the trails. It may last up to one minute. At the end you get this
<img width="422" height="517" alt="3hfd6zy5kD pngclean-finish" src="https://github.com/user-attachments/assets/6499a414-2108-4830-a231-31553d5b9d22" />

* press **OK** and the display is updated. See the updated image here after

* note 1:  it is streched, when displayed in normal conditions, 90% of the trails are gone!
  <img width="1204" height="774" alt="final_no_strech" src="https://github.com/user-attachments/assets/95cebbb6-257a-48b2-9ddf-5c7dec482daf" />

* note 2: some black artifacts are left by the algorithm on the edges of the picture. keep the picture as is, then add the stars back, then crop the edges

<img width="1164" height="816" alt="ZLlpq2dzE9 pngbrush30" src="https://github.com/user-attachments/assets/55c90858-a6b1-4cd7-a683-5ebc5589fc03" />


* **UNDO / REDO**: does what they say :-)
* **UNDO / REDO**: does what they say :-)
* **CLEAN RADIUS (px)**: play with this if you see that some trails are still not completely cleaned. In the final image I used a 30px cleaning brush.
* **SOFTNESS / BLEND %**: increases§decreases the blending effect. The higher the SOFTNESSslider , the longer the transition on the edges. With a RADIUS of 20 and a SOFTNESS of 50%,  10 central pixels are fully replaced, and 2x5 pixels on teh edges are progressively blended.
* **SAVE trailless image**: when you're happy with the final result. The image is saved with the same name postfixed with "**_TrailLess**"
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














