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
<img width="452" height="681" alt="start" src="https://github.com/user-attachments/assets/464f5168-4283-44e8-99bd-e602cd7bf53b" />
(This is the actual layout of the control windows. Some screenshots here after may reflect an older version)




* **Optimal Reference:** Use a "Star-only" mask obtained from the stacked on stars picture (starmask.Fit). A normal starmask should work, but, if needed you can  slightly stretch a starmask to make  the stars bearely visible. You may also use a raw image saved in fit or tiff (not tested yet)
Note that **the stack reference shall be either the first comet image or the last one** So that the star is either at the beginning or at the end of the trail.
* The **Load Button** becomes green and shows the total amount of detected stars (e.g., `✅ 26 STARS DETECTED`).
* <img width="422" height="517" alt="stars" src="https://github.com/user-attachments/assets/77264139-4c0e-4411-93e5-886e1126caf4" />

* if the **show stars centers** is ticked, yellow circles are drawn in the picture. THey reflect the actual position of the **star trail radius** slider

* play with the **sigma slider** to dynamically increase (slider to the left)/decrease (to the right) the star count to get a reasonable amount (e.g., `✅ 117 STARS DETECTED`). This is the number of trails that will be erased in the picture, don't target too high numbers !
When you start seeing clusters of stars around the comet nucleus, this means you are at the limit of stars detection noise...
<img width="1081" height="792" alt="cluster" src="https://github.com/user-attachments/assets/eae85c10-91d3-462e-941b-ea86fb548798" />


### 2. Defining the Trail (The Vector)
* **DISPLAY STRETCH**: if you do not see the trails enough, play with the display stretch slider. It is just used for the display!
* **Ctrl + Left Click**: Set the start of a star trail (P1 : Blue point). It shall be placed at the center of one yellow circle
  <img width="494" height="443" alt="blue" src="https://github.com/user-attachments/assets/31524142-9d7f-41fd-81a2-490e5c791710" />

* **Shift + Left Click**: Set the end of the same star trail (P2 : Green line). Keep some margin and make is a bit longer than the trail...
* This defines the **length and angle** applied to all detected stars.
* start and end points can be modified if needed (just repeat the Ctrl + click or shift+click)
<img width="1148" height="695" alt="trail" src="https://github.com/user-attachments/assets/6f551c97-7759-4d7d-92d4-825035eb6853" />

* **GLOBAL TRAIL RADIUS (px)**: play with this if you see that some trails are still not completely cleaned. This value applies to all stars but the bright ones.
In the final image I used a 30px cleaning brush.

### 3. Local Protection (Comet Mask)
* **Alt + Left Click**: Draw a "Restore" mask over the aeras you want to keep unchanged (eg: comet's head or tail).
* **Mouse Wheel**: Change the brush size.
* Red circles indicate areas that will be restored from original data, keeping the area intact.
<img width="1142" height="720" alt="before_OK" src="https://github.com/user-attachments/assets/48ae61fc-bd96-4ec5-8ae6-1934efe8150c" />

### 4. manage bright strars (large trails)
* bright stars generate large trails that are larger than the standard 20px or so trail width. You can deposit a larger circle to tell the script there is here a large trail please erase it with a larger path...
* **Shift + control and drag**: Draw an orange circle you have to drag over the yellow circle of a bright star.
* **Mouse Wheel**: Change the brush size.
* **Shift + control + left click** an orange circle is drawn, its diameter will be used for this star to compute the trail.
<img width="332" height="386" alt="orange" src="https://github.com/user-attachments/assets/5a1b9d86-11ab-436e-b8b8-71a93c7cbe4b" />


### 5. run the script
* **RUN**:this will run the sript and erase the trails. It may last up to one minute. At the end you get this
<img width="422" height="517" alt="3hfd6zy5kD pngclean-finish" src="https://github.com/user-attachments/assets/6499a414-2108-4830-a231-31553d5b9d22" />
<img width="1194" height="840" alt="comet_before" src="https://github.com/user-attachments/assets/ba7466d8-97e4-4c44-96e7-78a54a4bfc86" />

* press **OK** and the display is updated. See the updated image here after
<img width="1069" height="734" alt="after_ok" src="https://github.com/user-attachments/assets/87f63618-3994-4a39-8ec0-92feaad2dc4f" />
<img width="1194" height="840" alt="comet_done" src="https://github.com/user-attachments/assets/f8b8f561-6e87-4ad2-bcdc-2195fa958bf0" />

* during the run, some logs are displayed inside the SIRIL's log window. It should finish with 0 stars left. If not please contact me :-)
<img width="435" height="277" alt="log" src="https://github.com/user-attachments/assets/1bbad3a5-c0f6-4ef8-a5d1-8405498720bd" />


* note 1:  it is streched, when displayed in normal conditions, 95% of the trails are gone! Only the really bright stars with diffraction spikes are not corrected by the script see the red arrows (the spikes are shifting during comet stacking making a wide trail...) 
  <img width="1189" height="812" alt="comet_final" src="https://github.com/user-attachments/assets/4bda23b7-e1b0-479a-8aea-81ac1e0e4437" />


* note 2: I did my best to manage sides of the picture but some black artifacts may be left by the algorithm on the edges of the picture. keep the picture as is, then add the stars back, then crop the edges!

* **UNDO / REDO**: do what they say :-)
* **SOFTNESS / BLEND %**: increases§decreases the blending effect. The higher the SOFTNESSslider , the longer the transition on the edges. With a RADIUS of 20 and a SOFTNESS of 50%,  10 central pixels are fully replaced, and 2x5 pixels on teh edges are progressively blended.
* **SAVE trailless image**: when you're happy with the final result. The image is saved with the same name postfixed with "**_TrailLess**"
  <img width="488" height="172" alt="saved" src="https://github.com/user-attachments/assets/c9c84723-39c2-49ec-9788-ec6226b3a5f0" />

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























