### 1. Clone the repository

```bash
git clone https://github.com/swapnilbhadra2004/water-monitoring-site.git
cd water-monitoring-site
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

(OR manually)

```bash
pip install streamlit pyserial pandas numpy scikit-learn
```

---

### 3. Connect hardware

* Connect your microcontroller (Arduino / ESP, etc.)
* Note the correct COM port (e.g., COM3, COM12 or /dev/ttyUSB0)

---

### 4. Run the Streamlit app

```bash
streamlit run IoT.py
```

---

### 5. Configure inside the dashboard

* Enter/select the correct COM port
* Set baud rate (same as your microcontroller code)
* Click **Connect**

---

### ⚠️ Important Notes

* Ensure your device is sending data in this format:

  ```
  time,tds_value,status
  ```

  Example:

  ```
  12,250,OK
  ```
* Close Arduino Serial Monitor before running the app (port conflict issue)
* App runs locally (hardware required; won’t work on cloud hosting)

---
