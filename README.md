# Braille Device — Sesli Konuşmayı Braille'e Çeviren Taşınabilir Cihaz

---

## 🇹🇷 Türkçe

### Proje Açıklaması
Bu proje, görme ve işitme engelli bireyler için çevrimdışı çalışan, konuşmayı gerçek zamanlı
Braille'e dönüştüren taşınabilir bir cihaz yazılımıdır. Raspberry Pi üzerinde çalışacak şekilde
tasarlanmış olup internet bağlantısı gerektirmez.

**Özellikler:**
- 🎙️ Gerçek zamanlı ses tanıma (VOSK — çevrimdışı)
- 🔇 RMS tabanlı gürültü kapısı ve bandpass filtresi (300–3400 Hz)
- 📝 Son 10 kelimeyi tutan kayan kelime tamponu
- ⌨️ İngilizce Braille alfabesi görsel ekranı (Tkinter)
- 🔑 Anahtar kelime tespiti ("fire", "help", "danger" vb.)
- 📳 Haptic motor geri bildirimi (Raspberry Pi GPIO + masaüstü simülasyonu)

### Kurulum

1. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. VOSK İngilizce modelini indirin ve proje klasörüne çıkarın:
   - İndirme linki: https://alphacephei.com/vosk/models
   - Kullanılan model: `vosk-model-small-en-us-0.15`
   - Model klasörü `main.py` ile aynı dizinde olmalıdır.

### Çalıştırma

```bash
python main.py
```

### Raspberry Pi Notları
- `scipy` yoksa bandpass filtresi numpy FFT ile otomatik olarak simüle edilir.
- `RPi.GPIO` yoksa haptic motor konsol çıktısı ile simüle edilir.
- GPIO pin 18 (BCM) haptic motor için kullanılmaktadır.

---

## 🇬🇧 English

### Project Description
This project is a portable, fully offline assistive device that converts live speech into
real-time Braille output for deafblind individuals. It is designed to run on a Raspberry Pi
without any internet connection.

**Features:**
- 🎙️ Real-time speech recognition (VOSK — offline)
- 🔇 RMS-based noise gate and bandpass filter (300–3400 Hz)
- 📝 Rolling 10-word memory buffer
- ⌨️ English Braille alphabet visual display (Tkinter)
- 🔑 Keyword detection ("fire", "help", "danger", etc.)
- 📳 Haptic motor feedback (Raspberry Pi GPIO + desktop simulation fallback)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the VOSK English model and extract it to the project directory:
   - Download link: https://alphacephei.com/vosk/models
   - Model used: `vosk-model-small-en-us-0.15`
   - The model folder must be in the same directory as `main.py`.

### Running

```bash
python main.py
```

### Raspberry Pi Notes
- If `scipy` is not available, the bandpass filter is automatically simulated using numpy FFT.
- If `RPi.GPIO` is not available, haptic motor output is simulated via console messages.
- GPIO pin 18 (BCM) is used for the haptic motor.