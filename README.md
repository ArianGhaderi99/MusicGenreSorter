# 🎵 MusicGenreSorter

A deep learning project that classifies music by genre from raw audio and automatically sorts audio files into genre-named folders.

---

## English

### Overview
**MusicGenreSorter** is a Convolutional Neural Network (CNN) based audio classification project. It listens to a song, predicts its genre, and automatically copies the file into a folder named after that genre — turning a messy music collection into an organized one.

This project was built as a personal portfolio piece to demonstrate an end-to-end deep learning workflow: data preparation, feature extraction, model training, evaluation, and a practical real-world application.

### Features
- Classifies audio into **10 genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- Trained on the classic **GTZAN** genre dataset
- Uses **Mel-Spectrograms** as input features (a visual/frequency representation of audio)
- Lightweight **CNN architecture** with Batch Normalization and Dropout to prevent overfitting
- Automatically creates a folder per predicted genre and sorts files into it
- Achieves **75% accuracy** on the held-out test set

### How It Works
1. **Feature Extraction** — Each audio file is converted into a Mel-Spectrogram using `librosa`.
2. **Normalization** — Features are normalized (zero mean, unit variance) for stable training.
3. **Model Training** — A CNN (3 convolutional blocks + Batch Normalization + Global Average Pooling + Dense layers) is trained on Google Colab (GPU) with Early Stopping to avoid overfitting.
4. **Prediction** — For a new, unseen song, the same preprocessing pipeline is applied, and the trained model predicts the most likely genre.
5. **Sorting** — The song is automatically copied into a folder named after its predicted genre.

### Model Architecture
```
Input (128 x time x 1) — Mel-Spectrogram
→ Conv2D(16) → BatchNorm → MaxPooling
→ Conv2D(32) → BatchNorm → MaxPooling
→ Conv2D(64) → BatchNorm → MaxPooling
→ GlobalAveragePooling2D
→ Dense(64) → Dropout(0.5)
→ Dense(10, softmax)
```
Total trainable parameters: ~28,000 (a compact model, chosen deliberately to fit a dataset of ~1,000 samples without overfitting).

### Dataset
- **GTZAN Genre Collection** — 1,000 audio tracks (30 seconds each), 100 tracks per genre, 10 genres.
- Note: The dataset itself is not included in this repository due to its size. You can download it from Kaggle: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

### Tech Stack
- Python
- TensorFlow / Keras
- librosa (audio processing)
- scikit-learn
- Google Colab (GPU training)

### Project Structure
```
MusicGenreSorter/
│
├── genre_model.keras       # Trained CNN model
├── label_encoder.pkl       # Encodes genre names <-> numeric labels
├── norm_params.pkl         # Mean/std used for normalization + padding length
├── MusicGenreSorter.ipynb  # Full notebook: training + prediction + sorting
└── README.md
```

### How to Use
1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install tensorflow librosa scikit-learn numpy
   ```
3. Load the trained model and helper files (`genre_model.keras`, `label_encoder.pkl`, `norm_params.pkl`).
4. Run the classification + sorting function on any audio file:
   ```python
   classify_and_sort('path/to/your/song.mp3')
   ```
5. The song will be automatically copied into a folder named after its predicted genre (e.g. `sorted_songs/jazz/`).

### Results
- Test accuracy: **75%**
- The model performs very well on tracks similar to the GTZAN dataset, and reasonably on external tracks — with room for improvement via data augmentation or a larger dataset.

### Future Improvements
- Data augmentation (pitch shift, time stretch, noise injection) to improve generalization
- Transfer learning with a pretrained audio model


---

## فارسی

### معرفی پروژه
**MusicGenreSorter** یک پروژه‌ی دسته‌بندی صوتی مبتنی بر شبکه‌ی عصبی کانولوشنی (CNN) است. این پروژه به یک آهنگ گوش می‌دهد، ژانر آن را تشخیص می‌دهد و به‌طور خودکار فایل را در پوشه‌ای با نام همان ژانر کپی می‌کند — یعنی یک مجموعه‌ی موسیقی درهم را تبدیل به یک مجموعه‌ی مرتب می‌کند.

این پروژه به‌عنوان یک نمونه‌کار شخصی ساخته شده تا یک فرآیند کامل یادگیری عمیق را نشان دهد: آماده‌سازی داده، استخراج ویژگی، آموزش مدل، ارزیابی و یک کاربرد واقعی و عملی.

### امکانات
- دسته‌بندی صدا به **۱۰ ژانر**: بلوز، کلاسیک، کانتری، دیسکو، هیپ‌هاپ، جز، متال، پاپ، رگی، راک
- آموزش‌دیده روی دیتاست معروف **GTZAN**
- استفاده از **Mel-Spectrogram** به‌عنوان ویژگی ورودی (یک بازنمایی فرکانسی/تصویری از صدا)
- معماری **CNN سبک** همراه با Batch Normalization و Dropout برای جلوگیری از Overfitting
- ساخت خودکار پوشه برای هر ژانر پیش‌بینی‌شده و انتقال فایل به آن
- دقت **۷۵٪** روی داده‌ی تست

### نحوه‌ی عملکرد
۱. **استخراج ویژگی** — هر فایل صوتی با کتابخانه‌ی `librosa` به یک Mel-Spectrogram تبدیل می‌شود.
۲. **نرمالایز کردن** — ویژگی‌ها نرمالایز می‌شوند (میانگین صفر، انحراف معیار یک) تا آموزش پایدارتر باشد.
۳. **آموزش مدل** — یک CNN (سه بلوک کانولوشنی + Batch Normalization + Global Average Pooling + لایه‌های Dense) روی Google Colab (با GPU) و با استفاده از Early Stopping آموزش داده شده تا از Overfitting جلوگیری شود.
۴. **پیش‌بینی** — برای یک آهنگ جدید، همان مراحل پیش‌پردازش اعمال می‌شود و مدل آموزش‌دیده محتمل‌ترین ژانر را پیش‌بینی می‌کند.
۵. **مرتب‌سازی** — آهنگ به‌طور خودکار در پوشه‌ای با نام ژانر پیش‌بینی‌شده کپی می‌شود.

### معماری مدل
```
ورودی (۱۲۸ × زمان × ۱) — Mel-Spectrogram
→ Conv2D(16) → BatchNorm → MaxPooling
→ Conv2D(32) → BatchNorm → MaxPooling
→ Conv2D(64) → BatchNorm → MaxPooling
→ GlobalAveragePooling2D
→ Dense(64) → Dropout(0.5)
→ Dense(10, softmax)
```
مجموع پارامترهای قابل‌آموزش: حدود ۲۸,۰۰۰ (یک مدل کوچک و فشرده، که عمداً برای همخوانی با حجم داده‌ی حدود ۱۰۰۰ نمونه و جلوگیری از Overfitting انتخاب شده است).

### دیتاست
- **GTZAN Genre Collection** — ۱۰۰۰ فایل صوتی (هرکدام ۳۰ ثانیه)، ۱۰۰ فایل به ازای هر ژانر، ۱۰ ژانر.
- توجه: به دلیل حجم بالا، خود دیتاست در این ریپازیتوری قرار داده نشده. می‌توانید آن را از کگل دانلود کنید: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

### تکنولوژی‌های استفاده‌شده
- پایتون (Python)
- TensorFlow / Keras
- librosa (پردازش صوت)
- scikit-learn
- Google Colab (آموزش با GPU)

### ساختار پروژه
```
MusicGenreSorter/
│
├── genre_model.keras       # مدل CNN آموزش‌دیده
├── label_encoder.pkl       # تبدیل اسم ژانر به عدد و برعکس
├── norm_params.pkl         # مقادیر میانگین/انحراف معیار و طول padding
├── MusicGenreSorter.ipynb  # نوت‌بوک کامل: آموزش + پیش‌بینی + مرتب‌سازی
└── README.md
```

### نحوه‌ی استفاده
۱. این ریپازیتوری را کلون کنید.
۲. کتابخانه‌های موردنیاز را نصب کنید:
   ```bash
   pip install tensorflow librosa scikit-learn numpy
   ```
۳. مدل آموزش‌دیده و فایل‌های کمکی (`genre_model.keras`، `label_encoder.pkl`، `norm_params.pkl`) را بارگذاری کنید.
۴. تابع دسته‌بندی و مرتب‌سازی را روی هر فایل صوتی اجرا کنید:
   ```python
   classify_and_sort('path/to/your/song.mp3')
   ```
۵. آهنگ به‌طور خودکار در پوشه‌ای با نام ژانر پیش‌بینی‌شده کپی می‌شود (مثلاً `sorted_songs/jazz/`).

### نتایج
- دقت روی داده‌ی تست: **۷۵٪**
- مدل روی فایل‌های مشابه دیتاست GTZAN عملکرد خوبی دارد و روی فایل‌های خارج از دیتاست هم عملکرد قابل‌قبولی دارد — با جای بهبود از طریق Data Augmentation یا دیتاست بزرگ‌تر.

### بهبودهای آینده
- استفاده از Data Augmentation (تغییر سرعت، Pitch، افزودن نویز) برای تعمیم‌پذیری بهتر
- استفاده از Transfer Learning با یک مدل صوتی از‌پیش‌آموزش‌دیده

