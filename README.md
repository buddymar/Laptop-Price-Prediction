# 💻 Cracking the Code: Predicting Laptop Prices with Analysis and Machine Learning Model
<br>

**Platform**: Jupyter Notebook | [View Notebook on nbviewer](https://nbviewer.org/github/buddymar/Laptop-Price-Prediction/blob/main/Laptop%20Price%20Prediction.ipynb) | [View Notebook on Github](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/Laptop%20Price%20Prediction.ipynb)<br>
**Programming Language**: Python <br>
**Libraries**: Pandas, NumPy, Matplotlib, Seaborn, sklearn, SHAP <br>
**Source Data**: Kaggle <br>
<br>

**Table of Contents**
- [Introduction](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#-introduction)
- [Data Source](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#-data-source)
- [Data Preprocessing](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#-data-preprocessing)
	- [Data Cleaning](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#-data-preprocessing)
      - [Missing Values](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#missing-values)
      - [Duplicated Values](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#duplicated-values)
	- [Data Transformation](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#feature-engineering)
      - [Feature Engineering](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#feature-engineering)
- [Exploratory Data Analysis](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#-exploratory-data-analysis)
  - [Feature Target](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#laptop-price)
      - [Laptop Price](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#laptop-price)
  - [Univariate & Bivariate Analysis](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#company)
      - [Company](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#company)
      - [Laptop Type](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#laptop-type)
      - [Display Size](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#display-size)
      - [RAM](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#ram)
      - [Operation System](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#operation-system)
      - [Screen Resolution](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#screen-resolution)
      - [CPU Brand](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#cpu-brand)
      - [CPU Speed](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#cpu-speed)
      - [GPU Brand](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#gpu-brand)
      - [Memory](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#memory)
      - [Weight](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#weight)
  - [Handling Outliers](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#handling-outliers)
      - [Laptop Price](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#1-laptop-price)
      - [RAM 32 GB](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#2-ram-32-gb)
      - [Resolution 2560x1440](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#3-resolution-2560x1440)
  - [Features Correlations](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#features-correlations)
- [Predictive Modeling](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#-predictive-modeling)
  - [Modeling](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#modeling)
  - [Model Interpretation](https://github.com/buddymar/Laptop-Price-Prediction/blob/main/README.md#model-interpretation)
<br>

---

## 📌 **Introduction**

<p align="center">
    <kbd> <img width="1000" alt="mvp banner" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/laptop_banner.jpg"> </kbd> <br>
</p>

Laptop telah menjadi kebutuhan primer di kalangan umum, didorong oleh perkembangan digital yang cepat dan dinamis. Saat membeli laptop, banyak komponen dan fitur yang memengaruhi harga totalnya, termasuk CPU, GPU, dan lainnya. Oleh karena itu, penting untuk memahami dampak setiap komponen ini pada harga laptop. Dengan informasi ini, kita dapat mengetahui komponen yang paling berpengaruh pada harga laptop, mengestimasi harga komponen tertentu, dan bahkan memperkirakan harga total laptop berdasarkan spesifikasinya.

Notebook ini bertujuan untuk menganalisis dan memodelkan faktor-faktor yang memengaruhi harga laptop di pasaran. Analisis ini akan mengidentifikasi fitur yang paling signifikan dalam menentukan harga laptop, serta membangun model untuk memprediksi harga berdasarkan fitur-fitur yang ada dalam dataset.

<br>

## 📌 **Data Source**

Berikut ini feature dan deskripsi dari dataset yang digunakan pada analisis ini.

Table 1 — Feature Engineering
 **Feature** | **Explanation** |
-----------------|--------------|
Company | Perusahaan produsen laptop
Product | Brand dan model
TypeName | Tipe laptop (notebook, gaming, dll.) 
Inches | Ukuran layar
ScreenResolution | Resolusi layar
Cpu | Central Processing Unit (CPU)
Ram | Ukuran RAM laptop 
Memory | Ukuran & tipe memori laptop (HDD, SSD, dll.) 
GPU | Graphics Processing Units (GPU)
OpSys | Sistem operasi
Weight | Berat laptop
Price_euros | Harga laptop (€)

<br>

## 📌 **Data Preprocessing**

The data cleaning section will involve various processes such as correcting errors, adjusting data types, handling missing values, managing duplicates, and so on. In the data transformation section, various processes will be performed, including adding new columns or features, handling outliers, encoding variables, correcting errors, and creating and transforming new columns.

<br>

### Missing Values

Jumlah baris dengan missing data pada dataset: 0

<br>

### Duplicated Values

Jumlah baris dengan data duplikat pada dataset: 0

<br>

### Feature Engineering

Pada bagian ini, akan dilakukan *feature engineering* terhadap beberapa fitur untuk mendapatkan dan menambah fitur-fitur laptop yang ada pada dataset ini.

Table 2 — Feature Engineering
 **Feature** | **New Engineered Feature** |
-----------------|--------------|
Screen Resolution | Resolution, IPS, Touchscreen
CPU | Cpu_GHz, Cpu_brand
Memory | Memory_1, SSD_1 (GB), HDD_1 (GB), Hybrid_1 (GB), Flash_1 (GB), Memory_2, Memory_2 (GB), Total_Memory (GB) 
GPU | Gpu_brand

<br>

## 📌 **Exploratory Data Analysis**

### Laptop Price

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/a_target.png"> </kbd> <br>
</p>

Interquartil (IQR) distribusi harga laptop berada di angka sekitar 600-1500 €, dengan rata-rata harga laptop ada di angka 1124 €. Harga laptop termahal ada di angka 6099 €.

Terdapat beberapa laptop yang harganya melebihi batas ekstrim atas grafik boxplot. Hal ini dapat menjadi perhatian saat analisis pada bagian selanjutnya akan adanya potensi outlier laptop mahal pada dataset ini.

<br>

### Company

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_company.png"> </kbd> <br>
</p>

**Key Points:**
- Dari 19 brand laptop yang ada di dataset, **brand dengan distribusi terbanyak adalah Lenovo, Dell, dan HP**
- Terdapat 11 brand laptop yang penjualannya minim, dibawah 10 unit laptop per company.
- Empat brand laptop yang memiliki penjualan terbanyak memiliki rataan harga yang relatif sama di angka ~1000 €.
- Dari brand-brand laptop dengan penjualan diatas 10 unit, **brand Acer memiliki rataan harga laptop paling murah.**

<br>

### Laptop Type

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_type.png"> </kbd> <br>
</p>

**Key Points:**

- **Lebih dari setengah tipe laptop yang terjual adalah tipe Notebook (55.8%).** 
- **Salah satu faktor banyaknya penjualan laptop tipe Notebook adalah rataan harganya yang lebih murah dibandingkan tipe laptop lain (kecuali Netbook).** Namun, tetap ada juga laptop high-end dengan harga yang mahal di tipe Notebook ini. Harga tertinggi berada di angka ~5000 €.
- Tipe laptop dengan rataan harga termahal adalah laptop Workstation di angka ~2000 €.
- **Tipe Netbook memiliki rataan dan rentang harga laptop termurah.** 75% dari tipe laptop ini berada di rentang harga ~175-745 €. Meskipun harganya yang murah dibandingkan tipe laptop lain, Netbook masih memiliki total penjualan yang minim.

<br>

### Display Size

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_display.png"> </kbd> <br>
</p>     
      
**Key Points:**

- **Ukuran layar laptop yang paling umum adalah 15.6"** (51.65% dari total laptop di dataset)
- Ukuran layar terkecil (11.6") memiliki rataan dan rentang harga laptop termurah, sementara ukuran layar terbesar (17.3") memiliki rataan dan rentang harga termahal.
- **Dari grafik boxplot dapat dilihat bahwa tidak ada korelasi yang signifikan terlihat dari ukuran layar terhadap harga laptop.**

<br>

### RAM

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_ram.png"> </kbd> <br>
</p> 

**Key Points:**

- **Mayoritas laptop-laptop di dataset ini menggunakan RAM 4, 8, dan 16 GB**
- **Terdapat korelasi positif yang terlihat antara kapasitas RAM dan harga laptop, dimana semakin besar kapasitas RAM yang digunakan maka semakin mahal harga laptop tersebut.**

<br>

### Operation System

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_opsys.png"> </kbd> <br>
</p> 

**Key Points:**

- **Mayoritas laptop pada dataset menggunakan Windows 10 sebagai Operation System (82.27%).** Rataan harga berada di angka ~1000 €.
- Untuk laptop yang menggunakan OpSys selain Windows 10, terdapat empat OpSys yang memiliki rataan harga lebih rendah yaitu Linux, Chrome OS, Android dan laptop tanpa OS.
- Sementara itu, ada dua OpSys yang memiliki rataan harga lebih tinggi dibandingkan Windows 10 & 10S yaitu Windows 7 dan macOS.

<br>

### Screen Resolution

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_reso.png"> </kbd> <br>
</p> 

**Key Points:**
- **Mayoritas resolusi layar yang digunakan pada laptop di dataset adalah 1920x1080 (66%) dan 1366x768 (24%)**
- **Terlihat korelasi positif antara resolusi dan harga laptop**, dimana semakin besar resolusi layar maka rataan dari harga laptop cenderung lebih mahal. Namun, terdapat pengecualian pada resolusi 2560x1440 karena rataan harga resolusi ini justru lebih mahal dibandingkan resolusi 3200x1800.
- Tiga laptop termahal sama-sama menggunakan resolusi tertinggi di dataset yaitu 3840x2160 (4K Ultra HD)

<br>

### CPU Brand

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_cpu.png"> </kbd> <br>
</p> 

**Key Points:**
- **Komposisi persentase brand CPU pada dataset ini adalah Intel (95.16%), AMD (4.76%), Samsung (0.08%)**
- Sub-brand yang paling banyak digunakan adalah Intel Core series, terutama untuk Intel Core dengan indikator U (mobile power efficient)
- **Untuk Intel Core series, brand modifier i7 menjadi CPU Brand yang paling banyak digunakan**
- Sub-brand Intel Celeron, Pentium dan Atom memiliki rataan harga yang relatif paling murah dibandingkan brand lain
- **Untuk Intel Core series pada brand modifier yang sama, urutan indikator U-HQ-HK adalah urutan indikator dengan rataan harga laptop termurah sampai yang termahal**
- **Pada brand AMD, sub-brand AMD Ryzen memiliki rataan harga yang lebih mahal dibandingkan AMD non-Ryzen**

<br>

### CPU Speed

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_cpughz.png"> </kbd> <br>
</p> 

**Key Points:**
- **Kecepatan CPU yang paling sering digunakan pada laptop di dataset berada di rentang 2.3-2.8 GHz, dimana kecepatan CPU yang paling banyak digunakan ada di angka 2.5 GHz**
- **Plot regresi menunjukan adanya korelasi positif antara kecepatan CPU dan harga laptop.** Namun, terdapat poin-poin yang dapat diamati terkait plot regresi ini. Plot regresi menunjukan bahwa semakin tinggi kecepatan CPU maka semakin mahal harga laptop tersebut. Akan tetapi lima laptop yang memiliki kecepatan CPU paling tinggi (3.6 GHz) justru memiliki harga yang relatif murah (kurang dari 1000 €). Bahkan terdapat cukup banyak laptop dengan kecepatan CPU dibawah 2 GHz yang memiliki harga yang lebih mahal dibandingkan kelima laptop tersebut.

<br>

### GPU Brand

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_gpu.png"> </kbd> <br>
</p> 

**Key Points:**
- **Brand GPU yang paling banyak digunakan adalah Intel Graphics (55.49%)**
- Brand GPU yang memiliki rataan dan kisaran harga laptop yang paling mahal adalah Nvidia Quadro
- Selain Nvidia Quadro, brand Nvidia GeForce GTX dan AMD RX/Pro juga memiliki rataan harga yang lebih tinggi dibandingkan Intel Graphics
- **Brand GPU yang memiliki rataan dan kisaran harga laptop yang paling murah adalah AMD Radeon**

<br>

### Memory

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_memory1.png"> </kbd> <br>
</p> 

**Key Points:**
- **Memori utama yang paling banyak digunakan pada laptop adalah memori SSD (64.70%)**
- **Memori SSD sebagai memori utama memiliki rataan harga yang lebih mahal dibandingkan tipe memori utama lainnya**. Bahkan, semua laptop yang harganya diatas 3000 € menggunakan memori SSD.
- Sementara itu, tipe memori dengan rataan harga paling murah adalah memori Flash Storage.
- Terkait memori tambahan, hanya 15.96% dari semua laptop yang memberikan memori tambahan.
- Hampir semua memori tambahan yang diberikan adalah memori tipe HDD.

<br>

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_memory1.png"> </kbd> <br>
</p> 

**Key Points:**
1. SSD
    - **Ukuran memori SSD yang paling banyak digunakan adalah SSD 256 GB**
    - **Terdapat korelasi positif pada memori SSD, dimana semakin besar ukuran memori yang digunakan maka akan semakin mahal juga harga laptop tersebut**
    - Terdapat potensi outlier, dimana laptop dengan memori SSD 8 GB memiliki harga yang relatif mahal diatas 2000 €
2. HDD
    - Range ukuran Memori HDD relatif jauh lebih besar dibandingkan tipe memori lainnya (500-2000 GB)
    - Pada tipe memori ini, terlihat tidak ada pengaruh signifikan antara ukuran memori terhadap harga laptop
3. Hybrid
    - Memori tipe ini jarang digunakan sebagai memori utama laptop. Selain itu, terlihat juga tidak ada pengaruh antara ukuran tipe memori ini terhadap harga laptop
4. Flash Storage
    - Ukuran memori tipe Flash Storage relatif lebih kecil dibandingkan memori lainnya, dimana ukuran memori yang paling banyak digunakan adalah 32-64 GB
    - Pada memori tipe ini, terdapat juga korelasi positif terhadap harga laptop, dimana semakin besar ukuran memori maka semakin mahal harga laptop

<br>

### Weight

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/b_weight.png"> </kbd> <br>
</p> 

**Key Points:**
- Laptop-laptop pada dataset ini paling banyak memiliki berat pada rentang ~1.8-2.3 kg.
- **Dari plot regresi dapat dilihat bahwa ada korelasi positif yang tidak terlalu signifikan antara berat laptop dengan harga laptop** 
- Dapat dilihat juga bahwa untuk laptop-laptop yang beratnya diatas 3.5 kg mayoritas memiliki harga diatas 1000 €.

<br>

### Handling Outliers

### 1. Laptop Price

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/c_price.png"> </kbd> <br>
</p> 

Dari empat laptop ini, setiap fitur laptop yang digunakan adalah fitur-fitur yang premium sehingga masih wajar harga empat laptop ini relatif lebih mahal dibandingkan laptop-laptop lain. RAM, brand CPU, brand GPU, tipe memori utama yang digunakan dari empat laptop ini memiliki rataan harga yang lebih mahal seperti yang diketahui dari analisa sebelumnya. Dikarenakan tidak adanya kontradiksi antara fitur yang digunakan terhadap harga keempat laptop ini, maka empat laptop ini tidak akan dibuang dari dataset.

<br>

### 2. RAM 32 GB

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/c_ram32.png"> </kbd> <br>
</p> 

Laptop ini memiliki fitur-fitur dengan harga premium. Selain RAM 32 GB, laptop Gaming ini juga menggunakan CPU Intel Core i7 HK, GPU Nvidia GeForce GTX, dan memori utama SSD 256 GB. Bahkan laptop ini juga memiliki memori tambahan berupa SSD 256 GB. Berdasarkan fitur-fitur yang digunakan, harga laptop ini dapat dikatakan terlalu murah. Maka dari itu, laptop ini dapat dianggap sebagai outlier dan akan dibuang dari dataset.

<br>

### 3. Resolution 2560x1440

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/c_reso.png"> </kbd> <br>
</p> 

Fitur yang digunakan oleh laptop ini adalah fitur dengan rataan harga yang relatif murah. Chuwi, brand yang mengeluarkan laptop ini memang memiliki rataan harga laptop yang murah pada dataset. Selain itu, fitur lain yang digunakan seperti CPU Intel Celeron dan GPU Intel Graphics merupakan fitur dengan harga yang murah juga. Ditambah lagi, tipe memori utama yang digunakan adalah Flash Storage dengan ukuran hanya 64 GB. Tidak mengherankan kalau harga laptop ini jauh lebih murah dibandingkan laptop dengan resolusi 2560x1440 lainnya. Maka dari itu, untuk sementara laptop ini tidak akan dibuang dari dataset.

<br>

### Features Correlations

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/d_corr.png"> </kbd> <br>
</p> 

<br>

## 📌 **Predictive Modeling**

Untuk memprediksi harga laptop pada pemodelan ini, tiga model akan dibandingkan yaitu: Linear Regression (Ridge), Extra Tree Regressor, and Gradient Boosting Regressor. Model dengan skoring terbaik akan digunakan sebagai model akhir dari analisis ini.

<br>

### Modeling

Table 3 — Model Scoring
 **Model** | **RMSE** | **R-squared** | **Best Hyperparameters** |
-----------------|--------------|--------------|--------------|
Ridge | 280 € | 82.07% | 'reg__alpha': 300
ExtraTreesRegressor | 267 € | 83.64% | 'max_depth': 40, 'min_samples_split': 3
GradientBoostingRegressor | 225 € | 88.43% | 'min_samples_leaf': 2, 'min_samples_split': 15

<br>

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/1_allmodel.png"> </kbd> <br>
</p>

Dari ketiga pemodelan yang dilakukan untuk memprediksi harga laptop, model `Gradient Boosting` memiliki score RMSLE, RMSE, dan R2 yang terbaik dibandingkan model lainnya. 

<br>

### Model Interpretation

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/2_shap.png"> </kbd> <br>
</p>

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/2_beeshap.png"> </kbd> <br>
</p>

Dari lima fitur yang paling berpengaruh, dapat dilihat bahwa semakin besar `RAM`, `besar SSD`, dan `kecepatan GPU` maka semakin besar juga tambahan harga pada laptop. Sementara itu, semakin kecil `berat laptop` maka harga cenderung semakin besar. Untuk tipe laptop, laptop dengan `tipe Notebook` diberikan pengurangan harga laptop dibandingkan dengan laptop tipe lainnya. 

Sebagai contoh, berikut ini interpretasi model pada salah satu laptop di *test data*.

<p align="center">
    <kbd> <img width="1000" alt="share" src="https://raw.githubusercontent.com/buddymar/Laptop-Price-Prediction/main/assets/2_example.png"> </kbd> <br>
</p>

Pada model ini, semua laptop diberi harga awal sebesar 1111.604 €. Laptop diatas diprediksi oleh model memiliki harga sebesar 575.8 €, dimana harga aktual laptop tersebut adalah 689 €.

Lima fitur yang paling berpengaruh dari semakin kecilnya harga laptop prediksi dibandingkan dengan harga awal model adalah besar `RAM`, `kecepatan CPU`, `besar memori SSD`, `tipe laptop Notebook`, dan `resolusi 1366x768` yang dimiliki. 

Dapat dilihat bahwa fitur-fitur ini menjadi pinalti/pengurang harga pada laptop tersebut karena ukurannya yang kecil seperti `RAM` hanya 4 GB, `kecepatan CPU` 2 GHz, dan `ukuran SSD` hanya 128 GB. Selain ukuran fitur yang kecil, tipe fitur yang dimiliki laptop ini juga tipe yang dilabel sebagai pinalti harga seperti `tipe Notebook` dan `resolusi 1366x768`.
