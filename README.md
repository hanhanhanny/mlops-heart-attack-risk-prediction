# Submission 1: Machine Learning Pipeline - Heart Attack Risk Prediction
Nama: Hanny

Username dicoding: hanhanhanny

![heart attack](https://d35oenyzp35321.cloudfront.net/Heart_Failure_In_Youngsters_498e3b1f38.jpg)

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Heart Attack Risk Prediction](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset/data) |
| Masalah | Serangan jantung adalah salah satu penyebab utama kematian di seluruh dunia. Dengan memahami faktor risiko dan gejala awal, kita dapat mengurangi angka kejadian dan meningkatkan peluang penyelamatan nyawa. Masalah yang diangkat dalam proyek ini adalah prediksi risiko serangan jantung pada individu berdasarkan data medis dan faktor risiko tertentu. |
| Solusi machine learning | Untuk membantu mencegah penyakit serangan jantung, saya mengembangkan model machine learning yang dapat memprediksi risiko serangan jantung pada pasien. Dengan prediksi ini, langkah-langkah pencegahan dapat diambil lebih awal untuk mengurangi risiko dan mencegah kondisi yang lebih parah. |
| Metode pengolahan | Dataset yang digunakan memiliki 25 fitur sebagai faktor risiko terkait serangan jantung. 19 fitur dipilih untuk membangun model prediksi risiko serangan jantung, dengan target fitur adalah 'heart attack risk' (dengan nilai 1 untuk memiliki risiko dan 0 untuk tidak memiliki risiko). Data kemudian dibagi menjadi set training dan testing dengan rasio 80:20. Proses transformasi yang dilakukan meliputi standardisasi untuk memastikan semua fitur berada pada skala yang sama. |
| Arsitektur model | Arsitektur model yang digunakan adalah Neural Network dengan lapisan input berbentuk (19,). Model ini terdiri dari beberapa lapisan Dense dan Dropout, dengan jumlah lapisan dan unit yang diacak menggunakan Tuner untuk mendapatkan kombinasi terbaik. Lapisan output memiliki bentuk (1,) dengan fungsi aktivasi sigmoid, karena tujuan prediksi adalah untuk menghasilkan output biner (ya atau tidak). |
| Metrik evaluasi | Metrik evaluasi mencakup ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy. Metrik ini memberikan gambaran menyeluruh tentang kinerja model. |
| Performa model | Performa model tidak cukup baik dengan memiliki loss sebesar 0.6545 dan val_binary_accuracy sebesar 0.6673. Namun, performa ini dapat ditingkatkan dengan lebih banyak data, peningkatan kualitas data, dan penyesuaian arsitektur model lebih lanjut. |
