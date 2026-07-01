# Hedonik Fiyat Modeli ile Konut Fiyatı Analizi

Ames, Iowa konut satış verisi kullanılarak, konut özelliklerinin (yaşam
alanı, arsa büyüklüğü, genel kalite, ev yaşı, mahalle vb.) satış fiyatı
üzerindeki etkisini **log-lineer hedonik regresyon modeli** ile tahmin
eden bir ekonometri projesi.

## Motivasyon

Hedonik fiyatlandırma teorisi, heterojen bir malın (bu örnekte konut)
fiyatının, o malı oluşturan örtük özelliklerin (attribute) doğrusal bir
kombinasyonu olarak açıklanabileceğini öne sürer (Rosen, 1974). Bu
proje, ekonometri derslerinde öğrenilen OLS regresyon, log-dönüşüm,
çoklu doğrusal bağlantı (multicollinearity) ve değişen varyans
(heteroskedasticity) kavramlarının gerçek veri üzerinde uygulanmasını
amaçlar.

## Veri Seti

[Ames Housing Dataset (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
— `train.csv` dosyasını indirip `data/` klasörüne yerleştirin.

## Metodoloji

1. **Veri temizleme**: eksik gözlemler çıkarılır, uç değerler
   (%0.5 alt/üst) kırpılır.
2. **Model**:

   ```
   log(Fiyat) = β0 + β1·YaşamAlanı + β2·ArsaAlanı + β3·GenelKalite
                + β4·GenelDurum + β5·EvYaşı + β6·BodrumAlanı
                + β7·GarajKapasitesi + β8·BanyoSayısı + β9·YatakOdası
                + β10·MerkeziKlima + Mahalle sabit etkileri + ε
   ```

   Bağımlı değişkenin logaritması alınarak katsayılar yaklaşık
   semi-elastikiyet olarak yorumlanır.
3. **Sağlam standart hatalar**: heteroskedastisiteye karşı dayanıklı
   HC3 kovaryans tahmincisi kullanılır.
4. **Tanı testleri**:
   - Breusch-Pagan testi (değişen varyans)
   - Jarque-Bera testi (artıkların normalliği)
   - VIF (çoklu doğrusal bağlantı)

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python src/hedonic_model.py
```

Çıktılar `outputs/` klasörüne kaydedilir:
- `model_summary.txt` — tam regresyon çıktısı
- `model_diagnostics.png` — gerçek vs. tahmin ve artık grafikleri

## Bulgular

Model, konut fiyatındaki varyansın **%88.6'sını** açıklamaktadır (R² = 0.886, n = 1418).
Tüm standart hatalar heteroskedastisiteye dayanıklı HC3 tahmincisiyle hesaplanmıştır.

İstatistiksel olarak anlamlı (p < 0.05) başlıca etkiler:

- **Genel kalite puanı**: 1 birimlik artış, diğer değişkenler sabitken fiyatı
  yaklaşık **%7.2** artırmaktadır (ceteris paribus).
- **Genel durum puanı**: 1 birimlik artış, fiyatı yaklaşık **%6.3** artırmaktadır.
- **Merkezi klima**: klimalı evler, klimasız evlere göre yaklaşık **%5.7** daha
  pahalı fiyatlanmaktadır.
- **Garaj kapasitesi**: her ek araçlık garaj alanı, fiyatı yaklaşık **%5.3**
  artırmaktadır.
- **Ev yaşı**: her ek yaş, fiyatı yaklaşık **%0.3** azaltmaktadır.
- **Yatak odası sayısı**: yaşam alanı sabit tutulduğunda, oda sayısındaki artış
  fiyatı yaklaşık **%2.7** azaltmaktadır — bu, aynı büyüklükteki bir evde daha
  fazla (ve dolayısıyla daha küçük) odaların piyasa tarafından daha az
  değerli bulunduğuna işaret etmektedir.
- **Mahalle sabit etkileri**: NridgHt mahallesi referans mahalleye göre
  yaklaşık **%7.1** daha pahalı, OldTown mahallesi ise yaklaşık **%8.1** daha
  ucuz fiyatlanmaktadır — konumun fiyatlama üzerinde güçlü ve anlamlı bir
  etkisi olduğu görülmektedir.

**Model tanı sonuçları**: Artıkların (residuals) normal dağılımdan saptığı
(Jarque-Bera testi, p < 0.001) ve hafif sağa çarpık olduğu gözlemlenmiştir;
bu genellikle konut fiyatı gibi verilerde beklenen bir durumdur. Değişken
şişirme faktörü (condition number) yüksek çıkmış olup bazı değişkenler
arasında (örn. yaşam alanı ile bodrum alanı) orta düzey çoklu doğrusal
bağlantı olabileceğine işaret etmektedir; bu nedenle katsayılar bireysel
olarak değil, birbirleriyle birlikte yorumlanmalıdır.

## Kullanılan Araçlar

Python, pandas, statsmodels, scikit-learn, matplotlib, seaborn

## Referans

Rosen, S. (1974). Hedonic Prices and Implicit Markets: Product
Differentiation in Pure Competition. *Journal of Political Economy*,
82(1), 34-55.
