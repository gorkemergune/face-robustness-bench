# Benchmark Raporu: Filtreli, Kötü ve Sahte "İkiz" Fotoğraflarda Yüz Tanıma

**Soru:** Bir yüz tanıma sistemi kişiyi temiz fotoğraflarıyla tanıdıktan sonra, aynı kişinin filtreli / makyajlı / efektli, kötü çekilmiş ve yapay zekâ ile üretilmiş "ikiz" fotoğraflarını nasıl değerlendiriyor?  
**Modeller:** ArcFace (InsightFace `buffalo_l`) ve Qwen2.5-VL 7B (Ollama, yerel)  
**Pilot veri:** 1 kişi (anonim), 35 temiz fotoğraf, 97 test fotoğrafı · 9 Ekim 2026

## Sonuçlar

| Grup                           | ArcFace             | Qwen2.5-VL          |
|--------------------------------|---------------------|---------------------|
| Değişiklikli (dirty)           | %100 (28/28)        | %100 (28/28)        |
| Kötü (bad)                     | %92,9 (52/56)       | %100 (56/56)        |
| Sahte ikiz (fake), düşük = iyi | %84,6 kabul (11/13) | %84,6 kabul (11/13) |

- **Değişiklikli (dirty):** makyaj, filtre ve efekt uygulanmış fotoğraflar. Oran, kişinin tanındığı fotoğrafların yüzdesi.
- **Kötü (bad):** düşük kalite, yarım yüz veya yanında başka biri olan fotoğraflar. Oran yine tanınma yüzdesi.
- **Sahte ikiz (fake):** Gemini ile kişiye benzetilmiş ama kişi olmayan yüzler. Oran, sistemin bunları yanlışlıkla kişi sandığı yüzde, yani düşük olması iyidir.
- **Referans:** değişiklik içermeyen 14 temiz test fotoğrafı her iki modelde de %100 tanındı.

## Temel bulgular

1. **Makyaj, filtre ve efekt:** İki model de bu değişikliklere karşı dayanıklı. Ortalama benzerlik temiz fotoğraflara göre ArcFace'te %95, Qwen'de %99 oranında korundu.
2. **Kötü fotoğraflar:** Qwen2.5-VL daha dayanıklı. ArcFace 2 fotoğrafta yüz bulamadı, 2 fotoğrafta ise yanındaki kişinin yüzünü seçtiği için kişiyi tanımadı. Fotoğrafta birden fazla yüz varsa ArcFace en büyük yüzü kullanıyor ve bu yüz başka birine ait olabiliyor.
3. **Sahte ikizler:** İki model de ikizlerin büyük çoğunluğunu kişinin kendisi sandı. ArcFace'te eşiği yükseltmek ikiz kabulünü azaltıyor, ama kişinin kendi kötü fotoğraflarının tanınması da düşüyor. Örneğin eşik 0,50 olduğunda ikiz kabulü %30,8'e iniyor, kötü fotoğraf tanıma ise %73,2'ye düşüyor. Hiçbir eşik ikizleri kişiden temiz biçimde ayıramadı.

## Yöntem (kısa)

- **Şablon:** Temiz fotoğrafların %60'ı (21) kişinin şablonunu oluşturdu, %40'ı (14) referans test olarak ayrıldı (sabit seed 42).
- **ArcFace:** yüz bulunur, hizalanır ve 512 boyutlu vektöre çevrilir. Skor, şablona kosinüs benzerliğidir.
- **Qwen2.5-VL:** her test fotoğrafı, kişinin 3 farklı şablon fotoğrafıyla ikili gösterildi ve *"Aynı kişi mi? Makyaj, filtre, efekt ve kozmetik değişiklikleri yok say."* diye soruldu (temperature 0, sabit seed).
- **Karar eşiği:** her model için ayrı ayarlandı. Eşik, LFW veri setindeki yabancı yüzlerin en fazla %1'ini kabul edecek şekilde seçildi (ArcFace 1000, Qwen 100 yabancı skoru). İkiz fotoğraflar eşik ayarına katılmadı.
- **Sızıntı önlemi:** temiz bir fotoğrafın birebir kopyası olan 3 test fotoğrafı değerlendirmeden çıkarıldı.

## Kısıtlar

- **Az veri:** Tek kişi ve grup başına 13–56 fotoğraf var. Sonuçlar yol göstericidir; %95 güven aralıkları geniştir (ör. ikiz kabulü için %58–96).
- **İkizler gerçek değil:** Sahte ikizler yapay zekâ ile üretildi. Bazılarında yüz çok az değiştirilmiş olabilir.
- **Hız farkı:** Qwen2.5-VL karşılaştırma başına ~26 saniye sürüyor; ArcFace fotoğraf başına 1 saniyenin altında.

## Tekrar üretmek için

```bash
python scripts/prepare_user_data.py --person <kişi_id>      # fotoğrafları dönüştür ve numarala
python -m frbench.cli run --data data --backends arcface,qwen_vl --qwen-max-impostors 100
python -m frbench.cli effects                                # fotoğraf bazında ayrıntılı rapor (yerel)
```

Fotoğraflar, fotoğraf bazında sonuçlar ve önbellekler gizlilik nedeniyle repoda yer almaz (`.gitignore`).
