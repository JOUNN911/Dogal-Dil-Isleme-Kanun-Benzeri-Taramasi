import os
import re
import glob
import pandas as pd
from gensim.models import Word2Vec
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime

# NLTK veri paketlerini indirme
nltk.download('punkt')

# Stop words listesi
STOP_WORDS = set([
    "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç",
    "birşey", "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa",
    "diye", "eğer", "en", "gibi", "hem", "hep", "hepsi", "her", "hiç",
    "için", "ile", "ise", "kez", "ki", "kim", "mı", "mu", "mü", "nasıl",
    "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani",
    "bir", "kadar", "sonra", "diğer", "şimdi", "zaman", "böyle", "tarafından"
])

# Klasör yapısı oluşturma
os.makedirs("word2vec_models/stemmed", exist_ok=True)
os.makedirs("word2vec_models/lemmatized", exist_ok=True)
os.makedirs("zipf_analizi", exist_ok=True)
os.makedirs("temizlenmis_veriler", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)
os.makedirs("tfidf_models", exist_ok=True)
os.makedirs("benzer_metinler", exist_ok=True)


# Klasördeki tüm .txt dosyalarını al
txt_files = sorted(glob.glob("gazeteler/*.txt"))  # glob modülünün glob fonksiyonunu kullan


def clean_text(text):
    """Metin temizleme fonksiyonu (nokta ve virgül korunur)"""
    text = text.lower()
    # Nokta ve virgül dışındaki tüm noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s.,ğüşıöçĞÜŞİÖÇ]', ' ', text)
    # Sayıları kaldır
    text = re.sub(r'\d+', ' ', text)
    # Fazla boşlukları sadeleştir
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def normalize_and_tokenize(text):
    """Tokenizasyon fonksiyonu"""
    words = word_tokenize(text)
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def advanced_stem(word, method=1):
    """Gelişmiş stemming fonksiyonu (8 farklı yöntem)"""
    word = word.lower()

    # Tüm yöntemler için ortak kontrol
    if len(word) <= 3:  # Çok kısa kelimeleri olduğu gibi döndür
        return word

    # Yöntem 1: Temel ekleri kaldır
    if method == 1:
        suffixes = ['lar', 'ler', 'da', 'de', 'ta', 'te']
    # Yöntem 2: İyelik ekleri
    elif method == 2:
        suffixes = ['ın', 'in', 'un', 'ün', 'ım', 'im', 'um', 'üm']
    # Yöntem 3: Hal ekleri
    elif method == 3:
        suffixes = ['a', 'e', 'ı', 'i', 'u', 'ü']
    # Yöntem 4: Fiil çekim ekleri
    elif method == 4:
        suffixes = ['yor', 'miş', 'müş', 'di', 'dı', 'du', 'dü']
    # Yöntem 5: Çoğul ve iyelik kombinasyonu
    elif method == 5:
        suffixes = ['ları', 'leri', 'ların', 'lerin']
    # Yöntem 6: Fiil birleşik ekler
    elif method == 6:
        suffixes = ['ıyor', 'iyor', 'uyor', 'üyor']
    # Yöntem 7: Sıfat ekleri
    elif method == 7:
        suffixes = ['lı', 'li', 'lu', 'lü']
    # Yöntem 8: Zarf ekleri
    elif method == 8:
        suffixes = ['ca', 'ce', 'ça', 'çe']
    else:
        return word  # Geçersiz yöntem numarası

    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def advanced_lemma(word, method=1):
    """Basit lemmatization fonksiyonu (8 farklı yöntem)"""
    word = word.lower()

    # Tüm yöntemler için ortak kontrol
    if len(word) <= 3:  # Çok kısa kelimeleri olduğu gibi döndür
        return word

    # Yöntem 1: Temel kök bulma
    if method == 1:
        if word.endswith(('lar', 'ler')):
            return word[:-3]
    # Yöntem 2: Fiil kökleri
    elif method == 2:
        if word.endswith(('mak', 'mek')):
            return word[:-3]
    # Yöntem 3: İsim kökleri
    elif method == 3:
        if word.endswith(('lık', 'lik', 'luk', 'lük')):
            return word[:-3]
    # Yöntem 4: Sıfat kökleri
    elif method == 4:
        if word.endswith(('sız', 'siz', 'suz', 'süz')):
            return word[:-3]
    # Yöntem 5: Zarf kökleri
    elif method == 5:
        if word.endswith(('ca', 'ce', 'ça', 'çe')):
            return word[:-2]
    # Yöntem 6: Fiil çekimleri
    elif method == 6:
        if word.endswith(('ıyor', 'iyor', 'uyor', 'üyor')):
            return word[:-4]
    # Yöntem 7: İyelik ekleri
    elif method == 7:
        if word.endswith(('ım', 'im', 'um', 'üm')):
            return word[:-2]
    # Yöntem 8: Hal ekleri
    elif method == 8:
        if word.endswith(('dan', 'den', 'tan', 'ten')):
            return word[:-3]
    else:
        return word  # Geçersiz yöntem numarası

    return word


def process_text(text):
    """Metin işleme pipeline'ı"""
    cleaned = clean_text(text)
    tokens = normalize_and_tokenize(cleaned)

    # 8 farklı stemming yöntemi
    stems = []
    for i in range(1, 9):
        stems.append([advanced_stem(t, i) for t in tokens])

    # 8 farklı lemmatization yöntemi
    lemmas = []
    for i in range(1, 9):
        lemmas.append([advanced_lemma(t, i) for t in tokens])

    return tokens, lemmas, stems

def process_input_text(input_text):
    print("\n--- Örnek Metin Girişi ---")
    print(f"Girdi: {input_text}")

    tokens, lemmas, stems = process_text(input_text)

    print("\nToken'lar:")
    print(tokens)

    print("\nStemming Sonuçları (1. yöntem):")
    print(stems[0])

    print("\nLemmatization Sonuçları (1. yöntem):")
    print(lemmas[0])

    # TF-IDF vektör modeli yüklü mü kontrol et
    try:
        tfidf_model = joblib.load('tfidf_models/tfidf_lemmatized.model')
        transformed = tfidf_model.transform([' '.join(lemmas[0])])
        print("\nTF-IDF Vektörü (1. lemmatized yöntem):")
        print(transformed.toarray())
    except Exception as e:
        print(f"TF-IDF modeli yüklenemedi: {str(e)}")

    # Word2Vec örneği
    try:
        w2v_model = Word2Vec.load('word2vec_models/lemmatized/lemma_model_1.model')
        print("\nWord2Vec örnek vektörler:")
        for word in lemmas[0]:
            if word in w2v_model.wv:
                print(f"{word}: {w2v_model.wv[word][:5]}")  # İlk 5 değeri yazdır
    except Exception as e:
        print(f"Word2Vec modeli yüklenemedi: {str(e)}")


def apply_zipfs_law(words, output_prefix):
    """Gelişmiş Zipf analizi"""
    word_counts = Counter(words)
    most_common = word_counts.most_common(1000)

    ranks = np.arange(1, len(most_common) + 1)
    freqs = [count for (word, count) in most_common]

    plt.figure(figsize=(12, 6))
    plt.plot(ranks, freqs, 'b-', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Zipf Kanunu - {output_prefix} (Log-Log)')
    plt.xlabel('Log(Sıralama)')
    plt.ylabel('Log(Frekans)')
    plt.grid(True, which="both", ls="-")
    plt.savefig(f"zipf_analizi/{output_prefix}_loglog_zipf.png", dpi=300)
    plt.close()

    # En sık kullanılan 100 kelimeyi kaydet
    with open(f"temizlenmis_veriler/{output_prefix}_sik_kelimeler.txt", "w", encoding="utf-8") as f:
        for word, count in most_common[:100]:
            f.write(f"{word}: {count}\n")


def create_tfidf_models(lemmas, stems):
    """TF-IDF modellerini oluşturup 'tfidf_models' klasörüne kaydeder"""
    lemma_texts = [' '.join(lemma) for lemma in lemmas]
    stem_texts = [' '.join(stem) for stem in stems]

    def has_valid_words(text_list):
        for text in text_list:
            words = text.split()
            if any(len(word) > 2 for word in words):
                return True
        return False

    if not has_valid_words(lemma_texts):
        lemma_texts = ["örnek metin tfidf için"]
    if not has_valid_words(stem_texts):
        stem_texts = ["örnek metin tfidf için"]

    lemma_vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=1,
        token_pattern=r'\b[^\d\W]{3,}\b',
        ngram_range=(1, 2)
    )

    stem_vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=1,
        token_pattern=r'\b[^\d\W]{3,}\b',
        ngram_range=(1, 2)
    )

    try:
        lemma_tfidf = lemma_vectorizer.fit_transform(lemma_texts)
        joblib.dump(lemma_vectorizer, 'tfidf_models/tfidf_lemmatized.model')
        pd.DataFrame(lemma_tfidf.toarray(),
                     columns=lemma_vectorizer.get_feature_names_out()).to_csv(
            'tfidf_models/lemma_tfidf_matrix.csv', index=False)
    except Exception as e:
        print(f"Lemma TF-IDF oluşturulurken hata: {str(e)}")
        lemma_tfidf = None

    try:
        stem_tfidf = stem_vectorizer.fit_transform(stem_texts)
        joblib.dump(stem_vectorizer, 'tfidf_models/tfidf_stemmed.model')
        pd.DataFrame(stem_tfidf.toarray(),
                     columns=stem_vectorizer.get_feature_names_out()).to_csv(
            'tfidf_models/stem_tfidf_matrix.csv', index=False)
    except Exception as e:
        print(f"Stem TF-IDF oluşturulurken hata: {str(e)}")
        stem_tfidf = None

    return lemma_tfidf, stem_tfidf

def compare_input_with_dataset(input_text):
    print("\n--- Veri Seti Karşılaştırması Başlıyor ---")

    # Tüm dosyalar için önceden işlenmiş veriler
    all_lemmas = []
    all_stems = []

    for file in txt_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                _, lemmas, stems = process_text(content)
                all_lemmas.append(lemmas[0])
                all_stems.append(stems[0])
        except Exception as e:
            print(f"{file} dosyasında hata: {e}")

    # Girdi metni işlenir
    _, input_lemmas, input_stems = process_text(input_text)

    # TF-IDF modelleri yüklenir
    try:
        tfidf_lemma_model = joblib.load('tfidf_models/tfidf_lemmatized.model')
        tfidf_stem_model = joblib.load('tfidf_models/tfidf_stemmed.model')
    except Exception as e:
        print(f"TF-IDF modelleri yüklenemedi: {str(e)}")
        return

    tfidf_lemma_matrix = tfidf_lemma_model.transform(
        [' '.join(doc) for doc in all_lemmas[0:1]])  # sadece örnek olarak 1 tanesini tutuyor
    tfidf_stem_matrix = tfidf_stem_model.transform([' '.join(doc) for doc in all_stems[0:1]])

    input_lemma_vec = tfidf_lemma_model.transform([' '.join(input_lemmas[0])])
    input_stem_vec = tfidf_stem_model.transform([' '.join(input_stems[0])])

    # TF-IDF Benzerliği
    lemma_similarities = []
    stem_similarities = []
    file_texts = []

    for i, txt_file in enumerate(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                file_text = f.read()
                tokens, lemmas, stems = process_text(file_text)
                file_texts.append(file_text)

                # TF-IDF vektörleri
                lemma_vec = tfidf_lemma_model.transform([' '.join(lemmas[0])])
                stem_vec = tfidf_stem_model.transform([' '.join(stems[0])])

                lemma_score = cosine_similarity(input_lemma_vec, lemma_vec)[0][0]
                stem_score = cosine_similarity(input_stem_vec, stem_vec)[0][0]

                lemma_similarities.append((txt_file, lemma_score))
                stem_similarities.append((txt_file, stem_score))
        except Exception as e:
            print(f"{txt_file} okunurken hata oluştu: {str(e)}")

    print("\n--- En Benzer 5 Dosya (TF-IDF - Lemmatized) ---")
    for file, score in sorted(lemma_similarities, key=lambda x: x[1], reverse=True)[:5]:
        print(f"{file} - Benzerlik: {score:.4f}")

    print("\n--- En Benzer 5 Dosya (TF-IDF - Stemmed) ---")
    for file, score in sorted(stem_similarities, key=lambda x: x[1], reverse=True)[:5]:
        print(f"{file} - Benzerlik: {score:.4f}")

    # Word2Vec ile benzerlik: Ortalama vektörler üzerinden
    try:
        w2v_lemma_model = Word2Vec.load('word2vec_models/lemmatized/lemma_model_1.model')
        w2v_stem_model = Word2Vec.load('word2vec_models/stemmed/stem_model_1.model')

        def avg_vector(words, model):
            vectors = [model.wv[word] for word in words if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

        input_vec_lemma = avg_vector(input_lemmas[0], w2v_lemma_model)
        input_vec_stem = avg_vector(input_stems[0], w2v_stem_model)

        lemma_sim_w2v = []
        stem_sim_w2v = []

        for i, txt_file in enumerate(txt_files):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    _, lemmas, stems = process_text(file_text)

                    file_vec_lemma = avg_vector(lemmas[0], w2v_lemma_model)
                    file_vec_stem = avg_vector(stems[0], w2v_stem_model)

                    lemma_sim = cosine_similarity([input_vec_lemma], [file_vec_lemma])[0][0]
                    stem_sim = cosine_similarity([input_vec_stem], [file_vec_stem])[0][0]

                    lemma_sim_w2v.append((txt_file, lemma_sim))
                    stem_sim_w2v.append((txt_file, stem_sim))
            except Exception as e:
                print(f"{txt_file} - W2V hatası: {str(e)}")

        print("\n--- En Benzer 5 Dosya (Word2Vec - Lemmatized) ---")
        for file, score in sorted(lemma_sim_w2v, key=lambda x: x[1], reverse=True)[:5]:
            print(f"{file} - Benzerlik: {score:.4f}")

        print("\n--- En Benzer 5 Dosya (Word2Vec - Stemmed) ---")
        for file, score in sorted(stem_sim_w2v, key=lambda x: x[1], reverse=True)[:5]:
            print(f"{file} - Benzerlik: {score:.4f}")

    except Exception as e:
        print(f"Word2Vec benzerlik hesaplamada hata: {str(e)}")

        def write_top_matches(matches, prefix):
            os.makedirs("benzer_metinler", exist_ok=True)
            for idx, (file_path, score) in enumerate(matches[:5]):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    filename = f"benzer_metinler/{prefix}_{idx + 1}_benzerlik_{score:.4f}.txt"
                    with open(filename, "w", encoding="utf-8") as out:
                        out.write(content)
                except Exception as e:
                    print(f"{file_path} kaydedilirken hata: {str(e)}")
    # En benzer 5 metni kaydet (TF-IDF - Lemmatized'e göre)
    top5_lemma = sorted(lemma_similarities, key=lambda x: x[1], reverse=True)[:5]

    for i, (file_path, score) in enumerate(top5_lemma, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            output_path = os.path.join("benzer_metinler", f"benzer_{i}_{Path(file_path).stem}.txt")
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(f"Benzerlik Skoru: {score:.4f}\n\n")
                out_f.write(content)
        except Exception as e:
            print(f"{file_path} dosyası kaydedilirken hata: {str(e)}")
    # En benzer 5 metni dosyaya yaz
    from datetime import datetime

    def save_top_matches(similarities, method_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"benzer_metinler/{timestamp}_{method_name}_top5"
        os.makedirs(out_dir, exist_ok=True)

        top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
        for i, (filepath, score) in enumerate(top5, 1):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                filename = os.path.basename(filepath)
                new_filename = f"{i:02d}_{score:.4f}_{filename}"
                with open(os.path.join(out_dir, new_filename), 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
            except Exception as e:
                print(f"Dosya kaydederken hata oluştu: {filepath} - {e}")

    save_top_matches(lemma_similarities, "tfidf_lemma")
    save_top_matches(stem_similarities, "tfidf_stem")
    save_top_matches(lemma_sim_w2v, "w2v_lemma")
    save_top_matches(stem_sim_w2v, "w2v_stem")

    # En benzer 5 metni dosyaya yaz
    def save_top_matches(similarities, method_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"benzer_metinler/{timestamp}_{method_name}_top5"
        os.makedirs(out_dir, exist_ok=True)

        top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
        for i, (filepath, score) in enumerate(top5, 1):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                filename = os.path.basename(filepath)
                new_filename = f"{i:02d}_{score:.4f}_{filename}"
                with open(os.path.join(out_dir, new_filename), 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
            except Exception as e:
                print(f"Dosya kaydederken hata oluştu: {filepath} - {e}")

    save_top_matches(lemma_similarities, "tfidf_lemma")
    save_top_matches(stem_similarities, "tfidf_stem")
    save_top_matches(lemma_sim_w2v, "w2v_lemma")
    save_top_matches(stem_sim_w2v, "w2v_stem")


def generate_similar_texts(input_text, num_similar=5, output_dir="benzer_metinler"):
    """
    TF-IDF ve Word2Vec modellerini kullanarak girilen metne benzer yeni metinler üretir
    ve bunları belirtilen klasöre kaydeder.

    Args:
        input_text (str): Karşılaştırılacak örnek metin
        num_similar (int): Üretilecek benzer metin sayısı (varsayılan: 5)
        output_dir (str): Çıktı klasörü (varsayılan: "benzer_metinler")
    """
    # Gerekli modülleri yükle
    from gensim.models import Word2Vec
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import os
    import random
    from datetime import datetime
    import joblib

    # Klasörü oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Girdi metnini işle
    _, input_lemmas, input_stems = process_text(input_text)

    # TF-IDF ve Word2Vec modellerini yükle
    try:
        # TF-IDF modelleri
        tfidf_lemma_model = joblib.load('tfidf_models/tfidf_lemmatized.model')
        tfidf_stem_model = joblib.load('tfidf_models/tfidf_stemmed.model')

        # Word2Vec modelleri
        w2v_lemma_model = Word2Vec.load('word2vec_models/lemmatized/lemma_model_1.model')
        w2v_stem_model = Word2Vec.load('word2vec_models/stemmed/stem_model_1.model')
    except Exception as e:
        print(f"Modeller yüklenirken hata oluştu: {str(e)}")
        return

    # Benzer metinler üretmek için fonksiyonlar
    def generate_from_tfidf(input_vec, model, feature_names, num_words=50):
        """TF-IDF vektöründen benzer kelimelerle metin üret"""
        # En yüksek ağırlıklı kelimeleri al
        sorted_items = np.argsort(input_vec.toarray().flatten())[::-1]
        top_words = [feature_names[i] for i in sorted_items[:num_words]]

        # Kelimeleri karıştırıp rastgele birleştir
        random.shuffle(top_words)
        generated_text = ' '.join(top_words[:random.randint(20, num_words)])
        return generated_text

    def generate_from_word2vec(input_words, model, num_words=20):
        """Word2Vec modelinden benzer kelimelerle metin üret"""
        generated_words = []
        if not input_words:
            return ""

        # Modelde bulunan kelimeleri filtrele
        valid_words = [word for word in input_words if word in model.wv]
        if not valid_words:
            return ""

        # Başlangıç kelimesi rastgele seç
        current_word = random.choice(valid_words)
        generated_words.append(current_word)

        # Zincir oluştur
        for _ in range(num_words - 1):
            try:
                # En benzer kelimeleri bul
                similar_words = model.wv.most_similar(current_word, topn=5)
                next_word = random.choice(similar_words)[0]
                generated_words.append(next_word)
                current_word = next_word
            except:
                break

        return ' '.join(generated_words)

    # TF-IDF ile metin üret (lemmatized)
    input_lemma_vec = tfidf_lemma_model.transform([' '.join(input_lemmas[0])])
    tfidf_lemma_text = generate_from_tfidf(
        input_lemma_vec,
        tfidf_lemma_model,
        tfidf_lemma_model.get_feature_names_out()
    )

    # TF-IDF ile metin üret (stemmed)
    input_stem_vec = tfidf_stem_model.transform([' '.join(input_stems[0])])
    tfidf_stem_text = generate_from_tfidf(
        input_stem_vec,
        tfidf_stem_model,
        tfidf_stem_model.get_feature_names_out()
    )

    # Word2Vec ile metin üret (lemmatized)
    w2v_lemma_text = generate_from_word2vec(
        input_lemmas[0],
        w2v_lemma_model
    )

    # Word2Vec ile metin üret (stemmed)
    w2v_stem_text = generate_from_word2vec(
        input_stems[0],
        w2v_stem_model
    )

    # Tüm üretilen metinleri birleştir
    generated_texts = {
        "tfidf_lemma": tfidf_lemma_text,
        "tfidf_stem": tfidf_stem_text,
        "w2v_lemma": w2v_lemma_text,
        "w2v_stem": w2v_stem_text,
    }

    # Metinleri dosyaya yaz
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for method, text in generated_texts.items():
        if text.strip():  # Boş olmayan metinleri kaydet
            filename = os.path.join(output_dir, f"{timestamp}_{method}_generated.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Orijinal Metin:\n{input_text}\n\n")
                f.write(f"Üretilen Metin ({method}):\n{text}")

    print(f"{len(generated_texts)} adet benzer metin '{output_dir}' klasörüne kaydedildi.")
    return generated_texts

def process_files():
    """Dosya işleme fonksiyonu"""
    all_tokens = []
    all_lemmas = [[] for _ in range(8)]  # 8 lemmatization yöntemi
    all_stems = [[] for _ in range(8)]  # 8 stemming yöntemi

    def split_into_sentences(tokens_list):
        """Noktalara göre cümlelere ayırır, her cümle bir kelime listesi olur"""
        sentences = []
        current_sentence = []
        for word in tokens_list:
            if word == ".":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(word)
        if current_sentence:
            sentences.append(current_sentence)
        return sentences

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens, lemmas, stems = process_text(text)
                all_tokens.extend(tokens)

                for i in range(8):
                    all_lemmas[i].extend(lemmas[i])
                    all_stems[i].extend(stems[i])
        except Exception as e:
            print(f"Hata: {txt_file} - {str(e)}")

    # Zipf analizlerini uygula
    apply_zipfs_law(all_tokens, "orijinal")
    for i in range(8):
        apply_zipfs_law(all_lemmas[i], f"lemma_{i + 1}")
        apply_zipfs_law(all_stems[i], f"stem_{i + 1}")

    # Word2Vec modellerini oluştur ve kaydet
    for i in range(8):
        # Cümlelere göre böl
        lemma_sentences = split_into_sentences(all_lemmas[i])
        stem_sentences = split_into_sentences(all_stems[i])

        # Lemmatized modeller
        lemma_model = Word2Vec(lemma_sentences, vector_size=100, window=5, min_count=1, workers=4)
        lemma_model.save(f"word2vec_models/lemmatized/lemma_model_{i + 1}.model")

        # Stemmed modeller
        stem_model = Word2Vec(stem_sentences, vector_size=100, window=5, min_count=1, workers=4)
        stem_model.save(f"word2vec_models/stemmed/stem_model_{i + 1}.model")

    # TF-IDF modellerini oluştur
    create_tfidf_models(all_lemmas, all_stems)

    # Temizlenmiş verileri kaydet
    with open("temizlenmis_veriler/anlam_butunlugu_bozan_kelimeler.txt", "w", encoding="utf-8") as f:
        f.write("STOP WORDS LİSTESİ:\n")
        f.write("\n".join(sorted(STOP_WORDS)))

        word_counts = Counter(all_tokens)
        f.write("\n\nEN SIK KARŞILAŞILAN ANLAMSIZ KELİMELER:\n")
        for word, count in word_counts.most_common(50):
            if word not in STOP_WORDS and len(word) < 4:  # Kısa ve muhtemelen anlamsız kelimeler
                f.write(f"{word}: {count}\n")

    return all_lemmas, all_stems


def score_similar_texts(reference_text, scoring_dir="benzer_metinler"):
    """
    Benzer metinleri referans metne göre puanlar (1-5 arası)

    Args:
        reference_text (str): Karşılaştırma için referans metin
        scoring_dir (str): Puanlanacak dosyaların bulunduğu klasör
    """

    # Klasördeki tüm .txt dosyalarını al
    similar_files = glob.glob(os.path.join(scoring_dir, "*.txt"))  # glob modülünü kullan

    if not similar_files:
        print("Puanlanacak dosya bulunamadı.")
        return

    # Referans metni işle
    _, ref_lemmas, _ = process_text(reference_text)

    # TF-IDF modelini yükle
    try:
        tfidf_model = joblib.load('tfidf_models/tfidf_lemmatized.model')
    except Exception as e:
        print(f"TF-IDF modeli yüklenemedi: {str(e)}")
        return

    # Referans vektörü oluştur
    ref_vec = tfidf_model.transform([' '.join(ref_lemmas[0])])

    # Her dosyayı oku ve puanla
    results = []
    for file_path in similar_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Üretilen metni bul (dosya içeriğinden)
                generated_text = ""
                lines = content.split('\n')
                for line in lines:
                    if line.startswith("Üretilen Metin"):
                        generated_text = '\n'.join(lines[lines.index(line) + 1:])
                        break

                if not generated_text:
                    continue

                # Metni işle ve vektörleştir
                _, gen_lemmas, _ = process_text(generated_text)
                gen_vec = tfidf_model.transform([' '.join(gen_lemmas[0])])

                # Benzerlik skorunu hesapla
                similarity = cosine_similarity(ref_vec, gen_vec)[0][0]

                # Skoru 1-5 aralığına dönüştür
                score = min(5, max(1, round(similarity * 5)))

                results.append({
                    'dosya': os.path.basename(file_path),
                    'benzerlik': similarity,
                    'puan': score,
                    'metin': generated_text[:100] + "..."  # Önizleme
                })

        except Exception as e:
            print(f"{file_path} işlenirken hata: {str(e)}")

    # Sonuçları DataFrame'e dönüştür ve kaydet
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('benzerlik', ascending=False)

        # CSV olarak kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(scoring_dir, f"metin_puanlari_{timestamp}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')

        # Sonuçları göster
        print("\nMetin Puanlama Sonuçları:")
        print(df[['dosya', 'puan', 'benzerlik']].to_string(index=False))
        print(f"\nDetaylı sonuçlar kaydedildi: {output_file}")

        return df
    else:
        print("Puanlanabilir metin bulunamadı.")
        return None

if __name__ == "__main__":
    print("Metin işleme başlıyor...")
    lemmas, stems = process_files()
    print(f"İşlem tamamlandı. 8 lemmatized, 8 stemmed ve 2 TF-IDF modeli oluşturuldu.")

    # Örnek metin girişini test et
    sample_text = input("\nLütfen örnek bir cümle girin: ")
    process_input_text(sample_text)

    # Benzer metinler üret
    generated_texts = generate_similar_texts(sample_text)

    # Üretilen metinleri göster
    print("\nÜretilen Benzer Metinler:")
    for method, text in generated_texts.items():
        print(f"\n--- {method.upper()} ---")
        print(text[:500] + "..." if len(text) > 500 else text)

    # Üretilen metinleri puanla
    print("\n--- Metin Puanlama İşlemi ---")
    score_similar_texts(sample_text)

