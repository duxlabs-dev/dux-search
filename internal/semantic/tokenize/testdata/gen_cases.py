"""Builds cases.json: texts and the token ids Hugging Face `tokenizers` produces
for them with the model's tokenizer.json (add_special_tokens=True).

  uv venv .venv && uv pip install --python .venv/bin/python tokenizers
  .venv/bin/python gen_cases.py <model dir> [GOROOT] > cases.json

Sources of the texts (all redistributable): the model README (Apache-2.0), Go
standard library sources and docs (BSD-3-Clause, taken from GOROOT), this
script (hand-written Turkish, emoji, whitespace and edge-case strings).
Turkish and edge-case texts are written for this test; the rest are fixed
slices of the files above, so the same Go release reproduces the same file.
"""
import json, re, sys, os
from tokenizers import Tokenizer

model_dir = sys.argv[1]
goroot = sys.argv[2] if len(sys.argv) > 2 else "/usr/lib/go"
tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

cases = []
def add(group, text):
    cases.append({"group": group, "text": text})

def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def paragraphs(text, minlen=40):
    return [p.strip("\n") for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= minlen]

# English prose: model card.
readme = read(os.path.join(model_dir, "README.md"))
body = readme.split("---", 2)[-1]
prose = [p for p in paragraphs(body) if not p.lstrip().startswith(("```", "|", "<", "import", "const"))]
for p in prose[:30]:
    add("en-readme", p)
# English prose and HTML: Go memory model, godebug docs.
for name in ("go_mem.html", "godebug.md"):
    for p in paragraphs(read(os.path.join(goroot, "doc", name)))[:25]:
        add("en-doc", p)
# Code: Go source with tabs, braces, comments; shell, JSON and YAML-like text.
for rel in ("src/strings/strings.go", "src/net/http/cookie.go", "src/encoding/json/stream.go", "src/sort/sort.go"):
    src = read(os.path.join(goroot, rel))
    funcs = re.findall(r"(?ms)^func .*?^}\n", src)
    for fn in funcs[3:13]:
        add("code-go", fn[:1500])
for rel in ("src/make.bash", "src/run.bash"):
    p = os.path.join(goroot, rel)
    if os.path.exists(p):
        s = read(p)
        add("code-sh", s[:1200]); add("code-sh", s[1200:2600])
add("code-json", json.dumps({"id": "theme.dark", "title": "Dark theme", "keywords": ["dark", "karanlık"], "params": [{"name": "percent", "type": "int", "min": 0, "max": 100}]}, indent=2, ensure_ascii=False))
add("code-json", '{"hits":[{"kind":"action","id":"theme.dark","score":0.031,"semantic_score":0.76}],"took_ms":4}')
add("code-py", 'def embed(texts: list[str], dim: int = 768) -> np.ndarray:\n    """Return L2-normalized vectors."""\n    ids = tok.encode_batch(texts)\n    return v / np.linalg.norm(v, axis=1, keepdims=True)\n')
add("code-sql", "SELECT id, title\nFROM documents\nWHERE created_at > '2026-01-01' AND title LIKE '%fatura%'\nORDER BY created_at DESC LIMIT 20;")
add("code-html", "<!DOCTYPE html>\n<html lang=\"tr\"><head><meta charset=\"utf-8\"><title>Başlık</title></head>\n<body><div class=\"a\"><p>Merhaba <b>dünya</b></p><table><tr><td>1</td></tr></table><br/></div></body></html>")
add("code-cpp", '#include <vector>\nint main(int argc, char** argv) {\n  std::vector<int> v{1, 2, 3};\n  for (auto& x : v) { x *= 2; }\n  return 0;\n}\n')

# Turkish: launcher queries, app descriptions, prose, diacritics, mixed language.
tr_queries = ["karanlık", "gözlerim yoruldu", "kablosuz kulaklık", "fener", "sesi kıs", "parlaklığı %40 yap",
  "bildirimleri sustur", "pil tasarrufu", "ekranı kaydet", "15 dakika zamanlayıcı kur", "oda ışığını aç",
  "internet kapat", "bilgisayarı uyut", "pencereyi kapat", "mikrofonu kapat", "geçen haftaki fatura resmi",
  "hata ekran görüntüsü", "İstanbul'daki toplantı notları", "ışık", "ÇOCUK ŞARKILARI", "ığdır ılık süt",
  "dosya yöneticisi", "yazıcı ekle", "şifreyi değiştir", "uçak modu", "dil ve bölge ayarları", "Türkçe klavye düzeni",
  "bağlantı kesildi, yeniden dene", "kapat", "a", "ö", "I ı İ i", "iı", "wifi şifresi nedir?", "2026-10-30 tarihli rapor"]
for q in tr_queries:
    add("tr-query", q)
tr_prose = [
 "Karanlık tema, masaüstü renk düzenini koyu tonlara çevirir ve gece çalışırken gözleri yormaz.",
 "Gece Işığı açıldığında ekran renkleri sarımsı bir tona kayar; mavi ışık azalır, uykuya geçmek kolaylaşır.",
 "Rahatsız Etmeyin kipi bildirimleri susturur ve ekranda açılan uyarıları gizler. Sunum yaparken ya da film izlerken kullanışlıdır.",
 "Bluetooth açıldığında kablosuz kulaklıklar, hoparlörler ve klavyeler eşleştirilebilir.",
 "Dosyalarınızı anlamsal olarak arayabilirsiniz: \"geçen ayki kira sözleşmesi\" yazmanız yeterlidir; dosya adını hatırlamanız gerekmez.",
 "Çok güzel bir gün: çocuklar şarkı söylüyor, güneş parlıyor, kuşlar ötüyor. Öğleden sonra göğün rengi değişti ve yağmur başladı.",
 "Türkiye Cumhuriyeti'nin başkenti Ankara'dır; en kalabalık şehri ise İstanbul'dur. Ege kıyılarında zeytin yetiştirilir.",
 "Ayarlar > Ağ > Wi-Fi bölümünden kayıtlı ağları silebilir, parolayı yeniden girebilirsiniz.\nSorun sürerse yönlendiriciyi yeniden başlatın.",
 "Sayın kullanıcı, 12.345,67 TL tutarındaki fatura 15 Kasım 2026 tarihine kadar ödenmelidir. Gecikme halinde %2 faiz uygulanır.",
 "Toplantı notları:\n- Bütçe onaylandı (₺250.000)\n- Yeni sürüm 3.1.4 Cuma günü yayınlanacak\n- Test ekibi: Ayşe, Mehmet, Gökçe, Işıl\n",
 "Merhaba! Bu e-postayı size info@ornek.com.tr adresinden gönderiyoruz; https://ornek.com.tr/destek?konu=sifre&dil=tr adresini ziyaret edin.",
 "Yapay zekâ, makine öğrenmesi ve derin öğrenme yöntemleriyle metin, görüntü ve ses aynı vektör uzayına eşlenebilir.",
 "Kütüphane'den ödünç aldığım kitabı iade etmeyi unuttum; şimdi gecikme cezası ödemem gerekiyor galiba.",
 "ÇALIŞKAN ÖĞRENCİLER ÜZÜLMEZ. çalışkan öğrenciler üzülmez. Çalışkan Öğrenciler Üzülmez.",
 "Şu an: Açık — Enter: Kapat. Karanlık Tema [Şu an: Açık]",
 "Kullanıcı, ayarlar penceresinde \"anlamsal aramayı etkinleştir\" seçeneğini işaretledi ve modeli indirdi (842 MB).",
 "Dünyanın en uzun nehri hangisidir? Nil mi, Amazon mu? Kaynaklar farklı ölçümler verdiği için tartışma sürüyor.",
 "kedi köpek kuş balık at inek koyun keçi tavuk horoz hindi ördek kaz",
 "bir iki üç dört beş altı yedi sekiz dokuz on yüz bin milyon milyar",
 "Pazartesi, Salı, Çarşamba, Perşembe, Cuma, Cumartesi, Pazar; Ocak, Şubat, Mart, Nisan, Mayıs, Haziran, Temmuz, Ağustos, Eylül, Ekim, Kasım, Aralık.",
 "Ağaçların gölgesinde oturup çay içtik; semaverin başında ağır ağır sohbet ettik. Akşam olunca herkes evine döndü.",
 "Evlerinizdekilerle görüşebilirsiniz; gidebileceklerimizdendi, gelebilecekmişsiniz, yapamayacaklarımızdan. Çekoslovakyalılaştıramadıklarımızdan mısınız?",
 "Mühendislik fakültesi, bilgisayar bilimleri bölümü, yüksek lisans programı için başvurular 1 Haziran'da başlıyor.",
 "Şikâyet, hâlâ, kâğıt, mâni, rüzgâr, şûra, hükûmet, kâr, dâhil: şapkalı harflerle yazılan sözcükler.",
 "Not: Bu metin yalnızca test içindir. Gerçek kişilerle ilgisi yoktur; tüm adlar ve olaylar kurgusaldır.",
]
for p in tr_prose:
    add("tr-prose", p)
add("tr-prose", " ".join(tr_prose[:6]) * 12)  # long paragraph
add("tr-mixed", "Spotify'da \"Kırmızı Pazartesi\" çal, sonra volume'u 30'a getir ve Do Not Disturb'ü aç.")
add("tr-mixed", "git commit -m \"düzeltme: ışık seviyesi 100'ü aşınca taşma oluyordu\" && git push origin ana-dal")
add("tr-mixed", "Dockerfile'daki RUN komutu başarısız oldu: error: İzin reddedildi (permission denied)")

# Emoji and symbols.
emo = ["😀", "👍🏽", "🇹🇷🇺🇸", "👨‍👩‍👧‍👦", "❤️", "✨🎉", "🏳️‍🌈", "🧑🏻‍💻", "©®™ ½ ² ³ € £ ¥ ₺ § ¶", "→ ← ↑ ↓ ⇒ ≈ ≠ ≤ ≥ ∞ ∑ √ ∫",
       "Bugün hava çok güzel ☀️ ama yarın yağmur var 🌧️", "Harika iş! 👏👏👏 Teşekkürler 🙏", "🔥🔥🔥", "I ❤️ Linux 🐧", "text 😀😀 text😀text",
       "\u2603 \u2764 \u2705 \u274c", "𝕳𝖊𝖑𝖑𝖔 𝓦𝓸𝓻𝓵𝓭 𝟙𝟚𝟛", "ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３", "é é ñ ü ß ø å æ œ", "ﬁ ﬂ ﬃ ŉ ǆ ĳ"]
for e in emo:
    add("emoji", e)

# Other scripts.
other = ["The quick brown fox jumps over the lazy dog.", "Pack my box with five dozen liquor jugs!", "Привет, мир! Как дела?", "你好，世界！今天天气很好。", "こんにちは世界、お元気ですか",
         "안녕하세요 세계", "مرحبا بالعالم، كيف حالك؟", "שלום עולם", "नमस्ते दुनिया", "Γειά σου Κόσμε", "สวัสดีชาวโลก", "Hallo Welt, wie geht's? Größe, Straße, Müller.",
         "Bonjour le monde, ça va ? Où est l'hôpital ?", "¿Cómo estás? ¡Muy bien, gracias! Mañana vamos al niño.", "Zażółć gęślą jaźń", "Příliš žluťoučký kůň úpěl ďábelské ódy",
         "1234567890 3.14159 1e-9 0xDEADBEEF 1,000,000 ½ ¼ 99.9%", "a+b=c; (x*y)/z; {k: v}; [1, 2, 3]; <a, b>; a->b; a<=b; a>=b; a!=b; a==b; a&&b || !c",
         "@user #hashtag $VAR %PATH% ~/home /usr/bin/env python3 C:\\Users\\test\\file.txt", "foo_bar fooBar FooBar foo-bar foo.bar foo::bar foo/bar"]
for o in other:
    add("multi", o)

# Whitespace and newlines.
ws = ["", " ", "  ", "   leading", "trailing   ", "  both  ", "a  b", "a   b    c", "a\tb", "\ta", "a\n", "\n", "\n\n", "\n\n\n", "a\nb", "a\n\nb", "a\n\n\nb", "a \n b", " \n ", "line1\r\nline2\r\n", "\r", "a\rb",
      "\u00a0nbsp\u00a0", "zero\u200bwidth", "tab\t\tdouble", "x\u3000y", "\u2028\u2029", "a\x0bb\x0cc", "end with space ", "  \n  indented\n    more\n\n  ", ".", "..", "...", "a.", ".a", "-", "--", "---", "_", "__init__", "'", '"', "``", "`code`"]
for w in ws:
    add("whitespace", w)

# Special tokens and user-defined pieces appearing literally in text.
sp = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<bos>hello<eos>", "a<eos>b", "<|image|>", "<|image>x<image|>", "<|audio>a<audio|>", "<|video|>", "<|turn>user\nhi<turn|>\n", "<|think|> <|channel>x<channel|>",
      "<|tool_call>f()<tool_call|>", '<|"|>q<|"|>', "<unused0> <unused5>", "[multimodal]", "<start_of_turn>user", "<end_of_turn>", "<table><tr><td>", "<b>bold</b> <i>it</i> <br>", "<h1>", "<div class=\"x\">", "<<bos>>", "<bos", "bos>", "<|", "<|image", "<|image||>", "</s><s>", "<s>", "[CLS] [SEP] [PAD]", "<0x0A>", "<0x41>"]
for s in sp:
    add("special", s)

seen = set()
out = []
for c in cases:
    if c["text"] in seen:
        continue
    seen.add(c["text"])
    c["ids"] = tok.encode(c["text"]).ids
    out.append(c)
json.dump({"model": "google/embeddinggemma-2 tokenizer.json", "tokenizers": __import__("tokenizers").__version__, "cases": out}, sys.stdout, ensure_ascii=False, indent=1)
print(len(out), "cases", file=sys.stderr)
