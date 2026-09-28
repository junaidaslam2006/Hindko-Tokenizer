# -*- coding: utf-8 -*-
"""Non-Hindko check set for Track B (does the extension leave the base encoding of other text unchanged?).

Nothing is downloaded. Sources:
  english_wiki      12 Wikipedia-style English paragraphs WRITTEN FOR THIS CHECK (generated, not copied)
  python_code       10 whole modules of the local CPython 3.11 standard library (paths and sha256 recorded)
  other_scripts     short generated paragraphs in Devanagari (Hindi), Cyrillic, Greek, Han, Kana, Hangul,
                    plus a symbols/emoji line
  urdu_generated    9 Urdu news paragraphs WRITTEN FOR THIS CHECK (generated): same script as Hindko, so the
                    extension is EXPECTED to change their encoding; measured, not gated
  urdu_news_dev     every dev_permissive (validation split, permissive tier) newspaper or web document labelled
                    language_variety 'urdu' (dev_strict has no Urdu-labelled document)
  urdu_book_dev     every dev_permissive book document labelled 'urdu' (larger Urdu measurement)
  english_dev       every dev_permissive document labelled 'english' (English inside the Hindko corpus)
  bulk_code         (listed by path + sha256, read at evaluation time) every .py module of the local CPython 3.11
                    standard library outside test/, tests/, idlelib/ and site-packages/
  bulk_english      (listed likewise) every *.dist-info/METADATA file (package READMEs, mostly English prose)
                    in the local site-packages
Every text is passed through hp.normalize 1.0.1 (the canonical data form, applied to all evaluation text); the
normalised text is what is stored and encoded.

    python make_checkset.py      -> checkset.json, bulk_manifest.json
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tb_common as C  # noqa: E402

import sysconfig  # noqa: E402
PYLIB = sysconfig.get_paths()["stdlib"]

ENGLISH = [
    "Peshawar is the capital of the Khyber Pakhtunkhwa province of Pakistan and one of the oldest continuously "
    "inhabited cities in South Asia. Located near the eastern end of the Khyber Pass, it served for centuries as a "
    "trading post between Central Asia and the Indian subcontinent. According to the 2017 census, the city had a "
    "population of about 1.97 million, making it the sixth-largest city in the country.",
    "The Hazara Division is an administrative division of Khyber Pakhtunkhwa. It comprises the districts of "
    "Abbottabad, Haripur, Mansehra, Battagram, Torghar, Upper Kohistan, Lower Kohistan and Kolai-Palas. The region "
    "is known for its mountainous terrain, pine forests and hill stations such as Nathia Gali and Thandiani, which "
    "attract tourists during the summer months.",
    "Hindko is an Indo-Aryan language spoken by several million people in northwestern Pakistan. It is closely "
    "related to the Punjabi dialects of the Potohar plateau and is written in a modified Perso-Arabic script. "
    "Linguists have described several regional varieties, including those of Peshawar, Kohat and the Hazara "
    "Division, which differ in vocabulary and in the realisation of tone.",
    "Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into "
    "chemical energy. During the light-dependent reactions, water molecules are split and oxygen is released as a "
    "by-product. The energy captured in ATP and NADPH is then used in the Calvin cycle to fix carbon dioxide into "
    "sugars such as glucose (C6H12O6).",
    "The Roman Empire reached its greatest territorial extent in AD 117, under the emperor Trajan, when it covered "
    "roughly 5 million square kilometres. Its road network, which eventually exceeded 400,000 km, connected "
    "provinces from Britain to Mesopotamia and allowed troops, officials and merchants to travel quickly across "
    "the Mediterranean world.",
    "Python is a high-level, general-purpose programming language first released by Guido van Rossum in 1991. Its "
    "design philosophy emphasises code readability through the use of significant indentation. Python supports "
    "multiple programming paradigms, including structured, object-oriented and functional programming, and is "
    "often described as a \"batteries included\" language because of its comprehensive standard library.",
    "The Moon is Earth's only natural satellite. With a mean radius of 1,737.4 km, it is the fifth-largest moon in "
    "the Solar System. Its orbital period of about 27.3 days is equal to its rotation period, so the same side "
    "always faces Earth. The first crewed landing took place on 20 July 1969, during the Apollo 11 mission.",
    "Mount Everest, known locally as Sagarmatha or Chomolungma, is Earth's highest mountain above sea level. Its "
    "elevation of 8,848.86 m was most recently established in 2020 by Chinese and Nepali authorities. The "
    "international border between Nepal and China runs across its summit point.",
    "The FIFA World Cup is an international association football competition contested by the senior men's "
    "national teams of the members of FIFA. It has been held every four years since the inaugural tournament in "
    "1930, except in 1942 and 1946 because of the Second World War. Brazil has won the tournament a record five "
    "times.",
    "The printing press is a mechanical device for applying pressure to an inked surface resting upon a print "
    "medium, thereby transferring the ink. Johannes Gutenberg's movable-type press, developed around 1440, "
    "reduced the cost of books dramatically; by 1500, printing shops in Western Europe had produced more than "
    "twenty million volumes.",
    "Deoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to "
    "form a double helix. The two strands are held together by hydrogen bonds between the bases adenine (A), "
    "thymine (T), guanine (G) and cytosine (C). The structure was described by James Watson and Francis Crick in "
    "1953, using X-ray data obtained by Rosalind Franklin.",
    "The Indus River is one of the longest rivers in Asia, flowing for about 3,180 km from the Tibetan Plateau "
    "through Ladakh and Gilgit-Baltistan before turning south along the length of Pakistan to the Arabian Sea near "
    "Karachi. The Tarbela Dam, completed in 1976 in the Haripur and Swabi districts, is one of the largest "
    "earth-filled dams in the world.\nSee also: List of rivers of Pakistan; Indus Waters Treaty (1960).",
]

OTHER_SCRIPTS = [
    ("hindi_devanagari", "नई दिल्ली: भारतीय मौसम विज्ञान विभाग ने अगले दो दिनों में उत्तर भारत के कई राज्यों में भारी बारिश "
                         "की चेतावनी जारी की है। विभाग के अनुसार पहाड़ी इलाकों में भूस्खलन का खतरा बना हुआ है, इसलिए "
                         "यात्रियों को सावधानी बरतने की सलाह दी गई है।"),
    ("russian_cyrillic", "Москва — столица России и крупнейший по численности населения город страны. Город "
                         "расположен на реке Москве в центре Восточно-Европейской равнины. В 2021 году население "
                         "Москвы составило более 12,6 миллиона человек."),
    ("greek", "Η Αθήνα είναι η πρωτεύουσα και η μεγαλύτερη πόλη της Ελλάδας. Θεωρείται μία από τις αρχαιότερες "
              "πόλεις του κόσμου, με καταγεγραμμένη ιστορία που εκτείνεται σε περισσότερα από 3.400 χρόνια."),
    ("chinese_han", "北京是中华人民共和国的首都，也是全国的政治、文化和国际交往中心。北京有三千多年的建城史，"
                    "拥有故宫、天坛和长城等多处世界文化遗产。"),
    ("japanese_kana", "東京は日本の首都であり、世界有数の大都市です。毎年多くの観光客が浅草や渋谷を訪れ、"
                      "伝統的な文化と最新の技術の両方を楽しんでいます。"),
    ("korean_hangul", "서울은 대한민국의 수도이자 최대 도시이다. 한강을 중심으로 발달한 서울은 오랜 역사와 "
                      "현대적인 문화가 공존하는 도시로 알려져 있다."),
    ("symbols_emoji", "Temperature: 23.5 °C ± 0.2; area ≈ 1.2 km²; E = mc²; ∑ᵢ xᵢ ≥ 0; price € 12,99 → $ 14.10 "
                      "🙂🚀✅ — “quoted” ‘text’ … © 2026 · ½ ¾ ¼ • ★"),
]

URDU = [
    "اسلام آباد: وفاقی حکومت نے آئندہ مالی سال کے بجٹ میں تعلیم کے شعبے کے لیے مختص رقم میں پندرہ فیصد اضافے کا "
    "اعلان کیا ہے۔ وزیرِ خزانہ نے قومی اسمبلی میں خطاب کرتے ہوئے کہا کہ سرکاری اسکولوں میں بنیادی سہولیات کی فراہمی "
    "حکومت کی اولین ترجیح ہے۔",
    "لاہور: محکمہ موسمیات نے آئندہ چوبیس گھنٹوں کے دوران پنجاب کے بیشتر علاقوں میں تیز ہواؤں کے ساتھ بارش کی پیش "
    "گوئی کی ہے۔ شہریوں کو ہدایت کی گئی ہے کہ وہ غیر ضروری سفر سے گریز کریں اور بجلی کے کھمبوں سے دور رہیں۔",
    "کراچی: اسٹیٹ بینک آف پاکستان نے شرح سود کو بارہ فیصد پر برقرار رکھنے کا فیصلہ کیا ہے۔ مرکزی بینک کے اعلامیے کے "
    "مطابق مہنگائی کی شرح میں مسلسل کمی دیکھی جا رہی ہے، تاہم بیرونی ادائیگیوں کا دباؤ اب بھی برقرار ہے۔",
    "پشاور: خیبر پختونخوا کے وزیرِ صحت نے صوبے کے تمام اضلاع میں پولیو کے خلاف پانچ روزہ مہم کا افتتاح کر دیا۔ مہم "
    "کے دوران پچاس لاکھ سے زائد بچوں کو قطرے پلائے جائیں گے، جس کے لیے ہزاروں رضاکاروں کی خدمات حاصل کی گئی ہیں۔",
    "ایبٹ آباد: ہزارہ یونیورسٹی میں علاقائی زبانوں کے فروغ کے حوالے سے دو روزہ کانفرنس منعقد ہوئی جس میں ملک بھر سے "
    "ماہرینِ لسانیات نے شرکت کی۔ مقررین نے زور دیا کہ مادری زبانوں میں ابتدائی تعلیم بچوں کی ذہنی نشوونما کے لیے "
    "ناگزیر ہے۔",
    "کھیل: پاکستان کرکٹ ٹیم نے تیسرے ایک روزہ میچ میں سری لنکا کو چھ وکٹوں سے شکست دے کر سیریز دو ایک سے اپنے نام کر "
    "لی۔ کپتان نے میچ کے بعد گفتگو میں کہا کہ نوجوان کھلاڑیوں کی کارکردگی نے ٹیم کا اعتماد بحال کیا ہے۔",
    "معیشت: رواں مالی سال کے پہلے چھ ماہ میں ترسیلاتِ زر میں گزشتہ سال کے مقابلے میں اٹھارہ فیصد اضافہ ریکارڈ کیا گیا "
    "ہے۔ ماہرین کے مطابق بیرونِ ملک مقیم پاکستانیوں کی جانب سے رقوم کی منتقلی میں اضافہ زرمبادلہ کے ذخائر کے استحکام "
    "میں مدد دے گا۔",
    "کوئٹہ: بلوچستان حکومت نے صوبے میں پانی کی قلت پر قابو پانے کے لیے تین نئے ڈیموں کی تعمیر کی منظوری دے دی ہے۔ "
    "ترجمان کے مطابق منصوبوں پر مجموعی طور پر چالیس ارب روپے لاگت آئے گی اور یہ دو سال میں مکمل کیے جائیں گے۔",
    "مظفرآباد (۱۲ مارچ ۲۰۲۶ء): آزاد کشمیر میں برف باری کے بعد بند ہونے والی شاہراہیں 48 گھنٹوں میں کھول دی گئیں۔\n"
    "انتظامیہ کے مطابق 1,250 سے زائد مسافروں کو محفوظ مقامات پر منتقل کیا گیا۔",
]

PY_CURATED = ["json/encoder.py", "json/decoder.py", "textwrap.py", "heapq.py", "bisect.py", "fnmatch.py",
              "colorsys.py", "shlex.py", "string.py", "contextlib.py"]


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    norm = C.normalizer()
    sets = {}

    def add(set_name, doc_id, raw, prov):
        t = norm(raw)
        sets.setdefault(set_name, []).append({"id": doc_id, "text": t, "raw_sha256": sha(raw), "sha256": sha(t),
                                              "changed_by_normalize": t != raw, "provenance": prov})
    for i, t in enumerate(ENGLISH):
        add("english_wiki", "en%02d" % i, t, "generated for this check (Wikipedia-style)")
    for rel in PY_CURATED:
        p = os.path.join(PYLIB, rel.replace("/", os.sep))
        raw = open(p, encoding="utf-8").read()
        add("python_code", rel, raw, "CPython 3.11 stdlib %s (sha256 of file %s)" % (rel, C.sha256_file(p)))
    for name, t in OTHER_SCRIPTS:
        add("other_scripts", name, t, "generated for this check")
    for i, t in enumerate(URDU):
        add("urdu_generated", "ur%02d" % i, t, "generated for this check (Urdu news style)")
    dev = C.load_view("dev_permissive")
    for d in dev:
        if d["variety"] == "urdu" and d["source"] in ("newspaper", "web"):
            add("urdu_news_dev", d["uid"], d["text"], "dev_permissive (validation) %s uid %s" % (d["source"], d["uid"]))
        elif d["variety"] == "urdu" and d["source"] == "book":
            add("urdu_book_dev", d["uid"], d["text"], "dev_permissive (validation) book uid %s" % d["uid"])
        elif d["variety"] == "english":
            add("english_dev", d["uid"], d["text"], "dev_permissive (validation) %s uid %s" % (d["source"], d["uid"]))
    for s, docs in sets.items():
        for d in docs:
            d["has_arabic"] = C.has_arabic(d["text"])
    out = {"what": "Track B non-Hindko check set (curated part)", "normalize": C.verify_frozen()["normalize_version"],
           "sets": {s: {"docs": len(v), "chars": sum(len(d["text"]) for d in v),
                        "bytes": sum(len(d["text"].encode("utf-8")) for d in v),
                        "docs_with_arabic_script": sum(d["has_arabic"] for d in v)} for s, v in sets.items()},
           "docs": sets}
    C.dump_json(out, os.path.join(HERE, "checkset.json"))
    # bulk lists
    py = []
    for root, dirs, files in os.walk(PYLIB):
        rl = os.path.relpath(root, PYLIB)
        parts = rl.split(os.sep)
        if any(p in ("site-packages", "test", "tests", "idlelib", "__pycache__") for p in parts):
            dirs[:] = []
            continue
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(root, f)
                py.append({"path": p, "sha256": C.sha256_file(p), "bytes": os.path.getsize(p)})
    sp = os.path.join(PYLIB, "site-packages")
    md = []
    for d in sorted(os.listdir(sp)):
        p = os.path.join(sp, d, "METADATA")
        if d.endswith(".dist-info") and os.path.exists(p):
            md.append({"path": p, "sha256": C.sha256_file(p), "bytes": os.path.getsize(p)})
    C.dump_json({"what": "Track B bulk non-Hindko check lists (local files, read at evaluation time)",
                 "bulk_code": py, "bulk_english": md}, os.path.join(HERE, "bulk_manifest.json"))
    print(json.dumps(out["sets"], indent=1))
    print("bulk_code", len(py), sum(x["bytes"] for x in py), "bulk_english", len(md), sum(x["bytes"] for x in md))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
