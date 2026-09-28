"""Curated inventory of Hindko inflectional morphology (Perso-Arabic spelling).

This module is DATA: every item states its spelling as used in this corpus,
its gloss, and where the claim comes from. Corpus counts, examples and the
distributional dialect evidence are attached by build_inventory.py.

Source keys (full references in MORPHOLOGY.md):
  RRS2011  Raja, Rashid & Sohail (2011) 'A Brief Introduction of Hindko
           Language', Language in India 11(11): 471-482 (open access PDF;
           variety described: Muzaffarabad / Hazara belt). Consulted in full.
  SH1980   Shackle (1980) 'Hindko in Kohat and Peshawar', BSOAS 43(3):
           482-510. NOT read in full (paywalled); only the statements quoted
           from it on Wikipedia 'Kohati' (oblique -e, dative postposition a~,
           oblique plural -a~) are used.
  SH2010   Shackle (2010), as quoted on Wikipedia 'Lahnda': Lahnda varieties
           have a future tense in -s-.
  BC2019   Bashir & Conners (2019) 'A Descriptive Grammar of Hindko, Panjabi,
           and Saraiki' (De Gruyter Mouton). Only the publisher / LINGUIST
           List description was legitimately accessible (Abbottabad variety;
           covers phonology, orthography, morphology). NO claim below is
           attributed to its content, which was not consulted.
  BANO     Aftab Iqbal Bano, 'Aao Hindko zaban sikhne aan' (Hindko Bol Chal,
           Gandhara Hindko Academy, Peshawar, Farma 131), a Hindko phrasebook
           with Urdu glosses. It is IN this corpus (permissive tier, uids
           3249e9c0c57cb152, ce6a5f4580f650be, 7754db37b7b206bd, ...):
           lesson 5 = pronoun table, lesson 8 = present / past / future.
           Peshawari.
  CORPUS   Frequencies in the strict release by dialect group
           (build_counts.py); a distributional signal, not a grammar claim.

Dialect labels: 'Peshawari', 'Hazara', 'shared' (attested in both), or
'unknown'. dialect_lit is what a source says; build_inventory.py adds the
corpus-distribution label separately so the two can be compared.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Suffix inventory used by the induction engine.
#   suffix  : spelling (no harakat)
#   family  : VERB / NOUN / AGR (agreement: adjectives, possessives,
#             suppletive perfectives)
#   slot    : paradigm slot (for VERB: IPFV PFV FUT INF SBJV CONJ)
#   stem    : 'C' consonant-final stem, 'V' vowel-final stem (ends in
#             ا و ے ی ہ), 'any'
#   gloss   : Leipzig-style gloss
#   opt     : offsets INSIDE the suffix where a further split is
#             linguistically defensible (e.g. -د|ا participle + agreement,
#             -س|ی future + person). Splits there are neither rewarded nor
#             penalised by morph_eval.py.
#   target  : True if words with this suffix may enter the gold set
# ---------------------------------------------------------------------------
V = 'VERB'
VERB_SUFFIXES = [
    # imperfective participle (RRS2011: paR-daa 'reads'; BANO: کردا، ہوندا)
    dict(suffix='دا', slot='IPFV', stem='C', gloss='IPFV.PTCP.M.SG', opt=[1], target=True),
    dict(suffix='دی', slot='IPFV', stem='C', gloss='IPFV.PTCP.F.SG', opt=[1], target=True),
    dict(suffix='دے', slot='IPFV', stem='C', gloss='IPFV.PTCP.M.PL/OBL', opt=[1], target=True),
    dict(suffix='دیاں', slot='IPFV', stem='C', gloss='IPFV.PTCP.F.PL', opt=[1, 2], target=True),
    # after vowel-final stems the participle is -ndaa (ہوندا، جاندا، آندا)
    dict(suffix='ندا', slot='IPFV', stem='V', gloss='IPFV.PTCP.M.SG', opt=[1, 2], target=True),
    dict(suffix='ندی', slot='IPFV', stem='V', gloss='IPFV.PTCP.F.SG', opt=[1, 2], target=True),
    dict(suffix='ندے', slot='IPFV', stem='V', gloss='IPFV.PTCP.M.PL/OBL', opt=[1, 2], target=True),
    dict(suffix='ندیاں', slot='IPFV', stem='V', gloss='IPFV.PTCP.F.PL', opt=[1, 2, 3], target=True),
    # perfective participle (RRS2011: paR-yaa 'read (past)'); Hindko -iaa on
    # consonant stems (لکھیا، آخیا) where Urdu has -aa
    dict(suffix='یا', slot='PFV', stem='any', gloss='PFV.PTCP.M.SG', opt=[], target=True),
    dict(suffix='ی', slot='PFV', stem='C', gloss='PFV.PTCP.F.SG', opt=[], target=True),
    dict(suffix='ے', slot='PFV', stem='C', gloss='PFV.PTCP.M.PL | SBJV.3SG', opt=[], target=True),
    dict(suffix='یاں', slot='PFV', stem='any', gloss='PFV.PTCP.F.PL', opt=[1], target=True),
    # Urdu-like -aa perfective, also used in this corpus (BANO: پڑھا ایا; چلا، بیٹھا)
    dict(suffix='ا', slot='PFV', stem='C', gloss='PFV.PTCP.M.SG (-aa)', opt=[], target=True),
    dict(suffix='ئی', slot='PFV', stem='V', gloss='PFV.PTCP.F.SG', opt=[], target=True),
    dict(suffix='ئے', slot='PFV', stem='V', gloss='PFV.PTCP.M.PL', opt=[], target=True),
    dict(suffix='ئیاں', slot='PFV', stem='V', gloss='PFV.PTCP.F.PL', opt=[2], target=True),
    # -s- future (SH2010; RRS2011 paR-sii; BANO: دیساں 1SG, آسیں 2SG,
    # جاسی 3SG, جاسُن 3PL; 1PL written apart: جاسی ایں)
    dict(suffix='سی', slot='FUT', stem='any', gloss='FUT.3SG', opt=[1], target=True),
    dict(suffix='ساں', slot='FUT', stem='any', gloss='FUT.1SG', opt=[1], target=True),
    dict(suffix='سیں', slot='FUT', stem='any', gloss='FUT.2SG', opt=[1], target=True),
    dict(suffix='سن', slot='FUT', stem='any', gloss='FUT.3PL', opt=[1], target=True),
    dict(suffix='سو', slot='FUT', stem='any', gloss='FUT.2PL', opt=[1], target=True),
    # infinitive / verbal noun. Retroflex n is spelled نڑ in this corpus.
    # -نا/-نے after r-final stems (کرنا، کرنے); -نڑاں/-نڑیں Peshawari
    # direct/oblique (جانڑاں، ہونڑیں ولے); -نڑا/-نڑے elsewhere (books, Hazara)
    dict(suffix='نا', slot='INF', stem='any', gloss='INF | PRS.1 (-naa, BANO: دینا واں)', opt=[1], target=True),
    dict(suffix='نے', slot='INF', stem='any', gloss='INF.OBL', opt=[1], target=True),
    dict(suffix='نڑاں', slot='INF', stem='any', gloss='INF', opt=[2], target=True),
    dict(suffix='نڑا', slot='INF', stem='any', gloss='INF', opt=[2], target=True),
    dict(suffix='نڑیں', slot='INF', stem='any', gloss='INF.OBL', opt=[2], target=True),
    dict(suffix='نڑے', slot='INF', stem='any', gloss='INF.OBL', opt=[2], target=True),
    dict(suffix='نڑ', slot='INF', stem='any', gloss='INF.SHORT', opt=[], target=False),
    # Peshawari -naa~ after r-stems (کرناں); feminine agreement of the
    # infinitive in obligation clauses (BANO: پریکٹس کرنی وے 'must practise')
    dict(suffix='ناں', slot='INF', stem='any', gloss='INF', opt=[1], target=True),
    dict(suffix='نی', slot='INF', stem='any', gloss='INF.F.SG', opt=[1], target=True),
    dict(suffix='نیاں', slot='INF', stem='any', gloss='INF.F.PL', opt=[1, 2], target=True),
    dict(suffix='نڑی', slot='INF', stem='any', gloss='INF.F.SG', opt=[2], target=True),
    dict(suffix='نڑیاں', slot='INF', stem='any', gloss='INF.F.PL', opt=[2, 3], target=True),
    # subjunctive / imperative (consonant stems: کراں کریں کرے کرو کرن;
    # vowel stems take a glide: ہووے جاوے ہوون)
    dict(suffix='اں', slot='SBJV', stem='C', gloss='SBJV.1SG', opt=[], target=True),
    dict(suffix='یں', slot='SBJV', stem='C', gloss='SBJV.2SG', opt=[], target=True),
    dict(suffix='و', slot='SBJV', stem='any', gloss='SBJV/IMP.2PL', opt=[], target=True),
    dict(suffix='ن', slot='SBJV', stem='C', gloss='SBJV.3PL | INF.OBL', opt=[], target=True),
    dict(suffix='واں', slot='SBJV', stem='V', gloss='SBJV.1SG', opt=[1], target=True),
    dict(suffix='ویں', slot='SBJV', stem='V', gloss='SBJV.2SG', opt=[1], target=True),
    dict(suffix='وے', slot='SBJV', stem='V', gloss='SBJV.3SG', opt=[1], target=True),
    dict(suffix='ون', slot='SBJV', stem='V', gloss='SBJV.3PL', opt=[1], target=True),
    # conjunctive participle
    dict(suffix='کے', slot='CONJ', stem='any', gloss='CONJ.PTCP', opt=[], target=True),
]
# causative formatives (build a new vowel-final stem): کر -> کرا / کروا
CAUS_FORMATIVES = [
    dict(suffix='وا', gloss='CAUS2'),
    dict(suffix='ا', gloss='CAUS'),
]
# nominal
NOUN_SUFFIXES = [
    # RRS2011 (7): chowk/chowkan, zat/zatan, gal/gallan; SH1980: oblique plural -a~
    dict(suffix='اں', slot='PL', gloss='PL (F) | OBL.PL (M)', opt=[], target=True),
    # SH1980: pUttUr -> obl pUtre; masc -a nouns: obl.sg / dir.pl -e
    dict(suffix='ے', slot='OBL', gloss='M.OBL.SG | M.PL', opt=[], target=True),
    # Hindko masc oblique plural -iaa~ (Urdu -on): علاقیاں، بندیاں، حلقیاں
    dict(suffix='یاں', slot='OBL.PL', gloss='M.OBL.PL', opt=[1], target=True),
]
# agreement (adjectives, possessive pronouns, suppletive perfectives)
AGR_SUFFIXES = [
    dict(suffix='ا', slot='M.SG', gloss='M.SG', opt=[], target=True),
    dict(suffix='ی', slot='F.SG', gloss='F.SG', opt=[], target=True),
    dict(suffix='ے', slot='M.PL', gloss='M.PL/OBL', opt=[], target=True),
    dict(suffix='یاں', slot='F.PL', gloss='F.PL', opt=[1], target=True),
]
# suppletive perfective stems (glossing only; they are found by AGR evidence)
SUPPLETIVE_PFV = {'کیت': 'کر (do)', 'دت': 'دے (give)', 'لت': 'لے (take)'}

# Closed-class words never segmented (pronouns, postpositions, auxiliaries,
# particles). Several look like stem+suffix to the engine (اساں = اس+اں,
# جناں = جن+اں), which is exactly why they are listed.
FUNCTION_WORDS = set('''
دا دی دے دیاں نوں کو کی آں اں نال سنگ اچ بچ وچ وچوں بچوں اچوں تے توں تھیں سی کولوں تک تائیں وسے واسطے آسطے لئی نے
اساں تساں انہاں اناں اینہاں ایناں جنہاں جناں کنہاں کناں سانوں سوانوں تسانوں اسی تسی اسیں تسیں میں تو اوہ او اس ایس اوس
منوں مینوں تنوں تینوں سانے منے تنے ازا ازی ازے ازیاں اوندا ایندا ایندے جیدی موکو ایہہ ایہ اے وے ہے ہن ون ان واں ویں ایں
ایا ائی ائے ایاں آسا آسی آسے اسا سا پیا پئی پئے پئیاں گیا گئی گئے گئیاں کیا کہ جو جس جنہاں کوئی کسی ہک اک ہر
ہونڑ ہونڑاں ہوراں جد جدو جدوں تد تدو اتھے ایتھے اوتھے جتھے کتھے اتھا اوتھا ایتھا کدی کدے جیہڑا جیہڑی جیہڑے جیہڑیاں جڑا
کیہڑا کیہڑی کیہڑے کیہڑیاں کہڑا کہڑی کہڑے ایہو اوہو اینج اونج جیویں کیویں ہنڑ
ایہا ایہی ایہے ایہیاں جیڑا جیڑی جیڑے جیڑیاں جیہڑی جیہڑے جیہڑیاں جہڑا جہڑی جہڑے جہڑیاں نالے تلے
'''.split())
# ہونڑ ('now') and ہونڑاں (honorific after names) are listed because they are
# homographs of forms of ہو 'be' and would otherwise enter as INF forms.

# ---------------------------------------------------------------------------
# Free-standing items (postpositions, clitics, pronouns, auxiliaries) for the
# inventory table. 'words' are the corpus spellings that are counted.
# ---------------------------------------------------------------------------
FREE_ITEMS = [
    # genitive
    dict(id='P.GEN', section='postposition', words=['دا', 'دی', 'دے', 'دیاں'],
         gloss='GEN (agrees with possessum: M.SG / F.SG / M.PL,OBL / F.PL)',
         dialect_lit='shared', sources='BANO (اُس دا = اُس کا); CORPUS'),
    # dative / accusative
    dict(id='P.DAT.nu', section='postposition', words=['نوں'], gloss='DAT/ACC',
         dialect_lit='Peshawari (BANO: سانوں، تنوں, اُس نوں)', sources='BANO; RRS2011 (nadeem-nu, Muzaffarabad); CORPUS'),
    dict(id='P.DAT.aan', section='postposition', words=['آں'], gloss='DAT/ACC (also 1SG copula / PL written apart)',
         dialect_lit='Kohat/Peshawar (SH1980: Kohati a~ dative)', sources='SH1980 via Wikipedia; CORPUS'),
    dict(id='P.DAT.ko', section='postposition', words=['کو', 'موکو'], gloss='DAT/ACC',
         dialect_lit='unknown (also Urdu); corpus: Hazara-enriched', sources='CORPUS'),
    # comitative / instrumental
    dict(id='P.COM', section='postposition', words=['نال', 'سنگ'], gloss='with (COM/INS)',
         dialect_lit='shared', sources='CORPUS'),
    # locative
    dict(id='P.LOC.ich', section='postposition', words=['اچ'], gloss='in (LOC)',
         dialect_lit='Peshawari (newspaper standard)', sources='CORPUS'),
    dict(id='P.LOC.bich', section='postposition', words=['بچ'], gloss='in (LOC)',
         dialect_lit='shared (Peshawari and Hazara)', sources='CORPUS'),
    dict(id='P.LOC.vich', section='postposition', words=['وچ', 'وچوں'], gloss='in (LOC); وچوں from within (-وں ABL)',
         dialect_lit='book / Panjabi-like spelling; BANO uses وچ', sources='BANO; CORPUS'),
    dict(id='P.LOC.te', section='postposition', words=['تے'], gloss='on (LOC); homograph of تے "and"',
         dialect_lit='shared', sources='RRS2011 (-te locative); CORPUS'),
    # ablative
    dict(id='P.ABL.ton', section='postposition', words=['توں'], gloss='from (ABL); homograph of توں "you"',
         dialect_lit='shared', sources='CORPUS'),
    dict(id='P.ABL.thin', section='postposition', words=['تھیں'], gloss='from (ABL)',
         dialect_lit='Hazara (hp/lang.py R10 note: Hazara-Hindko form)', sources='hp/lang.py; CORPUS'),
    dict(id='P.ABL.si', section='postposition', words=['سی'], gloss='from (ABL); homograph of FUT.3SG سی and Hazara "was"',
         dialect_lit='Peshawari (BANO: سواڈے سی = آپ سے; دفتر سی = دفتر سے)', sources='BANO; CORPUS'),
    dict(id='P.ABL.kolon', section='postposition', words=['کولوں'], gloss='from / by',
         dialect_lit='shared (BANO: میرے کولوں = مجھ سے)', sources='BANO; CORPUS'),
    # benefactive
    dict(id='P.BEN', section='postposition', words=['وسے', 'واسطے', 'آسطے', 'لئی'], gloss='for (BEN)',
         dialect_lit='وسے Peshawari (BANO: میرے وسے = میرے لئے)', sources='BANO; CORPUS'),
    # ergative
    dict(id='P.ERG', section='postposition', words=['نے'], gloss='ERG',
         dialect_lit='shared (RRS2011 also reports ergative sun)', sources='RRS2011; BANO; CORPUS'),
    # pronouns
    dict(id='PR.1PL', section='pronoun', words=['اسی', 'اسیں', 'اساں', 'سانوں', 'سانے'],
         gloss='1PL: NOM اسی/اسیں, OBL/ERG اساں, DAT سانوں, ERG سانے',
         dialect_lit='اسی/سانے Peshawari (BANO: اسی = ہم, سانے = ہم نے); اسیں Hazara-enriched in corpus',
         sources='BANO; CORPUS'),
    dict(id='PR.2PL', section='pronoun', words=['تسی', 'تسیں', 'تساں', 'سوانوں'],
         gloss='2PL/HON: NOM تسی/تسیں, OBL تساں, DAT سوانوں',
         dialect_lit='تسی/سوانوں Peshawari (BANO: تسی = آپ, سوانوں = آپ کو)', sources='BANO; CORPUS'),
    dict(id='PR.3PL', section='pronoun', words=['انہاں', 'اناں', 'جنہاں', 'جناں', 'کنہاں'],
         gloss='3PL / REL / INTERR oblique (-ہاں)', dialect_lit='اناں spelling Peshawari newspaper; انہاں elsewhere',
         sources='BANO (اُنہاں = اُنھیں / اُن; کنہاں = کنہوں); CORPUS'),
    dict(id='PR.1SG', section='pronoun', words=['میں', 'منوں', 'مینوں', 'منے', 'موکو'],
         gloss='1SG: NOM میں, DAT منوں/مینوں/موکو, ERG منے',
         dialect_lit='منوں/منے Peshawari (BANO); مینوں Panjabi-like; موکو Hazara-enriched in corpus',
         sources='BANO; CORPUS'),
    dict(id='PR.2SG', section='pronoun', words=['تو', 'توں', 'تنوں', 'تینوں', 'تنے'],
         gloss='2SG: NOM تو/توں, DAT تنوں/تینوں, ERG تنے', dialect_lit='تنوں/تنے Peshawari (BANO)',
         sources='BANO; CORPUS'),
    dict(id='PR.3SG', section='pronoun', words=['اوہ', 'او', 'اس', 'ایس', 'ایہہ'],
         gloss='3SG / DEM', dialect_lit='shared', sources='BANO; CORPUS'),
    dict(id='PR.GEN3', section='pronoun', words=['ازا', 'ازی', 'ازے', 'اسدا', 'ایندا', 'ایندے'],
         gloss='3SG.GEN (his/her/its), agreeing', dialect_lit='ازا Peshawari (BANO: اُزا = اُس کا); ایندا Hazara-enriched in corpus',
         sources='BANO; CORPUS'),
    dict(id='PR.POSS', section='pronoun', words=['میرا', 'تیرا', 'ساڈا', 'سواڈا', 'تہاڈا', 'آپڑا', 'اپڑا', 'اپنا'],
         gloss='possessives, inflect like -aa adjectives (-ا/-ی/-ے/-یاں)',
         dialect_lit='سواڈا Peshawari HON (BANO: سُواڈا = آپ کا); آپڑا Peshawari newspaper; اپنا Urdu/Hazara',
         sources='BANO; CORPUS'),
    # copula / auxiliaries
    dict(id='AUX.PRS', section='auxiliary', words=['وے', 'اے', 'ہے', 'ون', 'ہن', 'ان', 'واں', 'ویں'],
         gloss='present copula: 3SG وے/اے, 3PL ون/ہن/ان, 1SG واں, 2SG ویں',
         dialect_lit='وے/ون Peshawari (BANO: کردا وے, ہسپتال وچ داخل اے)', sources='BANO; CORPUS'),
    dict(id='AUX.PST.P', section='auxiliary', words=['ایا', 'ائی', 'ائے', 'ایاں'],
         gloss='past copula (was): M.SG / F.SG / M.PL / F.PL',
         dialect_lit='Peshawari (BANO: گیا ایا = گیا تھا; گئے ائے = گئے تھے)', sources='BANO; CORPUS'),
    dict(id='AUX.PST.H', section='auxiliary', words=['آسا', 'اسا', 'سا', 'آسی', 'آسے'],
         gloss='past copula (was), Hazara', dialect_lit='Hazara (corpus: over-represented in Hazara web/speech sources)',
         sources='CORPUS'),
    dict(id='AUX.PROG', section='auxiliary', words=['پیا', 'پئی', 'پئے', 'پئیاں'],
         gloss='progressive auxiliary (کردا پیا وے "is doing")',
         dialect_lit='Peshawari (BANO: پیا دینا واں; کے پئے کردین)', sources='BANO; CORPUS'),
]

# ---------------------------------------------------------------------------
# Inventory rows for bound morphology (counts/examples added from the engine)
# ---------------------------------------------------------------------------
BOUND_ITEMS = [
    dict(id='N.PL', section='noun', suffix='-اں', gloss='PL of F nouns (DIR+OBL); OBL.PL of consonant-final M nouns',
         dialect_lit='shared', sources='RRS2011 (chowkan, zatan, gallan); SH1980 (obl.pl -a~); BANO (شاگرداں, کتاباں)'),
    dict(id='N.OBL', section='noun', suffix='-ے', gloss='OBL.SG / DIR.PL of M -aa/-ah nouns (بندہ -> بندے)',
         dialect_lit='shared', sources='SH1980 (pUttUr -> pUtre); CORPUS'),
    dict(id='N.OBL.PL', section='noun', suffix='-یاں', gloss='OBL.PL of M -aa/-ah nouns (علاقہ -> علاقیاں; Urdu -وں)',
         dialect_lit='shared', sources='CORPUS; BANO writes the same ending apart: بچے اں، جملے آں'),
    dict(id='N.F.PL', section='noun', suffix='-(ی)اں', gloss='PL of F -ii nouns (کڑی -> کڑیاں)',
         dialect_lit='shared', sources='CORPUS'),
    dict(id='ADJ.AGR', section='adjective', suffix='-ا / -ی / -ے / -یاں', gloss='agreement of -aa adjectives, possessives',
         dialect_lit='shared (lexical choice differs: بڈا Peshawari, وڈا general)', sources='BANO; CORPUS'),
    dict(id='V.IPFV', section='verb', suffix='-دا / -دی / -دے / -دیاں ; -ندا ... after vowels',
         gloss='imperfective participle', dialect_lit='shared',
         sources='RRS2011 (paR-daa, likh-Daa); BANO (کردا، ہوندا)'),
    dict(id='V.PFV', section='verb', suffix='-یا / -ی / -ے / -یاں ; -یا / -ئی / -ئے / -ئیاں after vowels',
         gloss='perfective participle (suppletive کیت- دت- لت- گ-)', dialect_lit='shared',
         sources='RRS2011 (paR-yaa, tur-yaa); BANO (پڑھا، آیا، گیا)'),
    dict(id='V.FUT', section='verb', suffix='-سی 3SG / -ساں 1SG / -سیں 2SG / -سن 3PL / -سو 2PL',
         gloss='-s- future; often written as a separate word (ہو سی، کر سُن)', dialect_lit='shared (Lahnda trait)',
         sources='SH2010 via Wikipedia; RRS2011 (paR-sii); BANO (دیساں، آسیں، جاسی، جاسُن)'),
    dict(id='V.INF', section='verb', suffix='-نڑاں / -نڑا / -نا (after ر) ; OBL -نڑیں / -نڑے / -نے',
         gloss='infinitive / verbal noun; retroflex n spelled نڑ', dialect_lit='-نڑاں/-نڑیں Peshawari; -نڑا/-نڑے books+Hazara',
         sources='CORPUS; BANO (دینا واں = PRS.1SG, کرنی وے)'),
    dict(id='V.SBJV', section='verb', suffix='-اں 1SG / -یں 2SG / -ے 3SG / -و 2PL / -ن 3PL ; vowel stems -واں -ویں -وے -ون',
         gloss='subjunctive / imperative (ہووے، جاوے: Hindko glide forms)', dialect_lit='shared',
         sources='BANO (کردیوے، ہووے); CORPUS'),
    dict(id='V.CAUS', section='verb', suffix='-ا- / -وا-', gloss='causative formatives (کرا-, کروا-), then vowel-stem endings',
         dialect_lit='shared', sources='CORPUS'),
    dict(id='V.CONJ', section='verb', suffix='-کے', gloss='conjunctive participle (کرکے, جاکے); often written apart',
         dialect_lit='shared', sources='BANO (جاکے); CORPUS'),
    dict(id='V.COP.CLITIC', section='verb', suffix='-ین / -ن (کردین، گئین، سکدین)',
         gloss='contracted 3PL copula on participles (کردے ہن -> کردین)', dialect_lit='Peshawari (BANO: کہندین، کھاندین، آگئین)',
         sources='BANO; CORPUS', not_in_gold='contraction boundary is not segmentable cleanly (ے -> ی)'),
    dict(id='ORTH.RETRO_N', section='orthography', suffix='نڑ (ṇ)', gloss='retroflex nasal written ن + ڑ; ݨ (U+0768) almost unused',
         dialect_lit='all sources of this corpus', sources='CORPUS (_probe_chars.py)'),
]
