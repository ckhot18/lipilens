"""
devanagari_to_modi.py

Character-level mapping from Devanagari (used to write Marathi) to the
Modi Unicode block (U+11600-U+1165F).

WHY THIS WORKS: Modi's Unicode block was designed by the Unicode Consortium
to structurally MIRROR Devanagari - independent vowels, then consonants,
then dependent vowel signs (matras), then virama/anusvara/visarga, then
punctuation, then digits - in the same phonetic order. So this is a
character substitution table, not a linguistic translation. Because Modi
matras and virama combine with consonants the same way Devanagari ones do
(Unicode rendering engines handle the reordering automatically), a plain
character-by-character swap produces correctly shaped Modi text as long as
the font (Noto Sans Modi) implements the right OpenType rules - which it
does.

Mapping built directly from the official Unicode code chart:
https://www.unicode.org/charts/nameslist/n_11600.html
cross-referenced against the Devanagari block (U+0900-U+097F).

Only characters that are actually assigned in the Modi block (79 code
points) are mapped. Devanagari characters with no Modi equivalent (nukta
forms for Persian/English loan sounds, candra vowels not used in Marathi,
etc.) are left untranslated and logged, so you can see if your corpus
contains something unexpected.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Independent vowels (Devanagari codepoint -> Modi codepoint)
# ---------------------------------------------------------------------------
VOWELS = {
    0x0905: 0x11600,  # अ  -> Modi A
    0x0906: 0x11601,  # आ  -> Modi AA
    0x0907: 0x11602,  # इ  -> Modi I
    0x0908: 0x11603,  # ई  -> Modi II
    0x0909: 0x11604,  # उ  -> Modi U
    0x090A: 0x11605,  # ऊ  -> Modi UU
    0x090B: 0x11606,  # ऋ  -> Modi Vocalic R
    0x0960: 0x11607,  # ॠ  -> Modi Vocalic RR
    0x090C: 0x11608,  # ऌ  -> Modi Vocalic L
    0x0961: 0x11609,  # ॡ  -> Modi Vocalic LL
    0x090F: 0x1160A,  # ए  -> Modi E
    0x0910: 0x1160B,  # ऐ  -> Modi AI
    0x0913: 0x1160C,  # ओ  -> Modi O
    0x0914: 0x1160D,  # औ  -> Modi AU
}

# ---------------------------------------------------------------------------
# Consonants. NOTE: ळ (Lla) is out of alphabetical sequence in the Modi
# block - it comes AFTER Ha, not after La - this matches the official chart.
# ---------------------------------------------------------------------------
CONSONANTS = {
    0x0915: 0x1160E,  # क  Ka
    0x0916: 0x1160F,  # ख  Kha
    0x0917: 0x11610,  # ग  Ga
    0x0918: 0x11611,  # घ  Gha
    0x0919: 0x11612,  # ङ  Nga
    0x091A: 0x11613,  # च  Ca
    0x091B: 0x11614,  # छ  Cha
    0x091C: 0x11615,  # ज  Ja
    0x091D: 0x11616,  # झ  Jha
    0x091E: 0x11617,  # ञ  Nya
    0x091F: 0x11618,  # ट  Tta
    0x0920: 0x11619,  # ठ  Ttha
    0x0921: 0x1161A,  # ड  Dda
    0x0922: 0x1161B,  # ढ  Ddha
    0x0923: 0x1161C,  # ण  Nna
    0x0924: 0x1161D,  # त  Ta
    0x0925: 0x1161E,  # थ  Tha
    0x0926: 0x1161F,  # द  Da
    0x0927: 0x11620,  # ध  Dha
    0x0928: 0x11621,  # न  Na
    0x092A: 0x11622,  # प  Pa
    0x092B: 0x11623,  # फ  Pha
    0x092C: 0x11624,  # ब  Ba
    0x092D: 0x11625,  # भ  Bha
    0x092E: 0x11626,  # म  Ma
    0x092F: 0x11627,  # य  Ya
    0x0930: 0x11628,  # र  Ra
    0x0932: 0x11629,  # ल  La
    0x0935: 0x1162A,  # व  Va
    0x0936: 0x1162B,  # श  Sha
    0x0937: 0x1162C,  # ष  Ssa
    0x0938: 0x1162D,  # स  Sa
    0x0939: 0x1162E,  # ह  Ha
    0x0933: 0x1162F,  # ळ  Lla  (Marathi-specific retroflex L)
}

# ---------------------------------------------------------------------------
# Dependent vowel signs (matras) - attach to the preceding consonant.
# ---------------------------------------------------------------------------
MATRAS = {
    0x093E: 0x11630,  # ा  AA
    0x093F: 0x11631,  # ि  I
    0x0940: 0x11632,  # ी  II
    0x0941: 0x11633,  # ु  U
    0x0942: 0x11634,  # ू  UU
    0x0943: 0x11635,  # ृ  Vocalic R
    0x0944: 0x11636,  # ॄ  Vocalic RR
    0x0962: 0x11637,  # ॢ  Vocalic L
    0x0963: 0x11638,  # ॣ  Vocalic LL
    0x0947: 0x11639,  # े  E
    0x0948: 0x1163A,  # ै  AI
    0x094B: 0x1163B,  # ो  O
    0x094C: 0x1163C,  # ौ  AU
}

# ---------------------------------------------------------------------------
# Signs, punctuation, digits
# ---------------------------------------------------------------------------
SIGNS = {
    0x0902: 0x1163D,  # ं  Anusvara
    0x0903: 0x1163E,  # ः  Visarga
    0x094D: 0x1163F,  # ्  Virama / halant
    0x0964: 0x11641,  # ।  Danda
    0x0965: 0x11642,  # ॥  Double Danda
    0x0970: 0x11643,  # ॰  Abbreviation sign
}

DIGITS = {0x0966 + i: 0x11650 + i for i in range(10)}  # ० - ९  ->  Modi digits

# ---------------------------------------------------------------------------
# Combined lookup table
# ---------------------------------------------------------------------------
DEVANAGARI_TO_MODI: dict[int, int] = {}
for table in (VOWELS, CONSONANTS, MATRAS, SIGNS, DIGITS):
    DEVANAGARI_TO_MODI.update(table)


def transliterate(text: str, strict: bool = False) -> tuple[str, list[str]]:
    """
    Convert a Devanagari (Marathi) string into Modi Unicode.

    Returns (modi_text, warnings) where warnings lists any characters that
    had no mapping (left unchanged in the output - e.g. spaces, Latin
    characters, numerals typed in ASCII, or rare nukta forms).

    strict=True raises an error instead of passing unmapped characters
    through untouched - useful when validating your corpus, not when
    running the pipeline.
    """
    out_chars = []
    warnings = []
    for ch in text:
        cp = ord(ch)
        if cp in DEVANAGARI_TO_MODI:
            out_chars.append(chr(DEVANAGARI_TO_MODI[cp]))
        elif ch.isspace() or not (0x0900 <= cp <= 0x097F):
            # Not Devanagari at all (space, punctuation, Latin, etc.) - keep as-is
            out_chars.append(ch)
        else:
            msg = f"No Modi mapping for U+{cp:04X} ({ch!r})"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            out_chars.append(
                ch
            )  # pass through unchanged so text isn't silently dropped
    return "".join(out_chars), warnings


if __name__ == "__main__":
    # Quick self-test with a few real Marathi words/sentences
    samples = [
        "नमस्कार",  # namaskar (hello)
        "महाराष्ट्र",  # Maharashtra
        "शिवाजी महाराज",  # Shivaji Maharaj
        "पुणे शहर खूप सुंदर आहे.",  # Pune city is very beautiful.
        "मोडी लिपी ही ऐतिहासिक आहे.",
    ]
    for s in samples:
        modi, warns = transliterate(s)
        print(f"{s}  ->  {modi}")
        for w in warns:
            print("   WARNING:", w)
