#!/usr/bin/env python3
import argparse
import json
import os
import sys
from huggingface_hub import dataset_info
from datasets import load_dataset

# Výchozí doporučený set postavený na oficiálních ISO 639-3 kódech FineWeb-2
DEFAULT_LANGS = ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani"]
DEFAULT_RATIOS = {
    "eng_Latn": 0.32,  # Angličtina (Poznámka: FineWeb-2 odkazuje pro čistou EN na původní FineWeb, 
    "spa_Latn": 0.17,  # ale v datasetu najdete eng_Latn pro specifické mixy textů)
    "deu_Latn": 0.17,
    "rus_Cyrl": 0.17,
    "cmn_Hani": 0.17
}

# Koeficienty pro přepočet znaků na tokeny (znaky / koeficient = tokeny)
TOKEN_ESTIMATORS = {
    "Latn": 4.1,   # Latinka (EN, ES, DE, atd.)
    "Cyrl": 3.8,   # Azbuka (RU, UKR, atd.)
    "Hani": 1.5,   # Čínské znaky (CMN)
    "Jpan": 1.7,   # Japonština
    "Hang": 2.5,   # Korejština
    "Arab": 3.5,   # Arabské písmo
    "Deva": 3.2    # Devanagari (Hindština)
}


def fetch_available_subsets():
    """Dynamicky načte všechny dostupné subsety z Hugging Face Hubu."""
    print("Načítám seznam jazyků přímo z Hugging Face Hubu...")
    try:
        info = dataset_info("HuggingFaceFW/fineweb-2", repo_type="dataset")
        # Filtrujeme pouze platné textové subsety (vynecháváme pomocné '_removed' a 'und_')
        subsets = sorted([
            config.config_name for config in info.card_data.get("configs", [])
            if config.config_name and not config.config_name.endswith("_removed") and not config.config_name.startswith("und_")
        ])
        if not subsets:
            # Záložní hardcoded list pro případ výpadku HF API metadat
            return ["deu_Latn", "spa_Latn", "fra_Latn", "rus_Cyrl", "cmn_Hani", "ces_Latn", "por_Latn", "ita_Latn"]
        return subsets
    except Exception as e:
        print(f"Varování: Nepodařilo se načíst metadata z HF Hubu: {e}", file=sys.stderr)
        return ["deu_Latn", "spa_Latn", "fra_Latn", "rus_Cyrl", "cmn_Hani"]


def list_languages(available_subsets):
    print("\n=== Dostupné jazykové subsety ve FineWeb-2 ===")
    for subset in available_subsets:
        marker = " [VÝCHOZÍ SUSET V MIXU]" if subset in DEFAULT_LANGS else ""
        print(f"  - {subset}{marker}")
    print(f"\nCelkem nalezeno: {len(available_subsets)} jazykově-písmových konfigurací.")
    print("==============================================\n")


def parse_mixing_ratios(ratio_str, selected_langs):
    """Zpracuje poměry míchání zadané uživatelem ve formátu kód:váha,kód:váha"""
    if not ratio_str:
        chosen_defaults = {l: DEFAULT_RATIOS.get(l, 1.0) for l in selected_langs}
        total = sum(chosen_defaults.values())
        return {l: chosen_defaults[l] / total for l in selected_langs}

    try:
        pairs = ratio_str.split(",")
        ratios = {}
        for pair in pairs:
            lang, weight = pair.split(":")
            if lang not in selected_langs:
                print(f"Varování: Poměr pro '{lang}' byl zadán, ale jazyk není vybrán. Ignoruji.")
                continue
            ratios[lang] = float(weight)

        total_weight = sum(ratios.values())
        for lang in ratios:
            ratios[lang] /= total_weight

        for lang in selected_langs:
            if lang not in ratios:
                ratios[lang] = 0.0
        return ratios
    except Exception as e:
        print(f"Chyba při parsování poměrů. Používám rovnoměrné rozdělení. Detail: {e}")
        return {l: 1.0 / len(selected_langs) for l in selected_langs}


def main():
    parser = argparse.ArgumentParser(description="Dynamický FineWeb-2 Multilingvální Mixer pro Tokenizer Studie")
    parser.add_argument("--list", action="store_true", help="Vypíše všechny dostupné jazykové subsety na HF a ukončí se")
    parser.add_argument("--langs", type=str, help="Čárkou oddělené přesné FineWeb-2 subsety (např. eng_Latn,deu_Latn,ces_Latn)")
    parser.add_argument("--target_tokens", type=int, default=20_000_000, help="Cílový celkový počet tokenů výsledného korpusu")
    parser.add_argument("--ratios", type=str, help="Vlastní poměry míchání, např. deu_Latn:0.5,rus_Cyrl:0.5")
    parser.add_argument("--output", type=str, default="multilingual_corpus.jsonl", help="Název výstupního souboru (.jsonl)")
    parser.add_argument("--min_char_len", type=int, default=200, help="Minimální délka dokumentu ve znacích")

    args = parser.parse_args()

    # Načtení reálného seznamu z Hugging Face
    available_subsets = fetch_available_subsets()

    if args.list:
        list_languages(available_subsets)
        sys.exit(0)

    # 1. Validace a výběr jazyků
    if args.langs:
        selected_langs = [l.strip() for l in args.langs.split(",")]
        for l in selected_langs:
            if l not in available_subsets:
                print(f"Chyba: Subset '{l}' neexistuje ve FineWeb-2. Spusťte skript s --list pro zobrazení všech kódů.")
                sys.exit(1)
    else:
        print(f"Nebyly zadány žádné jazyky. Používám výchozí mix: {', '.join(DEFAULT_LANGS)}")
        selected_langs = [l for l in DEFAULT_LANGS if l in available_subsets]

    # 2. Nastavení vah
    mixing_ratios = parse_mixing_ratios(args.ratios, selected_langs)

    print("\nPlán složení výsledného korpusu:")
    for lang in selected_langs:
        print(f"  - {lang}: {mixing_ratios[lang]*100:.1f}%")
    print(f"Cílová velikost: {args.target_tokens:,} tokenů\n")

    target_tokens_per_lang = {lang: int(args.target_tokens * mixing_ratios[lang]) for lang in selected_langs}

    # 3. Inicializace streamování z HF
    iterators = {}
    for lang in selected_langs:
        print(f"Otevírám síťový stream pro: {lang}...")
        try:
            # stream=True zajistí, že stahujeme data po kusech za běhu a neukládáme gigabajty na disk
            ds = load_dataset("HuggingFaceFW/fineweb-2", name=lang, split="train", streaming=True)
            iterators[lang] = iter(ds)
        except Exception as e:
            print(f"Chyba při připojování k subsetu {lang}: {e}")
            sys.exit(1)

    # 4. Proporcionální míchání a zápis do JSONL
    collected_tokens = {lang: 0 for lang in selected_langs}
    active_langs = list(selected_langs)

    print(f"\nZahajuji stahování vzorků a zápis do {args.output}...")

    with open(args.output, "w", encoding="utf-8") as f_out:
        while active_langs:
            for lang in list(active_langs):
                if collected_tokens[lang] >= target_tokens_per_lang[lang]:
                    active_langs.remove(lang)
                    continue

                try:
                    doc = next(iterators[lang])
                    text = doc.get("text", "")

                    if len(text) < args.min_char_len:
                        continue

                    # Detekce písma z názvu subsetu (např. z deu_Latn získáme Latn)
                    script_type = lang.split("_")[1] if "_" in lang else "Latn"
                    multiplier = TOKEN_ESTIMATORS.get(script_type, 4.0)
                    estimated_tokens = max(1, int(len(text) / multiplier))

                    output_obj = {
                        "text": text,
                        "dataset_subset": lang,
                        "estimated_tokens": estimated_tokens
                    }
                    f_out.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

                    collected_tokens[lang] += estimated_tokens

                except StopIteration:
                    print(f"\n[Varování] Data pro subset {lang} byla vyčerpána před dosažením limitu.")
                    active_langs.remove(lang)

    print("\n=== Hotovo ===")
    total_est_tokens = sum(collected_tokens.values())
    print(f"Uloženo do: {os.path.abspath(args.output)}")
    print(f"Celkem získáno tokenů (odhad): {total_est_tokens:,} / {args.target_tokens:,}")
    for lang in selected_langs:
        print(f"  - {lang}: {collected_tokens[lang]:,} tokenů (Cíl bol: {target_tokens_per_lang[lang]:,})")


if __name__ == "__main__":
    main()
