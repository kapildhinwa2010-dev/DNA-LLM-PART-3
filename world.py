"""Domains, relation algebra and query generation for the HELIX prototype."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from helix import RelSpec

# --------------------------------------------------------------------------
# Relation algebra (the "rule layer" — stored in the artifact, ~0 bytes/fact)
# --------------------------------------------------------------------------
REL_SPECS = [
    # ---- forward relations -------------------------------------------------
    RelSpec("capital_of", [
        (r"(?:what is |what's |whats )?the capital (?:city )?of ([a-z0-9 '\-]+)", "1", None),
        (r"([a-z0-9 '\-]+)'s capital (?:city )?is", "1", None),
        (r"capital of ([a-z0-9 '\-]+)", "1", None),
    ]),
    RelSpec("continent_of", [
        (r"what continent is ([a-z0-9 '\-]+) in", "1", None),
        (r"the continent of ([a-z0-9 '\-]+)", "1", None),
    ]),
    RelSpec("country_of", [
        (r"(?:which|what) country is ([a-z0-9 '\-]+) in", "1", None),
        (r"([a-z0-9 '\-]+) is located in which country", "1", None),
        (r"where is ([a-z0-9 '\-]+) located", "1", None),
    ]),
    RelSpec("symbol_of", [
        (r"(?:what is |what's |whats )?the (?:chemical )?symbol (?:of|for) ([a-z0-9 '\-]+)", "1", None),
        (r"chemical symbol for ([a-z0-9 '\-]+)", "1", None),
    ]),
    RelSpec("number_of", [
        (r"atomic number of ([a-z0-9 '\-]+)", "1", None),
        (r"what is the atomic number of ([a-z0-9 '\-]+)", "1", None),
    ]),
    RelSpec("category_of", [
        (r"what (?:type|kind|category) of element is ([a-z0-9 '\-]+)", "1", None),
        (r"category of ([a-z0-9 '\-]+)", "1", None),
    ]),
    # ---- inverse relations (derived from the forward ones) ------------------
    RelSpec("country_with_capital", [
        (r"(?:which|what) country has ([a-z0-9 '\-]+) as (?:its|the) capital", "1", None),
        (r"([a-z0-9 '\-]+) is the capital of which country", "1", None),
    ], inverse_of="capital_of"),
    RelSpec("element_with_symbol", [
        (r"(?:which|what) element has (?:the )?symbol ([a-z0-9 '\-]+)", "1", None),
        (r"symbol ([a-z0-9 '\-]+) belongs to which element", "1", None),
    ], inverse_of="symbol_of"),
    RelSpec("element_with_number", [
        (r"(?:which|what) element has (?:the )?atomic number ([a-z0-9 '\-]+)", "1", None),
    ], inverse_of="number_of"),
    RelSpec("element_with_category", [
        (r"(?:which|what) elements are ([a-z0-9 '\-]+)", "1", None),
    ], inverse_of="category_of"),
    # ---- composed (multi-hop) relations ------------------------------------
    RelSpec("capital_of_country", [
        (r"what is the capital of the country of ([a-z0-9 '\-]+)", "1", None),
        (r"capital of the country (?:containing|of) ([a-z0-9 '\-]+)", "1", None),
    ], chain=("country_of", "capital_of")),
    RelSpec("continent_of_country", [
        (r"what continent is the country of ([a-z0-9 '\-]+) in", "1", None),
    ], chain=("country_of", "continent_of")),
]

# --------------------------------------------------------------------------
# Domain A : capitals + chemical elements  (fact-dense)
# --------------------------------------------------------------------------
COUNTRIES = """France|Paris|Europe
Japan|Tokyo|Asia
Mexico|Mexico City|North America
Germany|Berlin|Europe
Italy|Rome|Europe
Spain|Madrid|Europe
Portugal|Lisbon|Europe
Greece|Athens|Europe
Poland|Warsaw|Europe
Sweden|Stockholm|Europe
Norway|Oslo|Europe
Finland|Helsinki|Europe
Denmark|Copenhagen|Europe
Netherlands|Amsterdam|Europe
Belgium|Brussels|Europe
Switzerland|Bern|Europe
Austria|Vienna|Europe
Ireland|Dublin|Europe
United Kingdom|London|Europe
Russia|Moscow|Europe
Ukraine|Kyiv|Europe
Turkey|Ankara|Asia
Israel|Jerusalem|Asia
India|New Delhi|Asia
Pakistan|Islamabad|Asia
Bangladesh|Dhaka|Asia
China|Beijing|Asia
Vietnam|Hanoi|Asia
Thailand|Bangkok|Asia
Indonesia|Jakarta|Asia
Malaysia|Kuala Lumpur|Asia
Philippines|Manila|Asia
South Korea|Seoul|Asia
North Korea|Pyongyang|Asia
Mongolia|Ulaanbaatar|Asia
Nepal|Kathmandu|Asia
Sri Lanka|Sri Jayawardenepura Kotte|Asia
Kazakhstan|Astana|Asia
Uzbekistan|Tashkent|Asia
Egypt|Cairo|Africa
Libya|Tripoli|Africa
Morocco|Rabat|Africa
Algeria|Algiers|Africa
Tunisia|Tunis|Africa
Sudan|Khartoum|Africa
Ethiopia|Addis Ababa|Africa
Nigeria|Abuja|Africa
Kenya|Nairobi|Africa
Ghana|Accra|Africa
Senegal|Dakar|Africa
Tanzania|Dodoma|Africa
Uganda|Kampala|Africa
Zimbabwe|Harare|Africa
South Africa|Pretoria|Africa
Angola|Luanda|Africa
Mozambique|Maputo|Africa
United States|Washington D.C.|North America
Canada|Ottawa|North America
Brazil|Brasilia|South America
Argentina|Buenos Aires|South America
Chile|Santiago|South America
Peru|Lima|South America
Colombia|Bogota|South America
Venezuela|Caracas|South America
Uruguay|Montevideo|South America
Bolivia|Sucre|South America
Ecuador|Quito|South America
Australia|Canberra|Oceania
New Zealand|Wellington|Oceania
Papua New Guinea|Port Moresby|Oceania
Fiji|Suva|Oceania"""

ELEMENTS = """H Hydrogen nonmetal|He Helium noble gas|Li Lithium alkali metal|Be Beryllium alkaline earth metal|B Boron metalloid|C Carbon nonmetal|N Nitrogen nonmetal|O Oxygen nonmetal|F Fluorine halogen|Ne Neon noble gas|Na Sodium alkali metal|Mg Magnesium alkaline earth metal|Al Aluminium post-transition metal|Si Silicon metalloid|P Phosphorus nonmetal|S Sulfur nonmetal|Cl Chlorine halogen|Ar Argon noble gas|K Potassium alkali metal|Ca Calcium alkaline earth metal|Sc Scandium transition metal|Ti Titanium transition metal|V Vanadium transition metal|Cr Chromium transition metal|Mn Manganese transition metal|Fe Iron transition metal|Co Cobalt transition metal|Ni Nickel transition metal|Cu Copper transition metal|Zn Zinc transition metal|Ga Gallium post-transition metal|Ge Germanium metalloid|As Arsenic metalloid|Se Selenium nonmetal|Br Bromine halogen|Kr Krypton noble gas|Rb Rubidium alkali metal|Sr Strontium alkaline earth metal|Y Yttrium transition metal|Zr Zirconium transition metal|Nb Niobium transition metal|Mo Molybdenum transition metal|Tc Technetium transition metal|Ru Ruthenium transition metal|Rh Rhodium transition metal|Pd Palladium transition metal|Ag Silver transition metal|Cd Cadmium transition metal|In Indium post-transition metal|Sn Tin post-transition metal|Sb Antimony metalloid|Te Tellurium metalloid|I Iodine halogen|Xe Xenon noble gas|Cs Caesium alkali metal|Ba Barium alkaline earth metal|La Lanthanum lanthanide|Ce Cerium lanthanide|Pr Praseodymium lanthanide|Nd Neodymium lanthanide|Pm Promethium lanthanide|Sm Samarium lanthanide|Eu Europium lanthanide|Gd Gadolinium lanthanide|Tb Terbium lanthanide|Dy Dysprosium lanthanide|Ho Holmium lanthanide|Er Erbium lanthanide|Tm Thulium lanthanide|Yb Ytterbium lanthanide|Lu Lutetium lanthanide|Hf Hafnium transition metal|Ta Tantalum transition metal|W Tungsten transition metal|Re Rhenium transition metal|Os Osmium transition metal|Ir Iridium transition metal|Pt Platinum transition metal|Au Gold transition metal|Hg Mercury transition metal|Tl Thallium post-transition metal|Pb Lead post-transition metal|Bi Bismuth post-transition metal|Po Polonium post-transition metal|At Astatine halogen|Rn Radon noble gas|Fr Francium alkali metal|Ra Radium alkaline earth metal|Ac Actinium actinide|Th Thorium actinide|Pa Protactinium actinide|U Uranium actinide|Np Neptunium actinide|Pu Plutonium actinide|Am Americium actinide|Cm Curium actinide|Bk Berkelium actinide|Cf Californium actinide|Es Einsteinium actinide|Fm Fermium actinide|Md Mendelevium actinide|No Nobelium actinide|Lr Lawrencium actinide|Rf Rutherfordium transition metal|Db Dubnium transition metal|Sg Seaborgium transition metal|Bh Bohrium transition metal|Hs Hassium transition metal|Mt Meitnerium unknown|Ds Darmstadtium unknown|Rg Roentgenium unknown|Cn Copernicium transition metal|Nh Nihonium unknown|Fl Flerovium unknown|Mc Moscovium unknown|Lv Livermorium unknown|Ts Tennessine unknown|Og Oganesson unknown"""

ALIASES = {
    "french republic": "France",
    "america": "United States",
    "usa": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "uae": "United Arab Emirates",
    "argentum": "Silver",
    "aurum": "Gold",
    "ferrum": "Iron",
    "cuprum": "Copper",
    "stannum": "Tin",
    "plumbum": "Lead",
    "hydrargyrum": "Mercury",
}

CLUES = {
    "french": "France", "japanese": "Japan", "german": "Germany", "italian": "Italy",
    "spanish": "Spain", "brazilian": "Brazil", "indian": "India", "chinese": "China",
    "russian": "Russia", "egyptian": "Egypt", "mexican": "Mexico", "canadian": "Canada",
    "australian": "Australia", "swedish": "Sweden", "norwegian": "Norway",
    "korean": "South Korea", "turkish": "Turkey", "nigerian": "Nigeria",
    "kenyan": "Kenya", "argentine": "Argentina", "portuguese": "Portugal",
    "polish": "Poland", "dutch": "Netherlands", "greek": "Greece",
}

FICTION = ["Wakanda", "Elbonia", "Freedonia", "Zarnovia", "NewHebron", "Mithgard"]


def domain_a_facts():
    facts = []
    for line in COUNTRIES.strip().splitlines():
        c, cap, cont = line.split("|")
        facts.append((c, "capital_of", cap))
        facts.append((c, "continent_of", cont))
        facts.append((cap, "country_of", c))
    for i, line in enumerate(ELEMENTS.strip().split("|"), start=1):
        sym, name, cat = line.split(" ", 2)
        facts.append((name, "symbol_of", sym))
        facts.append((name, "number_of", str(i)))
        facts.append((name, "category_of", cat))
    return facts


def domain_a_queries():
    """(category, prompt, expected_answer) triples."""
    q = []
    countries = [l.split("|") for l in COUNTRIES.strip().splitlines()]
    seen = set()
    for (c, cap, cont) in countries:
        if c in seen:
            continue
        seen.add(c)
        q.append(("direct", f"what is the capital of {c}?", cap))
        q.append(("chain", f"what is the capital of the country of {cap}?", cap))
        q.append(("reverse", f"which country has {cap} as its capital?", c))
        q.append(("distractor", f"i visited {cont} last year. anyway, what is the capital of {c}?", cap))
        q.append(("direct_continent", f"what continent is {c} in?", cont))
    for a, canon in ALIASES.items():
        pass
    # alias queries (only for aliases that map to a country)
    for a, canon in ALIASES.items():
        for (c, cap, cont) in countries:
            if c == canon:
                q.append(("alias", f"what is the capital of {a}?", cap))
                break
    # clue queries
    for clue, canon in CLUES.items():
        for (c, cap, cont) in countries:
            if c == canon:
                q.append(("clue", f"what is the capital of the {clue} country?", cap))
                break
    # chemistry
    for i, line in enumerate(ELEMENTS.strip().split("|"), start=1):
        sym, name, cat = line.split(" ", 2)
        q.append(("direct", f"what is the chemical symbol of {name}?", sym))
        q.append(("direct", f"what is the atomic number of {name}?", str(i)))
        q.append(("reverse", f"which element has the symbol {sym}?", name))
        q.append(("reverse", f"which element has the atomic number {i}?", name))
    # unknown
    for f in FICTION:
        q.append(("unknown", f"what is the capital of {f}?", None))
        q.append(("unknown", f"what is the chemical symbol of {f}?", None))
    return q


# --------------------------------------------------------------------------
# Domain B : procedurally generated provinces (rule-dense / "DNA as program")
# --------------------------------------------------------------------------
B_COUNTRIES = [
    ("Arden", "Avel", "West"), ("Bore", "Bril", "West"), ("Cald", "Corm", "North"),
    ("Doran", "Dorn", "North"), ("Elsi", "Els", "East"), ("Fenn", "Fen", "East"),
    ("Gora", "Gor", "South"), ("Hale", "Hal", "South"), ("Ivar", "Iva", "West"),
    ("Jork", "Jor", "West"), ("Kald", "Kal", "North"), ("Lund", "Lun", "North"),
    ("Mora", "Mor", "East"), ("Norn", "Nor", "East"), ("Oss", "Oss", "South"),
    ("Pell", "Pel", "South"),
]
B_N_PROV = 4096
B_SPAN = B_N_PROV // len(B_COUNTRIES)   # provinces per country


def domain_b_facts():
    facts = []
    for (c, cap, cont) in B_COUNTRIES:
        facts.append((c, "capital_of", cap))
        facts.append((c, "continent_of", cont))
    return facts


def domain_b_gen_rules():
    return {
        "country_of": [{"kind": "index_range", "quot": B_SPAN,
                        "table": [c for (c, _, _) in B_COUNTRIES]}],
    }


def domain_b_gen_entities():
    return [{"rx": r"\bp(\d{4})\b", "fmt": "P{:04d}"}]


def domain_b_queries():
    q = []
    for i in range(B_N_PROV):
        idx = i
        country = B_COUNTRIES[idx // B_SPAN][0]
        capital = B_COUNTRIES[idx // B_SPAN][1]
        cont = B_COUNTRIES[idx // B_SPAN][2]
        p = f"P{i:04d}"
        q.append(("direct", f"what is the capital of the country of {p}?", capital))
        q.append(("chain", f"which country is {p} located in?", country))
        q.append(("direct_continent", f"what continent is the country of {p} in?", cont))
    return q
