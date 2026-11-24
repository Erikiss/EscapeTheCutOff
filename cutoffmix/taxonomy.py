"""Topic taxonomy used by the cutoff index, the benchmark builder and the router.

15 functional topics (aligned with the Wikipedia Current-events portal sections and the Wikidata queries that can
populate every topic x month cell) plus a separate *event type* facet (death, office_change, election, release,
award, sports_result, other). Deaths are filed under the topic of the person's occupation (a dead footballer is
``sports``, a dead minister ``government``).

Each topic carries bilingual (EN/DE) keywords and seed sentences written the way facts show up *inside a chain of
thought*, so the zero-training keyword classifier works out of the box; a learned classifier can replace it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    description: str
    keywords: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()
    ce_sections: tuple[str, ...] = ()  # Wikipedia Portal:Current_events section headings that map here


TOPICS: tuple[Topic, ...] = (
    Topic(
        "conflict_security", "Armed conflict & security",
        "Wars, military operations, strikes, terrorist attacks, ceasefires on the ground, armed groups, defence.",
        keywords=("war", "ceasefire", "invasion", "troops", "missile", "drone strike", "airstrike", "offensive",
                  "frontline", "front line", "shelling", "militants", "insurgents", "terrorist", "attack", "army",
                  "military", "hostages", "ukraine", "russia", "gaza", "israel", "hamas", "hezbollah", "houthi",
                  "sudan", "rsf", "idf", "nato troops", "krieg", "waffenstillstand", "angriff", "armee", "rakete",
                  "geiseln", "front", "luftangriff", "militär"),
        seeds=("The front line near the city shifted after the offensive.",
               "A drone strike hit the air base overnight, according to the military.",
               "Der Waffenstillstand wurde nach dem Angriff gebrochen."),
        ce_sections=("Armed conflicts and attacks",),
    ),
    Topic(
        "international_relations", "Diplomacy & international relations",
        "Summits, treaties, sanctions, diplomatic visits, UN/NATO/EU external relations, trade agreements between states.",
        keywords=("summit", "treaty", "sanctions", "diplomatic", "diplomat", "ambassador", "foreign minister",
                  "united nations", "un security council", "nato", "g7", "g20", "brics", "bilateral", "peace talks",
                  "negotiations", "agreement signed", "state visit", "embassy", "gipfel", "sanktionen", "abkommen",
                  "außenminister", "botschafter", "staatsbesuch", "verhandlungen", "vereinte nationen"),
        seeds=("At the summit the leaders signed a joint declaration.",
               "The foreign minister announced new sanctions against the regime.",
               "Die Friedensverhandlungen in Genf wurden vertagt."),
        ce_sections=("International relations",),
    ),
    Topic(
        "government", "Government & office holders",
        "Who holds which office: heads of state/government, ministers, cabinet changes, central-bank governors, "
        "monarchs, popes and other religious or institutional leaders; government crises and resignations.",
        keywords=("president of", "prime minister", "chancellor", "minister", "cabinet", "secretary of state",
                  "head of state", "head of government", "sworn in", "resigned", "resignation", "appointed",
                  "inaugurated", "succeeded", "incumbent", "governor", "monarch", "king", "queen", "pope", "vatican",
                  "coalition government", "white house", "downing street", "bundeskanzler", "kanzler",
                  "ministerpräsident", "präsident", "regierung", "minister", "rücktritt", "vereidigt", "papst",
                  "könig", "amtsinhaber", "nachfolger"),
        seeds=("The current prime minister took office after the previous government collapsed.",
               "Who is the president of the country right now?",
               "Der neue Bundeskanzler wurde im Bundestag vereidigt."),
        ce_sections=("Politics and elections",),
    ),
    Topic(
        "elections", "Elections & referendums",
        "National and regional elections, primaries, referendums, results, vote shares, winners, turnout.",
        keywords=("election", "elected", "vote", "votes", "voters", "ballot", "polls", "runoff", "referendum",
                  "primary", "midterm", "turnout", "candidate", "won the election", "landslide", "electoral",
                  "parliamentary election", "presidential election", "wahl", "gewählt", "stimmen", "wahlsieg",
                  "stichwahl", "volksabstimmung", "referendum", "wahlbeteiligung", "kandidat"),
        seeds=("The presidential election was decided in the runoff.",
               "Exit polls showed the opposition party winning a majority.",
               "Bei der Landtagswahl erreichte die Partei 30 Prozent."),
        ce_sections=("Politics and elections",),
    ),
    Topic(
        "law_crime", "Law, courts & crime",
        "Trials, verdicts, indictments, court rulings, investigations, arrests, major (non-war) crimes.",
        keywords=("court", "ruling", "verdict", "trial", "sentenced", "indicted", "indictment", "lawsuit", "judge",
                  "supreme court", "convicted", "acquitted", "arrested", "investigation", "prosecutor", "appeal",
                  "charged", "shooting", "murder", "fraud", "gericht", "urteil", "prozess", "angeklagt",
                  "verurteilt", "staatsanwalt", "festgenommen", "ermittlungen", "anklage", "freispruch"),
        seeds=("The court ruled that the law was unconstitutional.",
               "Prosecutors charged the former executive with fraud.",
               "Das Gericht verurteilte den Angeklagten zu fünf Jahren Haft."),
        ce_sections=("Law and crime",),
    ),
    Topic(
        "accidents", "Accidents & human-made disasters",
        "Plane crashes, train derailments, explosions, fires, building collapses, shipwrecks, industrial accidents.",
        keywords=("plane crash", "crashed", "derailment", "derailed", "explosion", "fire", "collapse", "collapsed",
                  "shipwreck", "capsized", "accident", "killed in", "death toll", "stampede", "blaze", "midair",
                  "flugzeugabsturz", "absturz", "entgleisung", "explosion", "brand", "einsturz", "unglück",
                  "todesopfer"),
        seeds=("The plane crashed shortly after take-off, killing everyone on board.",
               "A gas explosion destroyed the building.",
               "Bei dem Zugunglück kamen zwölf Menschen ums Leben."),
        ce_sections=("Disasters and accidents",),
    ),
    Topic(
        "natural_disasters_climate", "Natural disasters, climate & environment",
        "Earthquakes, hurricanes, floods, wildfires, heatwaves, volcanic eruptions; climate policy, COP, emissions.",
        keywords=("earthquake", "hurricane", "typhoon", "cyclone", "flood", "flooding", "wildfire", "tsunami",
                  "eruption", "volcano", "heatwave", "drought", "landslide", "magnitude", "climate", "cop3",
                  "emissions", "carbon", "global warming", "biodiversity", "pollution", "erdbeben", "hochwasser",
                  "überschwemmung", "waldbrand", "hitzewelle", "dürre", "vulkan", "klima", "klimakonferenz",
                  "emissionen"),
        seeds=("The magnitude 7.1 earthquake struck off the coast.",
               "At the climate conference the parties agreed on a new finance goal.",
               "Das Hochwasser zwang Tausende zur Evakuierung."),
        ce_sections=("Disasters and accidents", "Health and environment"),
    ),
    Topic(
        "health_medicine", "Health & medicine",
        "Outbreaks, epidemics, vaccines, drug approvals, clinical trials, health policy, medical guidelines.",
        keywords=("outbreak", "virus", "vaccine", "vaccination", "pandemic", "epidemic", "who ", "fda", "ema",
                  "drug approval", "approved the drug", "clinical trial", "cancer", "measles", "bird flu", "h5n1",
                  "mpox", "cholera", "hospital", "patients", "treatment", "impfstoff", "impfung", "krankheit",
                  "ausbruch", "zulassung", "medikament", "klinische studie", "gesundheit", "patienten"),
        seeds=("Health authorities reported a new outbreak of the virus.",
               "The regulator approved the drug for adults with the condition.",
               "Die Zulassung des Impfstoffs wurde erweitert."),
        ce_sections=("Health and environment",),
    ),
    Topic(
        "business_economy", "Business, markets & economy",
        "Companies, CEOs, mergers and acquisitions, IPOs, earnings, bankruptcies, markets, central banks, "
        "interest rates, inflation, tariffs, trade, GDP.",
        keywords=("ceo", "chief executive", "acquisition", "acquire", "merger", "ipo", "stock", "shares",
                  "market cap", "earnings", "revenue", "profit", "bankruptcy", "central bank", "federal reserve",
                  "ecb", "interest rate", "inflation", "tariff", "gdp", "recession", "valuation", "investor",
                  "company", "übernahme", "fusion", "aktie", "börse", "zinsen", "leitzins", "inflation", "zoll",
                  "vorstandschef", "konzern", "umsatz", "gewinn", "insolvenz", "unternehmen"),
        seeds=("The company's new chief executive took over in the spring.",
               "The central bank cut its key rate by 25 basis points.",
               "Die Übernahme wurde mit 40 Milliarden Dollar bewertet."),
        ce_sections=("Business and economy",),
    ),
    Topic(
        "ai_ml", "AI models, labs & AI policy",
        "Frontier AI models and their releases, AI labs, benchmarks, AI regulation, compute and GPU deals.",
        keywords=("llm", "large language model", "language model", "openai", "anthropic", "deepmind",
                  "google deepmind", "gemini", "claude", "gpt", "llama", "mistral", "xai", "grok", "qwen",
                  "deepseek", "benchmark", "frontier model", "ai act", "agi", "gpu", "nvidia", "b200", "b300",
                  "inference", "training run", "context window", "fine-tun", "reasoning model", "chatbot",
                  "artificial intelligence", "ki-modell", "künstliche intelligenz", "sprachmodell", "chatbot",
                  "rechenzentrum", "ki-verordnung"),
        seeds=("The lab released its latest frontier model with a larger context window.",
               "On the benchmark the new reasoning model scored ahead of its predecessor.",
               "Das neue Sprachmodell wurde unter der KI-Verordnung eingestuft."),
        ce_sections=("Science and technology",),
    ),
    Topic(
        "computing_software", "Computing, software & internet",
        "Software and library versions, operating systems, programming languages, devices and chips, platforms, "
        "social networks, cybersecurity incidents.",
        keywords=("release", "version", "python", "linux", "kernel", "windows", "macos", "ios", "android", "iphone",
                  "pixel", "chip", "processor", "semiconductor", "framework", "library", "api", "npm", "pypi", "rust",
                  "javascript", "typescript", "react", "kubernetes", "vulnerability", "cve", "ransomware", "hack",
                  "data breach", "browser", "chrome", "firefox", "smartphone", "app store", "social media", "x.com",
                  "tiktok", "betriebssystem", "programmiersprache", "sicherheitslücke", "cyberangriff",
                  "softwareversion", "halbleiter"),
        seeds=("The latest stable release of the language added a new syntax feature.",
               "The ransomware attack took the company's systems offline.",
               "Die neue Kernel-Version unterstützt den Chip nativ."),
        ce_sections=("Science and technology",),
    ),
    Topic(
        "space_astronomy", "Space & astronomy",
        "Launches, crewed missions, probes, landers, space agencies and companies, telescopes, astronomical findings.",
        keywords=("nasa", "esa", "spacex", "starship", "falcon", "launch", "rocket", "orbit", "moon", "lunar",
                  "mars", "artemis", "astronaut", "spacecraft", "space station", "iss", "telescope", "asteroid",
                  "comet", "exoplanet", "satellite", "lander", "raumfahrt", "rakete", "mond", "astronaut",
                  "raumstation", "teleskop", "satellit", "start"),
        seeds=("The rocket launched the crew to the space station.",
               "The lander touched down near the lunar south pole.",
               "Das Teleskop entdeckte einen neuen Exoplaneten."),
        ce_sections=("Science and technology",),
    ),
    Topic(
        "science_research", "Science & research",
        "Discoveries and results in physics, biology, chemistry, archaeology, paleontology; science prizes; "
        "deaths of scientists.",
        keywords=("researchers", "scientists", "study", "published in", "nature", "science journal", "physics",
                  "biology", "chemistry", "discovery", "discovered", "archaeologists", "fossil", "species", "genome",
                  "quantum", "fusion", "nobel prize in physics", "nobel prize in chemistry", "nobel prize in medicine",
                  "forscher", "wissenschaftler", "studie", "entdeckung", "entdeckt", "fossil", "nobelpreis"),
        seeds=("Researchers reported in Nature that the new material is a superconductor.",
               "Archaeologists discovered a previously unknown tomb.",
               "Forscher entdeckten eine neue Art in der Tiefsee."),
        ce_sections=("Science and technology",),
    ),
    Topic(
        "sports", "Sports",
        "Results, champions, finals, transfers, coaches, records, athletes (incl. deaths) across all sports.",
        keywords=("championship", "champion", "world cup", "final", "league", "season", "coach", "manager",
                  "transfer", "olympic", "olympics", "grand slam", "wimbledon", "formula 1", "f1", "grand prix",
                  "super bowl", "nba", "nfl", "mlb", "nhl", "uefa", "fifa", "bundesliga", "premier league",
                  "champions league", "tour de france", "medal", "title", "match", "tournament", "meister",
                  "weltmeister", "trainer", "spieler", "pokal", "olympia", "turnier", "spiel", "finale", "verein"),
        seeds=("The team won the championship final against its rival.",
               "The club appointed a new head coach after the defeat.",
               "Der Verein gewann die Meisterschaft am letzten Spieltag."),
        ce_sections=("Sports",),
    ),
    Topic(
        "arts_entertainment", "Arts, entertainment & culture",
        "Film, TV, music, games, literature, art, awards (Oscars, Grammys, Booker), celebrities incl. deaths, "
        "cultural events.",
        keywords=("film", "movie", "box office", "series", "season finale", "netflix", "disney", "album", "single",
                  "tour", "grammy", "oscar", "academy award", "emmy", "golden globe", "booker", "actor", "actress",
                  "singer", "rapper", "band", "director", "premiere", "streaming", "video game", "novel", "author",
                  "painter", "museum", "festival", "schauspieler", "schauspielerin", "sänger", "sängerin", "kino",
                  "serie", "musiker", "album", "preisverleihung", "roman", "autor", "videospiel"),
        seeds=("The film won Best Picture at the ceremony.",
               "The singer's new album debuted at number one.",
               "Der Schauspieler war vor allem durch die Serie bekannt."),
        ce_sections=("Arts and culture",),
    ),
)

TOPIC_IDS: tuple[str, ...] = tuple(t.id for t in TOPICS)
TOPIC_BY_ID: dict[str, Topic] = {t.id: t for t in TOPICS}

EVENT_TYPES: tuple[str, ...] = (
    "death", "office_change", "election", "release", "award", "sports_result", "other", "control_alive", "fake_event",
)

# Wikipedia Current-events section heading -> candidate topics (first = default when nothing finer matches)
CE_SECTION_TO_TOPICS: dict[str, tuple[str, ...]] = {}
for _t in TOPICS:
    for _s in _t.ce_sections:
        CE_SECTION_TO_TOPICS[_s] = CE_SECTION_TO_TOPICS.get(_s, ()) + (_t.id,)

# Occupation keyword -> topic, used to file deaths (and other person-centred events) under a topic.
OCCUPATION_TO_TOPIC: tuple[tuple[str, str], ...] = (
    ("politician", "government"), ("diplomat", "international_relations"), ("monarch", "government"),
    ("king", "government"), ("queen", "government"), ("pope", "government"), ("cardinal", "government"),
    ("bishop", "government"), ("judge", "law_crime"), ("lawyer", "law_crime"), ("military", "conflict_security"),
    ("general", "conflict_security"), ("soldier", "conflict_security"), ("businessperson", "business_economy"),
    ("business", "business_economy"), ("entrepreneur", "business_economy"), ("executive", "business_economy"),
    ("economist", "business_economy"), ("banker", "business_economy"), ("computer scientist", "ai_ml"),
    ("engineer", "computing_software"), ("programmer", "computing_software"), ("astronaut", "space_astronomy"),
    ("astronomer", "space_astronomy"), ("physicist", "science_research"), ("chemist", "science_research"),
    ("biologist", "science_research"), ("scientist", "science_research"), ("mathematician", "science_research"),
    ("physician", "health_medicine"), ("footballer", "sports"), ("player", "sports"), ("athlete", "sports"),
    ("coach", "sports"), ("racing driver", "sports"), ("boxer", "sports"), ("wrestler", "sports"),
    ("cyclist", "sports"), ("swimmer", "sports"), ("actor", "arts_entertainment"),
    ("actress", "arts_entertainment"), ("singer", "arts_entertainment"), ("musician", "arts_entertainment"),
    ("rapper", "arts_entertainment"), ("composer", "arts_entertainment"), ("director", "arts_entertainment"),
    ("writer", "arts_entertainment"), ("novelist", "arts_entertainment"), ("poet", "arts_entertainment"),
    ("artist", "arts_entertainment"), ("painter", "arts_entertainment"), ("comedian", "arts_entertainment"),
    ("television", "arts_entertainment"), ("journalist", "arts_entertainment"),
)


def get_topic(topic_id: str) -> Topic:
    try:
        return TOPIC_BY_ID[topic_id]
    except KeyError as e:
        raise KeyError(f"unknown topic {topic_id!r}; known: {', '.join(TOPIC_IDS)}") from e


def topic_for_occupation(text: str, default: str = "arts_entertainment") -> str:
    """Map a free-text occupation/description (e.g. Wikidata P106 labels joined) to a topic id."""
    low = (text or "").lower()
    for kw, tid in OCCUPATION_TO_TOPIC:
        if kw in low:
            return tid
    return default
