"""Topic taxonomy used by the cutoff index and the router.

Design constraints
------------------
* Coarse enough that every topic x month cell can be populated from public, verifiable sources
  (Wikidata / Wikipedia current-events portal / the apoorvumang knowledge-cutoff benchmark).
* Fine enough that frontier models plausibly differ in *topic-specific* recency
  (e.g. AI-lab news vs. European politics vs. sports results).
* Each topic carries bilingual (EN/DE) keywords and seed sentences so a zero-training lexical
  classifier works out of the box; a learned classifier can replace it later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    description: str
    keywords: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()  # short example sentences (as they might appear inside a chain of thought)
    wikipedia_sections: tuple[str, ...] = ()  # Portal:Current_events section headings that map here
    parent: str | None = None


TOPICS: tuple[Topic, ...] = (
    Topic(
        "politics_us",
        "US politics & government",
        "US federal/state politics, elections, White House, Congress, Supreme Court appointments, cabinet changes.",
        keywords=("white house", "congress", "senate", "senator", "house of representatives", "republican", "democrat",
                  "president of the united states", "governor", "cabinet secretary", "us election", "midterm", "gop",
                  "trump", "vance", "speaker of the house", "attorney general", "pentagon", "washington", "us-präsident",
                  "repräsentantenhaus", "us-wahl", "kongress"),
        seeds=("The current US Secretary of State is", "After the midterm elections the House majority",
               "The Senate confirmed the nominee for"),
        wikipedia_sections=("Politics and elections",),
    ),
    Topic(
        "politics_europe",
        "European politics",
        "Politics of European countries and the EU: governments, chancellors, prime ministers, elections, coalitions.",
        keywords=("bundestag", "bundeskanzler", "chancellor", "european commission", "european parliament", "eu council",
                  "prime minister of", "élysée", "downing street", "coalition", "cdu", "spd", "afd", "grüne", "fdp",
                  "bundesregierung", "landtag", "brussels", "brüssel", "eu-kommission", "europawahl", "macron", "merz",
                  "starmer", "meloni", "sánchez", "tusk", "orbán", "parliamentary election", "minister"),
        seeds=("The German federal government under the chancellor", "The new French prime minister",
               "In the EU Commission the portfolio for"),
        wikipedia_sections=("Politics and elections",),
    ),
    Topic(
        "politics_world",
        "World politics (non-US/EU)",
        "Heads of state/government, elections and political crises in Asia, Africa, Latin America, Middle East, Oceania.",
        keywords=("president of", "prime minister", "election in", "coup", "junta", "parliament", "head of state",
                  "india", "japan", "china", "brazil", "mexico", "argentina", "south africa", "nigeria", "indonesia",
                  "pakistan", "turkey", "iran", "south korea", "australia", "canada", "präsident", "staatschef",
                  "regierungschef", "wahl in"),
        seeds=("The newly elected president of Brazil", "Japan's prime minister announced",
               "After the coup the military government"),
        wikipedia_sections=("Politics and elections",),
    ),
    Topic(
        "geopolitics_conflict",
        "Geopolitics, conflicts & diplomacy",
        "Wars, ceasefires, military operations, sanctions, treaties, summits, international organisations.",
        keywords=("war", "ceasefire", "invasion", "troops", "missile", "drone strike", "sanctions", "nato", "un security",
                  "united nations", "peace talks", "treaty", "summit", "ukraine", "russia", "gaza", "israel", "hamas",
                  "hezbollah", "taiwan strait", "frontline", "offensive", "krieg", "waffenstillstand", "angriff",
                  "sanktionen", "gipfel", "friedensverhandlungen", "armee"),
        seeds=("The ceasefire negotiations between", "NATO members agreed at the summit",
               "The front line near the city shifted after"),
        wikipedia_sections=("Armed conflicts and attacks", "International relations"),
    ),
    Topic(
        "business_finance",
        "Business, markets & economy",
        "Companies, CEOs, mergers & acquisitions, IPOs, earnings, central banks, interest rates, inflation, trade.",
        keywords=("ceo", "chief executive", "acquisition", "merger", "acquire", "ipo", "stock", "shares", "market cap",
                  "earnings", "revenue", "central bank", "federal reserve", "ecb", "interest rate", "inflation", "tariff",
                  "gdp", "recession", "bankruptcy", "valuation", "übernahme", "aktie", "börse", "zinsen", "leitzins",
                  "vorstandschef", "konzern", "umsatz", "insolvenz", "zoll"),
        seeds=("The company's new chief executive", "The central bank raised its key rate to",
               "The acquisition was valued at"),
        wikipedia_sections=("Business and economy",),
    ),
    Topic(
        "tech_ai",
        "AI models, labs & AI policy",
        "Frontier AI models and releases, AI labs, benchmarks, AI regulation, compute/GPU deals.",
        keywords=("llm", "language model", "openai", "anthropic", "google deepmind", "deepmind", "gemini", "claude",
                  "gpt", "llama", "mistral", "xai", "grok", "benchmark", "frontier model", "ai act", "agi", "gpu",
                  "nvidia", "b200", "b300", "inference", "training run", "parameters", "context window", "fine-tun",
                  "reasoning model", "ki-modell", "künstliche intelligenz", "sprachmodell", "rechenzentrum"),
        seeds=("The latest frontier model released by the lab", "On the benchmark the new model scored",
               "The AI Act obligations for general-purpose models"),
        wikipedia_sections=("Science and technology",),
    ),
    Topic(
        "tech_software",
        "Software, hardware & consumer tech",
        "Software/library versions, operating systems, programming languages, devices, chips, platforms, cybersecurity.",
        keywords=("release", "version", "python", "linux", "kernel", "windows", "macos", "ios", "android", "iphone",
                  "pixel", "chip", "processor", "framework", "library", "api", "npm", "pypi", "rust", "javascript",
                  "typescript", "react", "kubernetes", "vulnerability", "cve", "browser", "chrome", "firefox", "update",
                  "smartphone", "betriebssystem", "programmiersprache", "sicherheitslücke"),
        seeds=("The latest stable Python release is", "The new iPhone model ships with",
               "The kernel version introduced support for"),
        wikipedia_sections=("Science and technology",),
    ),
    Topic(
        "science_space",
        "Science, space & research",
        "Scientific discoveries, space missions and launches, physics/biology results, prizes in science.",
        keywords=("nasa", "esa", "spacex", "starship", "launch", "mission", "orbit", "moon", "mars", "telescope",
                  "discovery", "researchers", "study published", "nature", "science journal", "physics", "nobel prize in",
                  "artemis", "asteroid", "fusion", "quantum", "raumfahrt", "rakete", "forscher", "entdeckung", "studie"),
        seeds=("The space agency's mission to the Moon", "Researchers reported in Nature that",
               "The telescope observed"),
        wikipedia_sections=("Science and technology",),
    ),
    Topic(
        "health_environment",
        "Health, medicine, climate & environment",
        "Disease outbreaks, drug approvals, public health, climate policy, COP summits, environment, energy transition.",
        keywords=("outbreak", "virus", "vaccine", "pandemic", "who ", "fda", "drug approval", "clinical trial", "cancer",
                  "measles", "bird flu", "h5n1", "climate", "cop", "emissions", "carbon", "renewable", "heatwave",
                  "biodiversity", "pollution", "impfstoff", "krankheit", "klima", "emission", "erneuerbare", "gesundheit"),
        seeds=("Health authorities reported a new outbreak of", "The regulator approved the drug for",
               "At the climate conference the parties agreed"),
        wikipedia_sections=("Health and environment",),
    ),
    Topic(
        "disasters",
        "Disasters & accidents",
        "Earthquakes, hurricanes, floods, wildfires, plane crashes, industrial accidents and their death tolls.",
        keywords=("earthquake", "hurricane", "typhoon", "cyclone", "flood", "wildfire", "tsunami", "eruption", "volcano",
                  "plane crash", "derailment", "explosion", "collapse", "death toll", "magnitude", "evacuated",
                  "erdbeben", "hochwasser", "überschwemmung", "waldbrand", "flugzeugabsturz", "unglück", "todesopfer"),
        seeds=("The magnitude 7 earthquake struck", "The hurricane made landfall near", "The plane crashed shortly after"),
        wikipedia_sections=("Disasters and accidents",),
    ),
    Topic(
        "law_crime",
        "Law, courts & crime",
        "Trials, verdicts, court rulings, indictments, investigations, major crimes and attacks (non-war).",
        keywords=("court", "ruling", "verdict", "trial", "sentenced", "indicted", "lawsuit", "judge", "supreme court",
                  "convicted", "acquitted", "arrested", "investigation", "prosecutor", "appeal", "shooting", "murder",
                  "gericht", "urteil", "prozess", "angeklagt", "verurteilt", "staatsanwalt", "festgenommen"),
        seeds=("The court ruled that", "The defendant was sentenced to", "Prosecutors charged"),
        wikipedia_sections=("Law and crime",),
    ),
    Topic(
        "sports",
        "Sports",
        "Results, champions, transfers, coaches, records and athlete news across football, US sports, tennis, F1, Olympics.",
        keywords=("championship", "champion", "world cup", "final", "league", "season", "coach", "manager", "transfer",
                  "olympic", "grand slam", "wimbledon", "formula 1", "f1", "grand prix", "super bowl", "nba", "nfl",
                  "mlb", "nhl", "uefa", "fifa", "bundesliga", "premier league", "champions league", "tour de france",
                  "meister", "weltmeister", "trainer", "spieler", "tor", "pokal", "olympia"),
        seeds=("The team won the championship final against", "The new head coach of the club",
               "The driver clinched the title at the Grand Prix"),
        wikipedia_sections=("Sports",),
    ),
    Topic(
        "entertainment",
        "Film, TV, music & celebrities",
        "Movies, series, box office, albums, tours, awards (Oscars, Grammys, Emmys), celebrity news and deaths.",
        keywords=("film", "movie", "box office", "series", "season finale", "netflix", "disney", "album", "single",
                  "tour", "grammy", "oscar", "academy award", "emmy", "golden globe", "actor", "actress", "singer",
                  "rapper", "band", "director", "premiere", "streaming", "schauspieler", "sänger", "kino", "serie",
                  "musiker", "preisverleihung"),
        seeds=("The film won Best Picture at the", "The singer's new album debuted at",
               "The actor, known for the series"),
        wikipedia_sections=("Arts and culture",),
    ),
    Topic(
        "culture_society",
        "Culture, religion & society",
        "Religion (popes, church leaders), literature, art, education, social movements, demographics, royalty.",
        keywords=("pope", "vatican", "church", "archbishop", "religious", "literature", "novel", "booker", "museum",
                  "art", "royal", "king", "queen", "monarch", "university", "protest", "society", "migration",
                  "papst", "kirche", "könig", "königin", "literaturnobelpreis", "museum", "gesellschaft", "proteste"),
        seeds=("The pope appointed", "The novel won the literary prize", "The king opened the session"),
        wikipedia_sections=("Arts and culture", "Religion"),
    ),
)

TOPIC_IDS: tuple[str, ...] = tuple(t.id for t in TOPICS)
TOPIC_BY_ID: dict[str, Topic] = {t.id: t for t in TOPICS}

# Wikipedia current-events headings -> candidate topics (first is the default when no finer signal exists).
WIKIPEDIA_SECTION_TO_TOPICS: dict[str, tuple[str, ...]] = {}
for _t in TOPICS:
    for _s in _t.wikipedia_sections:
        WIKIPEDIA_SECTION_TO_TOPICS.setdefault(_s, ())
        WIKIPEDIA_SECTION_TO_TOPICS[_s] = WIKIPEDIA_SECTION_TO_TOPICS[_s] + (_t.id,)


def get_topic(topic_id: str) -> Topic:
    try:
        return TOPIC_BY_ID[topic_id]
    except KeyError as e:
        raise KeyError(f"unknown topic {topic_id!r}; known: {', '.join(TOPIC_IDS)}") from e
