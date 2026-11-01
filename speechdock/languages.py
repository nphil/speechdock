"""Whisper's supported language codes and names."""

LANGUAGES = dict(item.split(":", 1) for item in (
    "en:english zh:chinese de:german es:spanish ru:russian ko:korean fr:french ja:japanese "
    "pt:portuguese tr:turkish pl:polish ca:catalan nl:dutch ar:arabic sv:swedish it:italian "
    "id:indonesian hi:hindi fi:finnish vi:vietnamese he:hebrew uk:ukrainian el:greek ms:malay "
    "cs:czech ro:romanian da:danish hu:hungarian ta:tamil no:norwegian th:thai ur:urdu "
    "hr:croatian bg:bulgarian lt:lithuanian la:latin mi:maori ml:malayalam cy:welsh sk:slovak "
    "te:telugu fa:persian lv:latvian bn:bengali sr:serbian az:azerbaijani sl:slovenian "
    "kn:kannada et:estonian mk:macedonian br:breton eu:basque is:icelandic hy:armenian "
    "ne:nepali mn:mongolian bs:bosnian kk:kazakh sq:albanian sw:swahili gl:galician "
    "mr:marathi pa:punjabi si:sinhala km:khmer sn:shona yo:yoruba so:somali af:afrikaans "
    "oc:occitan ka:georgian be:belarusian tg:tajik sd:sindhi gu:gujarati am:amharic "
    "yi:yiddish lo:lao uz:uzbek fo:faroese ht:haitian ps:pashto tk:turkmen nn:nynorsk "
    "mt:maltese sa:sanskrit lb:luxembourgish my:myanmar bo:tibetan tl:tagalog mg:malagasy "
    "as:assamese tt:tatar haw:hawaiian ln:lingala ha:hausa ba:bashkir jw:javanese su:sundanese "
    "yue:cantonese"
).split())

_ALIASES = {"burmese": "my", "valencian": "ca", "flemish": "nl", "haitian creole": "ht", "letzeburgesch": "lb", "pushto": "ps", "panjabi": "pa", "moldavian": "ro", "moldovan": "ro", "sinhalese": "si", "castilian": "es", "mandarin": "zh"}


def normalize_language(value: str) -> str:
    value = value.strip().lower()
    if value == "auto":
        return value
    if value in LANGUAGES:
        return value
    if value in _ALIASES:
        return _ALIASES[value]
    for code, name in LANGUAGES.items():
        if value == name:
            return code
    raise ValueError(f"Unsupported language: {value}")
