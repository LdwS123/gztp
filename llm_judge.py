"""
LLM Judge — score viral d'un clip avec un modèle local via Ollama.

Stratégie 2026 : on calque les frameworks documentés des hooks qui performent
réellement sur Reels/TikTok (open loop, contradiction, hot take, forbidden
knowledge, self-aware, timeframe, negative bias) au lieu d'inventer des
critères abstraits.

API Ollama : http://localhost:11434/api/generate
Modèle par défaut : qwen2.5:7b
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"


# Catégories d'angle pour diversifier la sélection finale.
# Inspirées des familles de hooks qui marchent (Captain Hook AI / ReelRise 2026).
VALID_ANGLES = {
    "story",          # anecdote / récit perso avec arc
    "hot_take",       # opinion contrariante, "controversial truth"
    "confession",     # vulnérable, self-aware, "I screwed up"
    "reaction",       # rire, choc, surprise, banter, drama
    "advice",         # leçon concrète, framework, lifehack
    "contradiction",  # tension entre 2 idées opposées
    "behind_scenes",  # insider / forbidden knowledge
    "other",
}


@dataclass
class ViralScore:
    overall: float          # 0-10, score global agrégé
    hook: float             # le début accroche ?
    payoff: float           # y a-t-il une chute / punch / révélation ?
    emotion: float          # intensité émotionnelle
    clarity: float          # compréhensible sans contexte ?
    shareability: float     # donne envie de partager ?
    ending: float           # finit sur une phrase complète + impact (pas mid-thought)
    angle: str = "other"    # catégorie pour diversification
    reasoning: str = ""     # 1 phrase qui justifie


@dataclass
class GeneratedContent:
    hook: str               # accroche courte (≤ 9 mots, MAJUSCULES)
    caption: str            # caption complète pour le post
    hashtags: List[str]     # 5-8 hashtags ciblés
    start_excerpt: str      # ~6 mots EXACTS du transcript où démarrer le clip
    end_excerpt: str        # PHRASE COMPLÈTE de conclusion (verbatim, finit par . ! ?)


# ---------------------------------------------------------------------------
# Bas niveau
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL,
                      json_mode: bool = False, temperature: float = 0.2,
                      timeout: int = 180, num_predict: int = 500) -> str:
    """Appelle Ollama et renvoie la réponse texte."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "").strip()


def _safe_json_loads(text: str) -> Optional[dict]:
    """Tente de parser du JSON, en récupérant le premier objet trouvé."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# Scoring viral
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are a top short-form video editor who has produced 1000+ viral
Instagram Reels and TikToks for podcasts. You know the 2026 short-form playbook cold.

LANGUAGE RULE — ABSOLUTE: the source content is ENGLISH-ONLY. Your reasoning,
your angle label, and EVERY string you output MUST be in English (US idioms).
Never reply in French, Spanish, or any other language even if the transcript
contains a foreign word.

THE 6 FRAMEWORKS THAT WORK ON REELS/TIKTOK IN 2026 (reward when present):
  1. CONTRADICTION   — "I'm X but I think I'm the problem" (unresolved tension)
  2. HOT TAKE        — "the influencer era is dead", "soulmates are propaganda"
  3. FORBIDDEN       — "they keep this a secret", "this site feels illegal"
  4. SELF-AWARE      — surgical vulnerability ("I knew summer was coming and I just kept eating")
  5. TIMEFRAME       — "3 months ago I had 0 followers, today 211K"
  6. NEGATIVE BIAS   — "stop doing X", "the #1 mistake people make with Y"

CALIBRATION (use the FULL 0-10 range — DO NOT cluster everything around 5-7):
  0-2  = trash: filler, recap, generic platitudes, mid-thought fragment, no specifics
  3-4  = weak: a glimmer but no real hook OR no real payoff
  5-6  = mid: clear topic but predictable, would play once and be forgotten
  7-8  = strong: would realistically pull 50K-200K views, friends would screenshot
  9-10 = banger: genuinely surprising/emotional/quotable, would hit 1M+ and get stitched

REWARD HARD: specific numbers, named people/companies/places, contrarian takes,
  visible reactions (laughter, "no way", cursing), back-and-forth banter, vulnerable
  confessions, drama, named enemies, money figures, time-bound results, before/after
  contrasts, sentences that work as a tweet on their own.
PUNISH HARD: starts with filler ("yeah so", "I mean", "you know like", "and basically"),
  ends mid-sentence or mid-thought, abstract concepts without examples, recap of what
  was said earlier, hedging ("maybe", "I think we should kind of"), generic life advice.

ENDING IS CRITICAL: a clip that doesn't land on a complete punch/idea CANNOT score
  above 5 overall. The last 3 seconds decide if someone replays + shares."""


JUDGE_PROMPT_TEMPLATE = """Score this podcast clip transcript on 6 dimensions, 0-10 integers.
Use the FULL 0-10 range. Most clips are 4-6. Reserve 8+ for clips you'd actually
post on your own page.

CLIP TRANSCRIPT:
\"\"\"
{text}
\"\"\"{dialogue_block}{signals_block}

Dimensions:
1. **hook** — does the FIRST sentence stop a scroll in 1.5s? Does it match one of
   the 6 viral frameworks (contradiction / hot take / forbidden / self-aware /
   timeframe / negative bias)? Generic openers = 0-3.
2. **payoff** — is there an actual punchline, reveal, twist, named outcome, or
   counterintuitive insight? Or does it just trail off into "yeah" / "anyway" / vibes?
3. **emotion** — funny, shocking, controversial, vulnerable, angry, intense?
   Audible reaction in transcript (laughs, "wait WHAT", swearing) is a huge bonus.
4. **clarity** — can a stranger who never heard the podcast understand it standalone?
5. **shareability** — would someone screenshot, DM, stitch, or quote-tweet this?
6. **ending** — does the LAST sentence land cleanly (complete idea, period, punchline)?
   If it cuts mid-thought or ends on filler ("and yeah", "so…", "you know") → 0-3.

Also pick the dominant ANGLE among:
  story | hot_take | confession | reaction | advice | contradiction | behind_scenes | other

Respond ONLY with JSON, no markdown, no prose:
{{"hook":<0-10>,"payoff":<0-10>,"emotion":<0-10>,"clarity":<0-10>,"shareability":<0-10>,"ending":<0-10>,"angle":"<one of the 8>","reasoning":"<one short concrete sentence — what would or wouldn't make this go viral>"}}"""


def score_clip(
    text: str,
    model: str = DEFAULT_MODEL,
    dialogue: Optional[str] = None,
    speaker_turns: int = 0,
    has_laughter: bool = False,
    has_punchline: bool = False,
    duo_intensity: float = 0.0,
) -> Optional[ViralScore]:
    """Demande au LLM de scorer un clip. Renvoie None si parsing échoue.

    Args:
      text: transcript brut
      dialogue: version dialoguée "SPK_A: ... SPK_B: ..." (si diarization OK)
      speaker_turns: nb de changements de speaker dans le clip
      has_laughter / has_punchline / duo_intensity: signaux audio détectés
        que le LLM ne peut PAS deviner du texte seul. On les lui donne
        explicitement pour qu'il calibre mieux son score.
    """
    dialogue_block = ""
    if dialogue and speaker_turns >= 2:
        dialogue_block = (
            f"\n\nSAME CLIP, ATTRIBUTED BY SPEAKER (use this to detect banter, "
            f"interruptions, punchlines, debates):\n\"\"\"\n{dialogue[:1500]}\n\"\"\""
        )

    sig_lines = []
    if speaker_turns >= 3:
        sig_lines.append(f"- DIALOGUE: {speaker_turns} speaker changes "
                         f"(active back-and-forth, not a monologue)")
    if has_laughter:
        sig_lines.append("- LAUGHTER detected in audio (real reaction, not just text)")
    if has_punchline:
        sig_lines.append("- PUNCHLINE PATTERN detected (setup → short reaction → return)")
    if duo_intensity > 0.4:
        sig_lines.append(f"- DUO INTENSITY: {duo_intensity:.2f} "
                         "(tight banter / debate)")
    signals_block = ""
    if sig_lines:
        signals_block = (
            "\n\nAUDIO SIGNALS DETECTED (the listener will HEAR these — "
            "they boost virality):\n" + "\n".join(sig_lines)
        )

    prompt = JUDGE_SYSTEM + "\n\n" + JUDGE_PROMPT_TEMPLATE.format(
        text=text[:1800],
        dialogue_block=dialogue_block,
        signals_block=signals_block,
    )
    try:
        raw = _ollama_generate(prompt, model=model, json_mode=True, temperature=0.2)
    except Exception as exc:  # noqa: BLE001
        print(f"   ⚠️  LLM error: {exc}")
        return None

    data = _safe_json_loads(raw)
    if not data:
        return None

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    hook = _f("hook")
    payoff = _f("payoff")
    emotion = _f("emotion")
    clarity = _f("clarity")
    shareability = _f("shareability")
    ending = _f("ending")

    angle = str(data.get("angle", "other")).strip().lower()
    if angle not in VALID_ANGLES:
        angle = "other"

    # Score global pondéré : hook + shareability + ending sont les facteurs
    # qui prédisent le mieux la viralité réelle d'un clip podcast.
    overall = (
        hook * 2.0
        + shareability * 2.0
        + ending * 1.5
        + payoff * 1.5
        + emotion * 1.5
        + clarity * 1.0
    ) / 9.5

    # Cap dur : si l'ending est faible (<4), on ne dépasse pas 6.5 overall,
    # car un clip qui finit en queue de poisson ne se share pas.
    if ending < 4:
        overall = min(overall, 6.5)

    return ViralScore(
        overall=round(overall, 2),
        hook=hook,
        payoff=payoff,
        emotion=emotion,
        clarity=clarity,
        shareability=shareability,
        ending=ending,
        angle=angle,
        reasoning=str(data.get("reasoning", ""))[:220],
    )


# ---------------------------------------------------------------------------
# Génération de hook / caption / hashtags
# ---------------------------------------------------------------------------

CONTENT_PROMPT_TRASH = """You are the most brutally effective viral hook writer
on Reels/TikTok in 2026. Your job is ONE thing: stop the scroll in 0.8 seconds.
Your hooks pull 1M+ views because they are PROVOCATIVE, UNHINGED, and borderline
trash. Subtlety is death. Context is for the caption, not the hook.

LANGUAGE RULE — ABSOLUTE: target audience is ENGLISH-SPEAKING (US/UK).
ALL output (hook, caption, hashtags, excerpts) MUST be in English.
Never use French, Spanish, or any other language. Hashtags are lowercase English.

═══════════════════════════════════════════════════════════════════════
1) hook — on-screen overlay text. MAX 8 WORDS, ALL CAPS, no period, no emoji.

   ⚡ THE MISSION : the hook must feel like someone GRABBING you by the collar.
   It can be loosely inspired by the clip — even TANGENTIAL — as long as it
   punches in the face. Curiosity gap > literal description.

   ✅ ENCOURAGED patterns (pick one, push it to 10/10 intensity):

   • CALLOUT / ACCUSATION   "YOU'RE BROKE BECAUSE OF THIS"
                            "STOP LYING TO YOURSELF BRO"
                            "NOBODY IS GOING TO SAVE YOU"
   • CONFESSION TRASH       "I RUINED MY LIFE AT 22"
                            "I FAKED IT FOR 3 YEARS"
                            "EVERYONE HATES ME NOW"
   • HOT TAKE EXTREME       "COLLEGE IS A SCAM"
                            "NICE PEOPLE FINISH LAST"
                            "SUCCESS IS OVERRATED"
   • OUTRAGE / SHOCK        "HE SAID WHAT?!"
                            "THIS IS WHY MEN LEAVE"
                            "SHE COOKED HIM ALIVE"
   • FORBIDDEN / SECRET     "THEY DON'T WANT YOU TO KNOW"
                            "THIS SHOULD BE ILLEGAL"
                            "DELETE THIS BEFORE THEY SEE"
   • DARK QUESTION          "WHY ARE WE STILL POOR?"
                            "WHO RAISED YOU LIKE THIS?"
                            "WHY DO GIRLS DO THIS?"
   • STAKES + THREAT        "DO THIS OR STAY BROKE"
                            "READ OR REGRET IT"
                            "WATCH BEFORE IT'S GONE"
   • IDENTITY JAB           "WANNABE FOUNDERS DO THIS"
                            "LOSERS THINK LIKE THIS"
                            "BROKE PEOPLE LOVE THIS"

   ✅ THE HOOK CAN BE A TANGENT — it doesn't have to describe the clip.
   If the clip is about YC interviews and the speaker mentions rejection,
   a hook like "REJECTED? GOOD. HERE'S WHY" is fair game even if he never
   said those exact words. Vibe > literal.

   ⚠️  STILL BANNED — these are the ONE failure mode we forbid:
   • Specific DOLLAR AMOUNTS, PERCENTAGES, or YEARS that are NOT in the
     transcript. "$4M" or "5 YEARS" is only OK if it appears verbatim.
     (Vibes are free, fake numbers are lies.)
   • Soft corporate openers: "MEET THE FOUNDERS", "LISTEN TO THIS",
     "IMAGINE IF", "EVERY ENTREPRENEUR NEEDS". These don't stop scrolls.
   • Life-coach mush: "YOU'VE GOT THIS", "BELIEVE IN YOURSELF".

═══════════════════════════════════════════════════════════════════════
2) caption — 1-2 sentences, first person, 1 emoji max, no hashtags inside.
   This is where you ANCHOR the hook back to the actual clip content so the
   viewer doesn't feel baited too hard. Tease, don't spoil.

═══════════════════════════════════════════════════════════════════════
3) hashtags — 5-7 real multi-word hashtags (lowercase, no #).
   Mix 2-3 niche + 2-3 medium + 1 broad.
   GOOD: "ycombinator", "startupstory", "founderlife", "buildinpublic"
   BAD: single common words, nonsense ("podcastclip").

═══════════════════════════════════════════════════════════════════════
4) start_excerpt — EXACT first ~6 words (verbatim from transcript) where
   the clip should start. Skip filler ("yeah so", "and like", "I mean",
   "okay so"). Must begin a sentence.

═══════════════════════════════════════════════════════════════════════
5) end_excerpt — EXACT last full sentence (verbatim from transcript) where
   the clip should end.
   • MUST end in . ! or ? in the transcript.
   • MUST NOT end on filler.
   • Pick the strongest PUNCH sentence — the share-worthy one.
   • 6-12 verbatim words.

═══════════════════════════════════════════════════════════════════════
CLIP TRANSCRIPT:
\"\"\"
{text}
\"\"\"

Respond ONLY with JSON, no markdown, no prose:
{{"hook":"<MAX 8 WORDS ALL CAPS — MAKE IT TRASH, MAKE IT HIT>","caption":"<1-2 sentences, max 1 emoji>","hashtags":["tag1","tag2","tag3","tag4","tag5"],"start_excerpt":"<verbatim first ~6 words>","end_excerpt":"<verbatim last complete sentence, 6-12 words>"}}"""


# Garde l'ancien comportement disponible sous le nom CONTENT_PROMPT_GROUNDED
# au cas où on veuille revenir à du plus safe.
CONTENT_PROMPT_GROUNDED = """You are a top viral content writer for podcast
Reels/TikToks in 2026. Generate JSON with the fields below.

1) hook — MAX 8 WORDS, ALL CAPS, no period, no emoji. Must be derivable from
   the transcript. No invented numbers/names/places.
2) caption — 1-2 sentences, first person, 1 emoji max, no hashtags inside.
3) hashtags — 5-7 real multi-word hashtags (lowercase, no #).
4) start_excerpt — EXACT first ~6 words, skip filler.
5) end_excerpt — EXACT last complete sentence, 6-12 verbatim words.

CLIP TRANSCRIPT:
\"\"\"
{text}
\"\"\"

Respond ONLY with JSON:
{{"hook":"<MAX 8 WORDS ALL CAPS>","caption":"<1-2 sentences, max 1 emoji>","hashtags":["tag1","tag2","tag3","tag4","tag5"],"start_excerpt":"<verbatim first ~6 words>","end_excerpt":"<verbatim last complete sentence, 6-12 words>"}}"""


# Défaut = trash mode (c'est ce qu'on veut pour scroll-stop viralité 2026).
CONTENT_PROMPT = CONTENT_PROMPT_TRASH


# Mots/expressions à rejeter en début de start_excerpt (filler, mid-thought).
_BAD_START_TOKENS = {
    "and", "but", "so", "yeah", "okay", "ok", "right", "well", "like",
    "i mean", "you know", "basically", "actually", "anyway", "or",
    "because", "cause", "cuz", "though", "then", "also", "plus",
}

# Patterns interdits dans un hook — UNIQUEMENT les ouvertures corporate molles.
# En mode trash on garde une tolérance totale sur les autres patterns (callout,
# confession, outrage, identity jab…).
_VAGUE_HOOK_PATTERNS = [
    r"^MEET\b",
    r"^LISTEN\b(?! TO YOUR)",  # autorise "LISTEN TO YOUR MOM" etc.
    r"^WATCH\b(?! BEFORE)",    # autorise "WATCH BEFORE IT'S GONE"
    r"^IMAGINE IF\b",
    r"^EVERY [A-Z]+ NEEDS",
    r"^HOW TO BE\b",
    r"^YOU('VE| HAVE) GOT THIS",
    r"^BELIEVE IN YOURSELF",
]


_NUMBER_RE = re.compile(r"\$?\d[\d,.]*[kKmMbB]?")


def _hook_is_strong(hook: str) -> bool:
    if not hook:
        return False
    if len(hook.split()) < 2:
        return False
    for pat in _VAGUE_HOOK_PATTERNS:
        if re.search(pat, hook):
            return False
    return True


def _hook_is_grounded(hook: str, transcript: str) -> bool:
    """Vérifie que les chiffres/montants présents dans le hook sont aussi
    dans le transcript. Évite les hallucinations type 'LOST $4M' sans source.
    """
    if not hook or not transcript:
        return True
    nums_in_hook = _NUMBER_RE.findall(hook)
    if not nums_in_hook:
        return True
    transcript_norm = transcript.lower().replace(",", "")
    for num in nums_in_hook:
        norm = num.lower().lstrip("$").replace(",", "")
        if not norm:
            continue
        if norm in transcript_norm:
            continue
        digits = re.sub(r"[^\d]", "", norm)
        if digits and digits in transcript_norm:
            continue
        return False
    return True


def _start_excerpt_clean(excerpt: str) -> str:
    """Retire le filler en tête du start_excerpt (le LLM oublie parfois)."""
    if not excerpt:
        return excerpt
    tokens = excerpt.split()
    while tokens:
        joined = " ".join(tokens[:2]).lower().rstrip(",.!?;:")
        first = tokens[0].lower().rstrip(",.!?;:")
        if joined in _BAD_START_TOKENS or first in _BAD_START_TOKENS:
            tokens.pop(0)
            continue
        break
    return " ".join(tokens)


def generate_content(text: str, model: str = DEFAULT_MODEL,
                     retries: int = 2,
                     avoid_hooks: Optional[List[str]] = None) -> Optional[GeneratedContent]:
    """Génère hook, caption, hashtags via le LLM. Re-essaie si hook faible,
    halluciné ou déjà utilisé sur un autre clip."""
    avoid_set = {h.strip().upper() for h in (avoid_hooks or []) if h}
    last_data = None
    for attempt in range(retries + 1):
        prompt = CONTENT_PROMPT.format(text=text[:1800])
        if avoid_set:
            prompt += (
                "\n\nAVOID these exact hooks (already used on other clips, must be different):\n"
                + "\n".join(f"  • {h}" for h in sorted(avoid_set))
            )
        # Température haute = hooks plus déchaînés / provocants (mode trash).
        # On monte progressivement si on doit re-générer.
        temp = 0.85 + 0.15 * attempt
        try:
            raw = _ollama_generate(prompt, model=model, json_mode=True,
                                    temperature=temp, num_predict=500)
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  LLM content error: {exc}")
            return None

        data = _safe_json_loads(raw)
        if not data:
            continue
        last_data = data

        hook = str(data.get("hook", "")).strip().strip('"').strip(".").upper()
        if (
            _hook_is_strong(hook)
            and _hook_is_grounded(hook, text)
            and hook not in avoid_set
        ):
            break

    data = last_data
    if not data:
        return None

    hook = str(data.get("hook", "")).strip().strip('"').strip(".").upper()
    if len(hook.split()) > 9:
        hook = " ".join(hook.split()[:9])
    caption = str(data.get("caption", "")).strip().strip('"')

    raw_tags = data.get("hashtags", [])
    if isinstance(raw_tags, str):
        raw_tags = [t for t in re.split(r"[\s,]+", raw_tags) if t]
    tags = []
    for t in raw_tags[:8]:
        t = re.sub(r"[^a-z0-9]", "", str(t).lower())
        if t and len(t) >= 3 and t not in {tag.lstrip("#") for tag in tags}:
            tags.append(f"#{t}")
    if not tags:
        tags = ["#podcast", "#reels", "#shorts"]

    start_excerpt = _start_excerpt_clean(
        str(data.get("start_excerpt", "")).strip().strip('"')
    )
    end_excerpt = str(data.get("end_excerpt", "")).strip().strip('"')

    return GeneratedContent(
        hook=hook, caption=caption, hashtags=tags,
        start_excerpt=start_excerpt, end_excerpt=end_excerpt,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def is_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Vérifie que Ollama tourne et que le modèle est disponible."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        names = [m.get("name", "") for m in data.get("models", [])]
        return any(model in n or n.startswith(model.split(":")[0]) for n in names)
    except Exception:  # noqa: BLE001
        return False
