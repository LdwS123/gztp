# 🎬 Générateur de Reels pour Podcast

**Drop ta vidéo de podcast → reçois des clips courts viraux** prêts à styliser dans ton outil de montage.

100% local, 100% gratuit. Aucune API payante.

---

## 🤝 Qui fait quoi

| Étape | Qui |
|---|---|
| 1. Transcription audio (Whisper local) | 🤖 le script |
| 2. Détection des moments viraux | 🤖 le script |
| 3. Découpage des clips + recadrage 9:16 | 🤖 le script |
| 4. Génération du HOOK + caption + hashtags | 🤖 le script |
| 5. **Stylisation des sous-titres / overlay** | 👤 toi (CapCut, Submagic…) |
| 6. Publication sur Insta/TikTok | 👤 toi |

---

## ⚡ Usage

```bash
./reels.sh "/chemin/vers/podcast.mp4"
```

Astuce: glisse-dépose ton fichier dans le terminal pour insérer le chemin.

**Première fois sur une vidéo:** ça prend du temps (transcription Whisper, ~5-15 min pour 1h d'audio).
**Re-run sur la même vidéo:** instantané (transcription mise en cache dans `transcripts/`).

### Conseil: commence par un dry-run

```bash
./reels.sh "podcast.mp4" --dry-run
```

Ça transcrit + analyse + affiche les moments choisis (avec hook/caption/hashtags) **sans découper la vidéo**. Si les choix te plaisent, tu relances sans `--dry-run` pour générer les .mp4.

## 📦 Ce que tu reçois pour chaque clip

Dans `reels_output/` :

```
reel_01_xxxx.mp4    ← clip vertical 1080×1920, AUDIO + VIDÉO uniquement (pas de subs)
reel_01_xxxx.srt    ← sous-titres mots-par-mots, à importer dans CapCut/Submagic
reel_01_xxxx.txt    ← HOOK + CAPTION + HASHTAGS à copier-coller
reel_02_xxxx.mp4
reel_02_xxxx.srt
reel_02_xxxx.txt
…
summary.json        ← récap structuré de tous les clips
```

Workflow type:
1. Tu prends `reel_01_xxxx.mp4` + `reel_01_xxxx.srt`
2. Tu les ouvres dans CapCut / Submagic / Premiere
3. Tu stylises les sous-titres comme tu veux + tu rajoutes le HOOK en haut
4. Tu copies le contenu de `reel_01_xxxx.txt` dans ton post Insta/TikTok

## ⚙️ Options

```bash
./reels.sh "podcast.mp4" --max-clips 12 --min-len 20 --max-len 40
```

| Option | Défaut | Description |
|---|---|---|
| `--max-clips` | 10 | Nombre max de reels |
| `--min-len` | 15 | Durée min en secondes |
| `--max-len` | 45 | Durée max en secondes |
| `--model` | `base` | Whisper: `tiny` `base` `small` `medium` `large-v3` |
| `--horizontal` | off | Garder le format d'origine (pas de recadrage 9:16) |
| `--dry-run` | off | Juste l'analyse, pas de découpage |
| `--transcript F` | — | Utiliser une transcription externe (`.srt`/`.vtt`/`.json`) au lieu de Whisper |

### Choisir le modèle Whisper

| Modèle | Vitesse (1h audio) | Qualité |
|---|---|---|
| `tiny` | ~2 min | basique |
| `base` | ~5 min | bon (par défaut) |
| `small` | ~10 min | très bon |
| `medium` | ~30 min | excellent |
| `large-v3` | ~1h | broadcast |

Pour la 1ère fois et tester, **reste sur `base`**. Si tu vois que la transcription rate des trucs, monte à `small`.

## 🧠 Comment le script choisit les moments (v3 — full stack viral)

Le pipeline `make_reels_v3.py` combine **5 sources de signal** pour identifier
les vrais moments viraux (pas juste les segments thématiquement cohérents) :

### 1. Détection topique (ClipsAI)
Frontières propres début/fin de phrase via TextTiling sur embeddings.

### 2. Viral arc detector ⭐ (nouveau)
Au lieu de partir des **sujets**, on part des **réactions du public** :
- On détecte les pics audio (rires soutenus, applaudissements, montées d'énergie)
- Autour de chaque pic on construit un arc : `[setup avant ~60s] [PIC] [after-laugh ~8s]`
- Snap aux frontières de phrase pour des cuts propres
- Cap dur à `max_len * 1.15` (slack pour ne pas couper la chute)

C'est ce qui résout le problème **"les moments viraux dépassent toujours
ma fenêtre"** — un arc peut faire 80-90s si le moment le mérite.

### 3. Scoring heuristique
- ✅ Hooks d'ouverture (`écoute`, `the secret`, `imagine`…)
- ✅ Triggers viraux (`incroyable`, `crazy`, `dingue`…)
- ✅ Questions, exclamations, chiffres concrets
- ✅ Storytelling perso (`I was…`, `j'étais…`)
- ❌ Pénalité fillers, pénalité bad starters

### 4. LLM judge local (Ollama qwen2.5:7b)
Note 0-10 sur 6 dimensions : hook, payoff, emotion, clarity, shareability, ending.
Pondéré pour pénaliser dur les clips qui finissent en queue de poisson.
Génère aussi le hook 8 mots, la caption, les hashtags, et **start_excerpt /
end_excerpt verbatim** pour aligner le clip vidéo pile sur la phrase punch.

### 5. Signaux audio avancés (optionnels) 🆕
- **`--emotion`** : speechbrain wav2vec2 IEMOCAP → arousal par seconde
- **`--diarization`** : pyannote → turn-taking density (échanges vifs = viraux)
- **`--prosody`** : librosa → variation de pitch + énergie (cris, montée de voix)
- **`--all-signals`** : raccourci pour les 3

Bonus jusqu'à +15 points sur le score final.

## ⚡ Usage

```bash
./reels.sh "podcast.mp4"                       # full stack (v3)
./reels.sh "podcast.mp4" --dry-run             # juste l'analyse
./reels.sh "podcast.mp4" --all-signals         # +emotion +diarization +prosody
./reels.sh "podcast.mp4" --legacy              # ancien pipeline v1
```

## 📁 Structure du projet

```
GZTP/
├── make_reels.py       # v1 — pipeline heuristique pur (--legacy)
├── make_reels_v2.py    # v2 — ClipsAI + WhisperX + scoring
├── make_reels_v3.py    # v3 — full stack (LLM judge + viral arcs + signaux)
├── llm_judge.py        # interface Ollama (score + génération de contenu)
├── viral_arcs.py       # 🆕 détection d'arcs autour des pics audio
├── viral_signals.py    # 🆕 emotion / diarisation / prosodie (optionnels)
├── reels.sh            # lanceur shell (v3 par défaut, --legacy pour v1)
├── requirements.txt
├── README.md
├── reels_v3_output/    # sortie v3 : .mp4 + .srt + .txt + summary.json
├── transcripts/        # cache Whisper, audio analysis, signaux viraux
└── venv312/            # env Python (clipsai + speechbrain + librosa…)
```

## 🛠️  Setup

```bash
python3.12 -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt

# LLM judge (optionnel mais recommandé)
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b

# Diarisation (optionnel) — récupère un token sur huggingface.co/settings/tokens
# et accepte la licence sur huggingface.co/pyannote/speaker-diarization-3.1
export HF_TOKEN=hf_xxxxxxxxxxxx
```
