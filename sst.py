import os

# Disable Hugging Face symlinks on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
import torchaudio
import whisper

from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from IndicTransToolkit.processor import IndicProcessor
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    print("Warning: HF_TOKEN environment variable is not set.")


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("Loading models...")

# ------------------------------------------------------------
# Whisper
# Used ONLY for language detection + English transcription
# ------------------------------------------------------------

language_detector = whisper.load_model("tiny", device=DEVICE)


# ------------------------------------------------------------
# AI4Bharat IndicConformer
# Speech -> Original Indian Language Text
# ------------------------------------------------------------

asr_model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True,
    token=HF_TOKEN,
)
asr_model = asr_model.to(DEVICE)
asr_model.eval()


# ------------------------------------------------------------
# AI4Bharat IndicTrans2
# Indian Language -> English
# ------------------------------------------------------------

translator_model_name = (
    "ai4bharat/indictrans2-indic-en-dist-200M"
)

translator_tokenizer = AutoTokenizer.from_pretrained(
    translator_model_name,
    trust_remote_code=True,
    token=HF_TOKEN,
)

translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    translator_model_name,
    trust_remote_code=True,
    torch_dtype=(
        torch.float16
        if DEVICE == "cuda"
        else torch.float32
    ),
    token=HF_TOKEN,
).to(DEVICE)

translator_model.eval()


# ------------------------------------------------------------
# IndicTrans2 Processor
# ------------------------------------------------------------

ip = IndicProcessor(inference=True)

print("Models loaded successfully.")
print("Using device:", DEVICE)


# ============================================================
# LANGUAGE MAPS
# ============================================================

# IMPORTANT:
# These are the language codes expected by IndicConformer.

ASR_LANGUAGE_MAP = {
    "hi": "hi",       # Hindi
    "bn": "bn",       # Bengali
    "ta": "ta",       # Tamil
    "te": "te",       # Telugu
    "mr": "mr",       # Marathi
    "gu": "gu",       # Gujarati
    "kn": "kn",       # Kannada
    "ml": "ml",       # Malayalam
    "pa": "pa",       # Punjabi
    "or": "or",       # Odia
    "as": "as",       # Assamese
    "ur": "ur",       # Urdu
    "ne": "ne",       # Nepali
}


# IMPORTANT:
# These are the language codes expected by IndicTrans2.

TRANSLATION_LANGUAGE_MAP = {
    "hi": "hin_Deva",     # Hindi
    "bn": "ben_Beng",     # Bengali
    "ta": "tam_Taml",     # Tamil
    "te": "tel_Telu",     # Telugu
    "mr": "mar_Deva",     # Marathi
    "gu": "guj_Gujr",     # Gujarati
    "kn": "kan_Knda",     # Kannada
    "ml": "mal_Mlym",     # Malayalam
    "pa": "pan_Guru",     # Punjabi
    "or": "ory_Orya",     # Odia
    "as": "asm_Beng",     # Assamese
    "ur": "urd_Arab",     # Urdu
    "ne": "npi_Deva",     # Nepali
}


# ============================================================
# TRANSLATION HELPER
# ============================================================

def translate_to_english(text, src_lang):
    """
    Translate an Indian-language sentence to English
    using IndicTrans2.
    """

    if not text or not text.strip():
        return ""

    text = text.strip()

    # --------------------------------------------------------
    # IndicTrans2 preprocessing
    # --------------------------------------------------------

    batch = ip.preprocess_batch(
        [text],
        src_lang=src_lang,
        tgt_lang="eng_Latn",
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = translator_tokenizer(
        batch,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    # Move tensors to model device
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Generate translation
    # --------------------------------------------------------

    with torch.no_grad():

        generated_tokens = translator_model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    decoded = translator_tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # --------------------------------------------------------
    # IndicTrans2 postprocessing
    # --------------------------------------------------------

    translations = ip.postprocess_batch(
        decoded,
        lang="eng_Latn",
    )

    return translations[0].strip()


# ============================================================
# SPEECH -> TEXT -> ENGLISH
# ============================================================

def speech_to_english(audio_path):
    """
    Complete pipeline:

        Audio
          ↓
        Whisper language detection
          ↓
        IndicConformer for Indian languages
          ↓
        IndicTrans2
          ↓
        English

    For English:
        Audio
          ↓
        Whisper transcription
          ↓
        English
    """

    # ========================================================
    # 1. DETECT LANGUAGE
    # ========================================================

    result = language_detector.transcribe(
        audio_path,
        task="transcribe",
        fp16=False,
    )

    detected_language = result["language"]

    print("Detected language:", detected_language)


    # ========================================================
    # 2. ENGLISH
    # ========================================================

    # If Whisper detects English, there is absolutely no
    # reason to run IndicConformer or IndicTrans2.

    if detected_language == "en":

        transcript = result["text"].strip()

        print("Original text:", transcript)
        print("English translation:", transcript)

        return {
            "language": "en",
            "original_text": transcript,
            "english_text": transcript,
        }


    # ========================================================
    # 3. CHECK SUPPORTED INDIAN LANGUAGE
    # ========================================================

    if detected_language not in ASR_LANGUAGE_MAP:

        raise ValueError(
            f"Language '{detected_language}' is not supported."
        )


    # ========================================================
    # 4. LOAD AUDIO
    # ========================================================

    wav, sample_rate = torchaudio.load(audio_path)

    # --------------------------------------------------------
    # Stereo -> Mono
    # --------------------------------------------------------

    if wav.shape[0] > 1:

        wav = torch.mean(
            wav,
            dim=0,
            keepdim=True,
        )

    # --------------------------------------------------------
    # Resample -> 16 kHz
    # --------------------------------------------------------

    if sample_rate != 16000:

        wav = torchaudio.functional.resample(
            wav,
            sample_rate,
            16000,
        )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Keep the tensor as:
    #
    #       [1, samples]
    #
    # Do NOT use:
    #
    #       wav.squeeze(0)
    #
    # because that produces:
    #
    #       [samples]
    #
    # which caused your previous NeMo/TorchScript
    # "Dimension out of range" error.
    # --------------------------------------------------------

    wav = wav.float()

    # Move audio to same device as ASR model
    wav = wav.to(DEVICE)


    # ========================================================
    # 5. INDICCONFORMER
    # ========================================================

    # IndicConformer uses short language codes:
    #
    # hi
    # ur
    # bn
    # ta
    # etc.

    asr_language = ASR_LANGUAGE_MAP[
        detected_language
    ]

    with torch.no_grad():

        transcript = asr_model(
            wav,
            asr_language,
            "ctc",
        )

    # Make sure result is a string
    if isinstance(transcript, (list, tuple)):

        transcript = transcript[0]

    transcript = str(transcript).strip()

    print("Original text:", transcript)


    # ========================================================
    # 6. INDICTRANS2
    # ========================================================

    # IndicTrans2 uses DIFFERENT language codes:
    #
    # hi  -> hin_Deva
    # ur  -> urd_Arab
    # bn  -> ben_Beng
    # etc.

    translation_language = TRANSLATION_LANGUAGE_MAP[
        detected_language
    ]

    english_text = translate_to_english(
        transcript,
        translation_language,
    )

    print("English translation:", english_text)


    # ========================================================
    # 7. RETURN
    # ========================================================

    return {
        "language": detected_language,
        "original_text": repr(transcript),
        "english_text": english_text,
    }


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("Speech Translation System")
    print("==========================")
    print("Device:", DEVICE)
    print()

    audio_file = input(
        "Enter path to audio file: "
    ).strip()

    if not os.path.exists(audio_file):

        print(
            f"Error: File does not exist: {audio_file}"
        )

    else:

        try:

            result = speech_to_english(
                audio_file
            )

            print()
            print("==========================")
            print("RESULT")
            print("==========================")
            print(
                "Language:",
                result["language"],
            )
            print(
                "Original:",
                result["original_text"],
            )
            print(
                "English:",
                result["english_text"],
            )

        except Exception as e:

            print()
            print("Error:", str(e))
