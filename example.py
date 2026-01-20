import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths
xtts_checkpoint = "./model.pth"
xtts_config = "./config.json"
xtts_vocab = "./vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

# Patch to support custom languages (skip validation)
def patch_xtts_language_support(model, custom_language="sk"):
    """
    Patch XTTS model to support custom trained languages by bypassing tokenizer validation.
    The tokenizer is the actual bottleneck - it validates language in preprocess_text().
    
    Args:
        model: XTTS model instance
        custom_language: Language code to support (e.g., "sk" for Slovak)
    """
    # Patch the tokenizer's preprocess_text to skip validation
    if hasattr(model, 'tokenizer'):
        original_preprocess = model.tokenizer.preprocess_text
        
        def preprocess_text_no_validation(txt, lang):
            """Skip language validation in preprocess_text"""
            # Get the list of valid language codes from the tokenizer
            valid_langs = getattr(model.tokenizer, 'supported_languages', None)
            
            # If language not in list but is our custom language, temporarily add it
            if lang == custom_language:
                if hasattr(model.tokenizer, 'supported_languages') and lang not in model.tokenizer.supported_languages:
                    model.tokenizer.supported_languages.append(lang)
            
            # Call original, which will now have our language
            try:
                return original_preprocess(txt, lang)
            except NotImplementedError:
                # If still fails, bypass completely by simulating the preprocessing
                # Most TTS preprocessing just normalizes text for the target language
                return txt.strip()
        
        model.tokenizer.preprocess_text = preprocess_text_no_validation
        print(f"✓ Patched tokenizer.preprocess_text() for '{custom_language}'")
    
    # Also patch the tokenizer's encode method to handle our language
    if hasattr(model, 'tokenizer'):
        original_encode = model.tokenizer.encode
        
        def encode_with_custom_language(txt, lang):
            """Encode text with custom language support"""
            # If encoding with our custom language, try to use closest supported language as fallback
            try:
                return original_encode(txt, lang)
            except NotImplementedError:
                if lang == custom_language:
                    # Try with English as base, since tokenizer is language-specific
                    # The actual character encoding should be mostly compatible
                    print(f"⚠ Tokenizer encode failed for '{lang}', falling back to 'en' for token encoding...")
                    return original_encode(txt, 'en')
                else:
                    raise
        
        model.tokenizer.encode = encode_with_custom_language
        print(f"✓ Patched tokenizer.encode() for '{custom_language}'")
    
    print(f"✓ Language validation bypassed for '{custom_language}'")
    return model


def get_xtts_model(language="sk", device="cpu"):
    if language == "sk":
        xtts_checkpoint = "./best_model.pth"
        config = XttsConfig()
        config.load_json("./config.json")
        xtts_model = Xtts.init_from_config(config)
        xtts_model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path="./vocab.json", use_deepspeed=False)
    else:
        xtts_checkpoint = "./XTTS-v2/model.pth"
        config = XttsConfig()
        config.load_json("./XTTS-v2/config.json")
        xtts_model = Xtts.init_from_config(config)
        xtts_model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path="./XTTS-v2/vocab.json", use_deepspeed=False)

    xtts_model.to(device)
    
    return xtts_model

def get_speaker_embedding(model, speaker_audio_file):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )
    return gpt_cond_latent, speaker_embedding

def synthesize_speech(model, text, gpt_cond_latent, speaker_embedding, lang="sk"):
    """
    Synthesize speech with custom language support.
    Uses patched model that bypasses language validation.
    """
    tts_texts = sent_tokenize(text)

    wav_chunks = []
    for sentence in tqdm(tts_texts):
        wav_chunk = model.inference(
            text=sentence,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
    
    return out_wav, model.config.audio.output_sample_rate

if __name__ == "__main__":
    # Patch the model on startup
    XTTS_MODEL = get_xtts_model(language="sk", device=device)
    XTTS_MODEL = patch_xtts_language_support(XTTS_MODEL, custom_language="sk")

    print("Model loaded successfully!")
    print("✓ Ready to synthesize Slovak speech\n")

    # Inference
    tts_text = "Ahoj, toto je ukážka prevodu textu na reč pomocou modelu XTTS. Dúfam, že sa vám to páči!"
    speaker_audio_file = "recording.wav"
    lang = "sk"
    
    #tts_text = "Hello, this is a demonstration of text-to-speech conversion using the XTTS model. I hope you like it!"
    #speaker_audio_file = "recording.wav"
    #lang = "en"
    
    gpt_cond_latent, speaker_embedding = get_speaker_embedding(XTTS_MODEL, speaker_audio_file)
    
    synthesized_wav, sample_rate = synthesize_speech(
        XTTS_MODEL,
        tts_text,
        gpt_cond_latent,
        speaker_embedding,
        lang=lang,
    )
    
    torchaudio.save("synthesized_output_sk.wav", synthesized_wav, sample_rate)
    print("Synthesis complete! Audio saved to 'synthesized_output_sk.wav'")