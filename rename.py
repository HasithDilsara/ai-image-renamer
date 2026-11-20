import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import logging

# ---------------- CONFIG ----------------
TARGET_FOLDER = "../UpscaleBot/Upscaled"
MODEL_ID = "microsoft/Florence-2-large"

logging.getLogger("transformers").setLevel(logging.ERROR)

# --------------- MODEL SETUP ---------------
def setup_model():
    print(f"Loading {MODEL_ID}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Using device: {device.upper()}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    return model, processor, device, dtype

# --------------- TITLE + HASHTAGS BUILDER ---------------
def build_title_and_tags(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    banned = {
        'generated','generation','ai','artificial','created',
        'rendered','synthetic','illustration','art','design','stockphoto'
    }

    stopwords = {
        'the','a','an','this','that','these','those','image','photo',
        'picture','graphic','beautiful','high','quality','detailed',
        'style','view','isolated','close','up'
    }

    words = [w for w in text.split() if w not in stopwords and w not in banned]

    # Stock-priority keywords
    priority = [
        'technology','business','smart','agent','robotic','financial',
        'new','holiday','background','pattern','abstract','nature',
        'blue','white'
    ]

    ordered = []
    for p in priority:
        if p in words:
            ordered.append(p)
            words.remove(p)

    ordered.extend(words)

    # ---- FORCE 5–10 WORD TITLE ----
    title_words = ordered[:10]
    if len(title_words) < 5:
        title_words.extend(['background', 'scene', 'detail'])

    title_words = title_words[:10]

    title = " ".join(title_words).capitalize()

    # ---- EXACTLY 4 HASHTAGS ----
    hashtags = []
    for w in title_words:
        if len(w) > 5:
            tag = f"#{w.capitalize()}"
            if tag not in hashtags:
                hashtags.append(tag)
        if len(hashtags) == 5:
            break

    return title, " ".join(hashtags)

# --------------- MAIN RENAME LOGIC ---------------
def run_rename():
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
        print(f"Folder created: {TARGET_FOLDER}")
        return

    model, processor, device, dtype = setup_model()

    print("\n--- STARTING RENAME ---\n")

    for filename in os.listdir(TARGET_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        full_path = os.path.join(TARGET_FOLDER, filename)
        print(f"Analyzing: {filename}")

        try:
            image = Image.open(full_path).convert("RGB")

            # Florence requires ONLY the task token
            prompt = "<CAPTION>"

            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(device, dtype)

            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False
            )

            generated_text = processor.batch_decode(
                output_ids,
                skip_special_tokens=False
            )[0]

            parsed = processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )

            raw_description = parsed[prompt]

            title, hashtags = build_title_and_tags(raw_description)

            ext = os.path.splitext(filename)[1]
            new_name = f"{title} {hashtags}{ext}"

            # Safety filename length
            if len(new_name) > 220:
                new_name = new_name[:200] + ext

            if new_name != filename:
                os.rename(full_path, os.path.join(TARGET_FOLDER, new_name))
                print(f"  -> Renamed to: {new_name}\n")
            else:
                print("  -> Name already optimal\n")

        except Exception as e:
            print(f"  -> Error: {e}\n")

# --------------- ENTRY POINT ---------------
if __name__ == "__main__":
    run_rename()
