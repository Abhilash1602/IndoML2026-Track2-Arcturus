"""Build (a) the external category-matched noise bank for training and (b) the unseen-noise eval set.

Sources (permissive licences only): FSD50K clips licensed CC0 or CC-BY (per-clip metadata);
MUSAN music without vocals (singing_music). No NC data.
  train bank : FSD50K *dev* split  -> data/ext_noise/<category>/*.wav (16 kHz, 0.2-1.2 s energetic segments)
  eval pool  : FSD50K *eval* split -> data/eval_unseen/ (104 held-out clean refs + external noise events mixed by the
               organiser recipe: noise only inside 1-4 events of 0.2-1.1 s, per-event SNR from the empirical list)
Licences of every used clip -> data/ext_noise/LICENSES.csv.
Usage: python -I train/build_ext_noise.py <fsd50k_root> <musan_root|-> <repo_root>
"""
import csv, json, os, random, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

FSD, MUSAN, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
SR = 16000
CATS = {
    "human_non_speech": {"Breathing", "Cough", "Sneeze", "Laughter", "Giggle", "Chuckle_and_chortle", "Sigh", "Gasp",
                         "Burping_and_eructation", "Respiratory_sounds", "Chewing_and_mastication", "Clapping", "Finger_snapping"},
    "vehicle_traffic": {"Car", "Car_passing_by", "Bus", "Truck", "Motorcycle", "Motor_vehicle_(road)", "Traffic_noise_and_roadway_noise",
                        "Vehicle_horn_and_car_horn_and_honking", "Engine", "Engine_starting", "Idling", "Accelerating_and_revving_and_vroom",
                        "Train", "Rail_transport", "Siren", "Aircraft", "Bicycle_bell", "Subway_and_metro_and_underground"},
    "animal": {"Dog", "Bark", "Cat", "Meow", "Purr", "Bird", "Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet", "Crow",
               "Chicken_and_rooster", "Fowl", "Livestock_and_farm_animals_and_working_animals", "Insect", "Cricket", "Frog", "Growling",
               "Gull_and_seagull", "Domestic_animals_and_pets", "Wild_animals"},
    "baby_child": {"Crying_and_sobbing", "Screaming"},
    "phone_signal_alarm": {"Telephone", "Ringtone", "Alarm", "Doorbell", "Church_bell", "Bell", "Tick-tock", "Clock"},
    "appliance_machine": {"Mechanical_fan", "Microwave_oven", "Printer", "Drill", "Power_tool", "Sawing", "Typing", "Computer_keyboard",
                          "Typewriter", "Toilet_flush", "Water_tap_and_faucet", "Sink_(filling_or_washing)", "Dishes_and_pots_and_pans",
                          "Mechanisms", "Buzz", "Hammer", "Blender"},
    "singing_music": {"Guitar", "Acoustic_guitar", "Electric_guitar", "Piano", "Drum", "Drum_kit", "Percussion", "Tabla", "Harmonica",
                      "Accordion", "Organ", "Trumpet", "Bowed_string_instrument", "Brass_instrument", "Plucked_string_instrument",
                      "Wind_instrument_and_woodwind_instrument", "Marimba_and_xylophone", "Harp", "Keyboard_(musical)"},
}
EXCLUDE = {"Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking",
           "Conversation", "Chatter", "Crowd", "Singing", "Male_singing", "Female_singing", "Whispering", "Speech_synthesizer",
           "Shout", "Yell", "Cheering"}
OK_LIC = ("creativecommons.org/licenses/by/", "creativecommons.org/publicdomain/zero/")
ORG_W = {"human_non_speech": 670, "vehicle_traffic": 626, "animal": 548, "baby_child": 231, "phone_signal_alarm": 78,
         "appliance_machine": 32, "singing_music": 28}
PER_CAT_TRAIN, SEG_PER_CLIP = 3000, 3
rng = random.Random(0); nrng = np.random.default_rng(0)


def clips(split):
    gt = list(csv.DictReader(open(f"{FSD}/meta/FSD50K.ground_truth/{split}.csv")))
    info = json.load(open(f"{FSD}/meta/FSD50K.metadata/{split}_clips_info_FSD50K.json"))
    out = {c: [] for c in CATS}
    for r in gt:
        labs = set(r["labels"].split(",")); lic = info.get(r["fname"], {}).get("license", "")
        if labs & EXCLUDE or not lic.startswith(("http://" + OK_LIC[0], "http://" + OK_LIC[1], "https://" + OK_LIC[0], "https://" + OK_LIC[1])):
            continue
        hit = [c for c, s in CATS.items() if labs & s]
        if len(hit) == 1:                                   # unambiguous category only
            out[hit[0]].append((r["fname"], lic, info[r["fname"]].get("uploader", "")))
    return out


def load16(p):
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def segments(x, k, lo=0.2, hi=1.2):
    """Up to k energetic segments (mean frame RMS above the clip median) of random length in [lo, hi] s."""
    f = 800; nf = len(x) // f
    if nf < 6: return []
    rms = np.sqrt((x[:nf * f].reshape(nf, f) ** 2).mean(1)); thr = np.median(rms); out = []
    for _ in range(4 * k):
        L = int(rng.uniform(lo, hi) * SR)
        if L >= len(x): continue
        a = rng.randint(0, len(x) - L); s = x[a:a + L]
        if np.sqrt((s ** 2).mean()) > max(thr, 1e-4) and np.abs(s).max() > 1e-3: out.append(s)
        if len(out) >= k: break
    return out


def build_train_bank(audio_dir):
    lic_rows = []
    for cat, lst in clips("dev").items():
        rng.shuffle(lst); od = f"{ROOT}/data/ext_noise/{cat}"; os.makedirs(od, exist_ok=True); n = 0
        for fname, lic, up in lst:
            if n >= PER_CAT_TRAIN: break
            p = f"{audio_dir}/{fname}.wav"
            if not os.path.exists(p): continue
            for j, s in enumerate(segments(load16(p), SEG_PER_CLIP)):
                sf.write(f"{od}/fsd{fname}_{j}.wav", s, SR, subtype="PCM_16"); n += 1
            lic_rows.append(("FSD50K-dev", fname, lic, up, cat))
        print(cat, "train segments", n, "from", len(lst), "eligible clips", flush=True)
    return lic_rows


def build_musan_music():
    rows = []
    if MUSAN == "-": return rows
    od = f"{ROOT}/data/ext_noise/singing_music"; n = 0
    for sub in ["rfm"]:   # only rfm is uniformly CC BY 3.0 (others include BY-SA / BY-ND)
        d = f"{MUSAN}/music/{sub}"; ann = f"{d}/ANNOTATIONS"
        if not os.path.exists(ann): continue
        for line in open(ann):
            t = line.split()
            if len(t) < 3 or t[2].upper() != "N": continue      # vocals flag must be N
            p = f"{d}/{t[0]}.wav"
            if not os.path.exists(p): continue
            for j, s in enumerate(segments(load16(p), 2)):
                sf.write(f"{od}/musan_{t[0]}_{j}.wav", s, SR, subtype="PCM_16"); n += 1
            rows.append(("MUSAN-music", t[0], "CC BY 3.0 (MUSAN rfm, Kevin MacLeod / incompetech)", "", "singing_music"))
            if n >= PER_CAT_TRAIN: break
    print("musan instrumental segments", n, flush=True)
    return rows


def build_unseen(audio_dir):
    pool = {}
    for cat, lst in clips("eval").items():
        segs = []
        for fname, lic, up in lst[:400]:
            p = f"{audio_dir}/{fname}.wav"
            if os.path.exists(p): segs += segments(load16(p), 2, 0.2, 1.1)
        pool[cat] = segs; print(cat, "eval segments", len(segs), flush=True)
    cats = [c for c in ORG_W if pool[c]]; w = np.array([ORG_W[c] for c in cats], float); w /= w.sum()
    snrs = [s for r in json.load(open(f"{ROOT}/train/synth_recipe_rows.json")) for s in r["snrs"]]
    hold = json.load(open(f"{ROOT}/train/ft_data_lists.json"))["hold"]
    od = f"{ROOT}/data/eval_unseen"; os.makedirs(od, exist_ok=True); pairs = []
    for d in hold:
        c, sr = sf.read(d["clean"], dtype="float32"); assert sr == SR
        x = c.copy(); ev = []; T = len(c)
        for _ in range(rng.choice([1, 2, 3, 4, 4])):
            nz = rng.choice(pool[cats[nrng.choice(len(cats), p=w)]])
            if len(nz) >= T: continue
            for _try in range(20):
                a = rng.randint(0, T - len(nz)); b = a + len(nz)
                if all(b <= s * SR or a >= e * SR for s, e in ev): break
            else:
                continue
            pc = float((c[a:b] ** 2).mean()); pc = pc if pc > 1e-7 else float((c ** 2).mean())
            x[a:b] += nz * np.sqrt(pc / ((nz ** 2).mean() * 10 ** (rng.choice(snrs) / 10) + 1e-12)); ev.append((a / SR, b / SR))
        pk = np.abs(x).max()
        if pk > 0.99: x, c = x / pk * 0.99, c / pk * 0.99
        sf.write(f"{od}/{d['id']}_noisy.wav", x.astype(np.float32), SR, subtype="FLOAT")
        sf.write(f"{od}/{d['id']}_clean.wav", c.astype(np.float32), SR, subtype="FLOAT")
        pairs.append({"id": d["id"], "noisy": f"{od}/{d['id']}_noisy.wav", "clean": f"{od}/{d['id']}_clean.wav",
                      "events": sorted(ev), "gt": d["gt"]})
    json.dump(pairs, open(f"{od}/pairs.json", "w"), ensure_ascii=False, indent=0)
    print("unseen eval pairs", len(pairs), "events", sum(len(p["events"]) for p in pairs), flush=True)


if __name__ == "__main__":
    what = os.environ.get("WHAT", "all")
    rows = []
    if what in ("all", "unseen"): build_unseen(f"{FSD}/FSD50K.eval_audio")
    if what in ("all", "train"):
        rows += build_train_bank(f"{FSD}/FSD50K.dev_audio"); rows += build_musan_music()
        with open(f"{ROOT}/data/ext_noise/LICENSES.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["source", "clip", "license", "uploader", "category"]); wr.writerows(rows)
    print("DONE", what)
