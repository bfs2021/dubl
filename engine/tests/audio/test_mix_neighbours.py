"""Персонажи говорят почти встык: хвост озвученной реплики не должен звучать оригиналом под чужой репликой.

Случай из проекта «Мегамозг»: Мегамозг (s1, озвучен) договаривает «…Мегамозгу?» до 18,6 с, Титан (s2, не озвучен)
начинает в 18,7 с. Раньше расширенные границы реплик перекрывались, кусок отдавался Титану и звучал оригиналом.
"""
from __future__ import annotations

import numpy as np

from dubl_engine.audio.io import read_wav, write_wav
from dubl_engine.audio.mix import SceneStems, mix_scene
from dubl_engine.audio.separate import voice_extents
from dubl_engine.models import MixSettings, Scene, SceneLine, Take, TrackSettings

from .synth_voice import SR, voice_track

DUR = 8.0


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def test_neighbour_extents_do_not_overlap():
    # фраза A звучит до 3,45 (субтитр — до 3,0), фраза B начинается с 3,3 (субтитр — с 3,6)
    a = voice_track([(1.0, 3.45)], DUR, seed=1, level=1.0, f0=120)
    b = voice_track([(3.55, 5.5)], DUR, seed=2, level=1.0, f0=220)
    ext = voice_extents(a + b, SR, [(1.0, 3.0), (3.6, 5.5)], max_ext=2.0, gap=0.6)
    assert ext[0][1] <= ext[1][0]                           # общий кусок поделён, а не отдан обоим
    assert 3.3 <= ext[0][1] <= 3.6                          # граница — в тихом месте между фразами


def test_dubbed_tail_does_not_leak_into_undubbed_neighbour(tmp_path):
    sd = tmp_path / "scene"
    (sd / "takes").mkdir(parents=True)
    va = voice_track([(1.0, 3.5)], DUR, seed=1, level=1.0, f0=120)       # оригинал A: хвост до 3,5
    vb = voice_track([(3.62, 5.6)], DUR, seed=2, level=1.0, f0=220)      # оригинал B
    paths = {k: write_wav(sd / f"{k}.wav", v, SR) for k, v in
             (("voices", va + vb), ("a", va), ("b", vb), ("background", np.zeros((int(DUR * SR), 2), np.float32)))}
    stems = SceneStems(voices=paths["voices"], background=paths["background"], per_speaker={"s1": paths["a"], "s2": paths["b"]},
                       takes_dir=sd / "takes", scene_dir=sd)
    write_wav(sd / "takes" / "quiet.wav", np.full(int((DUR + 1) * SR), 1e-4, np.float32), SR)   # «дубль» без речи
    take = Take(id="t1", line_id="l1", speaker_id="s1", file="takes/quiet.wav", duration=DUR, preroll=1.0, start_rel=0.0,
                selected=True)
    # B идёт в сцене первым — раньше перекрытие доставалось ему
    scene = Scene(id="sc001", title="t", start=0, end=DUR, dubbed_speakers=["s1"], takes=[take],
                  lines=[SceneLine(line_id="l0", speaker_id="s2", start_rel=6.5, end_rel=7.0, original="o", text="o"),
                         SceneLine(line_id="l1", speaker_id="s1", start_rel=1.0, end_rel=3.0, original="o", text="t"),
                         SceneLine(line_id="l2", speaker_id="s2", start_rel=3.75, end_rel=5.6, original="o", text="o")],
                  mix=MixSettings(keep_breaths=False, duck_background=False, background=TrackSettings(mute=True)))
    y, _ = read_wav(mix_scene(scene, stems, tmp_path / "m.wav"))
    y = y.mean(axis=1)
    assert _rms_db(y[int(3.05 * SR):int(3.45 * SR)]) < -50       # хвост A после дубля не звучит
    assert _rms_db(y[int(3.9 * SR):int(5.5 * SR)]) > -30         # реплика B (не озвучена) осталась
