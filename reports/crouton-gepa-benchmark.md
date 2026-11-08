# Crouton GEPA Benchmark — تقرير أداء stealth/crouton-alpha مع تدوير مفاتيح OpenRouter

**تاريخ التشغيل:** 2026-11-08 11:05:59–11:12:40 (EET، UTC+02:00)
**المستودع:** `/home/fares/.hermes/self-evolution` (فقط — لم يُلمس أي ملف خارج هذا المجلد)
**النموذج الأساسي:** `openrouter/stealth/crouton-alpha`
**النموذج الداعم:** `openrouter/reflection/beam-1.0-501b-a23b`
**المهارة المُطوَّرة:** `github` (`skills/software-development/github/SKILL.md` في hermes-agent)
**مصدر بيانات التقييم:** `~/.hermes/state.db` الحقيقي (قراءة فقط، `mode=ro`)
**بدون pytest داخل الحلقة:** `run_pytest=False` (الافتراضي) — لم يُشغَّل أي pytest أثناء GEPA.

## الخلاصة التنفيذية

1. **تدوير المفاتيح يعمل.** أضيف `OpenRouterKeyPool` + `RotatingLM` يقرآن
   `OPENROUTER_API_KEY` و `OPENROUTER_API_KEY_2 … _11` (11 مفتاحًا) من
   `~/.hermes/.env` عبر `os.environ`، ويدوّران بالتناوب (Round-Robin) بعد كل
   استدعاء ناجح، وفورًا عند Rate Limit / Quota (429/402/503/529)، ثم
   يبدّلان إلى النموذج الداعم بعد استنفاد كل المفاتيح على النموذج الأساسي.
2. **دورة GEPA حية اكتملت بنجاح تقنيًا** في 81.3 ثانية (تحسين) و401 ثانية
   إجمالًا (منها ~280 ثانية تعدين بيانات الجلسات)، مع 68 استدعاء LM ناجحًا
   عبر المفاتيح الـ 11 وصفر Rate Limit hits أثناء مرحلة GEPA.
3. **كفاءة crouton-alpha:** نجاح 33/33 عبر المفاتيح الـ 11 في اختبار زمن
   منفصل (prompts فريدة، `cache=False`)، وسيط زمني 1.876s، متوسط 2.585s
   (متأثر بـ outlier أول استدعاء 17.153s). هذا يجعله مناسبًا كمحرك أساسي لـ GEPA.
4. **النتيجة التطورية نفسها لم تتحسن على الـ holdout:** أساسي 0.456 →
   مُطوَّر 0.404 (Δ −0.052). القيد الجوهري: `SkillModule.skill_text` ليس
   معامل DSPy قابلًا للتحسين، لذا ملف المهارة المُعاد تجميعه مطابق للأصلي
   (MD5 متطابق) — انظر «قيود صريحة وخطوات تالية».

## ما الذي تغيّر في الكود

| ملف | التغيير |
|---|---|
| `evolution/core/openrouter_keys.py` (جديد، stdlib فقط) | `OPENROUTER_KEY_ENV_VARS` (11 اسمًا بالترتيب)، `get_openrouter_api_keys()`، `OpenRouterKeyPool`، `is_rate_limit_error()`، `key_fingerprint()` (لا يُسجَّل المفتاح الكامل أبدًا — فقط `***<آخر 4>`) |
| `evolution/core/openrouter_rotation.py` (جديد) | `RotatingLM(dspy.LM)` يدوّر المفتاح داخل `__call__`، `make_rotating_lm()` / `configure_rotating_lm()`، تبديل للنموذج الداعم مع `max_tokens=8192` (انظر قيد beam أدناه) |
| `evolution/core/config.py` | `FREE_PRIMARY_MODEL = openrouter/stealth/crouton-alpha`، `FREE_FALLBACK_MODEL = openrouter/reflection/beam-1.0-501b-a23b`؛ حقول `openrouter_api_keys` / `openrouter_key_rotation` / `openrouter_round_robin` في `EvolutionConfig` |
| `evolution/skills/evolve_skill.py` | بناء LM عبر `make_rotating_lm(...)` مع طباعة عدد المفاتيح وبصمتها، وإضافة إحصاءات التدوير إلى `metrics.json` |
| `evolution/core/fitness.py`، `dataset_builder.py`، `external_importers.py` | استبدال كل `dspy.LM(model)` الخام بـ `make_rotating_lm(model, fallback=…)` |
| `tests/core/test_openrouter_rotation.py` (جديد، 19 اختبارًا) | يثبت قراءة المتغيرات المرقمة 1–11 بالترتيب، تخطي الفارغ/المكرر، Round-Robin، التدوير عند 429/402 وعدم التدوير عند 401/400، التبديل للنموذج الداعم، والتكامل مع `EvolutionConfig` — بمفاتيح وهمية فقط، بدون شبكة |
| `output/github/20261108_111240/metrics.json` | مخرجات الدورة الحية (مُستثنى من git عبر `.gitignore: output/`) |

> ملاحظة عن كيفية قراءة المفتاح قبل التغيير: `_load_hermes_env()` في
> `config.py` كان يحمّل كل سطر `KEY=VALUE` من `~/.hermes/.env` إلى
> `os.environ`، لكن كل LM كان يُبنى كـ `dspy.LM(model, base_url=…)` **بدون**
> `api_key`، فيستخدم LiteLLM المفتاح الوحيد `OPENROUTER_API_KEY` ضمنيًا.
> أول 429 كان يُسقط الدورة كاملة رغم وجود 10 مفاتيح أخرى.

## نتائج الاختبارات

```
.venv/bin/pytest -q
164 passed, 1 warning in 0.90s
```

- قبل التغيير: 145 passed.
- بعد التغيير: **164 passed** (145 + 19 اختبار تدوير جديد).
- التحذير الوحيد مسبق الوجود: `pytest_asyncio` deprecation + `anyio`
  `CallSpec2` — لا علاقة له بهذا العمل.

## دورة GEPA الحية — التفاصيل المقاسة

أمر التشغيل (بدون أي أسرار، وبدون pytest داخل الحلقة):

```bash
env -u PYTHONPATH .venv/bin/python -m evolution.skills.evolve_skill \
  --skill github --iterations 3 --eval-source sessiondb \
  --hermes-repo /home/fares/.hermes/hermes-agent \
  --optimizer-model openrouter/stealth/crouton-alpha \
  --eval-model openrouter/stealth/crouton-alpha
```

### 1) مصدر البيانات: `~/.hermes/state.db`

| مؤشر | القيمة |
|---|---|
| حجم الملف | 224 MB |
| إجمالي رسائل `messages` | 40,412 (user 1,716 / assistant 19,021 / tool 19,674 / session_meta 1) |
| جلسات مميزة | 322 |
| أزواج user/assistant مستخرجة بعد تنقية الأسرار | **785** |
| مرشحون بعد الترشيح الاستدلالي (heuristic) | 45 |
| أمثلة ذات صلة بعد تقييم LLM | **7** (كلها `source=hermes`) |
| التقسيم | train 3 / val 1 / holdout 3 |

المهارة المستهدفة وقعت في فخاخ حقيقية من الجلسات: إدارة repo عبر `gh`،
دورة issue→PR، ومراجعة PR بأربعة أسئلة (نطاق/صدق/اختبارات/ادعاءات) —
وهي نفسها التي علّمها GEPA لاحقًا في نص المتنبئ.

### 2) تطور GEPA (3 تكرارات، 12 استدعاء metric)

| المرحلة | val (مثال واحد) | train subsample (3 أمثلة) |
|---|---|---|
| الأساسي (Iteration 0) | 0.508 | 1.214 (متوسط 0.405) |
| Iteration 1 (مقبول) | **0.659** | 1.282 (متوسط 0.427) |
| Iteration 2 (مقبول على train، مرفوض على val) | 0.527 | 1.443 (متوسط 0.481) |
| الأفضل محفوظ في Pareto | Iteration 1 (0.659) | — |

- زمن التحسين: **81.3s** (≈5.7–10.2s لكل rollout).
- النص المقترح من GEPA شدّد: قراءة ملف reference المطابق قبل أي workflow،
  التحقق من حالة remote بـ `gh` طازج بدل الذاكرة، وتنفيذ end-to-end بدل التوقف
  عند خطة، وأسئلة مراجعة PR الأربعة مع استشهادات سطر-بترقيم.

### 3) التقييم على holdout + القيود

| المقياس | الأساسي | المُطوَّر | التغيّر |
|---|---|---|---|
| Holdout score (3 أمثلة) | 0.456 | 0.404 | **−0.052** |
| حجم body | 2,036 chars | 2,036 chars | +0 |
| القيود (size/growth/non-empty/frontmatter) | ✓ | ✓ جميعها ناجحة | — |
| Iterations | — | 3 | — |
| Elapsed (GEPA فقط) | — | 81.3s | — |

ملف `evolved_skill.md` **مطابق تمامًا** لملف `baseline_skill.md`
(MD5 `fa680674fce5e5894c1abe087db6f808` لكليهما، `diff` فارغ).
السبب الجذري (موثق بصراحة، خارج نطاق هذه المهمة): في
`evolution/skills/skill_module.py:104-107`، `self.skill_text` سمة نصية عادية
وليست معامل DSPy؛ GEPA يحسّن تعليمات `predictor.predict` (توقيع
`TaskWithSkill`)، وعند الاستخراج في `evolve_skill.py` يُقرأ
`optimized_module.skill_text` — أي النص الأصلي — فيضيع تحسين المتنبئ.
درجة الـ holdout أعلاه تقيس المتنبئ المُحسَّن أثناء التقييم، لا body المهارة.

### 4) إحصاءات تدوير المفاتيح أثناء الدورة (`metrics.json`)

```json
{
  "openrouter_key_count": 11,
  "openrouter_key_fingerprints": ["***346f","***6c36","***1240","***72de","***ff0f","***b83a","***5e23","***0077","***7cee","***0ee2","***b2ed"],
  "openrouter_rotations": 68,
  "openrouter_rate_limit_hits": 0,
  "openrouter_successful_calls": 68,
  "fallback_model": "openrouter/reflection/beam-1.0-501b-a23b"
}
```

- 11 مفتاحًا حُملت من البيئة (البصمات أعلاه هي آخر 4 أحرف فقط).
- 68 استدعاء ناجحًا ⇒ 68 تدويرًا بالتناوب (≈6.2 استدعاء لكل مفتاح في المتوسط)،
  0 Rate Limit hits — لم يضطر المحرك إلى التبديل للنموذج الداعم في هذه الدورة.
- عند بدء GEPA كان المفتاح الحالي `***6c36` (المفتاح الثاني) لأن تجمع
  المفاتيح مشترك (singleton) بين LM تعدين البيانات وLM التحسين، فتقدم
  المؤشر أثناء التعدين — وهذا سلوك مقصود لتوزيع الحمل.

## كفاءة النموذجين — اختبار زمن منفصل (مقاس حيًا)

إعداد: 11 مفتاحًا، prompts فريدة (nonce لكل استدعاء)، `cache=False`
(كاش DSPy الافتراضي `cache=True` يجعل الاستدعاءات المتكررة المتطابقة
0.00s — مفيد داخل GEPA حيث تتكرر أمثلة train/val، ومضلِّل في القياس).

### Crouton-alpha — كل المفاتيح الـ 11 × 3 تكرارات (33 استدعاء)

| مؤشر | القيمة |
|---|---|
| نجاح | **33/33 (100%)** |
| المتوسط | 2.585s |
| **الوسيط** | **1.876s** |
| min / max | 1.001s / 17.153s |
| ملاحظة outlier | أول استدعاء (17.153s) — cold start / إعادة محاولة داخلية؛ بقية الاستدعاءات 1.0–5.9s |

### Beam-1.0 — مفاتيح 3، 4، 11 × 2 مع `max_tokens=256` (6 استدعاءات)

| مؤشر | القيمة |
|---|---|
| نجاح | **6/6 (100%)** |
| المتوسط / الوسيط | 0.659s / 0.621s |

> مفارقة لافتة: على prompts ناتجة قصيرة جدًا كان beam أسرع من crouton
> في هذه العينة الصغيرة، بينما crouton هو الوحيد المجاني ($0) والمتاح
> بميزانية توكنز كاملة عبر كل المفاتيح. لا تُبنى قرارات GEPA على 6 استدعاءات.

### Beam-1.0 — المفاتيح 1، 2 بميزانية افتراضية (قيد حرج موثق)

| المفتاح | النتيجة |
|---|---|
| 1 (`***346f`) | 402 — "Insufficient credits. This account never purchased credits." |
| 2 (`***6c36`) | 402 — "can only afford 1" token من 131,072 المطلوبة |
| 3–11 بميزانية افتراضية 131,072 | كلها 402 — القدرة 11,529–36,195 توكنز لكل مفتاح، أي **أقل من الميزانية الافتراضية دائمًا** |

**النتيجة العملية:** النموذج الداعم على OpenRouter هو مسار micro-cost
وليس `:free`، وميزانيته الافتراضية (131,072) تجعل **كل** الاستدعاءات تفشل
مهما كان المفتاح. لذلك عند التبديل إلى beam يحدّ `RotatingLM` تلقائيًا
`max_tokens=8192` (ما لم يحدّده المتصل صراحة) — حدّ يقع ضمن قدرة كل
مفتاح من 3–11. مع هذا الحدّ ومع `max_tokens=256` نجح 6/6 أعلاه.
المفاتيح 1–2 يبقيان غير صالحين لـ beam حتى مع الحدّ (رصيد صفري/توكنز واحد).

## قيود صريحة وخطوات تالية

1. **استخراج المهارة المطوَّرة مكسور upstream:** `SkillModule.skill_text`
   ليست معامل DSPy، فلا يصل تحسين GEPA إلى ملف `SKILL.md`. الإصلاح الجذري:
   جعل `skill_instructions` معاملًا قابلًا للتحسين (مثلًا
   `dspy.Predict` بحقل قابل للتعلم، أو قراءة `predictor` المُحسَّن عند
   الاستخراج) قبل أي نشر لمهارة مطوَّرة. **القاعدة تبقى: لا نشر بدون
   مراجعة بشرية عبر PR** — لم تُرسل أي PR في هذه الدورة.
2. **حجم تقييم ضئيل:** 7 أمثلة فقط (holdout 3) — Δ −0.052 داخل ضجيج العينة
   ولا دلالة إحصائية له. لاتخاذ قرار نشر يلزم ≥50 مثالًا وholdout ≥15.
3. **مفتاحا beam غير صالحين:** shards 1–2 بلا رصيد لـ beam؛ التدوير
   يتجاوزهما عمليًا لكنهما يستهلكان محاولة. النظر في علامتهما كـ
   crouton-only في قائمة أولويات التدوير مستقبلًا.
4. **تكلفة التعدين تهيمن:** ~280s لتعدين وتقييم 45 مرشحًا مقابل 81s
   للتحسين نفسه. تسريع الترشيح المسبق أو تخزين تقييمات الصلة سيقلّص الدورة
   إلى ~2 دقيقة.
5. **التكرارات القادمة المقترحة (مرتبة بالأثر):**
   - إصلاح استخراج `skill_text` (أعلاه) ثم إعادة تشغيل نفس الدورة للمقارنة العادلة.
   - رفع `eval_dataset_size` إلى 30–50 مع نفس المفاتيح (التدوير مثبت الآن).
   - تجربة beam كـ judge/reviewer فقط (لا كمحرك أساسي) مع `max_tokens` المحدود.

## الملفات المرجعية (داخل المستودع فقط)

- `output/github/20261108_111240/metrics.json` — مقاييس الدورة + إحصاءات التدوير
- `output/github/20261108_111240/baseline_skill.md` / `evolved_skill.md`
- `output/crouton_benchmark_raw.json` — خام اختبار الزمن (33+6+2 استدعاء)
- `datasets/skills/github/{train,val,holdout}.jsonl` — 7 أمثلة مُعدَّنة من state.db
- سجل التشغيل الكامل حُفظ في `/tmp/crouton_gepa_run.log` أثناء الجلسة (خارج المستودع، لم يُلتزَم)

---

*أُنتج هذا التقرير من مقاييس حية مقاسة في هذه الجلسة، لا من ذاكرة أو تخمين.
كل رقم أعلاه له مصدر: `metrics.json`، أو سجل GEPA، أو اختبار الزمن المنفصل.
لا يحتوي أي ملف مُلتزَم على مفاتيح API — البصمات (`***xxxx`) فقط.*
