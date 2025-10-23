# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# اسم أو مسار النموذج
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# --- تحميل النموذج (الكود الذي قدمته) ---
# يتم تحديد dtype=torch.bfloat16 للاستفادة من وحدات المعالجة الحديثة (GPUs)
# ويتم تفعيل Flash Attention 2 عبر attn_implementation="flash_attention_2"
# هذا يطلب من Transformers استخدام التنفيذ المحسن للانتباه إذا كان متاحًا ومدعومًا.
print(f"بدء تحميل النموذج: {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", # يوزع النموذج تلقائيًا على الـ GPU المتاح
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
print("تم تحميل النموذج بنجاح.")


# --- الخطوة 1: تحميل الـ Tokenizer ---
# يجب أن يكون الـ Tokenizer متوافقًا مع النموذج لترميز النص بشكل صحيح.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# بعض النماذج مثل Llama 3 لا تضبط رمز الحشو (padding token) افتراضيًا.
# من الجيد ضبطه، خاصة عند التعامل مع دفعات (batches).
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- الخطوة 2: إعداد المُدخل (Prompt) باستخدام قالب المحادثة ---
# نماذج "Instruct" تتوقع تنسيقًا محددًا للمدخلات.
# استخدام `apply_chat_template` يضمن أن النص يُنسق بالطريقة التي تدرب عليها النموذج.
messages = [
    {"role": "user", "content": "ما هي عاصمة الإمارات العربية المتحدة؟ وما هي أشهر معالمها السياحية؟"},
]

# تطبيق القالب وتحويل النص إلى أرقام (tokens) يفهمها النموذج
# ونقلها إلى نفس الجهاز الموجود عليه النموذج (GPU).
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)


# --- الخطوة 3: توليد الإجابة ---
print("\n...يتم الآن توليد الإجابة...\n")
# استدعاء دالة `generate` لتنفيذ عملية الاستدلال.
outputs = model.generate(
    input_ids,
    max_new_tokens=256,       # الحد الأقصى لعدد التوكينات الجديدة التي سيتم توليدها
    do_sample=True,           # تفعيل أخذ العينات للحصول على إجابات أكثر تنوعًا
    temperature=0.6,          # درجة الحرارة للتحكم في عشوائية الإجابة (الأقل أكثر تحديدًا)
    top_p=0.9,                # تقنية "nucleus sampling" لاختيار الكلمات التالية
)


# --- الخطوة 4: فك تشفير النتيجة ---
# المخرجات `outputs` تحتوي على المُدخل الأصلي + النص المُولَّد.
# لذا، نقوم بقص الجزء الخاص بالمُدخل للحصول على الإجابة الجديدة فقط.
response_ids = outputs[0][input_ids.shape[-1]:]
decoded_response = tokenizer.decode(response_ids, skip_special_tokens=True)


# --- الخطوة 5: طباعة النتيجة النهائية ---
print("--- السؤال ---")
print(messages[0]['content'])
print("\n--- إجابة النموذج ---")
print(decoded_response)
