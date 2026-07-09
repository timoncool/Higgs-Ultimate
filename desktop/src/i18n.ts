// ─────────────────────────────────────────────────────────────────────────────
// Lightweight EN→RU localization.
//
// The upstream app has no i18n and hard-codes English throughout index.html and
// main.ts. Rather than annotate every element, we keep a dictionary keyed by the
// exact English source string and translate in two ways:
//   • translateStaticDom() walks the DOM once on load, replacing matching text
//     nodes and the placeholder / title / aria-label / data-tip attributes.
//   • t(en) translates dynamic strings created in JS (toasts, chips, statuses).
// Only exact known strings are touched, so code snippets and numbers are safe.
//
// Default language is Russian. Switching language reloads so the original
// English markup is restored cleanly.
// ─────────────────────────────────────────────────────────────────────────────

export type Lang = "ru" | "en";

export const LANG_STORAGE_KEY = "higgsAudio.lang";

export function getLang(): Lang {
  return localStorage.getItem(LANG_STORAGE_KEY) === "en" ? "en" : "ru";
}

export function setLang(lang: Lang): void {
  localStorage.setItem(LANG_STORAGE_KEY, lang);
  window.location.reload();
}

// English source string → Russian.
export const RU: Record<string, string> = {
  // ── Topbar ──
  "Neural TTS · Voice Cloning · Audio Continuation": "Нейросетевой TTS · Клонирование голоса · Продолжение аудио",
  "API server status": "Статус API-сервера",
  "API server stopped": "API-сервер остановлен",
  "Show active download": "Показать активную загрузку",
  "Settings": "Настройки",
  "Engine unloaded": "Движок не загружен",
  "Engine loaded": "Движок загружен",
  "No model": "Нет модели",

  // ── Model panel ──
  "Model": "Модель",
  "Not loaded": "Не загружено",
  "Loaded": "Загружено",
  "Loading…": "Загрузка…",
  "Error": "Ошибка",
  "Select a model…": "Выберите модель…",
  "📁 Browse…": "📁 Обзор…",
  "⬇ Download": "⬇ Скачать",
  "Load Engine": "Загрузить движок",
  "Load Model": "Загрузить модель",
  "Unload": "Выгрузить",
  "⬇ Download Engine DLLs": "⬇ Скачать DLL движка",

  // ── Whisper panel ──
  "Choose model": "Выбрать модель",
  "Whisper model": "Модель Whisper",
  "Selected file": "Выбранный файл",
  "Select ggml-base.en.bin…": "Выберите файл ggml…",

  // ── Hardware panel ──
  "Hardware": "Оборудование",
  "Poll": "Опрос",
  "Hardware monitor polling interval.": "Интервал опроса монитора оборудования.",
  "Collapse hardware": "Свернуть оборудование",
  "Power": "Питание",
  "Waiting for GPU data": "Ожидание данных GPU",

  // ── Queue ──
  "Queue Manager": "Очередь",
  "Idle": "Простаивает",
  "Clear": "Очистить",
  "Nothing generating": "Ничего не генерируется",

  // ── Mode tabs ──
  "Voice Clone": "Клон голоса",
  "Continue Speech": "Продолжить речь",
  "Multi Speaker": "Мультиспикер",
  "Speaker Gallery": "Галерея голосов",
  "History": "История",

  // ── TTS / Clone / Finish shared ──
  "Text to speak": "Текст для озвучки",
  "Type or paste text here…": "Введите или вставьте текст…",
  "Saved speaker identity (optional)": "Сохранённый голос (необязательно)",
  "Saved speaker identity": "Сохранённый голос",
  "Optional. Pick one from Speaker Gallery, or ignore this and upload a one-off reference below.":
    "Необязательно. Выберите голос из Галереи или загрузите разовый референс ниже.",
  "Reference voice": "Референс-голос",
  "Drop audio file or click to browse": "Перетащите аудио или нажмите для выбора",
  "⤒ Drop audio file here, or click to browse": "⤒ Перетащите аудио сюда или нажмите для выбора",
  "⤒ Drop reference voice here, or click to browse": "⤒ Перетащите референс-голос сюда или нажмите для выбора",
  "⤒ Drop the clip to continue, or click to browse": "⤒ Перетащите клип для продолжения или нажмите для выбора",
  "mp3 · wav · flac · m4a — auto-cropped to 30s": "mp3 · wav · flac · m4a — авто-обрезка до 30с",
  "Play reference": "Воспроизвести референс",
  "Play speaker reference": "Воспроизвести референс голоса",
  "Normalize reference": "Нормализовать референс",
  "Reference transcript (optional)": "Транскрипт референса (необязательно)",
  "Reference transcript": "Транскрипт референса",
  "✦ Auto-transcribe": "✦ Автотранскрипт",
  "What the reference audio says…": "Что говорит референс-аудио…",
  "What this reference voice says…": "Что говорит этот референс-голос…",
  "ⓘ Improves cloning fidelity. Leave blank to skip. Only clone voices you have permission to use.":
    "ⓘ Повышает точность клонирования. Можно оставить пустым. Клонируйте только голоса, на которые есть разрешение.",

  // ── Speaker Gallery ──
  "+ Speaker": "+ Голос",
  "Download the standard Nerual Dreming voice pack and add all voices to the gallery":
    "Скачать стандартный войспак Nerual Dreming и добавить все голоса в галерею",
  "Standard voices": "Стандартные голоса",
  "Import": "Импорт",
  "Export": "Экспорт",
  "Delete": "Удалить",
  "Add or replace display picture": "Добавить или заменить фото",
  "Speaker name": "Имя голоса",
  "Remove photo": "Удалить фото",
  "Drop speaker reference audio or click to browse": "Перетащите референс-аудио голоса или нажмите для выбора",
  "Source / consent notes": "Источник / заметки о согласии",
  "Where this reference came from, usage permission, character notes, etc.":
    "Откуда взят референс, разрешение на использование, заметки о персонаже и т.п.",

  // ── Continue Speech ──
  "What should be spoken after the input audio…": "Что должно прозвучать после входного аудио…",
  "ⓘ The text to generate after the input audio ends.": "ⓘ Текст, который будет сгенерирован после окончания входного аудио.",
  "Optional. Use a saved speaker as the continuation source, or upload a one-off clip below.":
    "Необязательно. Возьмите сохранённый голос как источник продолжения или загрузите разовый клип ниже.",
  "Audio to continue": "Аудио для продолжения",
  "Voice transcript": "Транскрипт голоса",
  "What the input audio says — type it or use auto-transcribe":
    "Что говорит входное аудио — впишите или используйте автотранскрипт",
  "Include whole audio": "Включить всё аудио",
  "ⓘ Transcript of the input audio. Improves continuation quality.":
    "ⓘ Транскрипт входного аудио. Улучшает качество продолжения.",

  // ── Multi Speaker ──
  "Speakers": "Спикеры",
  "Apply": "Применить",
  "Import IDs": "Импорт ID",
  "Export IDs": "Экспорт ID",
  "Speech Lines": "Реплики",
  "[Speaker1] First line\n[Speaker2] Reply line": "[Спикер1] Первая реплика\n[Спикер2] Ответная реплика",
  "Import tagged script": "Импорт размеченного сценария",
  "Clear lines": "Очистить реплики",
  "+ Line": "+ Реплика",

  // ── Advanced options ──
  "▸ Advanced options": "▸ Расширенные настройки",
  "Temperature": "Температура",
  "Default: 1.00. Controls sampling randomness. Higher is more varied; lower is more stable and repeatable.":
    "По умолчанию 1.00. Управляет случайностью сэмплирования. Больше — разнообразнее; меньше — стабильнее и воспроизводимее.",
  "Default: 50. Samples from only the top K likely tokens. 0 usually disables this filter.":
    "По умолчанию 50. Сэмплирует только из топ-K вероятных токенов. 0 обычно отключает фильтр.",
  "Default: 0.95. Nucleus sampling. Lower values keep output closer to the most likely tokens.":
    "По умолчанию 0.95. Nucleus-сэмплирование. Меньшие значения держат вывод ближе к самым вероятным токенам.",
  "Seed": "Сид",
  "Default: 0, Fixed. Controls repeatable randomness. Random changes every generation; increment and decrement step by 1 after each generation.":
    "По умолчанию 0, Фикс. Управляет воспроизводимой случайностью. «Случайно» меняется каждый раз; инкремент/декремент шагают на 1 после каждой генерации.",
  "Fixed": "Фикс.",
  "Increment": "Инкремент",
  "Decrement": "Декремент",
  "Random": "Случайно",
  "Randomize seed now": "Случайный сид сейчас",
  "Emotion": "Эмоция",
  "Default: None. Adds an optional emotion delivery tag when supported by the active Higgs backend/model.":
    "По умолчанию Нет. Добавляет тег эмоции, если поддерживается активным движком/моделью.",
  "None": "Нет",
  "Elation": "Ликование",
  "Amusement": "Веселье",
  "Enthusiasm": "Энтузиазм",
  "Determination": "Решимость",
  "Pride": "Гордость",
  "Contentment": "Удовлетворение",
  "Affection": "Нежность",
  "Relief": "Облегчение",
  "Contemplation": "Задумчивость",
  "Confusion": "Смятение",
  "Surprise": "Удивление",
  "Awe": "Трепет",
  "Longing": "Тоска",
  "Arousal": "Возбуждение",
  "Anger": "Гнев",
  "Fear": "Страх",
  "Disgust": "Отвращение",
  "Bitterness": "Горечь",
  "Sadness": "Грусть",
  "Shame": "Стыд",
  "Helplessness": "Беспомощность",
  "Style": "Стиль",
  "Default: None. Adds an optional style tag such as singing, shouting, or whispering when supported.":
    "По умолчанию Нет. Добавляет тег стиля (пение, крик, шёпот), если поддерживается.",
  "Singing": "Пение",
  "Shouting": "Крик",
  "Whispering": "Шёпот",
  "Speed": "Скорость",
  "Default: None. Adds an optional speaking-speed tag. Use only when the backend/model supports it.":
    "По умолчанию Нет. Добавляет тег скорости речи. Используйте, если движок/модель поддерживает.",
  "Very slow": "Очень медленно",
  "Slow": "Медленно",
  "Fast": "Быстро",
  "Very fast": "Очень быстро",
  "Pitch": "Высота тона",
  "Default: None. Adds an optional low/high pitch tag when supported by the backend/model.":
    "По умолчанию Нет. Добавляет тег низкой/высокой тональности, если поддерживается.",
  "Low": "Низкий",
  "High": "Высокий",
  "Expressiveness": "Выразительность",
  "Default: None. Adds an optional high/low expressiveness tag when supported.":
    "По умолчанию Нет. Добавляет тег высокой/низкой выразительности, если поддерживается.",
  "Max tokens": "Макс. токенов",
  "Default: 1024. Caps generated token count. Higher allows longer output but reserves more runtime KV/graph memory up front.":
    "По умолчанию 1024. Ограничивает число токенов. Больше — длиннее вывод, но больше памяти KV/графа резервируется заранее.",
  "Longform chunking": "Разбивка длинного текста",
  "Default: Off. Splits long text into smaller generations and joins the results. Useful for long passages.":
    "По умолчанию Выкл. Делит длинный текст на части и склеивает результаты. Полезно для длинных фрагментов.",
  "Chunk size": "Размер фрагмента",
  "Default: 45. Approximate text chunk size used only when longform chunking is enabled.":
    "По умолчанию 45. Примерный размер фрагмента текста, применяется только при включённой разбивке.",
  "Pause (sec)": "Пауза (сек)",
  "Default: 0.15 sec. Adds silence between chunks when longform chunking is enabled.":
    "По умолчанию 0.15 сек. Добавляет тишину между фрагментами при включённой разбивке.",
  "Speaker pause": "Пауза между спикерами",
  "Default: 0.15 sec. Multi Speaker only. Adds silence between generated speaker lines after they are stitched together.":
    "По умолчанию 0.15 сек. Только для мультиспикера. Добавляет тишину между репликами спикеров при склейке.",

  // ── API panel ──
  "Local API": "Локальный API",
  "Stopped": "Остановлен",
  "Running": "Работает",
  "API Info": "О API",
  "Start": "Старт",
  "Stop": "Стоп",
  "Host": "Хост",
  "Port": "Порт",
  "API key": "Ключ API",
  "Base URL": "Базовый URL",
  "Show API key": "Показать ключ API",
  "Generate key": "Сгенерировать ключ",
  "Copy base URL": "Копировать базовый URL",
  "Copy key": "Копировать ключ",
  "Copy curl": "Копировать curl",
  "Copy example": "Копировать пример",
  "Endpoints": "Эндпоинты",
  "Request Options": "Параметры запроса",
  "Command Centre": "Командный центр",
  "Pop out": "Открепить",
  "Copy": "Копировать",
  "All": "Все",
  "Info": "Инфо",
  "Warnings": "Предупреждения",
  "Errors": "Ошибки",
  "Requests": "Запросы",
  "Jobs": "Задачи",
  "Examples use the current Base URL and API key.": "Примеры используют текущий базовый URL и ключ API.",

  // ── History ──
  "Generation History": "История генераций",
  "Clear All": "Очистить всё",
  "No generations yet": "Пока нет генераций",

  // ── Generate / Output ──
  "▶ Generate": "▶ Генерировать",
  "⇉ Batch": "⇉ Пакет",
  "Batch: generate each line of text as a separate clip (Text to Speech and Voice Clone)":
    "Пакет: каждая строка текста генерируется как отдельный клип (TTS и Клон голоса)",
  "Add to Queue": "В очередь",
  "⏹ Cancel": "⏹ Отмена",
  "Output": "Результат",
  "Save audio": "Сохранить аудио",
  "Volume": "Громкость",

  // ── Settings popover ──
  "Theme": "Тема",
  "Dark": "Тёмная",
  "Light": "Светлая",
  "Accent": "Акцент",
  "Accent color": "Цвет акцента",
  "UI scale": "Масштаб интерфейса",
  "Language": "Язык",
  "Close settings": "Закрыть настройки",
  "Stream playback during generation": "Потоковое воспроизведение при генерации",
  "Requires a streaming-capable Higgs engine DLL. The current offline engine falls back to normal generation.":
    "Требуется потоковый DLL движка Higgs. Текущий офлайн-движок откатывается к обычной генерации.",
  "Minimize to tray on close": "Сворачивать в трей при закрытии",
  "Keeps the API and loaded model running in the background. Click the tray icon to restore.":
    "Оставляет API и загруженную модель работать в фоне. Клик по иконке в трее — восстановить.",
  "Quit app": "Выйти",

  // ── Download popover ──
  "Download Model": "Скачать модель",
  "Download Whisper Model": "Скачать модель Whisper",
  "Download Engine DLLs": "Скачать DLL движка",
  "Close": "Закрыть",
  "Model file": "Файл модели",
  "Paste HuggingFace GGUF URL…": "Вставьте ссылку на GGUF с HuggingFace…",
  "Paste whisper.cpp ggml .bin URL…": "Вставьте ссылку на ggml .bin whisper.cpp…",
  "Engine package URL…": "URL пакета движка…",
  "Fetch": "Скачать",
  "Pause": "Пауза",
  "Resume": "Продолжить",

  // ── Speaker export modal ──
  "Export Speakers": "Экспорт голосов",
  "Select all": "Выбрать все",
  "Select none": "Снять выбор",
  "Cancel": "Отмена",
  "Export selected": "Экспорт выбранных",

  // ── First run wizard ──
  "First Run Setup": "Первый запуск",
  "Engine, model, then generate": "Движок, модель, затем генерация",
  "Checking local engine and model files...": "Проверка локальных файлов движка и модели…",
  "Engine DLLs": "DLL движка",
  "Checking engine package...": "Проверка пакета движка…",
  "Browse": "Обзор",
  "Download": "Скачать",
  "Load": "Загрузить",
  "Higgs model folder": "Папка модели Higgs",
  "Checking downloaded model folders...": "Проверка скачанных папок моделей…",
  "Whisper auto-transcribe": "Автотранскрипт Whisper",
  "Optional. Needed only for auto reference transcripts.": "Необязательно. Нужно только для авто-транскриптов референса.",
  "Refresh": "Обновить",
  "Open Studio": "Открыть студию",

  // ── Engine diagnostics ──
  "Engine Dependency Check": "Проверка зависимостей движка",
  "Windows loader preflight": "Предпроверка загрузчика Windows",
  "Missing required DLLs": "Отсутствуют нужные DLL",
  "Detected DLLs": "Найденные DLL",
  "Search paths checked": "Проверенные пути поиска",
  "Copy diagnostics": "Копировать диагностику",
  "NVIDIA driver": "Драйвер NVIDIA",
  "VC++ runtime": "VC++ рантайм",
};

// Translate a dynamic string. Returns Russian when in RU mode and known,
// otherwise the original English (safe fallback).
export function t(en: string): string {
  if (getLang() !== "ru") return en;
  return RU[en] ?? en;
}

const ATTR_TO_TRANSLATE = ["placeholder", "title", "aria-label", "data-tip"];

// Walk the DOM once and swap known English strings for Russian.
export function translateStaticDom(root: ParentNode = document.body): void {
  if (getLang() !== "ru") return;

  // Text nodes.
  const walker = document.createTreeWalker(root as Node, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  let node = walker.nextNode();
  while (node) {
    nodes.push(node as Text);
    node = walker.nextNode();
  }
  for (const textNode of nodes) {
    const raw = textNode.nodeValue;
    if (!raw) continue;
    const key = raw.trim();
    if (!key) continue;
    const translated = RU[key];
    if (translated) textNode.nodeValue = raw.replace(key, translated);
  }

  // Attributes.
  const elements = (root as ParentNode).querySelectorAll<HTMLElement>("*");
  for (const el of elements) {
    for (const attr of ATTR_TO_TRANSLATE) {
      const value = el.getAttribute(attr);
      if (!value) continue;
      const translated = RU[value.trim()];
      if (translated) el.setAttribute(attr, translated);
    }
  }
}
