# Higgs Ultimate — русская портативная версия (движок Higgs Audio v3)

[![Stars](https://img.shields.io/github/stars/timoncool/Higgs-Audio-v3-Studio-rus?style=social)](https://github.com/timoncool/Higgs-Audio-v3-Studio-rus/stargazers)
[![License](https://img.shields.io/github/license/timoncool/Higgs-Audio-v3-Studio-rus)](LICENSE)

Мощный нейросетевой синтез речи (TTS), клонирование голоса и продолжение аудио на движке **Higgs Audio v3** — в полностью **портативной** сборке с **русским интерфейсом**.

Это форк отличного проекта [**Saganaki22/Higgs-Audio-v3-Studio**](https://github.com/Saganaki22/Higgs-Audio-v3-Studio) (нативный C++/CUDA-движок `audiocpp`, обёртка на Tauri). Огромная благодарность автору за движок и приложение 🙏 — вся тяжёлая работа по портированию Higgs Audio на быстрый нативный движок сделана им.

> Оригинальный README на английском: [README_EN.md](README_EN.md).

## Что добавлено в этой версии

- 🇷🇺 **Русский интерфейс** (переключатель RU/EN в настройках, по умолчанию русский).
- 📦 **Настоящая портативность** — модели, голоса, движок, настройки хранятся **рядом с `.exe`**, а не в профиле пользователя (`C:\Users\...`). Перенёс папку — перенёс всё; удалил папку — не осталось следов в системе.
- 🔊 **Регулятор громкости** на плеере результата.
- ⇉ **Пакетный режим** — каждая строка текста генерируется как отдельный клип (для TTS и клонирования голоса), все результаты попадают в Историю.
- 🎙️ **Стандартный войспак** — кнопка «Стандартные голоса» в Галерее: одним нажатием скачивает и добавляет ~50 готовых референс-голосов.
- 🩹 **Исправлена докачка моделей** — при вставке произвольной ссылки на GGUF теперь скачивается **вся папка модели** (веса + `config.json` + токенайзеры), а не только `.gguf`. Больше нет ошибки «missing config». Неполные папки помечаются «⚠ incomplete».
- 🌍 **Мультиязычный Whisper** по умолчанию (для авто-транскрипта клонов на русском).

## Возможности (из оригинала)

- Синтез речи (TTS), клонирование голоса из референса, продолжение речи, мультиспикер.
- Галерея голосов: сохранение личностей с фото, транскриптом, экспорт/импорт.
- Тонкая настройка: температура, top-k, top-p, seed, эмоция, стиль, скорость, тон, выразительность.
- Разбивка длинного текста, стриминг воспроизведения, экспорт WAV/MP3.
- Локальный HTTP API, монитор GPU (VRAM/загрузка/питание/RAM), тёмная/светлая тема.

## Установка

1. Скачайте архив из [**Releases**](https://github.com/timoncool/Higgs-Audio-v3-Studio-rus/releases) и распакуйте в любую папку (например, `D:\Higgs-Audio-Studio`).
2. Запустите `higgs-audio-studio.exe`.
3. При первом запуске мастер поможет скачать **движок** (DLL) и **модель** — всё сохранится рядом с `.exe`.

> Никакой установки в систему: Python/CUDA-тулкит не нужны, движок скачивается готовым.

## Системные требования

- **ОС:** Windows 10/11 x64, установленный [WebView2](https://developer.microsoft.com/microsoft-edge/webview2/) (обычно уже есть).
- **GPU:** NVIDIA с CUDA 13.x. Требования по VRAM зависят от кванта модели:

| Квант | VRAM |
|-------|------|
| Q4_K_M | ~8 ГБ |
| Q5_K | ~9 ГБ |
| Q6_K | ~10 ГБ |
| **Q8_0 (рекомендуется)** | ~12 ГБ |
| BF16 | ~16 ГБ |

- Для работы движка может понадобиться [драйвер NVIDIA](https://www.nvidia.com/Download/index.aspx), [CUDA 13.x](https://developer.nvidia.com/cuda-downloads) и [VC++ Runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe) — приложение подскажет, если чего-то не хватает.

## Благодарность автору

Оригинальный проект, нативный движок `audiocpp` и приложение: [**Saganaki22/Higgs-Audio-v3-Studio**](https://github.com/Saganaki22/Higgs-Audio-v3-Studio) (лицензия Apache 2.0). Веса модели Higgs Audio — от [Boson AI](https://huggingface.co/bosonai) (исследовательская/некоммерческая лицензия). Спасибо за огромную работу!

## Другие портативные нейросети

| Проект | Описание |
|--------|----------|
| [HiggsAudio-Studio](https://github.com/timoncool/HiggsAudio-Studio) | Higgs Audio TTS + LLM-режиссёр (Python-версия) |
| [Foundation Music Lab](https://github.com/timoncool/Foundation-Music-Lab) | Генерация музыки + таймлайн-редактор |
| [VibeVoice ASR](https://github.com/timoncool/VibeVoice_ASR_portable_ru) | Распознавание речи (ASR) |
| [LavaSR](https://github.com/timoncool/LavaSR_portable_ru) | Улучшение качества аудио |
| [Qwen3-TTS](https://github.com/timoncool/Qwen3-TTS_portable_rus) | Синтез речи (TTS) от Qwen |
| [SuperCaption Qwen3-VL](https://github.com/timoncool/SuperCaption_Qwen3-VL) | Генерация описаний изображений |
| [VideoSOS](https://github.com/timoncool/videosos) | AI-видеопродакшн в браузере |
| [RC Stable Audio Tools](https://github.com/timoncool/RC-stable-audio-tools-portable) | Генерация музыки и аудио |

## Авторы

- **Nerual Dreming** ([t.me/nerual_dreming](https://t.me/nerual_dreming)) — [neuro-cartel.com](https://neuro-cartel.com) | основатель [ArtGeneration.me](https://artgeneration.me)
- **Нейро-Софт** ([t.me/neuroport](https://t.me/neuroport)) — репаки и портативки нейросетей
- Оригинальный движок и приложение — **Saganaki22** ([Higgs-Audio-v3-Studio](https://github.com/Saganaki22/Higgs-Audio-v3-Studio))

---

> **Если проект полезен — поставьте звёздочку!** Это помогает другим находить проект и мотивирует на развитие.

## Star History

<a href="https://www.star-history.com/?repos=timoncool%2FHiggs-Audio-v3-Studio-rus&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=timoncool/Higgs-Audio-v3-Studio-rus&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=timoncool/Higgs-Audio-v3-Studio-rus&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=timoncool/Higgs-Audio-v3-Studio-rus&type=date&legend=top-left" />
 </picture>
</a>
