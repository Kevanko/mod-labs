#!/bin/bash
# Генерация отчёта лабы 2 на Linux (Ubuntu): картинки графов + DOCX.
# Запуск: ./generate_report.sh  или  bash generate_report.sh

set -e
cd "$(dirname "$0")"

echo "=== Лаба 2: генерация отчёта ==="

# Предпочтительно python3
PYTHON=""
for p in python3 python; do
    if command -v "$p" >/dev/null 2>&1; then
        PYTHON="$p"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Ошибка: не найден python3 или python."
    exit 1
fi

# Если есть venv — используем его (там уже могут быть зависимости)
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    echo "Используется виртуальное окружение .venv"
fi

# 1. Рисуем графы (нужны networkx, matplotlib)
echo ""
echo "1. Построение графов (images/)..."
if $PYTHON -c "import networkx, matplotlib" 2>/dev/null; then
    $PYTHON draw_graphs.py
else
    echo "   Установите зависимости: $PYTHON -m pip install networkx matplotlib"
    exit 1
fi

# 2. Собираем DOCX (нужен python-docx)
echo ""
echo "2. Генерация отчёта (lab-2-otchet.docx)..."
if $PYTHON -c "import docx" 2>/dev/null; then
    $PYTHON build_report.py
else
    echo "   Установите: $PYTHON -m pip install python-docx"
    exit 1
fi

echo ""
echo "Готово. Отчёт: $(pwd)/lab-2-otchet.docx"
