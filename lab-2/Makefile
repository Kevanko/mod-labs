# Лабораторная работа 2 (CPM). Сборка графов и отчёта.
# Использование: make [цель]
#   make          — полная сборка (venv + графы + отчёт)
#   make venv     — создать .venv и установить зависимости
#   make graphs   — построить изображения графов в images/
#   make report   — сгенерировать lab-2-otchet.docx
#   make run      — запустить lab-2.py (вывод в консоль)
#   make clean    — удалить артефакты сборки

PYTHON   := python3
VENV_PY  := .venv/bin/python
ifeq ($(wildcard .venv/bin/python),)
  RUNNER := $(PYTHON)
else
  RUNNER := $(VENV_PY)
endif

IMAGES   := images/graph_variant17.png images/graph_variant17.svg images/graph_custom.png images/graph_custom.svg
REPORT   := lab-2-otchet.docx

.PHONY: all venv graphs report run clean clean-venv help

all: report
	@echo "Готово: $(REPORT), $(IMAGES)"

# Виртуальное окружение и зависимости
venv:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -q -r requirements.txt
	@echo "Окружение .venv создано, зависимости установлены."

# Построение графов (нужны networkx, matplotlib)
graphs: $(IMAGES)

$(IMAGES): draw_graphs.py lab-2.py
	@mkdir -p images
	$(RUNNER) draw_graphs.py

# Генерация отчёта DOCX (нужен python-docx)
report: graphs
	$(RUNNER) build_report.py

# Запуск расчёта CPM в консоли
run:
	$(RUNNER) lab-2.py

# Удалить сгенерированные файлы
clean:
	rm -f $(IMAGES) $(REPORT)
	@echo "Удалены изображения и отчёт."

# Удалить также виртуальное окружение
clean-venv: clean
	rm -rf .venv
	@echo "Удалён .venv."

help:
	@echo "Цели:"
	@echo "  all     — полная сборка (по умолчанию)"
	@echo "  venv    — создать .venv и установить requirements.txt"
	@echo "  graphs  — построить графы (PNG 300 dpi и SVG) в images/"
	@echo "  report  — сгенерировать lab-2-otchet.docx"
	@echo "  run     — выполнить lab-2.py"
	@echo "  clean   — удалить изображения и отчёт"
	@echo "  clean-venv — clean + удалить .venv"
	@echo "  help    — этот вывод"
