# Лабораторная 4: выборки с возвращением, без возвращения, размещение m=n
# make       — запуск lab-4.py (12 таблиц)
# make out   — то же с сохранением вывода в output.txt

.PHONY: all out clean

all: lab-4.py
	python3 lab-4.py

out: lab-4.py
	python3 lab-4.py > output.txt
	@echo "Вывод записан в output.txt"

clean:
	rm -f output.txt
