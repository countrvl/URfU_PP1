# Проектный практикум 1

## Структура проекта: 

- dataset -содержит файлы датасета.  
     Файлы загружаются локально. Папка внесена в .gitignore
- lessons -содержит информацию с лекций, по номерам.
- tasks -содержит файлы "fio".ipynb c кодом выполнения заданий
- Файл main.ipynb -файл для сдачи.

## Начало работы:

1. Клонируем проект. 
2. Создаем отдельную ветку(branch) разработки
    ```
    git checkout -b "fio"
    ```
3. Создаем папку dataset в корне проекта и копируем в нее файлы датасета.
4. Загружаем зависимости из файла ./requirements.txt в корне проекта.
    ```
    pip install -r requirements.txt
    ```
5. В папке **tasks** cоздаем файл "fio".ipynb.
6. Работаем в своем файле.
    - 6.1 При необходимости обновляем requirements.txt.
      ```
      pip freeze > requirements.txt
      ```
      ### Внимание!
      Не заменяйте версии существующих зависимостей, указанные с >=, на конкретные версии через ==. 
      Новые зависимости добавляйте используя >= вместо ==. 
7. Для сохранения работы выполнить набор команд:   
    ```
    git add -A
    git checkout -m "свой коментарий"     
    git push origin "название своей ветки"
    ```

**!!!Не надо пушить в main!!!**
