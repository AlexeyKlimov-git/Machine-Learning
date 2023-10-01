# Пример инференса

После подготовки demo и обучения CNN:

```bash
python -m src.predict --checkpoint artifacts/cnn/model.pt --image data/demo/test/cat/0.png
```

Demo — цветовые картинки, не фотографии. Для полноценного результата используйте Oxford-IIIT Pet.
