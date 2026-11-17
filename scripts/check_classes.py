import tensorflow as tf
from pathlib import Path

train_dir = Path("dataset_split/train")
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print("\n" + "="*50)
print("INDEKS ASLI DARI GENERATOR TRAINING:")
print("="*50)
for k, v in sorted(train_generator.class_indices.items(), key=lambda item: item[1]):
    print(f"Indeks {v} : '{k}' (Rp {int(k):,})")
print("="*50)
