import tensorflow_datasets as tfds

# Will download (~300 MB) and prepare locally in ~/tensorflow_datasets
ds_train = tfds.load("visual_wake_words", split="train", as_supervised=True)
ds_val   = tfds.load("visual_wake_words", split="val", as_supervised=True)

# To check class distribution:
print(tfds.builder("visual_wake_words").info)
