"""
Data splitting and generator creation — matches original starter notebook.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from .config import IMG_SIZE, BATCH_SIZE, SEED, TEST_SIZE, VAL_SIZE, TARGET_COL


def split_data(df):
    """Stratified train/val/test split."""
    train_data, test_data = train_test_split(
        df,
        stratify=df[TARGET_COL],
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    train_data, val_data = train_test_split(
        train_data,
        stratify=train_data[TARGET_COL],
        test_size=VAL_SIZE,
        random_state=SEED,
    )

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data


def create_generators(train_data, val_data, test_data):
    """Create ImageDataGenerators with rescaling (no augmentation in baseline)."""
    rescale = 1./255
    datagen = ImageDataGenerator(rescale=rescale)

    train_generator = datagen.flow_from_dataframe(
        train_data,
        x_col='paths', y_col=TARGET_COL,
        shuffle=True,
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
    )
    val_generator = datagen.flow_from_dataframe(
        val_data,
        x_col='paths', y_col=TARGET_COL,
        shuffle=False,
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
    )
    test_generator = datagen.flow_from_dataframe(
        test_data,
        x_col='paths', y_col=TARGET_COL,
        shuffle=False,
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
    )

    return train_generator, val_generator, test_generator
