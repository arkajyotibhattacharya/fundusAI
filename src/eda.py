"""
Exploratory Data Analysis — plots from the original starter notebook.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from .config import TARGET_COL


def plot_sample_images(df):
    """Display one image per unique disease label."""
    unique_labels = df[TARGET_COL].unique()

    fig, ax = plt.subplots(1, len(unique_labels), figsize=(20, 5))

    for idx, label in enumerate(unique_labels):
        image_row = df[df[TARGET_COL] == label].iloc[0]
        img = plt.imread(image_row['paths'])
        ax[idx].imshow(img)
        ax[idx].set_title(f"Label: {label}")

    plt.tight_layout()
    plt.show()


def plot_label_distribution(df):
    """Show value counts of disease labels."""
    print(df[TARGET_COL].value_counts())


def plot_age_distribution(df):
    """Age histogram + diagnostic keyword counts."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Patient Age'], bins=30, kde=True)
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    if 'Left-Diagnostic Keywords' in df.columns:
        print(df['Left-Diagnostic Keywords'].value_counts())


def show_resize_example(test_data, test_generator, img_size, rescale, idx=4):
    """Show original vs resized/rescaled image side by side."""
    y = TARGET_COL
    sample_path = test_data.iloc[idx]['paths']
    sample_label = test_data.iloc[idx][y]

    batch_size = test_generator.batch_size
    batch_index = idx // batch_size
    sample_index_in_batch = idx % batch_size

    images, labels = test_generator[batch_index]
    resized_image = images[sample_index_in_batch]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(sample_path))
    plt.title(f"Original Image\nLabel: {sample_label}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(resized_image)
    plt.title(f"Resized {img_size}\n Rescaled {rescale}")
    plt.axis('off')

    plt.show()
