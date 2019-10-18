import os
import shutil

_TRAIN_FOLDER_ = "train/"  # Should be exists with all categorized images.
_VALIDATION_FOLDER_ = "validation/"
_TEST_FOLDER_ = "test/"

split_ratio = 0.15  # Images % from train to the target folder


def create_dataset(target_folder=None):
    train_folder = _TRAIN_FOLDER_
    create_folder(target_folder)
    print("loading...")
    for fname in os.listdir(train_folder):
        if fname != target_folder:
            images_list = os.listdir(train_folder + fname)
            folder_images_amount = len(images_list) * split_ratio
            create_folder(target_folder + fname)
            for imageName in images_list:
                if folder_images_amount > 0:
                    src_file_path = train_folder + fname + "/" + imageName
                    dst_file_path = target_folder + fname

                    shutil.move(src_file_path, dst_file_path)
                    folder_images_amount -= 1
                else:
                    break

    print("done")


def create_folder(folder=None):
    if folder is None:
        return

    if not os.path.exists(folder):
        os.makedirs(folder)


# Split train dataset images to validation and test.
if __name__ == '__main__':
    create_dataset(_VALIDATION_FOLDER_)
    create_dataset(_TEST_FOLDER_)
