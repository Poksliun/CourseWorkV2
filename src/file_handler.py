import os
import shutil
import logging

from PIL import Image


class FileHandler:
    dirs = {
        "zip_data_dir": {
            "train_data_zip": "resources/train_zip",
            "test_data_zip": "resources/test_zip"
        },
        "unzip_data_dirs": {
            "raw_data_dir": {
                "training_dir": "resources/raw_data/train_data",
                "test_dir": "resources/raw_data/test_data",
            },
            "proc_data_dir": {
                "training_dir": "resources/proc_data/train_data",
                "test_dir": "resources/proc_data/test_data"
            }
        }
    }
    data_classes = ("true", "false")

    @classmethod
    def __make_not_marker_dir(cls) -> None:
        """

        """
        for directories in cls.dirs["unzip_data_dirs"].values():
            for directory in directories.values():
                if os.path.exists(directory):
                    shutil.rmtree(directory)
                os.mkdir(directory)

    @classmethod
    def __unzip_data(cls, zip_dir: str, output_data_dir: str) -> None:
        """
        Разархивация необработанных изображений
        """
        for root, dirs, archives in os.walk(zip_dir):
            for archive in archives:
                shutil.unpack_archive(f"{zip_dir}/{archive}", output_data_dir)

    @classmethod
    def __check_empty_signature(cls) -> bool:
        """
        Проверяет есть ли подпись в месте для подписи
        :return: bool.
        """
        return None

    @classmethod
    def __cropping_image(cls, input_data_dir: str, output_data_dir: str) -> None:
        """
        Кроппинг и перенумерация изображений
        """
        true_counter = 0
        false_counter = 0
        for image in list(os.walk(input_data_dir))[0][2]:
            img = Image.open(f"{input_data_dir}/{image}")
            img_crop = img.crop((830, 1465, 1435, 1570)).convert('L')
            if "true" in image.lower():
                img_crop.save(f"{output_data_dir}/true-{true_counter}.jpg")
                true_counter += 1
            elif "false" in image.lower():
                img_crop.save(f"{output_data_dir}/false-{false_counter}.jpg")
                false_counter += 1

    @staticmethod
    def __create_data_directory(dir_name, create_dir: bool = True) -> None:
        """
        Создание директории для хранения обработанных изображений с
        разделением на true и false изображения подписей
        :param dir_name: Имя директории
        """
        if create_dir:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            os.makedirs(os.path.join(dir_name, "true"))
            os.makedirs(os.path.join(dir_name, "false"))

    @classmethod
    def __data_distribution(cls, start_index: int, end_index: int, input_dir: str, output_dir: str) -> None:
        """
        Распределение изображений в true и false директории
        :param start_index: Номер первого изображения (file_name-start_index.jpg)
        :param end_index: Номер последнего изображения минус один (file_name-start_index.jpg)
        :param output_dir: Имя директории
        """
        not_found_files = []
        for data_class in cls.data_classes:
            for i in range(start_index, end_index):
                try:
                    shutil.copy2(
                        os.path.join(input_dir, f"{data_class}-{i}.jpg"),
                        os.path.join(output_dir, data_class)
                    )
                except FileNotFoundError:
                    not_found_files.append(f"{data_class}-{i}.jpg")
                    continue
        if not_found_files:
            logging.warning(f"Not found files: {not_found_files}")

    @staticmethod
    def _dir_info(dir_name: str) -> None:
        """
        Выводит информацию о количестве и соотношении изображений внутри директорий (true, false)
        """
        return None

    @staticmethod
    def counting_files_in_dirs(*args: str) -> int:
        """
        Считает сумму файлов
        :param args:
        :return: int. Сумма файлов второго уровня вложенности
        """
        counter: int = 0
        for directory in list(args):
            for folder in list(os.walk(directory))[0][1]:
                counter += len(list(os.walk(f"{directory}/{folder}"))[0][2])
        return counter

    @staticmethod
    def get_image_size(image: str) -> tuple:
        img = Image.open(image)
        return img.size

    def train_data_preparation(self):
        self.__make_not_marker_dir()
        self.__unzip_data(
            zip_dir=self.dirs["zip_data_dir"]["train_data_zip"],
            output_data_dir=self.dirs["unzip_data_dirs"]["raw_data_dir"]["training_dir"]
        )
        self.__cropping_image(
            input_data_dir=self.dirs["unzip_data_dirs"]["raw_data_dir"]["training_dir"],
            output_data_dir=self.dirs["unzip_data_dirs"]["proc_data_dir"]["training_dir"]
        )

    def data_processing(self, dir_name: str, start_index: int, end_index: int):
        self.__create_data_directory(dir_name=dir_name)
        self.__data_distribution(
            start_index=start_index,
            end_index=end_index,
            input_dir=self.dirs["unzip_data_dirs"]["proc_data_dir"]["training_dir"],
            output_dir=dir_name
        )

    def adding_test_data(self, dir_name: str, count_files: int, create_new_dir: bool = True) -> None:
        """
        Позволяет добавить данные в тестовую выборку
        :param create_new_dir:
        :param count_files:
        :param dir_name:
        """
        self.__unzip_data(
            zip_dir=self.dirs["zip_data_dir"]["test_data_zip"],
            output_data_dir=self.dirs["unzip_data_dirs"]["raw_data_dir"]["test_dir"]
        )
        self.__cropping_image(
            input_data_dir=self.dirs["unzip_data_dirs"]["raw_data_dir"]["test_dir"],
            output_data_dir=self.dirs["unzip_data_dirs"]["proc_data_dir"]["test_dir"]
        )
        self.__create_data_directory(dir_name=dir_name, create_dir=create_new_dir)
        self.__data_distribution(
            start_index=0,
            end_index=count_files,
            input_dir=self.dirs["unzip_data_dirs"]["proc_data_dir"]["test_dir"],
            output_dir=dir_name
        )


handler = FileHandler()
