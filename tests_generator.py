import cv2
import math
import numpy as np
from pathlib import Path

class ImageRotator:
    def __init__(self, image_path: str, image_name: str):
        self.image_path = image_path
        self.image_name = image_name
        self.image = cv2.imread(image_path + '/' + image_name)

        if self.image is None:
            raise ValueError("Изображение не найдено или не может быть загружено.")

        self.height, self.width, _ = self.image.shape

    def generate_rotated_images(self, angles: list[float], output_dir: str) -> list[tuple[str, float], ...]:
        """
        Генерирует повернутые изображения и сохраняет их в указанной директории.

        :param angles: Список углов поворота в градусах.
        :param output_dir: Директория для сохранения повернутых изображений.

        :return: Список кортежей вида (output_path, angle)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        rotation_info = []

        for angle in angles:
            angle_rad = math.radians(angle)

            new_h = self.height * abs(math.cos(angle_rad)) + self.width * abs(math.sin(angle_rad))
            new_w = self.width * abs(math.cos(angle_rad)) - self.height * abs(math.sin(angle_rad))
            scale_value = max(new_h / self.height, new_w / self.width)

            center = (self.width // 2, self.height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1/scale_value)

            rotated_image = cv2.warpAffine(self.image, rotation_matrix, (self.width, self.height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            output_path = output_dir + f"/{self.image_name[:-4]}_rotated_{angle}.jpg"
            cv2.imwrite(output_path, rotated_image)

            rotation_info.append((output_path, angle))

        return rotation_info

if __name__ == "__main__":
    image_path = "files"
    image_names = ["TCGAN_page-0001.jpg", "TCGAN_page-0002.jpg", "TCGAN_page-0010.jpg", "TCGAN_page-0014.jpg", "example.jpg"]

    output_dir = "tests"

    angles = [-55, -30, -15, -5, 0, 5, 15, 30, 55]

    print("Повернутые изображения сохранены в:", output_dir)
    for image_name in image_names:
        generator = ImageRotator(image_path, image_name)
        rotation_info = generator.generate_rotated_images(angles, output_dir)

        for path, angle in rotation_info:
            print(f"{path}: {angle} градусов")