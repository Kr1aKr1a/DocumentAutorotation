import cv2
import math
import numpy as np

class ImageAutoRotator:
    def __init__(self):
        self.image_path = None
        self.median_angle = None
        self.rotation_matrix = None
        self.rotated_image = None

    def autorotate(self, image_path: str, output_path: str | None = None) -> cv2.UMat:
        """
        Автоматически поворачивает изображение документа, выравнивая его по горизонтали.

        Функция определяет угол наклона документа на изображении с помощью преобразования Хафа,
        поворачивает изображение на вычисленный угол, масштаблирует и обрезает лишнее. 
        Результат сохраняется по указанному пути или возвращается как объект изображения.

        :param image_path: Путь к исходному изображению.
        :param output_path: Путь для сохранения результата. Если None, результат не сохраняется.

        :return: Повернутое и выровненное изображение в формате cv2.UMat.

        Пример использования:
            >>> processor = ImageAutoRotator()
            >>> rotated_image = processor.autorotate("document.jpg", "rotated_document.jpg")
        """
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)

        if self.image is None:
            raise ValueError("Изображение не найдено или не может быть загружено.")

        canny_image = cv2.Canny(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), 50, 150, apertureSize=3)
        segments = cv2.HoughLinesP(canny_image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        angles = []
        for segment in segments:
            x1, y1, x2, y2 = segment[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            if -60 <= angle <= 60:
                angles.append(angle)
        
        self.median_angle = np.median(angles) if angles else 0

        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)

        angle_rad = math.radians(self.median_angle)
        new_h = height * abs(math.cos(angle_rad)) + width * abs(math.sin(angle_rad))
        new_w = width * abs(math.cos(angle_rad)) - height * abs(math.sin(angle_rad))

        scale_value = max(new_h / height, new_w / width)

        self.rotation_matrix = cv2.getRotationMatrix2D(center, self.median_angle, scale_value)
        self.rotated_image = cv2.warpAffine(
            self.image, 
            self.rotation_matrix, 
            (width, height), 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )

        if output_path is not None:
            cv2.imwrite(output_path, self.rotated_image)

        return self.rotated_image

    def get_angle(self) -> float | None:
        return -self.median_angle

if __name__ == "__main__":
    image_path = 'files/example_rotated.jpg'
    output_path = 'result/result.jpg'

    rotator = ImageAutoRotator()
    rotated_image = rotator.autorotate(image_path, output_path)

    print(f"Угол поворота картинки из примера: {rotator.get_angle()}")
    print(f"Результат автокоррекции угла наклона сохранён в {output_path}")

    angles = [-55, -30, -15, -5, 0, 5, 15, 30, 55]
    tests = ["TCGAN_page-0001.jpg", "TCGAN_page-0002.jpg", "TCGAN_page-0010.jpg", "TCGAN_page-0014.jpg", "example.jpg"]
    
    abs_error_sum = 0
    for image_name in tests:
        for angle in angles:
            rotator.autorotate(f'tests/{image_name[:-4]}_rotated_{angle}.jpg')
            abs_error_sum += abs(rotator.get_angle() - angle)

    print(f"MAE на {len(tests)} тестовых изображениях (в градусах): {abs_error_sum / (len(angles) * len(tests))}")
