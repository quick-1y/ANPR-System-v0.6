#!/usr/bin/env python3
# /app.py
import sys
import warnings
from pathlib import Path

from PyQt5 import QtWidgets

from anpr.config import Config
from anpr.infrastructure.logging_manager import LoggingManager, get_logger
from anpr.ui.main_window import MainWindow

# Silence noisy quantization warnings emitted by torch on repeated startups.
warnings.filterwarnings(
    "ignore",
    message="Please use quant_min and quant_max to specify the range for observers.",
    module="torch.ao.quantization.observer",
)
warnings.filterwarnings(
    "ignore",
    message="must run observer before calling calculate_qparams",
    module="torch.ao.quantization.observer",
)

logger = get_logger(__name__)


def main() -> None:
    """Entrypoint that wires settings, logging and the main window."""

    config = Config()
    LoggingManager(config.get_logging_config())
    logger.info("Запуск ANPR Desktop")

    app = QtWidgets.QApplication(sys.argv)
    model_paths = [
        Path("models/yolo/best.pt"),
        Path("models/ocr_crnn/crnn_ocr_model_int8_fx.pth"),
    ]
    missing_paths = [path for path in model_paths if not path.exists()]
    for path in missing_paths:
        logger.warning("Отсутствует файл модели: %s", path)
    if missing_paths:
        message = (
            "Не найдены файлы моделей:\n"
            + "\n".join(str(path) for path in missing_paths)
            + "\nПроверьте пути и перезапустите приложение."
        )
        QtWidgets.QMessageBox.critical(
            None,
            "Ошибка загрузки моделей",
            message,
        )
        sys.exit(1)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
