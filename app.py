
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from PyQt6.QtCore import (
    Qt,
    QSize,
    QPropertyAnimation,
    QEasingCurve
)

from PyQt6.QtGui import (
    QPixmap,
    QIcon,
    QFont
)

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QFrame,
    QProgressBar,
    QGraphicsDropShadowEffect
)

from model import CancerModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/best_model.pt"
IMAGE_SIZE = 96


class NovaApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.dark_mode = True
        self.current_image = None

        self.setWindowTitle(
            "Nova • Histopathology"
        )

        self.setMinimumSize(
            1050,
            700
        )

        self.resize(
            1200,
            760
        )

        self.load_model()

        self.build_ui()

        self.apply_theme()

    def load_model(self):
        self.model = CancerModel(
            pretrained=False
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        self.root_layout = QVBoxLayout(
            central
        )

        self.root_layout.setContentsMargins(
            32,
            28,
            32,
            22
        )

        self.root_layout.setSpacing(20)

        self.build_header()

        self.build_content()

        self.build_footer()

    def build_header(self):
        header = QHBoxLayout()

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)

        self.title = QLabel(
            "Nova"
        )

        self.title.setFont(
            QFont(
                "Segoe UI",
                26,
                QFont.Weight.Bold
            )
        )

        self.subtitle = QLabel(
            "Histopathology Analysis"
        )

        self.subtitle.setFont(
            QFont(
                "Segoe UI",
                11
            )
        )

        title_layout.addWidget(
            self.title
        )

        title_layout.addWidget(
            self.subtitle
        )

        header.addLayout(
            title_layout
        )

        header.addStretch()

        self.theme_button = QPushButton(
            "☀"
        )

        self.theme_button.setFixedSize(
            48,
            42
        )

        self.theme_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.theme_button.clicked.connect(
            self.toggle_theme
        )

        header.addWidget(
            self.theme_button
        )

        self.root_layout.addLayout(
            header
        )

    def build_content(self):
        content = QHBoxLayout()

        content.setSpacing(18)

        self.image_card = self.create_card()

        image_layout = QVBoxLayout(
            self.image_card
        )

        image_layout.setContentsMargins(
            22,
            22,
            22,
            22
        )

        image_layout.setSpacing(16)

        self.image_title = QLabel(
            "Image"
        )

        self.image_title.setFont(
            QFont(
                "Segoe UI",
                17,
                QFont.Weight.Bold
            )
        )

        image_layout.addWidget(
            self.image_title
        )

        self.preview = QLabel()

        self.preview.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.preview.setMinimumSize(
            550,
            470
        )

        self.preview.setText(
            "⌁\n\n"
            "Choose a histopathology image\n\n"
            "TIFF • PNG • JPG"
        )

        self.preview.setFont(
            QFont(
                "Segoe UI",
                12
            )
        )

        image_layout.addWidget(
            self.preview,
            1
        )

        self.choose_button = QPushButton(
            "＋  Choose Image"
        )

        self.choose_button.setFixedHeight(
            48
        )

        self.choose_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.choose_button.clicked.connect(
            self.open_image
        )

        image_layout.addWidget(
            self.choose_button
        )

        self.result_card = self.create_card()

        self.result_card.setFixedWidth(
            340
        )

        result_layout = QVBoxLayout(
            self.result_card
        )

        result_layout.setContentsMargins(
            28,
            28,
            28,
            28
        )

        result_layout.setSpacing(8)

        self.result_title = QLabel(
            "Analysis"
        )

        self.result_title.setFont(
            QFont(
                "Segoe UI",
                19,
                QFont.Weight.Bold
            )
        )

        self.result_subtitle = QLabel(
            "Model prediction"
        )

        result_layout.addWidget(
            self.result_title
        )

        result_layout.addWidget(
            self.result_subtitle
        )

        result_layout.addStretch()

        self.status_icon = QLabel(
            "◉"
        )

        self.status_icon.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.status_icon.setFont(
            QFont(
                "Segoe UI",
                46
            )
        )

        result_layout.addWidget(
            self.status_icon
        )

        self.result_label = QLabel(
            "Waiting"
        )

        self.result_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.result_label.setFont(
            QFont(
                "Segoe UI",
                23,
                QFont.Weight.Bold
            )
        )

        result_layout.addWidget(
            self.result_label
        )

        self.confidence = QLabel(
            "Choose an image to begin"
        )

        self.confidence.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.confidence.setWordWrap(
            True
        )

        result_layout.addWidget(
            self.confidence
        )

        result_layout.addSpacing(16)

        self.progress = QProgressBar()

        self.progress.setRange(
            0,
            100
        )

        self.progress.setValue(
            0
        )

        self.progress.setTextVisible(
            False
        )

        self.progress.setFixedHeight(
            8
        )

        result_layout.addWidget(
            self.progress
        )

        result_layout.addStretch()

        self.model_info = QLabel(
            "ResNet18\n"
            "Binary classification\n"
            "CUDA: "
            + (
                "Available"
                if torch.cuda.is_available()
                else "CPU"
            )
        )

        self.model_info.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        result_layout.addWidget(
            self.model_info
        )

        content.addWidget(
            self.image_card,
            1
        )

        content.addWidget(
            self.result_card
        )

        self.root_layout.addLayout(
            content,
            1
        )

    def build_footer(self):
        self.footer = QLabel(
            "Research / educational use • "
            "Not a medical diagnosis"
        )

        self.footer.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.root_layout.addWidget(
            self.footer
        )

    def create_card(self):
        card = QFrame()

        card.setObjectName(
            "Card"
        )

        shadow = QGraphicsDropShadowEffect()

        shadow.setBlurRadius(
            30
        )

        shadow.setOffset(
            0,
            8
        )

        card.setGraphicsEffect(
            shadow
        )

        return card

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow {
                    background: #0B0B0D;
                }

                QWidget {
                    color: #F5F5F7;
                    font-family: "Segoe UI";
                }

                QFrame#Card {
                    background: #17171A;
                    border: 1px solid #29292D;
                    border-radius: 22px;
                }

                QLabel {
                    background: transparent;
                }

                QPushButton {
                    background: #29292D;
                    color: #F5F5F7;
                    border: none;
                    border-radius: 14px;
                    padding: 8px 18px;
                }

                QPushButton:hover {
                    background: #36363B;
                }

                QPushButton:pressed {
                    background: #444449;
                }

                QProgressBar {
                    background: #29292D;
                    border: none;
                    border-radius: 4px;
                }

                QProgressBar::chunk {
                    background: #0A84FF;
                    border-radius: 4px;
                }
            """)

            self.title.setStyleSheet(
                "color: #F5F5F7;"
            )

            self.subtitle.setStyleSheet(
                "color: #A1A1A6;"
            )

            self.image_title.setStyleSheet(
                "color: #F5F5F7;"
            )

            self.preview.setStyleSheet(
                """
                color: #A1A1A6;
                background: #111113;
                border-radius: 18px;
                """
            )

            self.result_title.setStyleSheet(
                "color: #F5F5F7;"
            )

            self.result_subtitle.setStyleSheet(
                "color: #A1A1A6;"
            )

            self.status_icon.setStyleSheet(
                "color: #0A84FF;"
            )

            self.result_label.setStyleSheet(
                "color: #F5F5F7;"
            )

            self.confidence.setStyleSheet(
                "color: #A1A1A6;"
            )

            self.model_info.setStyleSheet(
                "color: #7C7C82;"
            )

            self.footer.setStyleSheet(
                "color: #6E6E73;"
            )

            self.theme_button.setText(
                "☀"
            )

        else:
            self.setStyleSheet("""
                QMainWindow {
                    background: #F5F5F7;
                }

                QWidget {
                    color: #1D1D1F;
                    font-family: "Segoe UI";
                }

                QFrame#Card {
                    background: #FFFFFF;
                    border: 1px solid #D2D2D7;
                    border-radius: 22px;
                }

                QLabel {
                    background: transparent;
                }

                QPushButton {
                    background: #E5E5EA;
                    color: #1D1D1F;
                    border: none;
                    border-radius: 14px;
                    padding: 8px 18px;
                }

                QPushButton:hover {
                    background: #D8D8DD;
                }

                QPushButton:pressed {
                    background: #C8C8CD;
                }

                QProgressBar {
                    background: #E5E5EA;
                    border: none;
                    border-radius: 4px;
                }

                QProgressBar::chunk {
                    background: #007AFF;
                    border-radius: 4px;
                }
            """)

            self.title.setStyleSheet(
                "color: #1D1D1F;"
            )

            self.subtitle.setStyleSheet(
                "color: #6E6E73;"
            )

            self.image_title.setStyleSheet(
                "color: #1D1D1F;"
            )

            self.preview.setStyleSheet(
                """
                color: #6E6E73;
                background: #F2F2F7;
                border-radius: 18px;
                """
            )

            self.result_title.setStyleSheet(
                "color: #1D1D1F;"
            )

            self.result_subtitle.setStyleSheet(
                "color: #6E6E73;"
            )

            self.status_icon.setStyleSheet(
                "color: #007AFF;"
            )

            self.result_label.setStyleSheet(
                "color: #1D1D1F;"
            )

            self.confidence.setStyleSheet(
                "color: #6E6E73;"
            )

            self.model_info.setStyleSheet(
                "color: #8E8E93;"
            )

            self.footer.setStyleSheet(
                "color: #8E8E93;"
            )

            self.theme_button.setText(
                "☾"
            )

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode

        self.apply_theme()

        if self.current_image:
            self.show_image(
                self.current_image
            )

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Histopathology Image",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg)"
        )

        if not path:
            return

        try:
            image = Image.open(
                Path(path)
            ).convert("RGB")

            self.current_image = image

            self.show_image(
                image
            )

            self.predict(
                image
            )

        except Exception as error:
            self.result_label.setText(
                "Error"
            )

            self.confidence.setText(
                str(error)
            )

    def show_image(self, image):
        image = image.copy()

        image.thumbnail(
            (600, 500),
            Image.Resampling.LANCZOS
        )

        image.save(
            "_nova_preview.png"
        )

        pixmap = QPixmap(
            "_nova_preview.png"
        )

        self.preview.setPixmap(
            pixmap
        )

        self.preview.setScaledContents(
            False
        )

    def predict(self, image):
        self.result_label.setText(
            "Analyzing..."
        )

        self.confidence.setText(
            "Running Nova model"
        )

        self.status_icon.setText(
            "◌"
        )

        QApplication.processEvents()

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        ).to(DEVICE)

        with torch.no_grad():
            logits = self.model(
                tensor
            )

            probability = torch.sigmoid(
                logits
            ).item()

        cancer_probability = probability
        normal_probability = 1.0 - probability

        cancer_percent = (
            cancer_probability * 100
        )

        if cancer_probability >= 0.5:
            self.result_label.setText(
                "Cancer detected"
            )

            self.status_icon.setText(
                "●"
            )

            self.status_icon.setStyleSheet(
                "color: #FF453A;"
            )

            self.result_label.setStyleSheet(
                "color: #FF453A;"
            )

        else:
            self.result_label.setText(
                "Non-cancer"
            )

            self.status_icon.setText(
                "●"
            )

            self.status_icon.setStyleSheet(
                "color: #30D158;"
            )

            self.result_label.setStyleSheet(
                "color: #30D158;"
            )

        self.confidence.setText(
            f"Cancer probability  •  "
            f"{cancer_percent:.2f}%"
        )

        self.progress.setValue(
            int(cancer_percent)
        )


def main():
    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "Nova Histopathology"
    )

    window = NovaApp()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()

