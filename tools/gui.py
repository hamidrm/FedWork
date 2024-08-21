import sys
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QGridLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QHBoxLayout, QFileDialog, QMessageBox, QDialog, QFormLayout
)
from PyQt5.QtCore import pyqtSlot

class MethodDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Method")
        self.layout = QFormLayout(self)

        # Method attributes
        self.name = QLineEdit(self)
        self.type = QComboBox(self)
        self.type.addItems(["FedPoll", "FedAvg", "FedPAQ"])
        self.platform = QComboBox(self)
        self.platform.addItems(["cuda", "cpu"])
        self.epochs_num = QSpinBox(self)
        self.epochs_num.setMaximum(1000)
        self.args = QLineEdit(self)

        self.layout.addRow("Method Name:", self.name)
        self.layout.addRow("Type:", self.type)
        self.layout.addRow("Platform:", self.platform)
        self.layout.addRow("Number of Epochs:", self.epochs_num)
        self.layout.addRow("Arguments:", self.args)

        self.buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_add.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.buttons.addWidget(self.btn_add)
        self.buttons.addWidget(self.btn_cancel)
        self.layout.addRow(self.buttons)

    def get_method_details(self):
        return {
            'name': self.name.text(),
            'type': self.type.currentText(),
            'platform': self.platform.currentText(),
            'epochs_num': self.epochs_num.value(),
            'args': self.args.text()
        }

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Federated Learning XML Configuration Generator'
        self.methods = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        mainLayout = QVBoxLayout(self)

        # Section for basic network settings
        self.networkSettingsLayout = QGridLayout()
        mainLayout.addLayout(self.networkSettingsLayout)
        self.setupNetworkSettings()

        # Section for dataset settings
        self.datasetSettingsLayout = QGridLayout()
        mainLayout.addLayout(self.datasetSettingsLayout)
        self.setupDatasetSettings()

        # Buttons for adding methods and generating XML
        self.btnAddMethod = QPushButton("Add Method", self)
        self.btnAddMethod.clicked.connect(self.addMethod)
        self.btnGenerateXML = QPushButton("Generate XML", self)
        self.btnGenerateXML.clicked.connect(self.generateXml)

        mainLayout.addWidget(self.btnAddMethod)
        mainLayout.addWidget(self.btnGenerateXML)

        self.show()

    def setupNetworkSettings(self):
        self.name = QLineEdit()
        self.numNodes = QSpinBox()
        self.port = QSpinBox()
        self.ip = QLineEdit()
        self.networkSettingsLayout.addWidget(QLabel("Name:"), 0, 0)
        self.networkSettingsLayout.addWidget(self.name, 0, 1)
        self.networkSettingsLayout.addWidget(QLabel("Number of Nodes:"), 1, 0)
        self.networkSettingsLayout.addWidget(self.numNodes, 1, 1)
        self.networkSettingsLayout.addWidget(QLabel("Port:"), 2, 0)
        self.networkSettingsLayout.addWidget(self.port, 2, 1)
        self.networkSettingsLayout.addWidget(QLabel("IP:"), 3, 0)
        self.networkSettingsLayout.addWidget(self.ip, 3, 1)

    def setupDatasetSettings(self):
        self.datasetType = QComboBox()
        self.datasetType.addItems(["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"])
        self.heterogeneous = QComboBox()
        self.heterogeneous.addItems(["True", "False"])
        self.nonIidLevel = QDoubleSpinBox()
        self.datasetSettingsLayout.addWidget(QLabel("Dataset Type:"), 0, 0)
        self.datasetSettingsLayout.addWidget(self.datasetType, 0, 1)
        self.datasetSettingsLayout.addWidget(QLabel("Heterogeneous:"), 1, 0)
        self.datasetSettingsLayout.addWidget(self.heterogeneous, 1, 1)
        self.datasetSettingsLayout.addWidget(QLabel("Non-IID Level:"), 2, 0)
        self.datasetSettingsLayout.addWidget(self.nonIidLevel, 2, 1)

    def addMethod(self):
        dialog = MethodDialog(self)
        if dialog.exec():
            method_details = dialog.get_method_details()
            method = ET.Element('method', name=method_details['name'], type=method_details['type'], platform=method_details['platform'])
            ET.SubElement(method, 'epochs_num').text = str(method_details['epochs_num'])
            ET.SubElement(method, 'args').text = method_details['args']
            self.methods.append(method)

    def generateXml(self):
        root = ET.Element('fedwork_cfg')
        for method in self.methods:
            root.append(method)
        file_path = QFileDialog.getSaveFileName(self, 'Save XML File', filter="XML files (*.xml)")[0]
        if file_path:
            tree = ET.ElementTree(root)
            tree.write(file_path)
            QMessageBox.information(self, 'Success', 'XML file has been saved successfully!')

# Run the app
app = QApplication(sys.argv)
ex = MainApp()
sys.exit(app.exec_())
