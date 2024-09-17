# XML Configuration File Format for Federated Learning

## Overview

The XML configuration file is used to set up and manage federated learning experiments. It specifies parameters for datasets, methods, local clients, and reporting settings. This document provides a detailed description of the XML structure and the purpose of each element and attribute.

## XML Structure

The XML file consists of several main sections:

1. **`<fedwork_cfg>`**: The root element that encompasses the entire configuration.
2. **`<dataset>`**: Defines dataset-related settings.
3. **`<localclients>`**: Configures local client settings for federated learning.
4. **`<method>`**: Specifies different federated learning methods.
5. **`<report>`**: Outlines the reporting and logging configuration.

### XML Elements and Attributes

#### `<fedwork_cfg>`

- **Attributes**:
  - `name`: A unique name for the configuration (e.g., `"fedpoll_40p_of_30c"`).
  - `num_of_nodes`: Number of nodes in the federated setup (e.g., `"10"`).
  - `net_port`: Port number for network communication (e.g., `"23335"`).
  - `net_ip`: IP address for network communication (e.g., `"127.0.0.1"`).
  - `eval_criterion`: Evaluation criterion for the model (e.g., `"CrossEntropyLoss"`).
  - `num_of_rounds`: Number of training rounds (e.g., `"100"`).

- **Child Elements**:
  - `<dataset>`: Contains dataset configuration.
  - `<localclients>`: Contains local client settings.
  - `<method>`: Defines different methods used in the federated learning experiment.
  - `<report>`: Contains settings for generating reports.

#### `<dataset>`

- **Attributes**:
  - `type`: Specifies the type of dataset (e.g., `"CIFAR10"`).

- **Child Elements**:
  - `<var>`: Defines dataset-related variables.
    - **Attributes**:
      - `name`: Name of the variable (e.g., `"heterogeneous"`).
    - **Text Content**: Value of the variable (e.g., `"True"`).

#### `<localclients>`

- **Attributes**:
  - `learning_rate`: Learning rate for local clients (e.g., `"0.001"`).
  - `momentum`: Momentum for the optimizer (e.g., `"0.9"`).
  - `weight_decay`: Weight decay for the optimizer (e.g., `"1e-7"`).
  - `optimizer`: Type of optimizer used (e.g., `"SGD"`).
  - `platform`: Computational platform (e.g., `"cuda"`).

- **Text Content**:
  - Represents the number of local clients (e.g., `"10"`).

#### `<method>`

- **Attributes**:
  - `name`: Name of the method (e.g., `"FedPollMethod"`).
  - `type`: Type of the method (e.g., `"FedPoll"`).
  - `platform`: Computational platform for the method (e.g., `"cuda"`).

- **Child Elements**:
  - `<arch>`: Defines the architecture of the method.
    - **Attributes**:
      - `type`: Type of the architecture (e.g., `"ResNet18"`).
  - `<var>`: Defines method-related variables.
    - **Attributes**:
      - `name`: Name of the variable (e.g., `"epochs_num"`).
    - **Text Content**: Value of the variable (e.g., `"3"`).

#### `<report>`

- **Attributes**:
  - `save_log`: Indicates whether to save logs (e.g., `"True"`).
  - `log_over_net`: Network address and port for logging (e.g., `"192.168.166.143,23456"`).

- **Child Elements**:
  - `<fig>`: Defines figures to be generated in the report.
    - **Attributes**:
      - `style`: Style and appearance settings for the plot (e.g., `"style=bmh;colors=black,green,blue,brown,black;linestyles=--,:,-.,-;linewidths=1,1,1,1,1;"`).
      - `name`: Name of the figure.
      - `caption`: Caption for the figure.
      - `x_axis`: Type of x-axis (`"round"` or `"time"`).
      - `y_axis`: Type of y-axis (e.g., `"EvaluationAccuracy"`).
      - `x_axis_title`: Title for the x-axis (e.g., `"# rounds"`).
      - `y_axis_title`: Title for the y-axis (e.g., `"Accuracy"`).
      - `methods`: Comma-separated list of methods to include in the figure (e.g., `"FedPollMethod,SCAFFOLDMethod,FedYogiMethod"`).
      - `labels`: Labels for different series in the figure (e.g., `"FedPoll-MaxMin,SCAFFOLD,FedYogi"`).

## Example XML Configuration

Here is an example of the XML configuration file:

```xml
<fedwork_cfg name="fedpoll_40p_of_30c" num_of_nodes="10" net_port="23335" net_ip="127.0.0.1" eval_criterion="CrossEntropyLoss" num_of_rounds="100">
    <dataset type="CIFAR10">
        <var name="heterogeneous">True</var>
        <var name="non_iid_level">0.4</var>
        <var name="train_batch_size">128</var>
        <var name="test_batch_size">256</var>
        <var name="num_workers">0</var>
        <var name="save_graph">True</var>
        <var name="enclosed_info">True</var>
    </dataset>

    <localclients learning_rate="0.001" momentum="0.9" weight_decay="1e-7" optimizer="SGD" platform="cuda">10</localclients>
    
    <method name="FedPollMethod" type="FedPoll" platform="cuda">
        <arch type="ResNet18"/>
        <var name="epochs_num">3</var>
        <var name="args">no_r=8,epsilon=0.001,contributors_percent=50</var>
    </method>
    
    <method name="SCAFFOLDMethod" type="SCAFFOLD" platform="cuda">
        <arch type="ResNet18"/>
        <var name="epochs_num">3</var>
        <var name="args">contributors_percent=50</var>
    </method>
    
    <method name="FedYogiMethod" type="FedYogi" platform="cuda">
        <arch type="ResNet18"/>
        <var name="epochs_num">3</var>
        <var name="args">beta1=0.9,beta2=0.99,contributors_percent=50,epsilon=1e-8,eta=0.001</var>
    </method>
    
    <report save_log="True" log_over_net="192.168.166.143,23456">
        <fig style="style=bmh;colors=black,green,blue;linestyles=--,:,-.;linewidths=1,1,1;" name="accu" caption=" " x_axis="round" y_axis="EvaluationAccuracy" y_axis_title="Accuracy" x_axis_title="# rounds" methods="FedPollMethod,SCAFFOLDMethod,FedYogiMethod" labels="FedPoll-MaxMin,SCAFFOLD,FedYogi"/>
        <fig style="style=bmh;colors=black,green,blue;linestyles=--,:,-.;linewidths=1,1,1;" name="loss" caption=" " x_axis="time" y_axis="EvaluationLoss" y_axis_title="Loss" x_axis_title="time" methods="FedPollMethod,SCAFFOLDMethod,FedYogiMethod" labels="FedPoll-MaxMin,SCAFFOLD,FedYogi"/>
    </report>
</fedwork_cfg>
