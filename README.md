## NMT_Basics
This repo deals with Three Basic Models for Neural Machine Translation Task.
The main purpose is to check the developments while comparing each model.
For a fairer comparision, some modifications are applied and as a result, some parts may differ from those in papers.

<br>

### Table of Contents
> &nbsp; **[Model desc](#model-desc)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[Configs](#configurations)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[Results](#results)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[How to Use](#how-to-use)** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **[References](#references)**
</br>


## Model desc

**[Sequence-to-Sequence](https://arxiv.org/abs/1409.3215)**
> As the name **"Sequence-to-Sequence"** suggests, it is an end-to-end sequence model.
The Architecture consists of Encoder and Decoder. In detail, the Encoder first makes Contetx Vectors from Input Sequences. 
And then the Decoder gets Encoder Outputs and Auto Regressive Values from Target sequences as an Input Values to return Target Sequences.
Before Sequence-to-Sequence Architecture was generally applied to various NLP Tasks, Statistical Models outperformed Neural End-to-End models.
This Architecture has proved its significance by opening Neural end-to-end Model Era.

<br>

**[Attention Mechanism](https://arxiv.org/abs/1409.0473)**
> The main idea of Attention Mechanism came from Human's Brain Cognition Process.
People live with a variety of information, but when faced with a specific problem, people usually focus on the information needed to solve the problem. We call this as an **Attention**.
The Architecture also use Encoder-Decoder architecture, but the difference is that the Decoder uses Attention Operation to make predictions.
By using Attention Mechanism, the model could avoid Bottle Neck problem, which results in Better performances in Quantative and Qualitive Evaluation at the same time.

<br>


**[Transformer](https://arxiv.org/abs/1706.03762)**
> Natural Language is inevitably a time-series data. In order to consider the time series aspect, the RNN structure was considered as the only option.
But **Transformer** broke this conventional prejudice and showed remarkable achievements by only using Attention Mechanism without any RNN Layer.
Existing RNN models always had two chronic problems. First is a vanishing gradient problem which is apparent as the sequence length gets longer. Second is Recurrent Operation process itself, which makes parallel processing difficult.
But the Transformer solved these problems only with Attentions. As a result, the architecture not only performs well in a variety of NLP tasks, but is also fast in speed.

<br>
<br>

## Configurations

> **Model Configs**

|  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `Seq2Seq` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `Attention` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `Transformer` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
| :--- | :---: | :---: | :---: |
| **`Input Dimension`** | 10,000 | 10,000 | 10,000 |
| **`Output Dimension`** | 10,000 | 10,000 | 10,000 |
| **`Embedding Dimension`** | 256 | 256 | 256 |
| **`Hidden Dimension`** | 512 | 512 | 512 |
| **`PFF Dimension`** | - | - | 1024 |
| **`N Layers`** | 2 | - | 3 |
| **`N Heads`** | - | - | 8 |
| **`Dropout Ratio`** | 0.5 | 0.5 | 0.1 |
| **`Model Params`** |  |  |  |
| **`Model Size`** |  |  |  |

<br>
<br>

> **Training Configs**

* **Batch Size:** &nbsp; **`32`** </br>
* **Learning Rate:** &nbsp; **`5e-4`** </br>
* **Num of Epochs:** &nbsp; **`10`** </br>
* **Label Smoothing:** &nbsp; **`0.1`** </br>
* **Optimizer:** &nbsp; **`Adam Optimizer`** </br>
* **Loss Function:** &nbsp; **`Cross Entropy Loss`** </br>
* **Tokenizer:** &nbsp; **`SentencePiece BPE Tokenzer`** </br>
* **Data:** &nbsp; **`downsized WMT14 EN-DE dataset (4.5M -> 300K)`** </br>

<br>
<br>

> **Model Parameter Initialization**

Different Initialization applied for Each Models.

* Uniform Initalization applied on Seq2Seq Model
* Normal Initalization applied on Attention Model
* Xarvier Initalization applied on Transformer Model

<center>
  <img src="https://user-images.githubusercontent.com/71929682/168110116-374d3ac9-48d6-41e3-a2ce-d216f2e76422.png" width="80%" height="60%">
</center>


<br>
<br>

## Results

> **Training Results**

<center>
  <img src="https://user-images.githubusercontent.com/71929682/201269096-2cc00b2f-4e8d-4071-945c-f5a3bfbca985.png" width="90%" height="70%">
</center>


</br>

> **Test Results**

</br>
</br>


## How to Use
**First clone git repo in your env**
```
git clone https://github.com/moon23k/NMT_Basic
```

<br>

**Download and Process Dataset via setup.py**
```
cd NMT_Basic
bash setup.py -downsize [True, False] -sort [True, False]
```

<br>

**Train models with "train.py" file (scheduler is optional)**
```
python3 train.py -model ['seq2seq', 'attention', 'transformer'] -scheduler ['constant', 'exponential', 'step', 'cosine_annealing_warm']
```

<br>

**Test trained models with "test.py" file**
```
python3 test.py -model ['seq2seq', 'attention', 'transformer']
```

<br>

**Test with user input sentence via trained models**
```
python3 inference.py -model ['seq2seq', 'attention', 'transformer']
```

<br>
<br>

## References
* [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
